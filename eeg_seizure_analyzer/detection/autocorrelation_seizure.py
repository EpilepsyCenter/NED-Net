"""Autocorrelation-based seizure detection.

Based on White et al. (2006, J. Neuroscience Methods).
Combines two complementary signals:

1. **Range autocorrelation**: compresses signal into min/max sub-windows
   and measures how much consecutive ranges overlap.  High overlap =
   rhythmic, repetitive activity characteristic of seizures.

2. **Spike frequency**: counts detected spikes per window.  Seizures
   have a sustained high spike rate (typically ≥2 Hz).

An analysis window is flagged as ictal when BOTH metrics exceed their
respective thresholds.  Consecutive flagged windows are grouped into
events.

This method achieved 96% PPV and 100% sensitivity on 75 seizures
across 8 rats in the original paper.
"""

from __future__ import annotations

import numpy as np

from eeg_seizure_analyzer.config import AutocorrelationParams
from eeg_seizure_analyzer.detection.base import DetectedEvent, DetectorBase
from eeg_seizure_analyzer.detection.boundary_utils import (
    refine_signal_rms,
    refine_spike_density,
)
from eeg_seizure_analyzer.detection.spike_utils import (
    Spike,
    compute_baseline,
    detect_spikes,
)
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


def _build_acorr_features(
    event_spikes: list[Spike],
    seg_spike_freq: list[float],
    seg_acorr: list[float],
    acorr_threshold: float,
    bl_mean: float,
    bl_std: float,
    threshold: float,
) -> dict:
    """Build features dict for an autocorrelation event."""
    # ISI stats
    isis_ms = []
    for i in range(len(event_spikes) - 1):
        isi = (event_spikes[i + 1].time_sec - event_spikes[i].time_sec) * 1000
        isis_ms.append(isi)
    mean_isi = float(np.mean(isis_ms)) if isis_ms else 0.0
    isi_cv = float(np.std(isis_ms) / np.mean(isis_ms)) if isis_ms and np.mean(isis_ms) > 0 else 0.0

    raw_amps = [s.amplitude for s in event_spikes]

    return {
        "detection_method": "autocorrelation",
        "seizure_subtype": "seizure",
        "n_spikes": len(event_spikes),
        "mean_spike_frequency_hz": round(float(np.mean(seg_spike_freq)), 2),
        "mean_isi_ms": round(mean_isi, 2),
        "spike_regularity": round(isi_cv, 3),
        "mean_amplitude_uv": round(float(np.mean(raw_amps)), 2) if raw_amps else 0.0,
        "max_amplitude_uv": round(float(np.max(raw_amps)), 2) if raw_amps else 0.0,
        "max_amplitude_x_baseline": round(float(np.max([s.amplitude for s in event_spikes]) / bl_mean), 2) if event_spikes and bl_mean > 0 else 0.0,
        "peak_acorr": round(float(np.max(seg_acorr)), 4),
        "mean_acorr": round(float(np.mean(seg_acorr)), 4),
        "acorr_threshold": round(acorr_threshold, 4),
        "spike_times": [s.time_sec for s in event_spikes],
        "spike_amplitudes": [s.amplitude for s in event_spikes],
        "spike_samples": [s.sample_idx for s in event_spikes],
        "baseline_mean": bl_mean,
        "baseline_std": bl_std,
        "threshold": threshold,
    }


class AutocorrelationDetector(DetectorBase):
    """Detect seizures using range autocorrelation + spike frequency."""

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        params: AutocorrelationParams | None = None,
        baseline_rms: float | None = None,
        baseline_std: float | None = None,
        **_kwargs,
    ) -> list[DetectedEvent]:
        if params is None:
            params = AutocorrelationParams()

        data = recording.get_channel_data(channel)
        fs = recording.fs

        # Step 1: bandpass filter
        filtered = bandpass_filter(data, fs, params.bandpass_low, params.bandpass_high)

        # Step 2: compute baseline
        if baseline_rms is not None:
            bl_mean = baseline_rms
            bl_std = baseline_std if baseline_std is not None else baseline_rms * 0.5
        else:
            bl_mean, bl_std = compute_baseline(
                filtered, fs,
                method=params.baseline_method,
                percentile=params.baseline_percentile,
                rms_window_sec=params.baseline_rms_window_sec,
            )

        # Step 3: detect individual spikes (reuse shared utility)
        spikes = detect_spikes(
            filtered, fs, bl_mean, bl_std,
            spike_amplitude_x_baseline=params.spike_amplitude_x_baseline,
            spike_min_amplitude_uv=params.spike_min_amplitude_uv,
            spike_refractory_ms=params.spike_refractory_ms,
            spike_prominence_x_baseline=params.spike_prominence_x_baseline,
            spike_max_width_ms=params.spike_max_width_ms,
            spike_min_width_ms=params.spike_min_width_ms,
        )

        # Compute threshold (same formula as spike detection)
        if bl_std > 0:
            threshold = bl_mean + params.spike_amplitude_x_baseline * bl_std
        else:
            threshold = params.spike_amplitude_x_baseline * bl_mean

        # Store spike detection info for inspector compatibility
        self._last_detection_info = {
            "channel": channel,
            "baseline_mean": bl_mean,
            "baseline_std": bl_std,
            "threshold": threshold,
            "all_spike_times": [s.time_sec for s in spikes],
            "all_spike_amplitudes": [s.amplitude for s in spikes],
            "all_spike_samples": [s.sample_idx for s in spikes],
        }

        # Step 4: compute range-autocorrelation metric per analysis window
        acorr_window_samples = int(params.acorr_window_sec * fs)
        acorr_step_samples = max(1, int(params.acorr_step_sec * fs))
        sub_n = params.subwindow_points
        look_n = params.lookahead_points

        n_windows = max(1, (len(filtered) - acorr_window_samples) // acorr_step_samples + 1)

        acorr_metrics = np.zeros(n_windows)
        spike_freqs = np.zeros(n_windows)
        window_times = np.zeros(n_windows)

        spike_times_arr = np.array([s.time_sec for s in spikes]) if spikes else np.array([])

        for i in range(n_windows):
            w_start = i * acorr_step_samples
            w_end = w_start + acorr_window_samples
            if w_end > len(filtered):
                w_end = len(filtered)
            window = filtered[w_start:w_end]
            t_start = w_start / fs
            t_end = w_end / fs
            window_times[i] = (t_start + t_end) / 2

            # Range-autocorrelation (White et al. metric 3)
            acorr_metrics[i] = _compute_range_autocorrelation(
                window, sub_n, look_n
            )

            # Spike frequency in this window
            if len(spike_times_arr) > 0:
                mask = (spike_times_arr >= t_start) & (spike_times_arr < t_end)
                n_spikes_in_window = int(np.sum(mask))
            else:
                n_spikes_in_window = 0
            spike_freqs[i] = n_spikes_in_window / max(params.acorr_window_sec, 1e-6)

        # Store autocorrelation timeseries for inspector
        self._last_detection_info["acorr_times"] = window_times.tolist()
        self._last_detection_info["acorr_values"] = acorr_metrics.tolist()
        self._last_detection_info["spike_freqs"] = spike_freqs.tolist()

        # Step 5: threshold both metrics
        # Autocorrelation baseline from quiet percentile
        acorr_sorted = np.sort(acorr_metrics)
        n_baseline = max(1, int(len(acorr_sorted) * (100 - params.baseline_percentile) / 100))
        baseline_acorr = acorr_sorted[:n_baseline]
        acorr_mean = float(np.mean(baseline_acorr))
        acorr_std = float(np.std(baseline_acorr))
        if acorr_std < 1e-12:
            acorr_std = acorr_mean * 0.1

        acorr_threshold = acorr_mean + params.acorr_threshold_z * acorr_std

        self._last_detection_info["acorr_threshold"] = acorr_threshold

        # Both conditions must hold
        ictal_mask = (acorr_metrics > acorr_threshold) & (spike_freqs >= params.min_spike_freq_hz)

        # Step 6: group consecutive flagged windows into events
        segments = _contiguous_segments(ictal_mask)

        events: list[DetectedEvent] = []

        for seg_start, seg_end in segments:
            onset_sec = float(window_times[seg_start] - params.acorr_window_sec / 2)
            offset_sec = float(window_times[min(seg_end, n_windows - 1)] + params.acorr_window_sec / 2)
            onset_sec = max(0.0, onset_sec)
            offset_sec = min(len(filtered) / fs, offset_sec)
            duration = offset_sec - onset_sec

            if duration < params.min_duration_sec:
                continue

            seg_acorr = acorr_metrics[seg_start : seg_end + 1]
            seg_spike_freq = spike_freqs[seg_start : seg_end + 1]

            # Spikes within this event
            event_spikes = [
                s for s in spikes
                if onset_sec <= s.time_sec <= offset_sec
            ]

            confidence = min(
                1.0,
                (float(np.mean(seg_acorr)) - acorr_threshold)
                / max(acorr_std, 1e-12) / 5.0 + 0.5,
            )

            events.append(
                DetectedEvent(
                    onset_sec=round(onset_sec, 4),
                    offset_sec=round(offset_sec, 4),
                    duration_sec=round(duration, 4),
                    channel=channel,
                    event_type="seizure",
                    confidence=round(max(0.1, confidence), 3),
                    severity=_classify_severity(duration),
                    features=_build_acorr_features(
                        event_spikes, seg_spike_freq, seg_acorr,
                        acorr_threshold, bl_mean, bl_std, threshold,
                    ),
                )
            )

        # Merge close events
        events = _merge_events(events, params.merge_gap_sec)

        # ── Boundary refinement ────────────────────────────────────
        if params.boundary_method != "none" and events:
            refined = []
            for ev in events:
                if params.boundary_method == "signal":
                    onset_s = int(ev.onset_sec * fs)
                    offset_s = int(ev.offset_sec * fs)
                    # Use first/last spike as anchors if available
                    ev_spikes = [s for s in spikes
                                 if ev.onset_sec <= s.time_sec <= ev.offset_sec]
                    anchor_on = ev_spikes[0].sample_idx if ev_spikes else onset_s
                    anchor_off = ev_spikes[-1].sample_idx if ev_spikes else offset_s
                    r_onset, r_offset = refine_signal_rms(
                        ev.onset_sec, ev.offset_sec,
                        filtered, fs, bl_mean,
                        rms_window_ms=params.boundary_rms_window_ms,
                        rms_threshold_x=params.boundary_rms_threshold_x,
                        max_trim_sec=params.boundary_max_trim_sec,
                        anchor_onset_sample=anchor_on,
                        anchor_offset_sample=anchor_off,
                    )
                    dur = r_offset - r_onset
                    if dur >= params.min_duration_sec:
                        ev.onset_sec = round(r_onset, 4)
                        ev.offset_sec = round(r_offset, 4)
                        ev.duration_sec = round(dur, 4)
                        refined.append(ev)

                elif params.boundary_method == "spike_density":
                    ev_spikes = [s for s in spikes
                                 if ev.onset_sec <= s.time_sec <= ev.offset_sec]
                    result = refine_spike_density(
                        ev_spikes, ev.onset_sec, ev.offset_sec,
                        boundary_window_sec=params.boundary_window_sec,
                        min_rate_hz=params.boundary_min_rate_hz,
                        min_amplitude_x=params.boundary_min_amplitude_x,
                    )
                    if result is not None:
                        r_onset, r_offset = result
                        dur = r_offset - r_onset
                        if dur >= params.min_duration_sec:
                            ev.onset_sec = round(r_onset, 4)
                            ev.offset_sec = round(r_offset, 4)
                            ev.duration_sec = round(dur, 4)
                            refined.append(ev)
                else:
                    refined.append(ev)
            events = refined

        return sorted(events, key=lambda e: e.onset_sec)


# ── Helpers ─────────────────────────────────────────────────────────


def _compute_range_autocorrelation(
    window: np.ndarray, sub_n: int, look_n: int
) -> float:
    """Compute the range-overlap autocorrelation metric (White et al. metric 3).

    For each group of ``sub_n`` consecutive points, compute its (min, max)
    range.  Then compute the overlap between that range and the range of
    the next ``look_n`` points.  Sum the overlaps across the window.

    High values indicate rhythmic, correlated activity (seizure).
    Low values indicate isolated spikes or random noise.
    """
    n = len(window)
    if n < sub_n + look_n:
        return 0.0

    n_groups = (n - look_n) // sub_n
    if n_groups < 1:
        return 0.0

    total_overlap = 0.0

    for g in range(n_groups):
        idx = g * sub_n
        # Current sub-window range
        seg = window[idx : idx + sub_n]
        seg_min = float(np.min(seg))
        seg_max = float(np.max(seg))

        # Lookahead range (next look_n points)
        look_start = idx + sub_n
        look_end = min(look_start + look_n, n)
        look_seg = window[look_start:look_end]
        look_min = float(np.min(look_seg))
        look_max = float(np.max(look_seg))

        # Overlap: min of maxes - max of mins
        hv = min(seg_max, look_max)
        lv = max(seg_min, look_min)
        overlap = max(0.0, hv - lv)
        total_overlap += overlap

    return total_overlap


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True segments → list of (start_idx, end_idx) inclusive."""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask) - 1))
    return segments


def _classify_severity(duration: float) -> str:
    if duration > 45:
        return "severe"
    elif duration > 15:
        return "moderate"
    return "mild"


def _merge_events(
    events: list[DetectedEvent], merge_gap_sec: float
) -> list[DetectedEvent]:
    """Merge events separated by less than merge_gap_sec."""
    if len(events) <= 1:
        return events

    events = sorted(events, key=lambda e: e.onset_sec)
    merged = [events[0]]

    for ev in events[1:]:
        prev = merged[-1]
        if ev.channel == prev.channel and ev.onset_sec - prev.offset_sec < merge_gap_sec:
            prev.offset_sec = max(prev.offset_sec, ev.offset_sec)
            prev.duration_sec = prev.offset_sec - prev.onset_sec
            prev.confidence = max(prev.confidence, ev.confidence)
            prev.severity = _classify_severity(prev.duration_sec)
            prev.features["n_spikes"] = (
                prev.features.get("n_spikes", 0) + ev.features.get("n_spikes", 0)
            )
            # Merge spike lists
            for key in ("spike_times", "spike_amplitudes", "spike_samples"):
                prev.features[key] = prev.features.get(key, []) + ev.features.get(key, [])
        else:
            merged.append(ev)

    return merged
