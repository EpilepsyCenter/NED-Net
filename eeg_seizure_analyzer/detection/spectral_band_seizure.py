"""Spectral-band seizure detection.

Based on Casillas-Espinosa et al. (2019, Epilepsia) — the "ASSYST" approach.
Seizures across rodent epilepsy models share a strong spectral peak in the
17–25 Hz band that is absent during interictal EEG.

Algorithm
---------
1. Compute band power in the target band (default 17–25 Hz) using Welch's
   method over a sliding window.
2. Compute band power in a reference band (default 1–50 Hz) for
   normalisation.
3. Spectral Band Index (SBI) = target_power / reference_power.
4. Compute baseline SBI distribution from the recording and threshold at
   ``mean + z × std`` (quiet percentile).
5. Group consecutive above-threshold windows into candidate events.
6. Apply minimum duration and merge close events.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from eeg_seizure_analyzer.config import SpectralBandParams
from eeg_seizure_analyzer.detection.base import DetectedEvent, DetectorBase
from eeg_seizure_analyzer.detection.boundary_utils import refine_signal_rms
from eeg_seizure_analyzer.detection.spike_utils import compute_baseline
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


class SpectralBandDetector(DetectorBase):
    """Detect seizures by monitoring power in a narrow spectral band."""

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        params: SpectralBandParams | None = None,
        **_kwargs,
    ) -> list[DetectedEvent]:
        if params is None:
            params = SpectralBandParams()

        data = recording.get_channel_data(channel)
        fs = recording.fs

        # Light bandpass to remove DC and very high frequency noise
        filtered = bandpass_filter(data, fs, params.ref_band_low, params.ref_band_high)

        # ── Compute SBI timeseries ──────────────────────────────────
        window_samples = int(params.window_sec * fs)
        step_samples = max(1, int(params.step_sec * fs))
        nperseg = min(window_samples, int(fs))  # Welch segment length

        n_windows = max(1, (len(filtered) - window_samples) // step_samples + 1)

        sbi_values = np.zeros(n_windows)
        sbi_times = np.zeros(n_windows)
        target_powers = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            if end > len(filtered):
                end = len(filtered)
            segment = filtered[start:end]

            sbi_times[i] = (start + window_samples // 2) / fs  # window centre

            freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), nperseg))

            # Target band power
            target_mask = (freqs >= params.band_low) & (freqs <= params.band_high)
            if np.any(target_mask):
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                target_power = float(np.trapezoid(psd[target_mask], dx=freq_res))
            else:
                target_power = 0.0

            # Reference band power (for normalisation)
            ref_mask = (freqs >= params.ref_band_low) & (freqs <= params.ref_band_high)
            if np.any(ref_mask):
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                ref_power = float(np.trapezoid(psd[ref_mask], dx=freq_res))
            else:
                ref_power = 1e-12

            target_powers[i] = target_power
            sbi_values[i] = target_power / max(ref_power, 1e-12)

        # ── Baseline & threshold ────────────────────────────────────
        if params.baseline_method == "first_n":
            # Use first 5 min
            n_baseline = max(1, int(5 * 60 / params.step_sec))
            baseline_sbi = sbi_values[:n_baseline]
        else:
            # Percentile: use quiet windows
            cutoff = float(np.percentile(sbi_values, 100 - params.baseline_percentile))
            baseline_sbi = sbi_values[sbi_values <= cutoff]
            if len(baseline_sbi) < 3:
                baseline_sbi = sbi_values

        sbi_mean = float(np.mean(baseline_sbi))
        sbi_std = float(np.std(baseline_sbi))
        if sbi_std < 1e-12:
            sbi_std = sbi_mean * 0.1

        threshold = sbi_mean + params.threshold_z * sbi_std

        # ── Store detection metadata ────────────────────────────────
        self._last_detection_info = {
            "channel": channel,
            "sbi_times": sbi_times.tolist(),
            "sbi_values": sbi_values.tolist(),
            "target_powers": target_powers.tolist(),
            "sbi_mean": sbi_mean,
            "sbi_std": sbi_std,
            "threshold": threshold,
            "baseline_mean": sbi_mean,
            "baseline_std": sbi_std,
            # Empty spike lists for compatibility with spike-train inspector
            "all_spike_times": [],
            "all_spike_amplitudes": [],
            "all_spike_samples": [],
        }

        # ── Threshold → candidate segments ──────────────────────────
        above = sbi_values > threshold

        segments = _contiguous_segments(above)

        # ── Build events ────────────────────────────────────────────
        events: list[DetectedEvent] = []

        for seg_start, seg_end in segments:
            onset_sec = float(sbi_times[seg_start])
            offset_sec = float(sbi_times[min(seg_end, n_windows - 1)])
            duration = offset_sec - onset_sec

            if duration < params.min_duration_sec:
                continue

            seg_sbi = sbi_values[seg_start : seg_end + 1]
            peak_sbi = float(np.max(seg_sbi))
            mean_sbi = float(np.mean(seg_sbi))

            # Confidence: how far above threshold (capped at 1.0)
            confidence = min(1.0, (mean_sbi - threshold) / max(sbi_std, 1e-12) / 5.0 + 0.5)

            events.append(
                DetectedEvent(
                    onset_sec=round(onset_sec, 4),
                    offset_sec=round(offset_sec, 4),
                    duration_sec=round(duration, 4),
                    channel=channel,
                    event_type="seizure",
                    confidence=round(confidence, 3),
                    severity=_classify_severity(duration),
                    features={
                        "detection_method": "spectral_band",
                        "seizure_subtype": "seizure",
                        "sbi_peak": round(peak_sbi, 4),
                        "sbi_mean": round(mean_sbi, 4),
                        "sbi_threshold": round(threshold, 4),
                        "band_hz": f"{params.band_low}-{params.band_high}",
                    },
                )
            )

        # ── Merge close events ──────────────────────────────────────
        events = _merge_events(events, params.merge_gap_sec)

        # ── Boundary refinement ────────────────────────────────────
        if params.boundary_method == "signal" and events:
            # Compute signal-level baseline for RMS reference
            sig_bl_mean, _ = compute_baseline(
                filtered, fs,
                method="percentile",
                percentile=params.baseline_percentile,
            )
            refined = []
            for ev in events:
                onset_s = int(ev.onset_sec * fs)
                offset_s = int(ev.offset_sec * fs)
                r_onset, r_offset = refine_signal_rms(
                    ev.onset_sec, ev.offset_sec,
                    filtered, fs, sig_bl_mean,
                    rms_window_ms=params.boundary_rms_window_ms,
                    rms_threshold_x=params.boundary_rms_threshold_x,
                    max_trim_sec=params.boundary_max_trim_sec,
                    anchor_onset_sample=onset_s,
                    anchor_offset_sample=offset_s,
                )
                dur = r_offset - r_onset
                if dur >= params.min_duration_sec:
                    ev.onset_sec = round(r_onset, 4)
                    ev.offset_sec = round(r_offset, 4)
                    ev.duration_sec = round(dur, 4)
                    refined.append(ev)
            events = refined

        return sorted(events, key=lambda e: e.onset_sec)


# ── Helpers ─────────────────────────────────────────────────────────


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
            # Keep max SBI values
            prev.features["sbi_peak"] = max(
                prev.features.get("sbi_peak", 0), ev.features.get("sbi_peak", 0)
            )
        else:
            merged.append(ev)

    return merged
