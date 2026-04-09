"""Spike-train-based seizure detection.

Implements the criteria described in Twele et al. (2017) for the
intrahippocampal kainate mouse model:

1. Detect individual spikes exceeding N× baseline amplitude.
2. Group spikes into trains based on inter-spike interval.
3. Classify each train as:
   - **HVSW** (high-voltage sharp wave): monomorphic, ≥3×baseline, ≥2 Hz, ≥5 s
   - **HPD** (hippocampal paroxysmal discharge): evolving pattern, ≥2×baseline,
     evolved phase ≥5 Hz, typically >20 s
   - **Electroclinical/convulsive**: very high amplitude, long duration,
     dramatic frequency evolution, often followed by post-ictal suppression
   - **unclassified**: trains meeting basic thresholds but not fitting the
     above categories

All amplitude thresholds are expressed as multiples of the recording's
robust baseline (median amplitude), so they adapt automatically to different
noise floors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_seizure_analyzer.config import SpikeTrainSeizureParams
from eeg_seizure_analyzer.detection.base import DetectedEvent, DetectorBase
from eeg_seizure_analyzer.detection.boundary_utils import (
    refine_signal_rms,
    refine_spike_density,
)
from eeg_seizure_analyzer.detection.spike_utils import (
    Spike as _Spike,
    compute_baseline,
    detect_spikes,
)
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


# ── Internal helpers ────────────────────────────────────────────────────


@dataclass
class _SpikeTrain:
    """A grouped train of spikes."""
    spikes: list           # list of _Spike
    onset_sec: float
    offset_sec: float
    duration_sec: float
    mean_amplitude_x: float
    max_amplitude_x: float
    mean_frequency_hz: float   # mean spike rate within train
    isi_cv: float              # coefficient of variation of inter-spike intervals
    amplitude_trend: float     # slope of amplitude over time (+ = increasing)
    frequency_trend: float     # slope of ISI over time (- = accelerating)


class SpikeTrainSeizureDetector(DetectorBase):
    """Detect seizures by finding and classifying spike trains.

    Parameters are provided via ``SpikeTrainSeizureParams``.
    """

    # ── Public API ──────────────────────────────────────────────────

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        params: SpikeTrainSeizureParams | None = None,
        baseline_rms: float | None = None,
        baseline_std: float | None = None,
    ) -> list[DetectedEvent]:
        if params is None:
            params = SpikeTrainSeizureParams()

        data = recording.get_channel_data(channel)
        fs = recording.fs

        # Step 1: bandpass filter
        filtered = bandpass_filter(data, fs, params.bandpass_low, params.bandpass_high)

        # Step 2: compute baseline (mean, std)
        if baseline_rms is not None:
            # Precomputed baseline (from chunked pipeline)
            bl_mean = baseline_rms
            bl_std = baseline_std if baseline_std is not None else baseline_rms * 0.5
        else:
            bl_mean, bl_std = self._compute_baseline(filtered, fs, params)
        # baseline_amp is the reference amplitude for relative thresholds
        # (amplitude_x, boundary refinement, classification)
        baseline_amp = bl_mean

        # Compute threshold (same formula as _detect_spikes)
        if bl_std > 0:
            threshold = bl_mean + params.spike_amplitude_x_baseline * bl_std
        else:
            threshold = params.spike_amplitude_x_baseline * bl_mean
        if params.spike_min_amplitude_uv > 0:
            threshold = max(threshold, params.spike_min_amplitude_uv)

        # Step 3: detect individual spikes
        spikes = self._detect_spikes(filtered, fs, baseline_amp, params,
                                     baseline_std=bl_std)

        # Store detection metadata for visualization
        # (all spikes, baseline, threshold — accessible via last_detection_info)
        self._last_detection_info = {
            "channel": channel,
            "baseline_mean": bl_mean,
            "baseline_std": bl_std,
            "threshold": threshold,
            "all_spike_times": [s.time_sec for s in spikes],
            "all_spike_amplitudes": [s.amplitude for s in spikes],
            "all_spike_samples": [s.sample_idx for s in spikes],
        }

        if len(spikes) < params.min_train_spikes:
            return []

        # Step 4: group spikes into trains
        trains = self._group_into_trains(spikes, params)

        # Step 5: refine boundaries
        trains = [
            self._refine_boundaries(t, filtered, fs, baseline_amp, params)
            for t in trains
        ]
        trains = [t for t in trains if t is not None]

        # Step 6: classify each train (or skip classification)
        events = []
        for train in trains:
            if params.classify_subtypes:
                event = self._classify_train(
                    train, filtered, fs, baseline_amp, channel, params
                )
            else:
                event = self._train_to_event(train, channel)
            if event is not None:
                # Store spike positions within this event for visualization
                event_spikes = [
                    s for s in spikes
                    if event.onset_sec <= s.time_sec <= event.offset_sec
                ]
                event.features["spike_times"] = [s.time_sec for s in event_spikes]
                event.features["spike_amplitudes"] = [s.amplitude for s in event_spikes]
                event.features["spike_samples"] = [s.sample_idx for s in event_spikes]
                event.features["baseline_mean"] = bl_mean
                event.features["baseline_std"] = bl_std
                event.features["threshold"] = threshold
                events.append(event)

        # Step 7: merge events closer than min_interevent_interval
        events = self._merge_close_events(events, params.min_interevent_interval_sec)

        return sorted(events, key=lambda e: e.onset_sec)

    # ── Step 2: baseline ────────────────────────────────────────────

    @staticmethod
    def _compute_baseline(
        data: np.ndarray, fs: float, params: SpikeTrainSeizureParams
    ) -> tuple[float, float]:
        """Compute baseline via shared utility.  See ``spike_utils.compute_baseline``."""
        return compute_baseline(
            data, fs,
            method=params.baseline_method,
            percentile=params.baseline_percentile,
            rms_window_sec=params.baseline_rms_window_sec,
            rolling_lookback_sec=getattr(params, "rolling_lookback_sec", 1800.0),
            rolling_step_sec=getattr(params, "rolling_step_sec", 300.0),
        )

    # ── Step 3: spike detection ─────────────────────────────────────

    @staticmethod
    def _detect_spikes(
        filtered: np.ndarray,
        fs: float,
        baseline_amp: float,
        params: SpikeTrainSeizureParams,
        baseline_std: float = 0.0,
    ) -> list[_Spike]:
        """Detect spikes via shared utility.  See ``spike_utils.detect_spikes``."""
        return detect_spikes(
            filtered, fs, baseline_amp, baseline_std,
            spike_amplitude_x_baseline=params.spike_amplitude_x_baseline,
            spike_min_amplitude_uv=params.spike_min_amplitude_uv,
            spike_refractory_ms=params.spike_refractory_ms,
            spike_prominence_x_baseline=params.spike_prominence_x_baseline,
            spike_min_prominence_uv=getattr(params, "spike_min_prominence_uv", 0.0),
            spike_max_width_ms=params.spike_max_width_ms,
            spike_min_width_ms=params.spike_min_width_ms,
        )

    # ── Step 4: grouping ────────────────────────────────────────────

    def _group_into_trains(
        self,
        spikes: list[_Spike],
        params: SpikeTrainSeizureParams,
    ) -> list[_SpikeTrain]:
        """Group spikes into trains based on max inter-spike interval."""
        max_isi_sec = params.max_interspike_interval_ms / 1000.0
        trains: list[_SpikeTrain] = []

        current_group: list[_Spike] = [spikes[0]]

        for i in range(1, len(spikes)):
            gap = spikes[i].time_sec - spikes[i - 1].time_sec
            if gap <= max_isi_sec:
                current_group.append(spikes[i])
            else:
                # Finish current group
                train = self._make_train(current_group, params)
                if train is not None:
                    trains.append(train)
                current_group = [spikes[i]]

        # Last group
        train = self._make_train(current_group, params)
        if train is not None:
            trains.append(train)

        return trains

    def _make_train(
        self, spikes: list[_Spike], params: SpikeTrainSeizureParams
    ) -> _SpikeTrain | None:
        """Create a SpikeTrain from a group of spikes, or None if too short."""
        if len(spikes) < params.min_train_spikes:
            return None

        onset = spikes[0].time_sec
        offset = spikes[-1].time_sec
        duration = offset - onset

        if duration < params.min_train_duration_sec:
            return None

        # Inter-spike intervals
        isis = [spikes[i + 1].time_sec - spikes[i].time_sec for i in range(len(spikes) - 1)]
        isis_arr = np.array(isis)
        mean_isi = float(np.mean(isis_arr)) if len(isis_arr) > 0 else 1.0
        isi_cv = float(np.std(isis_arr) / mean_isi) if mean_isi > 1e-10 else 0.0
        mean_freq = 1.0 / mean_isi if mean_isi > 1e-10 else 0.0

        # Amplitude stats
        amps = [s.amplitude_x for s in spikes]
        mean_amp_x = float(np.mean(amps))
        max_amp_x = float(np.max(amps))

        # Trends: fit simple linear regression over time
        times = np.array([s.time_sec for s in spikes])
        times_norm = times - times[0]

        # Amplitude trend
        if len(times_norm) > 2 and np.std(times_norm) > 1e-10:
            amp_trend = float(np.polyfit(times_norm, amps, 1)[0])
        else:
            amp_trend = 0.0

        # Frequency trend (slope of ISI → negative = accelerating)
        if len(isis_arr) > 2 and np.std(times_norm[:-1]) > 1e-10:
            freq_trend = float(np.polyfit(times_norm[:-1], isis, 1)[0])
        else:
            freq_trend = 0.0

        return _SpikeTrain(
            spikes=spikes,
            onset_sec=onset,
            offset_sec=offset,
            duration_sec=duration,
            mean_amplitude_x=mean_amp_x,
            max_amplitude_x=max_amp_x,
            mean_frequency_hz=mean_freq,
            isi_cv=isi_cv,
            amplitude_trend=amp_trend,
            frequency_trend=freq_trend,
        )

    # ── Step 5: boundary refinement ────────────────────────────────

    def _refine_boundaries(
        self,
        train: _SpikeTrain,
        filtered: np.ndarray,
        fs: float,
        baseline_amp: float,
        params: SpikeTrainSeizureParams,
    ) -> _SpikeTrain | None:
        """Refine train onset/offset using the selected method."""
        method = params.boundary_method

        if method == "none":
            return train
        elif method == "spike_density":
            return self._refine_spike_density(train, params)
        elif method == "signal":
            return self._refine_signal(train, filtered, fs, baseline_amp, params)
        else:
            return train

    def _refine_spike_density(
        self,
        train: _SpikeTrain,
        params: SpikeTrainSeizureParams,
    ) -> _SpikeTrain | None:
        """Trim edges using local spike rate + amplitude criteria."""
        spikes = train.spikes
        if len(spikes) < params.min_train_spikes:
            return None

        result = refine_spike_density(
            spikes, train.onset_sec, train.offset_sec,
            boundary_window_sec=params.boundary_window_sec,
            min_rate_hz=params.boundary_min_rate_hz,
            min_amplitude_x=params.boundary_min_amplitude_x,
        )
        if result is None:
            return None

        r_onset, r_offset = result
        trimmed = [s for s in spikes if r_onset <= s.time_sec <= r_offset]
        if len(trimmed) < params.min_train_spikes:
            return None
        return self._make_train(trimmed, params)

    def _refine_signal(
        self,
        train: _SpikeTrain,
        filtered: np.ndarray,
        fs: float,
        baseline_amp: float,
        params: SpikeTrainSeizureParams,
    ) -> _SpikeTrain | None:
        """Refine boundaries using the raw signal RMS envelope."""
        spikes = train.spikes
        if len(spikes) < params.min_train_spikes:
            return None

        onset_sec, offset_sec = refine_signal_rms(
            train.onset_sec, train.offset_sec,
            filtered, fs, baseline_amp,
            rms_window_ms=params.boundary_rms_window_ms,
            rms_threshold_x=params.boundary_rms_threshold_x,
            max_trim_sec=params.boundary_max_trim_sec,
            anchor_onset_sample=spikes[0].sample_idx,
            anchor_offset_sample=spikes[-1].sample_idx,
        )

        trimmed = [s for s in spikes if onset_sec <= s.time_sec <= offset_sec]

        if len(trimmed) < params.min_train_spikes:
            return None

        new_train = self._make_train(trimmed, params)
        if new_train is None:
            return None

        new_train.onset_sec = onset_sec
        new_train.offset_sec = offset_sec
        new_train.duration_sec = offset_sec - onset_sec
        return new_train

    # ── Step 6: classification ──────────────────────────────────────

    def _train_to_event(
        self, train: _SpikeTrain, channel: int
    ) -> DetectedEvent:
        """Convert a spike train to a generic seizure event (no subtype)."""
        return DetectedEvent(
            onset_sec=train.onset_sec,
            offset_sec=train.offset_sec,
            duration_sec=train.duration_sec,
            channel=channel,
            event_type="seizure",
            confidence=min(1.0, train.mean_amplitude_x / 5.0),
            severity=self._classify_severity_simple(train),
            features={
                "seizure_subtype": "seizure",
                "n_spikes": len(train.spikes),
                "max_amplitude_x_baseline": round(train.max_amplitude_x, 2),
                "mean_spike_frequency_hz": round(train.mean_frequency_hz, 2),
                "isi_cv": round(train.isi_cv, 3),
                "detection_method": "spike_train",
                "spike_amplitudes_x": [round(s.amplitude_x, 2) for s in train.spikes],
            },
        )

    @staticmethod
    def _classify_severity_simple(train: _SpikeTrain) -> str:
        """Simple severity based on duration only."""
        if train.duration_sec > 45:
            return "severe"
        elif train.duration_sec > 15:
            return "moderate"
        return "mild"

    def _classify_train(
        self,
        train: _SpikeTrain,
        filtered: np.ndarray,
        fs: float,
        baseline_amp: float,
        channel: int,
        params: SpikeTrainSeizureParams,
    ) -> DetectedEvent | None:
        """Classify a spike train as HVSW, HPD, electroclinical, or unclassified."""

        # Check for post-ictal suppression (for convulsive classification)
        has_postictal = self._check_postictal_suppression(
            filtered, fs, train, baseline_amp, params
        )

        # Build common features dict
        features = {
            "n_spikes": len(train.spikes),
            "mean_amplitude_x_baseline": round(train.mean_amplitude_x, 2),
            "max_amplitude_x_baseline": round(train.max_amplitude_x, 2),
            "mean_spike_frequency_hz": round(train.mean_frequency_hz, 2),
            "isi_cv": round(train.isi_cv, 3),
            "amplitude_trend": round(train.amplitude_trend, 4),
            "frequency_trend": round(train.frequency_trend, 4),
            "has_postictal_suppression": has_postictal,
            "spike_amplitudes_x": [round(s.amplitude_x, 2) for s in train.spikes],
        }

        # ── Electroclinical / convulsive ─────────────────────────────
        if (train.duration_sec >= params.convulsive_min_duration_sec
                and train.max_amplitude_x >= params.convulsive_min_amplitude_x
                and has_postictal):
            seizure_type = "electroclinical"
            severity = "severe"
            confidence = min(1.0, train.max_amplitude_x / params.convulsive_min_amplitude_x)

        # ── HPD ──────────────────────────────────────────────────────
        elif (train.duration_sec >= params.hpd_min_duration_sec
              and train.mean_amplitude_x >= params.hpd_min_amplitude_x
              and train.isi_cv > params.hvsw_max_evolution):
            # HPD: must show evolution (high CV of ISI)
            # Check if evolved phase has higher frequency
            n_half = len(train.spikes) // 2
            if n_half > 1:
                first_isis = [train.spikes[i + 1].time_sec - train.spikes[i].time_sec
                              for i in range(n_half - 1)]
                second_isis = [train.spikes[i + 1].time_sec - train.spikes[i].time_sec
                               for i in range(n_half, len(train.spikes) - 1)]
                first_freq = 1.0 / np.mean(first_isis) if first_isis and np.mean(first_isis) > 0 else 0
                second_freq = 1.0 / np.mean(second_isis) if second_isis and np.mean(second_isis) > 0 else 0

                features["first_half_freq_hz"] = round(float(first_freq), 2)
                features["second_half_freq_hz"] = round(float(second_freq), 2)

                # HPD evolves: either first or second half should reach hpd_min_frequency
                has_fast_phase = max(first_freq, second_freq) >= params.hpd_min_frequency_hz
            else:
                has_fast_phase = train.mean_frequency_hz >= params.hpd_min_frequency_hz

            if has_fast_phase:
                seizure_type = "HPD"
                severity = "moderate" if train.duration_sec < 30 else "severe"
                confidence = min(1.0, train.isi_cv / 0.5)  # higher CV = more confident
            else:
                # Evolving but not fast enough → still classify as seizure
                seizure_type = "HPD"
                severity = "moderate"
                confidence = 0.6

        # ── HVSW ─────────────────────────────────────────────────────
        elif (train.duration_sec >= params.hvsw_min_duration_sec
              and train.mean_amplitude_x >= params.hvsw_min_amplitude_x
              and train.mean_frequency_hz >= params.hvsw_min_frequency_hz
              and train.isi_cv <= params.hvsw_max_evolution):
            seizure_type = "HVSW"
            severity = "mild"
            confidence = min(1.0, train.mean_amplitude_x / params.hvsw_min_amplitude_x)

        # ── Unclassified seizure ─────────────────────────────────────
        # Meets basic train criteria but doesn't fit neatly into the above
        elif (train.mean_amplitude_x >= params.hpd_min_amplitude_x
              and train.mean_frequency_hz >= params.hvsw_min_frequency_hz):
            seizure_type = "unclassified"
            severity = "mild"
            confidence = 0.5

        else:
            # Doesn't meet seizure criteria
            return None

        features["seizure_subtype"] = seizure_type

        return DetectedEvent(
            onset_sec=train.onset_sec,
            offset_sec=train.offset_sec,
            duration_sec=train.duration_sec,
            channel=channel,
            event_type="seizure",
            confidence=round(confidence, 3),
            severity=severity,
            features=features,
        )

    def _check_postictal_suppression(
        self,
        filtered: np.ndarray,
        fs: float,
        train: _SpikeTrain,
        baseline_amp: float,
        params: SpikeTrainSeizureParams,
    ) -> bool:
        """Check if there is post-ictal suppression after the event.

        Post-ictal suppression = a period of very low amplitude (<0.5× baseline)
        immediately after the event.
        """
        suppression_samples = int(params.convulsive_postictal_suppression_sec * fs)
        end_idx = int(train.offset_sec * fs)
        check_start = end_idx
        check_end = min(len(filtered), end_idx + suppression_samples)

        if check_end - check_start < int(0.5 * fs):  # need at least 0.5s
            return False

        postictal_segment = filtered[check_start:check_end]
        postictal_amp = float(np.median(np.abs(postictal_segment)))

        return postictal_amp < 0.5 * baseline_amp

    # ── Step 6: merge ───────────────────────────────────────────────

    def _merge_close_events(
        self, events: list[DetectedEvent], min_gap_sec: float
    ) -> list[DetectedEvent]:
        """Merge events on the same channel separated by less than min_gap_sec."""
        if len(events) <= 1:
            return events

        events = sorted(events, key=lambda e: e.onset_sec)
        merged = [events[0]]

        for event in events[1:]:
            prev = merged[-1]
            if (event.channel == prev.channel
                    and event.onset_sec - prev.offset_sec < min_gap_sec):
                # Merge: keep the more severe classification
                severity_order = {"mild": 0, "moderate": 1, "severe": 2}
                prev_sev = severity_order.get(prev.severity or "mild", 0)
                curr_sev = severity_order.get(event.severity or "mild", 0)

                prev.offset_sec = max(prev.offset_sec, event.offset_sec)
                prev.duration_sec = prev.offset_sec - prev.onset_sec
                if curr_sev > prev_sev:
                    prev.severity = event.severity
                    prev.features["seizure_subtype"] = event.features.get("seizure_subtype", "unclassified")
                # Merge feature stats
                prev.features["n_spikes"] = (
                    prev.features.get("n_spikes", 0) + event.features.get("n_spikes", 0)
                )
                prev.features["max_amplitude_x_baseline"] = max(
                    prev.features.get("max_amplitude_x_baseline", 0),
                    event.features.get("max_amplitude_x_baseline", 0),
                )
                # Merge spike position lists for visualization
                for list_key in ("spike_times", "spike_amplitudes", "spike_samples"):
                    prev_list = prev.features.get(list_key, [])
                    curr_list = event.features.get(list_key, [])
                    prev.features[list_key] = prev_list + curr_list
                prev.confidence = max(prev.confidence, event.confidence)
            else:
                merged.append(event)

        return merged
