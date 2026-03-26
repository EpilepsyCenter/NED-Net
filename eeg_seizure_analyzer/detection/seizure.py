"""Rule-based seizure detection using line-length and energy features."""

from __future__ import annotations

import numpy as np

from eeg_seizure_analyzer.config import SeizureDetectionParams
from eeg_seizure_analyzer.detection.base import DetectedEvent, DetectorBase
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.features import (
    compute_percentile_baseline,
    compute_rolling_baseline,
    compute_zscore,
    line_length,
    signal_energy,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


class SeizureDetector(DetectorBase):
    """Detect seizures using line-length + energy z-score thresholding.

    Algorithm:
    1. Bandpass filter the signal
    2. Compute line-length and energy in sliding windows
    3. Compute baseline (percentile / rolling / first_n / manual)
    4. Z-score normalize against baseline
    5. Threshold crossing: both features must exceed their thresholds
    6. Refine onset/offset to lower z-score boundary
    7. Filter by minimum duration and merge close events
    8. Assign severity based on duration and peak energy
    """

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        params: SeizureDetectionParams | None = None,
        baseline_rms: float | None = None,
    ) -> list[DetectedEvent]:
        """Detect seizures on a single channel.

        Parameters
        ----------
        recording : EEGRecording
        channel : channel index
        params : detection parameters
        baseline_rms : optional precomputed baseline (for chunked pipeline).
            When provided, skips internal baseline computation.
        """
        if params is None:
            params = SeizureDetectionParams()

        data = recording.get_channel_data(channel)
        fs = recording.fs

        # Step 1: Bandpass filter
        filtered = bandpass_filter(data, fs, params.bandpass_low, params.bandpass_high)

        # Step 2: Compute features
        window_samples = int(params.line_length_window_sec * fs)
        step_samples = window_samples // 2
        step_sec = step_samples / fs

        ll = line_length(filtered, window_samples, step_samples)
        en = signal_energy(filtered, window_samples, step_samples)

        # Step 3: Compute baseline and z-score normalize
        method = params.baseline_method
        rolling_baselines = None

        if method in ("percentile", "robust"):
            if baseline_rms is None:
                baseline_rms = compute_percentile_baseline(
                    filtered, fs,
                    window_sec=params.baseline_rms_window_sec,
                    percentile=params.baseline_percentile,
                )
            ll_z = compute_zscore(ll, method="percentile", baseline_rms=baseline_rms)
            en_z = compute_zscore(en, method="percentile", baseline_rms=baseline_rms)

        elif method == "rolling":
            rolling_baselines = compute_rolling_baseline(
                filtered, fs,
                window_sec=params.baseline_rms_window_sec,
                percentile=params.baseline_percentile,
                lookback_sec=params.rolling_lookback_sec,
                step_sec=params.rolling_step_sec,
            )
            ll_z = compute_zscore(
                ll, method="rolling",
                rolling_baselines=rolling_baselines, step_sec=step_sec,
            )
            en_z = compute_zscore(
                en, method="rolling",
                rolling_baselines=rolling_baselines, step_sec=step_sec,
            )

        elif method in ("first_n", "manual"):
            baseline_indices = self._get_baseline_indices(
                len(ll), fs, step_samples, params
            )
            ll_z = compute_zscore(ll, method=method, baseline_indices=baseline_indices)
            en_z = compute_zscore(en, method=method, baseline_indices=baseline_indices)

        else:
            raise ValueError(f"Unknown baseline method: {method}")

        # Step 4: Threshold crossing (both must exceed)
        above_threshold = (ll_z > params.line_length_threshold_zscore) & (
            en_z > params.energy_threshold_zscore
        )

        # Step 5: Find contiguous regions and refine onset/offset
        events = self._extract_events(
            above_threshold, ll_z, en_z, fs, step_samples, channel, params
        )

        # Step 6: Filter by minimum duration
        events = [e for e in events if e.duration_sec >= params.min_duration_sec]

        # Step 7: Merge close events
        events = self._merge_events(events, params.merge_gap_sec)

        # Step 8: Assign severity and store baseline_rms
        for event in events:
            event.severity = self._classify_severity(event, params)
            event.features["baseline_rms"] = baseline_rms
            event.features["detection_method"] = "line_length_energy"

        return events

    def _get_baseline_indices(
        self,
        n_windows: int,
        fs: float,
        step_samples: int,
        params: SeizureDetectionParams,
    ) -> tuple[int, int] | None:
        """Compute baseline window indices for first_n / manual methods."""
        if params.baseline_method == "first_n":
            baseline_samples = int(params.baseline_duration_min * 60 * fs)
            baseline_windows = baseline_samples // step_samples
            return (0, min(baseline_windows, n_windows))
        elif params.baseline_method == "manual":
            if params.baseline_start_sec is None or params.baseline_end_sec is None:
                return (0, min(int(5 * 60 * fs / step_samples), n_windows))
            start_win = int(params.baseline_start_sec * fs / step_samples)
            end_win = int(params.baseline_end_sec * fs / step_samples)
            return (start_win, min(end_win, n_windows))
        return None

    def _extract_events(
        self,
        above_threshold: np.ndarray,
        ll_z: np.ndarray,
        en_z: np.ndarray,
        fs: float,
        step_samples: int,
        channel: int,
        params: SeizureDetectionParams,
    ) -> list[DetectedEvent]:
        """Extract events from threshold crossings with onset/offset refinement."""
        events = []
        step_sec = step_samples / fs

        # Find contiguous regions above threshold
        changes = np.diff(above_threshold.astype(int))
        onsets = np.where(changes == 1)[0] + 1
        offsets = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            onsets = np.concatenate([[0], onsets])
        if above_threshold[-1]:
            offsets = np.concatenate([offsets, [len(above_threshold)]])

        # Ensure matching pairs
        n_events = min(len(onsets), len(offsets))

        for i in range(n_events):
            onset_idx = int(onsets[i])
            offset_idx = int(offsets[i])

            # Refine onset: walk backward to lower threshold
            refined_onset = onset_idx
            while refined_onset > 0:
                if ll_z[refined_onset - 1] < params.onset_offset_zscore and en_z[refined_onset - 1] < params.onset_offset_zscore:
                    break
                refined_onset -= 1

            # Refine offset: walk forward to lower threshold
            refined_offset = offset_idx
            while refined_offset < len(ll_z) - 1:
                if ll_z[refined_offset] < params.onset_offset_zscore and en_z[refined_offset] < params.onset_offset_zscore:
                    break
                refined_offset += 1

            onset_sec = refined_onset * step_sec
            offset_sec = refined_offset * step_sec
            duration = offset_sec - onset_sec

            # Compute features for this event
            event_ll_z = ll_z[refined_onset:refined_offset]
            event_en_z = en_z[refined_onset:refined_offset]

            features = {
                "peak_line_length_zscore": float(np.max(event_ll_z)) if len(event_ll_z) > 0 else 0.0,
                "peak_energy_zscore": float(np.max(event_en_z)) if len(event_en_z) > 0 else 0.0,
                "mean_line_length_zscore": float(np.mean(event_ll_z)) if len(event_ll_z) > 0 else 0.0,
                "mean_energy_zscore": float(np.mean(event_en_z)) if len(event_en_z) > 0 else 0.0,
            }

            events.append(
                DetectedEvent(
                    onset_sec=onset_sec,
                    offset_sec=offset_sec,
                    duration_sec=duration,
                    channel=channel,
                    event_type="seizure",
                    confidence=min(1.0, float(np.mean(event_ll_z)) / params.line_length_threshold_zscore) if len(event_ll_z) > 0 else 0.0,
                    features=features,
                )
            )

        return events

    def _merge_events(
        self, events: list[DetectedEvent], gap_sec: float
    ) -> list[DetectedEvent]:
        """Merge events separated by less than gap_sec."""
        if len(events) <= 1:
            return events

        merged = [events[0]]
        for event in events[1:]:
            prev = merged[-1]
            if event.onset_sec - prev.offset_sec <= gap_sec and event.channel == prev.channel:
                # Merge: extend previous event
                prev.offset_sec = event.offset_sec
                prev.duration_sec = prev.offset_sec - prev.onset_sec
                prev.features["peak_line_length_zscore"] = max(
                    prev.features.get("peak_line_length_zscore", 0),
                    event.features.get("peak_line_length_zscore", 0),
                )
                prev.features["peak_energy_zscore"] = max(
                    prev.features.get("peak_energy_zscore", 0),
                    event.features.get("peak_energy_zscore", 0),
                )
            else:
                merged.append(event)

        return merged

    def _classify_severity(
        self, event: DetectedEvent, params: SeizureDetectionParams
    ) -> str:
        """Classify seizure severity based on duration and peak energy."""
        peak_energy = event.features.get("peak_energy_zscore", 0)

        if event.duration_sec > params.moderate_max_duration_sec or peak_energy > params.severe_energy_zscore:
            return "severe"
        elif event.duration_sec > params.mild_max_duration_sec:
            return "moderate"
        else:
            return "mild"
