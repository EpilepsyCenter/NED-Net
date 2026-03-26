"""Interictal spike (IIS) detection using z-score baseline, prominence, and width.

Uses the same baseline computation and morphological constraints as the
seizure spike-train detector, applied to interictal spike detection.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from eeg_seizure_analyzer.config import SpikeDetectionParams
from eeg_seizure_analyzer.detection.base import DetectedEvent, DetectorBase
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.features import (
    compute_zscore_baseline,
    compute_rolling_baseline,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


class SpikeDetector(DetectorBase):
    """Detect interictal spikes using z-score baseline + morphology checks.

    Algorithm:
    1. Bandpass filter (default 10-70 Hz)
    2. Compute baseline (mean, std) from quiet windows (percentile method)
    3. Find peaks exceeding mean + z × std with prominence and width constraints
    4. Morphology validation: amplitude, duration
    5. Store detection metadata for visualization (spike positions, baseline, threshold)
    """

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        params: SpikeDetectionParams | None = None,
    ) -> list[DetectedEvent]:
        if params is None:
            params = SpikeDetectionParams()

        data = recording.get_channel_data(channel)
        fs = recording.fs

        # Step 1: bandpass filter
        filtered = bandpass_filter(data, fs, params.bandpass_low, params.bandpass_high)

        # Step 2: compute baseline (mean, std)
        bl_mean, bl_std = self._compute_baseline(filtered, fs, params)
        baseline_amp = bl_mean

        # Step 3: compute threshold
        threshold = bl_mean + params.amplitude_threshold_zscore * bl_std
        if params.spike_min_amplitude_uv > 0:
            threshold = max(threshold, params.spike_min_amplitude_uv)

        # Step 4: detect spikes with prominence + width constraints
        spikes = self._detect_spikes(filtered, fs, baseline_amp, bl_std, threshold, params)

        # Store detection metadata for visualization
        self._last_detection_info = {
            "channel": channel,
            "baseline_mean": bl_mean,
            "baseline_std": bl_std,
            "threshold": threshold,
            "all_spike_times": [s["time_sec"] for s in spikes],
            "all_spike_amplitudes": [s["amplitude"] for s in spikes],
            "all_spike_samples": [s["sample_idx"] for s in spikes],
        }

        # Step 5: build DetectedEvent list
        events = []
        for sp in spikes:
            dur_ms = sp.get("duration_ms") or params.max_duration_ms
            onset_sec = sp["time_sec"] - (dur_ms / 2000)
            offset_sec = sp["time_sec"] + (dur_ms / 2000)
            duration_sec = offset_sec - onset_sec

            events.append(
                DetectedEvent(
                    onset_sec=max(0.0, onset_sec),
                    offset_sec=offset_sec,
                    duration_sec=duration_sec,
                    channel=channel,
                    event_type="spike",
                    confidence=min(1.0, sp["amplitude_x"] / 5.0),
                    features={
                        "amplitude": sp["amplitude"],
                        "amplitude_x_baseline": sp["amplitude_x"],
                        "peak_time_sec": sp["time_sec"],
                        "duration_ms": sp.get("duration_ms"),
                        "sample_idx": sp["sample_idx"],
                        "baseline_mean": bl_mean,
                        "baseline_std": bl_std,
                        "threshold": threshold,
                    },
                )
            )

        return sorted(events, key=lambda e: e.onset_sec)

    # ── Baseline computation ─────────────────────────────────────────

    def _compute_baseline(
        self, data: np.ndarray, fs: float, params: SpikeDetectionParams,
    ) -> tuple[float, float]:
        """Compute baseline (mean, std) from quiet windows."""
        method = params.baseline_method

        if method == "percentile":
            return compute_zscore_baseline(
                data, fs,
                window_sec=params.baseline_rms_window_sec,
                percentile=params.baseline_percentile,
            )

        if method == "rolling":
            rolling = compute_rolling_baseline(
                data, fs,
                window_sec=params.baseline_rms_window_sec,
                percentile=params.baseline_percentile,
                lookback_sec=params.rolling_lookback_sec,
                step_sec=params.rolling_step_sec,
            )
            means = [m for _, m, _ in rolling]
            stds = [s for _, _, s in rolling]
            return (float(np.median(means)), float(np.median(stds)))

        # "first_n"
        n = int(5 * 60 * fs)
        segment = data[:min(n, len(data))]
        bl_mean = float(np.mean(np.abs(segment)))
        bl_std = float(np.std(np.abs(segment)))
        if bl_mean < 1e-10:
            bl_mean = 1.0
        if bl_std < 1e-10:
            bl_std = bl_mean * 0.1
        return (bl_mean, bl_std)

    # ── Spike detection with morphology ──────────────────────────────

    def _detect_spikes(
        self,
        filtered: np.ndarray,
        fs: float,
        baseline_amp: float,
        baseline_std: float,
        threshold: float,
        params: SpikeDetectionParams,
    ) -> list[dict]:
        """Detect spikes with amplitude, prominence, and width constraints.

        Returns list of dicts with spike info (time_sec, sample_idx,
        amplitude, amplitude_x, duration_ms).
        """
        refractory_samples = max(1, int(params.refractory_ms * fs / 1000))

        # Prominence: must stand out from local context
        min_prominence = params.spike_prominence_x_baseline * baseline_amp

        # Width constraints (in samples)
        min_width_samples = max(1, int(params.min_duration_ms * fs / 1000))
        max_width_samples = max(min_width_samples + 1,
                                int(params.max_duration_ms * fs / 1000))

        abs_signal = np.abs(filtered)

        peaks, properties = find_peaks(
            abs_signal,
            height=threshold,
            distance=refractory_samples,
            prominence=min_prominence,
            width=(min_width_samples, max_width_samples),
        )

        spikes = []
        half_win = int(0.05 * fs)  # 50ms window for peak-to-trough

        for pk in peaks:
            win_start = max(0, pk - half_win)
            win_end = min(len(filtered), pk + half_win)
            segment = filtered[win_start:win_end]

            amplitude = abs(float(np.max(segment)) - float(np.min(segment)))

            # Absolute floor check
            if params.spike_min_amplitude_uv > 0 and amplitude < params.spike_min_amplitude_uv:
                continue

            # Estimate duration from zero crossings
            peak_local = pk - win_start
            duration_ms = self._estimate_spike_duration(segment, peak_local, fs)

            spikes.append({
                "sample_idx": int(pk),
                "time_sec": float(pk) / fs,
                "amplitude": amplitude,
                "amplitude_x": amplitude / baseline_amp if baseline_amp > 0 else 0.0,
                "duration_ms": duration_ms,
            })

        return spikes

    def _estimate_spike_duration(
        self, segment: np.ndarray, peak_idx: int, fs: float,
    ) -> float | None:
        """Estimate spike duration from zero crossings around the peak.

        Returns duration in milliseconds, or None if can't determine.
        """
        zc_before = None
        for i in range(peak_idx - 1, 0, -1):
            if segment[i] * segment[i - 1] <= 0:
                zc_before = i
                break

        zc_after = None
        for i in range(peak_idx, len(segment) - 1):
            if segment[i] * segment[i + 1] <= 0:
                zc_after = i
                break

        if zc_before is not None and zc_after is not None:
            duration_samples = zc_after - zc_before
            return duration_samples / fs * 1000

        return None
