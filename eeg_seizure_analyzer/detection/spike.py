"""Interictal spike (IIS) detection using z-score baseline, prominence, and width.

Uses the same baseline computation and morphological constraints as the
seizure spike-train detector, applied to interictal spike detection.

Confidence scoring incorporates:
- Amplitude relative to baseline
- Sharpness (rise/fall slope asymmetry — true IIS have fast rise, slower fall)
- Local SNR (amplitude vs surrounding background noise)
- After-slow-wave presence (classic IIS morphology)
- Phase duration ratio (rising vs falling asymmetry)

Isolation filtering rejects spikes that occur inside dense bursts (seizures).
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
    4. Morphology validation: amplitude, sharpness, duration
    5. Isolation filtering: reject spikes inside dense bursts (seizures)
    6. Confidence scoring: composite of amplitude, sharpness, local SNR,
       after-slow-wave, and phase ratio
    7. Store detection metadata for visualization
    """

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        params: SpikeDetectionParams | None = None,
        baseline_rms: float | None = None,
        baseline_std: float | None = None,
    ) -> list[DetectedEvent]:
        if params is None:
            params = SpikeDetectionParams()

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
        baseline_amp = bl_mean

        # Step 3: compute threshold
        threshold = bl_mean + params.amplitude_threshold_zscore * bl_std
        if params.spike_min_amplitude_uv > 0:
            threshold = max(threshold, params.spike_min_amplitude_uv)

        # Step 4: detect spikes with prominence + width + morphology
        spikes = self._detect_spikes(filtered, fs, baseline_amp, bl_std, threshold, params)

        # Step 5: isolation filtering — reject spikes in dense bursts
        spikes = self._apply_isolation_filter(spikes, fs, params)

        # Step 6: confidence scoring
        self._compute_confidence(spikes, filtered, fs, baseline_amp, bl_std, params)

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

        # Step 7: build DetectedEvent list
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
                    confidence=sp.get("confidence", 0.5),
                    features={
                        "amplitude": sp["amplitude"],
                        "amplitude_x_baseline": sp["amplitude_x"],
                        "peak_time_sec": sp["time_sec"],
                        "duration_ms": sp.get("duration_ms"),
                        "sample_idx": sp["sample_idx"],
                        "baseline_mean": bl_mean,
                        "baseline_std": bl_std,
                        "threshold": threshold,
                        # New morphological features
                        "sharpness": sp.get("sharpness"),
                        "local_snr": sp.get("local_snr"),
                        "after_slow_wave": sp.get("after_slow_wave"),
                        "phase_ratio": sp.get("phase_ratio"),
                        "neighbours": sp.get("neighbours", 0),
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

        Returns list of dicts with spike info including morphological features.
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
        # Use a wider analysis window to capture afterwaves (up to max_duration_ms
        # each side, clamped to at least 50ms).
        half_win = max(int(0.05 * fs),
                       int(params.max_duration_ms * fs / 1000))

        for pk in peaks:
            win_start = max(0, pk - half_win)
            win_end = min(len(filtered), pk + half_win)
            segment = filtered[win_start:win_end]

            amplitude = abs(float(np.max(segment)) - float(np.min(segment)))

            # Absolute floor check
            if params.spike_min_amplitude_uv > 0 and amplitude < params.spike_min_amplitude_uv:
                continue

            # Estimate duration: envelope-based (captures afterwaves)
            peak_local = pk - win_start
            duration_ms = self._estimate_spike_duration_envelope(
                segment, peak_local, fs, baseline_amp)
            # Cap to max_duration_ms
            duration_ms = min(duration_ms, params.max_duration_ms)

            # ── Morphological features ──────────────────────────────

            # Sharpness: ratio of max rising slope to max falling slope
            sharpness = self._compute_sharpness(segment, peak_local, fs)

            # Phase duration ratio: rising phase / falling phase duration
            phase_ratio = self._compute_phase_ratio(segment, peak_local)

            spikes.append({
                "sample_idx": int(pk),
                "time_sec": float(pk) / fs,
                "amplitude": amplitude,
                "amplitude_x": amplitude / baseline_amp if baseline_amp > 0 else 0.0,
                "duration_ms": duration_ms,
                "sharpness": sharpness,
                "phase_ratio": phase_ratio,
            })

        return spikes

    # ── Isolation filter ─────────────────────────────────────────────

    def _apply_isolation_filter(
        self,
        spikes: list[dict],
        fs: float,
        params: SpikeDetectionParams,
    ) -> list[dict]:
        """Remove spikes that occur inside dense bursts (likely seizures).

        A spike is rejected if it has more than ``isolation_max_neighbours``
        other spikes within ±``isolation_window_sec``.
        """
        if not spikes:
            return spikes

        times = np.array([s["time_sec"] for s in spikes])
        win = params.isolation_window_sec
        max_n = params.isolation_max_neighbours
        kept = []

        for i, sp in enumerate(spikes):
            t = sp["time_sec"]
            # Count neighbours within ±window (excluding self)
            neighbours = int(np.sum((np.abs(times - t) <= win) & (times != t)))
            sp["neighbours"] = neighbours
            if neighbours <= max_n:
                kept.append(sp)

        return kept

    # ── Confidence scoring ───────────────────────────────────────────

    def _compute_confidence(
        self,
        spikes: list[dict],
        filtered: np.ndarray,
        fs: float,
        baseline_amp: float,
        baseline_std: float,
        params: SpikeDetectionParams,
    ) -> None:
        """Compute composite confidence score for each spike (in-place).

        Components (each normalised to 0–1):
        - amplitude_score: sigmoid of amplitude_x (saturates around 8x baseline)
        - sharpness_score: higher is better (asymmetric rise/fall)
        - local_snr_score: spike amplitude vs local background RMS
        - after_slow_wave_score: presence of post-spike slow wave
        - phase_ratio_score: asymmetry of rising vs falling phase
        """
        snr_window_sec = 2.0  # window for local RMS (excluding spike)
        snr_window_samples = int(snr_window_sec * fs)

        for sp in spikes:
            pk = sp["sample_idx"]

            # ── Local SNR ────────────────────────────────────────
            # RMS of surrounding signal (±2s), excluding ±50ms around peak
            excl = int(0.05 * fs)
            snr_start = max(0, pk - snr_window_samples)
            snr_end = min(len(filtered), pk + snr_window_samples)
            # Build mask excluding spike region
            local_data = np.concatenate([
                filtered[snr_start:max(snr_start, pk - excl)],
                filtered[min(pk + excl, snr_end):snr_end],
            ])
            if len(local_data) > 10:
                local_rms = float(np.sqrt(np.mean(local_data ** 2)))
            else:
                local_rms = baseline_amp if baseline_amp > 0 else 1.0
            local_snr = sp["amplitude"] / local_rms if local_rms > 1e-10 else 0.0
            sp["local_snr"] = round(local_snr, 2)

            # ── After-slow-wave detection ────────────────────────
            # Look for a slow deflection 50–300ms after the peak
            asw = self._detect_after_slow_wave(filtered, pk, fs, baseline_amp)
            sp["after_slow_wave"] = asw

            # ── Normalised component scores (0–1) ───────────────
            amp_x = sp.get("amplitude_x", 0)
            # Sigmoid: score = 1 / (1 + exp(-k*(x - x0)))
            # x0=4 (half-max at 4x baseline), k=0.8
            amp_score = 1.0 / (1.0 + np.exp(-0.8 * (amp_x - 4.0)))

            # Sharpness: ideal IIS has ratio > 1.5 (fast rise, slow fall)
            # Score peaks at sharpness ~2–3, sigmoid centred at 1.5
            sharp = sp.get("sharpness") or 1.0
            sharp_score = 1.0 / (1.0 + np.exp(-2.0 * (sharp - 1.5)))

            # Local SNR: sigmoid centred at 5
            snr_score = 1.0 / (1.0 + np.exp(-0.6 * (local_snr - 5.0)))

            # After-slow-wave: binary (0 or 1) with some softening
            asw_score = 1.0 if asw else 0.0

            # Phase ratio: ideal ~0.3–0.5 (fast rise, slow fall)
            # or ~2–3 (slow rise, fast fall — inverted polarity)
            # Score is high when ratio deviates from 1.0 (symmetric)
            pr = sp.get("phase_ratio") or 1.0
            asymmetry = abs(np.log(pr + 1e-6))  # log(1)=0 → symmetric
            pr_score = min(1.0, asymmetry / 1.5)  # saturates at ~4.5x asymmetry

            # ── Weighted composite ────────────────────────────────
            w_sum = (params.w_amplitude + params.w_sharpness +
                     params.w_local_snr + params.w_after_slow_wave +
                     params.w_phase_ratio)
            if w_sum < 1e-6:
                w_sum = 1.0

            confidence = (
                params.w_amplitude * amp_score +
                params.w_sharpness * sharp_score +
                params.w_local_snr * snr_score +
                params.w_after_slow_wave * asw_score +
                params.w_phase_ratio * pr_score
            ) / w_sum

            sp["confidence"] = round(float(np.clip(confidence, 0.0, 1.0)), 3)

    # ── Morphology helpers ───────────────────────────────────────────

    def _compute_sharpness(
        self, segment: np.ndarray, peak_idx: int, fs: float,
    ) -> float:
        """Compute sharpness as the ratio of max rising slope to max falling slope.

        True IIS typically have a fast rise and slower fall, giving sharpness > 1.
        Values close to 1.0 indicate a symmetric waveform.
        """
        if peak_idx < 2 or peak_idx >= len(segment) - 2:
            return 1.0

        # Rising phase: before peak
        rising = segment[max(0, peak_idx - 10):peak_idx + 1]
        if len(rising) < 2:
            return 1.0
        rising_slopes = np.abs(np.diff(rising))
        max_rise = float(np.max(rising_slopes)) if len(rising_slopes) > 0 else 1.0

        # Falling phase: after peak
        falling = segment[peak_idx:min(len(segment), peak_idx + 11)]
        if len(falling) < 2:
            return 1.0
        falling_slopes = np.abs(np.diff(falling))
        max_fall = float(np.max(falling_slopes)) if len(falling_slopes) > 0 else 1.0

        if max_fall < 1e-10:
            return 1.0
        return round(max_rise / max_fall, 2)

    def _compute_phase_ratio(
        self, segment: np.ndarray, peak_idx: int,
    ) -> float:
        """Ratio of rising phase duration to falling phase duration.

        Determined by zero crossings. Symmetric spike → ratio ≈ 1.0.
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

        if zc_before is None or zc_after is None:
            return 1.0

        rising_dur = peak_idx - zc_before
        falling_dur = zc_after - peak_idx

        if falling_dur < 1:
            return 1.0
        return round(rising_dur / falling_dur, 2)

    def _detect_after_slow_wave(
        self,
        filtered: np.ndarray,
        peak_sample: int,
        fs: float,
        baseline_amp: float,
    ) -> bool:
        """Detect whether a slow wave follows the spike (50–300ms after peak).

        Classic IIS morphology: sharp spike followed by a slow-wave deflection
        in the opposite direction, with amplitude > 0.5× baseline.
        """
        start = peak_sample + int(0.05 * fs)   # 50ms after peak
        end = peak_sample + int(0.30 * fs)      # 300ms after peak
        end = min(end, len(filtered))

        if start >= end or start >= len(filtered):
            return False

        post_segment = filtered[start:end]
        peak_val = filtered[peak_sample]

        # Look for deflection in opposite direction
        if peak_val > 0:
            # Spike was positive → slow wave should be negative
            min_val = float(np.min(post_segment))
            return min_val < -0.5 * baseline_amp
        else:
            # Spike was negative → slow wave should be positive
            max_val = float(np.max(post_segment))
            return max_val > 0.5 * baseline_amp

    def _estimate_spike_duration_envelope(
        self,
        segment: np.ndarray,
        peak_idx: int,
        fs: float,
        baseline_amp: float,
    ) -> float:
        """Estimate spike duration using the signal envelope.

        Instead of relying on zero crossings (which only capture the
        initial deflection), this method looks for where the absolute
        signal drops back to the noise floor.  This captures the full
        spike-and-wave complex including afterwaves.

        The noise floor is defined as ``noise_factor × baseline_amp``.
        Starting from the peak, we walk backwards and forwards until
        the absolute signal stays below the noise floor for a short
        run (``settle_ms``).

        Returns duration in milliseconds (never None).
        """
        abs_seg = np.abs(segment)
        peak_amp = abs_seg[peak_idx]

        # Noise floor: use the higher of 3× baseline or 25% of peak amplitude.
        # This prevents the walk from extending into normal background
        # fluctuations while still capturing meaningful afterwaves.
        noise_floor = max(baseline_amp * 3.0, peak_amp * 0.25, 1e-10)
        settle_samples = max(4, int(0.015 * fs))  # 15ms settle

        # Walk backward from peak
        start_idx = peak_idx
        run = 0
        for i in range(peak_idx - 1, -1, -1):
            if abs_seg[i] < noise_floor:
                run += 1
                if run >= settle_samples:
                    start_idx = i + run
                    break
            else:
                run = 0
                start_idx = i
        else:
            start_idx = 0

        # Walk forward from peak
        end_idx = peak_idx
        run = 0
        for i in range(peak_idx + 1, len(abs_seg)):
            if abs_seg[i] < noise_floor:
                run += 1
                if run >= settle_samples:
                    end_idx = i - run
                    break
            else:
                run = 0
                end_idx = i
        else:
            end_idx = len(abs_seg) - 1

        duration_samples = max(1, end_idx - start_idx)
        return duration_samples / fs * 1000

    def _estimate_spike_duration(
        self, segment: np.ndarray, peak_idx: int, fs: float,
    ) -> float | None:
        """Legacy: estimate spike duration from zero crossings.

        Kept for backward compatibility but no longer the primary method.
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
