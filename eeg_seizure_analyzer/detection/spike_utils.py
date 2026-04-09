"""Shared spike detection utilities.

Extracted from SpikeTrainSeizureDetector so that multiple detectors
(spike-train, autocorrelation) can reuse the same spike front-end
without code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from eeg_seizure_analyzer.processing.features import (
    compute_zscore_baseline,
    compute_rolling_baseline,
)


@dataclass
class Spike:
    """A single detected spike."""

    sample_idx: int
    time_sec: float
    amplitude: float  # peak-to-trough absolute amplitude
    amplitude_x: float  # amplitude as multiple of baseline


def compute_baseline(
    data: np.ndarray,
    fs: float,
    method: str = "percentile",
    percentile: int = 15,
    rms_window_sec: float = 10.0,
    rolling_lookback_sec: float = 1800.0,
    rolling_step_sec: float = 300.0,
) -> tuple[float, float]:
    """Compute baseline (mean, std) from quiet windows.

    Returns (baseline_mean, baseline_std).  The spike threshold is
    typically ``baseline_mean + z × baseline_std``.

    Supports:
    - ``"percentile"`` (default): z-score baseline from quiet RMS windows.
    - ``"rolling"``: adaptive z-score baseline recomputed periodically.
      Returns median of rolling (mean, std) pairs.
    - ``"first_n"``: mean + std of first 5 minutes.
    """
    if method == "percentile":
        return compute_zscore_baseline(
            data, fs,
            window_sec=rms_window_sec,
            percentile=percentile,
        )

    if method == "rolling":
        rolling = compute_rolling_baseline(
            data, fs,
            window_sec=rms_window_sec,
            percentile=percentile,
            lookback_sec=rolling_lookback_sec,
            step_sec=rolling_step_sec,
        )
        means = [m for _, m, _ in rolling]
        stds = [s for _, _, s in rolling]
        return (float(np.median(means)), float(np.median(stds)))

    # "first_n"
    n = int(5 * 60 * fs)
    segment = data[: min(n, len(data))]
    bl_mean = float(np.mean(np.abs(segment)))
    bl_std = float(np.std(np.abs(segment)))
    if bl_mean < 1e-10:
        bl_mean = float(np.std(segment))
    if bl_mean < 1e-10:
        bl_mean = 1.0
    if bl_std < 1e-10:
        bl_std = bl_mean * 0.1
    return (bl_mean, bl_std)


def detect_spikes(
    filtered: np.ndarray,
    fs: float,
    baseline_amp: float,
    baseline_std: float = 0.0,
    *,
    spike_amplitude_x_baseline: float = 3.0,
    spike_min_amplitude_uv: float = 0.0,
    spike_refractory_ms: float = 50.0,
    spike_prominence_x_baseline: float = 1.5,
    spike_min_prominence_uv: float = 0.0,
    spike_max_width_ms: float = 70.0,
    spike_min_width_ms: float = 2.0,
) -> list[Spike]:
    """Detect spikes with amplitude, prominence, and width constraints.

    Threshold: ``baseline_amp + z × baseline_std`` (z-score multiplier).

    Parameters
    ----------
    filtered : np.ndarray
        Bandpass-filtered 1-D signal.
    fs : float
        Sampling rate (Hz).
    baseline_amp : float
        Baseline amplitude (mean of absolute signal in quiet periods).
    baseline_std : float
        Standard deviation of the baseline.
    spike_amplitude_x_baseline : float
        Z-score multiplier for threshold.
    spike_min_amplitude_uv : float
        Absolute amplitude floor (µV); 0 = disabled.
    spike_refractory_ms : float
        Minimum inter-spike interval (ms).
    spike_prominence_x_baseline : float
        Minimum prominence as × baseline.
    spike_min_prominence_uv : float
        Minimum prominence absolute floor (µV); 0 = use baseline-relative.
    spike_max_width_ms : float
        Maximum half-width (ms); rejects slow waves.
    spike_min_width_ms : float
        Minimum half-width (ms); rejects single-sample noise.

    Returns
    -------
    list[Spike]
    """
    if baseline_std > 0:
        abs_threshold = baseline_amp + spike_amplitude_x_baseline * baseline_std
    else:
        abs_threshold = spike_amplitude_x_baseline * baseline_amp

    # Apply absolute floor if set
    if spike_min_amplitude_uv > 0:
        abs_threshold = max(abs_threshold, spike_min_amplitude_uv)

    refractory_samples = max(1, int(spike_refractory_ms * fs / 1000))

    # Prominence
    min_prominence = spike_prominence_x_baseline * baseline_amp
    if spike_min_prominence_uv > 0:
        min_prominence = max(min_prominence, spike_min_prominence_uv)

    # Width constraints (in samples)
    min_width_samples = max(1, int(spike_min_width_ms * fs / 1000))
    max_width_samples = max(min_width_samples + 1, int(spike_max_width_ms * fs / 1000))

    abs_signal = np.abs(filtered)

    peaks, _properties = find_peaks(
        abs_signal,
        height=abs_threshold,
        distance=refractory_samples,
        prominence=min_prominence,
        width=(min_width_samples, max_width_samples),
    )

    spikes: list[Spike] = []
    half_win = int(0.05 * fs)  # 50 ms window for peak-to-trough

    for pk in peaks:
        win_start = max(0, pk - half_win)
        win_end = min(len(filtered), pk + half_win)
        segment = filtered[win_start:win_end]

        amplitude = abs(float(np.max(segment)) - float(np.min(segment)))

        if spike_min_amplitude_uv > 0 and amplitude < spike_min_amplitude_uv:
            continue

        spikes.append(
            Spike(
                sample_idx=int(pk),
                time_sec=float(pk) / fs,
                amplitude=amplitude,
                amplitude_x=amplitude / baseline_amp if baseline_amp > 0 else 0.0,
            )
        )

    return spikes
