"""Signal preprocessing: filtering, resampling, artifact rejection."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, decimate


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array (n_channels, n_samples).
    fs : float
        Sampling rate in Hz.
    low, high : float
        Bandpass corner frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered data, same shape as input.
    """
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if data.ndim == 1:
        return sosfiltfilt(sos, data).astype(np.float32)
    return np.array([sosfiltfilt(sos, ch) for ch in data], dtype=np.float32)


def highpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth highpass filter."""
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    if data.ndim == 1:
        return sosfiltfilt(sos, data).astype(np.float32)
    return np.array([sosfiltfilt(sos, ch) for ch in data], dtype=np.float32)


def notch_filter(
    data: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """Remove power line artifact with a notch filter.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array.
    fs : float
        Sampling rate.
    freq : float
        Notch frequency (50 Hz for EU, 60 Hz for US).
    quality : float
        Quality factor (higher = narrower notch).
    """
    b, a = iirnotch(freq, quality, fs)
    from scipy.signal import filtfilt

    if data.ndim == 1:
        return filtfilt(b, a, data).astype(np.float32)
    return np.array([filtfilt(b, a, ch) for ch in data], dtype=np.float32)


def downsample(
    data: np.ndarray,
    fs_orig: float,
    fs_target: float,
) -> tuple[np.ndarray, float]:
    """Decimate data to a target sampling rate.

    Uses scipy.signal.decimate which applies an anti-alias filter.

    Returns
    -------
    tuple[np.ndarray, float]
        (downsampled_data, actual_new_fs)
    """
    factor = int(round(fs_orig / fs_target))
    if factor <= 1:
        return data, fs_orig

    if data.ndim == 1:
        result = decimate(data, factor).astype(np.float32)
    else:
        result = np.array([decimate(ch, factor) for ch in data], dtype=np.float32)

    actual_fs = fs_orig / factor
    return result, actual_fs


def remove_artifacts(
    data: np.ndarray,
    fs: float,
    threshold_uv: float = 3000.0,
    window_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark and zero extreme amplitude segments.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    fs : float
        Sampling rate.
    threshold_uv : float
        Amplitude threshold — samples exceeding this are marked as artifacts.
    window_sec : float
        Extend artifact mask by this many seconds on each side.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (cleaned_data, artifact_mask) where mask is boolean array (True = artifact).
    """
    artifact_mask = np.abs(data) > threshold_uv

    # Extend mask by window
    window_samples = int(window_sec * fs)
    if window_samples > 0:
        extended = np.zeros_like(artifact_mask)
        for i in range(len(data)):
            if artifact_mask[i]:
                start = max(0, i - window_samples)
                end = min(len(data), i + window_samples + 1)
                extended[start:end] = True
        artifact_mask = extended

    cleaned = data.copy()
    cleaned[artifact_mask] = 0.0
    return cleaned, artifact_mask
