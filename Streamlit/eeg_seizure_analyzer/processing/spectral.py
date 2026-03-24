"""Spectral analysis: band power, spectrograms, PSD."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch, spectrogram as scipy_spectrogram

from eeg_seizure_analyzer.config import BANDS


def compute_psd(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density via Welch's method.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    fs : float
        Sampling rate.
    nperseg : int | None
        Segment length for Welch. Defaults to 4*fs (4-second windows).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frequencies, psd_values)
    """
    if nperseg is None:
        nperseg = min(int(4 * fs), len(data))
    freqs, psd = welch(data, fs=fs, nperseg=nperseg)
    return freqs, psd


def compute_band_powers(
    data: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
) -> pd.DataFrame:
    """Compute power in each frequency band over time.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    fs : float
        Sampling rate.
    bands : dict
        Frequency band definitions. Defaults to BANDS from config.
    window_sec : float
        Analysis window duration.
    step_sec : float
        Step between windows.

    Returns
    -------
    pd.DataFrame
        Columns: time_sec, band1_power, band2_power, ...
    """
    if bands is None:
        bands = BANDS

    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    n_windows = max(1, (len(data) - window_samples) // step_samples + 1)

    results = {"time_sec": np.zeros(n_windows)}
    for band_name in bands:
        results[band_name] = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > len(data):
            end = len(data)
        segment = data[start:end]

        results["time_sec"][i] = start / fs

        freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), int(fs)))

        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(band_mask):
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                results[band_name][i] = np.trapezoid(psd[band_mask], dx=freq_res)

    return pd.DataFrame(results)


def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    max_freq: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram for display.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    fs : float
        Sampling rate.
    nperseg : int | None
        Segment length. Defaults to 2*fs.
    noverlap : int | None
        Overlap. Defaults to 75% of nperseg.
    max_freq : float
        Maximum frequency to return (Hz).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (times, frequencies, power) where power is in dB (10*log10).
    """
    if nperseg is None:
        nperseg = min(int(2 * fs), len(data))
    if noverlap is None:
        noverlap = int(nperseg * 0.75)

    freqs, times, sxx = scipy_spectrogram(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap
    )

    # Limit to max_freq
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    sxx = sxx[freq_mask, :]

    # Convert to dB
    sxx_db = 10 * np.log10(sxx + 1e-12)

    return times, freqs, sxx_db
