"""Feature extraction for seizure and spike detection."""

from __future__ import annotations

import numpy as np


def line_length(data: np.ndarray, window_samples: int, step_samples: int | None = None) -> np.ndarray:
    """Sliding window line-length: sum(|x[i+1] - x[i]|) over window.

    Line-length captures high-frequency oscillatory activity characteristic
    of seizures. It is robust to baseline drift.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    window_samples : int
        Window size in samples.
    step_samples : int | None
        Step size in samples. Defaults to window_samples // 2.

    Returns
    -------
    np.ndarray
        Line-length values, one per window.
    """
    if step_samples is None:
        step_samples = window_samples // 2

    n_windows = max(1, (len(data) - window_samples) // step_samples + 1)
    result = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > len(data):
            end = len(data)
        segment = data[start:end]
        result[i] = np.sum(np.abs(np.diff(segment)))

    return result


def signal_energy(data: np.ndarray, window_samples: int, step_samples: int | None = None) -> np.ndarray:
    """Sliding window energy: sum(x^2) over window.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    window_samples : int
        Window size in samples.
    step_samples : int | None
        Step size. Defaults to window_samples // 2.

    Returns
    -------
    np.ndarray
        Energy values, one per window.
    """
    if step_samples is None:
        step_samples = window_samples // 2

    n_windows = max(1, (len(data) - window_samples) // step_samples + 1)
    result = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > len(data):
            end = len(data)
        result[i] = np.sum(data[start:end] ** 2)

    return result


def rms_envelope(data: np.ndarray, window_samples: int, step_samples: int | None = None) -> np.ndarray:
    """Root-mean-square envelope.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    window_samples : int
        Window size in samples.
    step_samples : int | None
        Step size. Defaults to window_samples // 2.

    Returns
    -------
    np.ndarray
        RMS values, one per window.
    """
    if step_samples is None:
        step_samples = window_samples // 2

    n_windows = max(1, (len(data) - window_samples) // step_samples + 1)
    result = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > len(data):
            end = len(data)
        segment = data[start:end]
        result[i] = np.sqrt(np.mean(segment ** 2))

    return result


def teager_energy(data: np.ndarray) -> np.ndarray:
    """Teager-Kaiser energy operator: x[n]^2 - x[n-1]*x[n+1].

    Proportional to the product of instantaneous amplitude and frequency.
    Excellent for detecting interictal spikes (brief, high-amplitude,
    high-frequency events).

    Parameters
    ----------
    data : np.ndarray
        1D signal array.

    Returns
    -------
    np.ndarray
        Teager energy, same length as input (edges padded with 0).
    """
    result = np.zeros_like(data)
    result[1:-1] = data[1:-1] ** 2 - data[:-2] * data[2:]
    return result


def compute_zscore_baseline(
    data: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    percentile: int = 15,
) -> tuple[float, float]:
    """Z-score baseline from quiet windows.

    Computes RMS in non-overlapping windows to identify the quietest
    periods (≤ Nth percentile), then computes statistics from the **raw
    signal samples** within those quiet windows:

    - ``baseline_mean``: mean of absolute signal values in quiet windows
      (≈ RMS of quiet periods).
    - ``baseline_std``: std of absolute signal values in quiet windows
      (captures the actual amplitude variability of the quiet signal).

    The detection threshold becomes:  ``mean + z × std``, where *std*
    reflects real sample-level variability rather than the artificially
    low variance of window-averaged RMS values.

    Parameters
    ----------
    data : 1D signal array (raw or filtered)
    fs : sampling rate
    window_sec : RMS window duration (seconds)
    percentile : which percentile to use as quiet-window cutoff (1–50, default 15)

    Returns
    -------
    tuple[float, float] — (baseline_mean, baseline_std)
    """
    win = int(window_sec * fs)
    if win < 1:
        win = 1
    n_windows = len(data) // win
    if n_windows < 1:
        abs_data = np.abs(data)
        return (max(float(np.mean(abs_data)), 1e-10),
                max(float(np.std(abs_data)), 1e-10))

    # Step 1: compute RMS per window to rank them
    rms_values = np.array([
        np.sqrt(np.mean(data[i * win:(i + 1) * win] ** 2))
        for i in range(n_windows)
    ])

    # Step 2: select quiet windows (≤ Nth percentile RMS)
    cutoff = float(np.percentile(rms_values, percentile))
    quiet_mask = rms_values <= cutoff
    if quiet_mask.sum() < 2:
        # Fallback: take the quietest 10% or at least 2
        n_keep = max(2, n_windows // 10)
        sorted_idx = np.argsort(rms_values)[:n_keep]
        quiet_mask = np.zeros(n_windows, dtype=bool)
        quiet_mask[sorted_idx] = True

    # Step 3: gather raw signal samples from quiet windows
    quiet_samples = np.concatenate([
        np.abs(data[i * win:(i + 1) * win])
        for i in range(n_windows) if quiet_mask[i]
    ])

    bl_mean = float(np.mean(quiet_samples))
    bl_std = float(np.std(quiet_samples))

    bl_mean = max(bl_mean, 1e-10)
    bl_std = max(bl_std, 1e-10)
    return (bl_mean, bl_std)


# Keep backward-compatible alias
def compute_percentile_baseline(
    data: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    percentile: int = 15,
) -> float:
    """Deprecated — use ``compute_zscore_baseline`` instead.

    Returns only the baseline_mean for backward compatibility.
    """
    mean, _std = compute_zscore_baseline(data, fs, window_sec, percentile)
    return mean


def compute_rolling_baseline(
    data: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    percentile: int = 15,
    lookback_sec: float = 1800.0,
    step_sec: float = 300.0,
) -> list[tuple[float, float, float]]:
    """Rolling adaptive z-score baseline.

    For each step position, looks back ``lookback_sec`` and computes the
    z-score baseline (mean, std of quiet windows) over that lookback.
    Handles slow drift over multi-day recordings.

    Parameters
    ----------
    data : 1D signal array
    fs : sampling rate
    window_sec : RMS sub-window for percentile calc
    percentile : percentile of RMS windows (quiet-window cutoff)
    lookback_sec : how far back to look (default 30 min)
    step_sec : how often to recompute (default 5 min)

    Returns
    -------
    list of (time_sec, baseline_mean, baseline_std) tuples
    """
    win = max(1, int(window_sec * fs))
    step = max(1, int(step_sec * fs))
    lookback = int(lookback_sec * fs)
    result: list[tuple[float, float, float]] = []

    for pos in range(0, len(data), step):
        w_start = max(0, pos - lookback)
        segment = data[w_start:pos] if pos > 0 else data[:step]
        if len(segment) < win:
            segment = data[:max(win, pos)]
        n_wins = max(1, len(segment) // win)
        rms_vals = np.array([
            np.sqrt(np.mean(segment[i * win:(i + 1) * win] ** 2))
            for i in range(n_wins)
        ])
        cutoff = float(np.percentile(rms_vals, percentile))
        quiet_mask = rms_vals <= cutoff
        if quiet_mask.sum() < 2:
            n_keep = max(2, n_wins // 10)
            sorted_idx = np.argsort(rms_vals)[:n_keep]
            quiet_mask = np.zeros(n_wins, dtype=bool)
            quiet_mask[sorted_idx] = True
        # Gather raw samples from quiet windows
        quiet_samples = np.concatenate([
            np.abs(segment[i * win:(i + 1) * win])
            for i in range(n_wins) if quiet_mask[i]
        ])
        bl_mean = max(float(np.mean(quiet_samples)), 1e-10)
        bl_std = max(float(np.std(quiet_samples)), 1e-10)
        result.append((pos / fs, bl_mean, bl_std))

    return result


def get_baseline_at_time(
    baselines: list[tuple[float, float, float]],
    time_sec: float,
) -> tuple[float, float]:
    """Look up the baseline (mean, std) for a given time point.

    Uses the most recent baseline at or before ``time_sec``.

    Parameters
    ----------
    baselines : list of (time_sec, mean, std) tuples
    time_sec : query time

    Returns
    -------
    tuple[float, float] — (baseline_mean, baseline_std)
    """
    if not baselines:
        return (1.0, 0.1)
    times = [t for t, _, _ in baselines]
    idx = max(0, int(np.searchsorted(times, time_sec, side="right")) - 1)
    _, mean, std = baselines[idx]
    return (mean, std)


def compute_zscore(
    feature: np.ndarray,
    method: str = "robust",
    baseline_indices: tuple[int, int] | None = None,
    baseline_mean: float | None = None,
    baseline_std: float | None = None,
    baseline_rms: float | None = None,
    rolling_baselines: list[tuple[float, float, float]] | None = None,
    step_sec: float | None = None,
) -> np.ndarray:
    """Z-score normalize a feature array.

    Parameters
    ----------
    feature : np.ndarray
        1D feature array (e.g., line-length over time).
    method : str
        "percentile" — uses precomputed baseline (mean, std)
        "rolling" — uses per-window rolling baseline (mean, std)
        "first_n" — mean + std of first N samples (use baseline_indices)
        "manual" — mean + std of specified range (use baseline_indices)
        "robust" — deprecated, alias for "percentile"
    baseline_mean : float | None
        Mean of quiet-window RMS for "percentile" method.
    baseline_std : float | None
        Std of quiet-window RMS for "percentile" method.
    baseline_rms : float | None
        Deprecated. If provided and baseline_mean is None, used as
        baseline_mean with baseline_std estimated as baseline_rms × 0.3.
    rolling_baselines : list[(time, mean, std)] | None
        Rolling baseline array for "rolling" method.
    step_sec : float | None
        Feature step size in seconds (needed to map feature indices to time
        for rolling baseline lookup).
    baseline_indices : tuple[int, int] | None
        (start, end) indices for "first_n" or "manual" methods.

    Returns
    -------
    np.ndarray
        Z-scored feature values.
    """
    import warnings

    # Handle deprecated baseline_rms parameter — preserves old behavior:
    # z = (feature - rms) / rms  (fold-change above baseline)
    if baseline_mean is None and baseline_rms is not None:
        baseline_mean = baseline_rms
        if baseline_std is None:
            baseline_std = baseline_rms  # preserves (f - rms) / rms

    if method == "robust":
        warnings.warn(
            "baseline_method='robust' is deprecated; use 'percentile' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if baseline_mean is not None:
            method = "percentile"
        else:
            MAD_SCALE = 1.4826
            center = np.median(feature)
            mad = np.median(np.abs(feature - center))
            scale = mad * MAD_SCALE
            if scale < 1e-10:
                return np.zeros_like(feature)
            return (feature - center) / scale

    if method == "percentile":
        if baseline_mean is None or baseline_std is None:
            raise ValueError(
                "baseline_mean and baseline_std required for method 'percentile'"
            )
        if baseline_std < 1e-10:
            return np.zeros_like(feature)
        return (feature - baseline_mean) / baseline_std

    if method == "rolling":
        if rolling_baselines is None or step_sec is None:
            raise ValueError(
                "rolling_baselines and step_sec required for method 'rolling'"
            )
        result = np.zeros_like(feature, dtype=np.float64)
        for i in range(len(feature)):
            t = i * step_sec
            bl_mean, bl_std = get_baseline_at_time(rolling_baselines, t)
            if bl_std < 1e-10:
                result[i] = 0.0
            else:
                result[i] = (feature[i] - bl_mean) / bl_std
        return result.astype(np.float32)

    if method in ("first_n", "manual"):
        if baseline_indices is None:
            raise ValueError(f"baseline_indices required for method '{method}'")
        start, end = baseline_indices
        baseline = feature[start:end]
        center = np.mean(baseline)
        scale = np.std(baseline)
        if scale < 1e-10:
            return np.zeros_like(feature)
        return (feature - center) / scale

    raise ValueError(f"Unknown method: {method}")
