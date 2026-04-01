"""Activity channel loading and movement artifact flagging.

Many rodent EEG setups include low-rate activity/EMG/accelerometer channels
recorded alongside high-rate EEG.  This module handles:
  1. Loading a different-rate activity channel from the same EDF file.
  2. Upsampling it to match the EEG sampling rate.
  3. Computing activity statistics (mean, std, z-score) for detected events.

Activity metrics are stored as features for later use by the ML model.
No classification is attempted — the model learns what matters.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from eeg_seizure_analyzer.detection.base import DetectedEvent


def load_activity_channel(
    path: str,
    channel_idx: int,
    target_fs: float,
) -> tuple[np.ndarray, float]:
    """Load a single channel from an EDF file and upsample to target_fs.

    Parameters
    ----------
    path : str
        Path to the EDF file.
    channel_idx : int
        Channel index of the activity/EMG channel.
    target_fs : float
        Target sampling rate (typically the EEG fs).

    Returns
    -------
    tuple[np.ndarray, float]
        (upsampled_data, original_fs)
    """
    import pyedflib

    f = pyedflib.EdfReader(path)
    try:
        original_fs = f.getSampleFrequency(channel_idx)
        raw = f.readSignal(channel_idx).astype(np.float32)
    finally:
        f.close()

    if original_fs == target_fs:
        return raw, original_fs

    # Create time vectors for interpolation
    duration_sec = len(raw) / original_fs
    t_original = np.arange(len(raw)) / original_fs
    n_target = int(duration_sec * target_fs)
    t_target = np.arange(n_target) / target_fs

    # Linear interpolation (sufficient for slow activity signals)
    interpolator = interp1d(
        t_original, raw, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    upsampled = interpolator(t_target).astype(np.float32)

    return upsampled, original_fs


def compute_activity_stats(
    activity: np.ndarray,
) -> tuple[float, float]:
    """Compute global mean and std of |activity| for z-score normalisation.

    Returns
    -------
    tuple[float, float]
        (global_mean, global_std)
    """
    abs_act = np.abs(activity)
    mean = float(np.mean(abs_act))
    std = float(np.std(abs_act))
    return (max(mean, 1e-10), max(std, 1e-10))


def flag_events_activity(
    events: list[DetectedEvent],
    activity: np.ndarray,
    fs: float,
    pad_sec: float = 2.0,
) -> list[DetectedEvent]:
    """Compute activity z-score for each event and store as a feature.

    For each event, computes the mean |activity| during the event window
    (± pad), then converts to a z-score relative to the global activity
    distribution.  Higher z-scores indicate more movement during the event.

    Stored features per event:
    - ``activity_zscore``: how many SDs above mean the activity is
    - ``mean_activity``: raw mean |activity| during event ± pad

    Parameters
    ----------
    events : list[DetectedEvent]
        Detected events to analyse.
    activity : np.ndarray
        Activity channel data (at its native sampling rate).
    fs : float
        Sampling rate of the activity array.
    pad_sec : float
        Padding around event boundaries for activity measurement.

    Returns
    -------
    list[DetectedEvent]
        Same events with features updated in-place.
    """
    global_mean, global_std = compute_activity_stats(activity)
    n_samples = len(activity)

    for event in events:
        start_idx = max(0, int((event.onset_sec - pad_sec) * fs))
        end_idx = min(n_samples, int((event.offset_sec + pad_sec) * fs))

        if end_idx > start_idx:
            mean_act = float(np.mean(np.abs(activity[start_idx:end_idx])))
        else:
            mean_act = 0.0

        zscore = (mean_act - global_mean) / global_std
        event.features["activity_zscore"] = round(zscore, 2)
        event.features["mean_activity"] = round(mean_act, 4)

    return events
