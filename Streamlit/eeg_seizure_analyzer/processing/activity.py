"""Activity channel loading and movement artifact flagging.

Many rodent EEG setups include low-rate activity/EMG/accelerometer channels
recorded alongside high-rate EEG.  This module handles:
  1. Loading a different-rate activity channel from the same EDF file.
  2. Upsampling it to match the EEG sampling rate.
  3. Computing a movement threshold.
  4. Flagging detected events that co-occur with elevated activity.
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


def compute_movement_threshold(
    activity: np.ndarray,
    percentile: float = 85.0,
) -> float:
    """Compute a threshold for movement detection.

    Parameters
    ----------
    activity : np.ndarray
        Activity channel data (upsampled to EEG rate).
    percentile : float
        Percentile to use as threshold (default 85th).

    Returns
    -------
    float
        Threshold value above which activity is considered movement.
    """
    # Use absolute values for activity (handles bipolar signals)
    return float(np.percentile(np.abs(activity), percentile))


def flag_movement_artifacts(
    events: list[DetectedEvent],
    activity: np.ndarray,
    fs: float,
    threshold: float,
    pad_sec: float = 2.0,
) -> list[DetectedEvent]:
    """Flag events that co-occur with elevated activity channel values.

    Sets ``movement_flag = True`` on events where the mean absolute activity
    during the event (plus padding) exceeds the threshold.  Events are
    flagged, not removed — the user decides how to handle them.

    Parameters
    ----------
    events : list[DetectedEvent]
        Detected events to check.
    activity : np.ndarray
        Activity channel data upsampled to EEG rate.
    fs : float
        Sampling rate of the activity array.
    threshold : float
        Movement threshold (from compute_movement_threshold).
    pad_sec : float
        Seconds of padding around event boundaries to check.

    Returns
    -------
    list[DetectedEvent]
        Same events with movement_flag updated in-place.
    """
    n_samples = len(activity)

    for event in events:
        start_idx = max(0, int((event.onset_sec - pad_sec) * fs))
        end_idx = min(n_samples, int((event.offset_sec + pad_sec) * fs))

        if end_idx <= start_idx:
            continue

        segment = np.abs(activity[start_idx:end_idx])
        mean_activity = float(np.mean(segment))

        if mean_activity > threshold:
            event.movement_flag = True

    return events
