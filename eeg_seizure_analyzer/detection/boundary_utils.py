"""Shared boundary refinement utilities.

Provides signal-based (RMS envelope) and spike-density boundary
refinement methods, extracted from SpikeTrainSeizureDetector so that
all detectors can share the same logic.
"""

from __future__ import annotations

import numpy as np


def refine_signal_rms(
    onset_sec: float,
    offset_sec: float,
    filtered: np.ndarray,
    fs: float,
    baseline_amp: float,
    *,
    rms_window_ms: float = 100.0,
    rms_threshold_x: float = 2.0,
    max_trim_sec: float = 5.0,
    anchor_onset_sample: int | None = None,
    anchor_offset_sample: int | None = None,
) -> tuple[float, float]:
    """Refine event boundaries using the signal RMS envelope.

    Starting from anchor points (first/last spike, or raw onset/offset),
    walks outward to find where RMS drops below threshold, then inward to
    find the precise electrographic onset/offset.

    Parameters
    ----------
    onset_sec, offset_sec : float
        Current event boundaries in seconds.
    filtered : ndarray
        Bandpass-filtered signal for the full channel.
    fs : float
        Sampling rate.
    baseline_amp : float
        Baseline amplitude (RMS or mean-abs) of quiet signal.
    rms_window_ms : float
        Short window (ms) for RMS envelope computation.
    rms_threshold_x : float
        RMS must exceed ``rms_threshold_x * baseline_amp`` to be
        considered "active".
    max_trim_sec : float
        Maximum seconds to search outward from the anchor.
    anchor_onset_sample, anchor_offset_sample : int | None
        If provided, search starts from these sample indices (e.g. first
        and last spike positions).  If None, onset/offset seconds are
        converted to sample indices.

    Returns
    -------
    (refined_onset_sec, refined_offset_sec) : tuple[float, float]
    """
    n = len(filtered)
    rms_win_samples = max(1, int(rms_window_ms * fs / 1000))
    threshold = rms_threshold_x * baseline_amp
    max_trim_samples = int(max_trim_sec * fs)

    # ── Compute cumulative-sum for fast RMS ──────────────────────────
    sq = filtered.astype(np.float64) ** 2
    cs = np.concatenate([[0], np.cumsum(sq)])

    def rms_at(idx: int) -> float:
        half = rms_win_samples // 2
        lo = max(0, idx - half)
        hi = min(n, idx + half)
        if hi <= lo:
            return 0.0
        return float(np.sqrt((cs[hi] - cs[lo]) / (hi - lo)))

    # Anchor samples
    if anchor_onset_sample is None:
        anchor_onset_sample = int(onset_sec * fs)
    if anchor_offset_sample is None:
        anchor_offset_sample = int(offset_sec * fs)

    anchor_onset_sample = max(0, min(anchor_onset_sample, n - 1))
    anchor_offset_sample = max(0, min(anchor_offset_sample, n - 1))

    step = max(1, rms_win_samples // 2)

    # ── Onset refinement ─────────────────────────────────────────────
    search_start = max(0, anchor_onset_sample - max_trim_samples)
    onset_sample = anchor_onset_sample

    # Walk backward from anchor to find where RMS drops below threshold
    for idx in range(anchor_onset_sample, search_start - 1, -rms_win_samples):
        if rms_at(idx) < threshold:
            onset_sample = idx
            break
    else:
        onset_sample = search_start

    # Walk forward to find the precise crossing above threshold
    for idx in range(onset_sample, anchor_onset_sample + 1, step):
        if rms_at(idx) >= threshold:
            onset_sample = idx
            break

    # ── Offset refinement ────────────────────────────────────────────
    search_end = min(n, anchor_offset_sample + max_trim_samples)
    offset_sample = anchor_offset_sample

    # Walk forward from anchor to find where RMS drops below threshold
    for idx in range(anchor_offset_sample, search_end + 1, rms_win_samples):
        if rms_at(idx) < threshold:
            offset_sample = idx
            break
    else:
        offset_sample = search_end

    # Walk backward to find where RMS last exceeds threshold
    for idx in range(offset_sample, anchor_offset_sample - 1, -step):
        if rms_at(idx) >= threshold:
            offset_sample = idx
            break

    return onset_sample / fs, offset_sample / fs


def refine_spike_density(
    spikes: list,
    onset_sec: float,
    offset_sec: float,
    *,
    boundary_window_sec: float = 2.0,
    min_rate_hz: float = 2.0,
    min_amplitude_x: float = 2.0,
) -> tuple[float, float] | None:
    """Refine boundaries using local spike rate and amplitude.

    Trims event edges to the first/last point where local spike rate
    and amplitude exceed the thresholds.

    Parameters
    ----------
    spikes : list
        List of spike objects with ``.time_sec`` and ``.amplitude_x``
        attributes.  Should be pre-filtered to the event region.
    onset_sec, offset_sec : float
        Current event boundaries.
    boundary_window_sec : float
        Sliding window for spike rate computation.
    min_rate_hz : float
        Minimum local spike rate at event edges.
    min_amplitude_x : float
        Minimum spike amplitude (× baseline) at event edges.

    Returns
    -------
    (refined_onset_sec, refined_offset_sec) | None
        Refined boundaries, or None if no valid boundary found.
    """
    if not spikes:
        return None

    win = boundary_window_sec

    # Trim onset: walk forward
    onset_idx = None
    for i in range(len(spikes)):
        t0 = spikes[i].time_sec
        n_in_window = sum(1 for s in spikes[i:] if s.time_sec <= t0 + win)
        local_rate = n_in_window / win if win > 0 else 0
        if local_rate >= min_rate_hz and spikes[i].amplitude_x >= min_amplitude_x:
            onset_idx = i
            break

    if onset_idx is None:
        return None

    # Trim offset: walk backward
    offset_idx = None
    for i in range(len(spikes) - 1, -1, -1):
        t0 = spikes[i].time_sec
        n_in_window = sum(1 for s in spikes[:i + 1] if s.time_sec >= t0 - win)
        local_rate = n_in_window / win if win > 0 else 0
        if local_rate >= min_rate_hz and spikes[i].amplitude_x >= min_amplitude_x:
            offset_idx = i
            break

    if offset_idx is None or offset_idx <= onset_idx:
        return None

    return spikes[onset_idx].time_sec, spikes[offset_idx].time_sec
