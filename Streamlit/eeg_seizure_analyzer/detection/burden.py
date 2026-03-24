"""Seizure burden computation and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent


@dataclass
class SeizureBurden:
    """Summary statistics for seizure burden."""

    total_seizure_time_sec: float
    n_seizures: int
    seizure_frequency_per_hour: float
    mean_duration_sec: float
    median_duration_sec: float
    max_duration_sec: float
    min_duration_sec: float
    recording_duration_sec: float
    percent_time_in_seizure: float
    hourly_counts: list[int] = field(default_factory=list)
    severity_counts: dict[str, int] = field(default_factory=dict)


def compute_burden(
    events: list[DetectedEvent],
    recording_duration_sec: float,
    channel: int | None = None,
) -> SeizureBurden:
    """Compute seizure burden from detected events.

    Parameters
    ----------
    events : list[DetectedEvent]
        Detected seizure events.
    recording_duration_sec : float
        Total recording duration in seconds.
    channel : int | None
        If provided, filter events to this channel only.

    Returns
    -------
    SeizureBurden
    """
    seizures = [e for e in events if e.event_type == "seizure"]
    if channel is not None:
        seizures = [e for e in seizures if e.channel == channel]

    if not seizures:
        return SeizureBurden(
            total_seizure_time_sec=0.0,
            n_seizures=0,
            seizure_frequency_per_hour=0.0,
            mean_duration_sec=0.0,
            median_duration_sec=0.0,
            max_duration_sec=0.0,
            min_duration_sec=0.0,
            recording_duration_sec=recording_duration_sec,
            percent_time_in_seizure=0.0,
            hourly_counts=_compute_hourly_counts([], recording_duration_sec),
            severity_counts={"mild": 0, "moderate": 0, "severe": 0},
        )

    durations = np.array([s.duration_sec for s in seizures])
    total_time = float(np.sum(durations))
    hours = recording_duration_sec / 3600

    severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
    for s in seizures:
        if s.severity in severity_counts:
            severity_counts[s.severity] += 1

    return SeizureBurden(
        total_seizure_time_sec=total_time,
        n_seizures=len(seizures),
        seizure_frequency_per_hour=len(seizures) / hours if hours > 0 else 0.0,
        mean_duration_sec=float(np.mean(durations)),
        median_duration_sec=float(np.median(durations)),
        max_duration_sec=float(np.max(durations)),
        min_duration_sec=float(np.min(durations)),
        recording_duration_sec=recording_duration_sec,
        percent_time_in_seizure=(total_time / recording_duration_sec * 100) if recording_duration_sec > 0 else 0.0,
        hourly_counts=_compute_hourly_counts(seizures, recording_duration_sec),
        severity_counts=severity_counts,
    )


def compute_spike_rate(
    events: list[DetectedEvent],
    recording_duration_sec: float,
    bin_sec: float = 60.0,
    channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spike rate over time.

    Parameters
    ----------
    events : list[DetectedEvent]
        Detected spike events.
    recording_duration_sec : float
        Total recording duration.
    bin_sec : float
        Bin width for rate computation (default 60s = per minute).
    channel : int | None
        Filter to this channel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (time_bins_sec, rate_per_min) arrays.
    """
    spikes = [e for e in events if e.event_type == "spike"]
    if channel is not None:
        spikes = [e for e in spikes if e.channel == channel]

    n_bins = max(1, int(np.ceil(recording_duration_sec / bin_sec)))
    counts = np.zeros(n_bins, dtype=np.float32)

    for spike in spikes:
        bin_idx = int(spike.onset_sec / bin_sec)
        if 0 <= bin_idx < n_bins:
            counts[bin_idx] += 1

    # Convert to rate per minute
    rate = counts / (bin_sec / 60)
    time_bins = np.arange(n_bins) * bin_sec

    return time_bins, rate


def _compute_hourly_counts(
    seizures: list[DetectedEvent], recording_duration_sec: float
) -> list[int]:
    """Count seizures per hour."""
    n_hours = max(1, int(np.ceil(recording_duration_sec / 3600)))
    counts = [0] * n_hours
    for s in seizures:
        hour_idx = int(s.onset_sec / 3600)
        if 0 <= hour_idx < n_hours:
            counts[hour_idx] += 1
    return counts
