"""Base classes for event detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from eeg_seizure_analyzer.io.base import EEGRecording


@dataclass
class DetectedEvent:
    """A detected EEG event (seizure or interictal spike)."""

    onset_sec: float
    offset_sec: float
    duration_sec: float
    channel: int
    event_type: str  # "seizure" or "spike"
    confidence: float = 1.0  # 0-1
    severity: str | None = None  # "mild", "moderate", "severe" (seizures only)
    features: dict = field(default_factory=dict)
    movement_flag: bool = False  # True if co-occurring with movement artifact
    animal_id: str = ""
    quality_metrics: dict = field(default_factory=dict)

    @property
    def midpoint_sec(self) -> float:
        return (self.onset_sec + self.offset_sec) / 2

    def to_dict(self) -> dict:
        """Export event as a flat dictionary for CSV/DataFrame export."""
        d = {
            "onset_sec": self.onset_sec,
            "offset_sec": self.offset_sec,
            "duration_sec": self.duration_sec,
            "channel": self.channel,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "severity": self.severity or "",
            "movement_flag": self.movement_flag,
            "animal_id": self.animal_id,
        }
        # Flatten features and quality_metrics into the dict
        for k, v in self.features.items():
            d[f"feat_{k}"] = v
        for k, v in self.quality_metrics.items():
            d[f"qm_{k}"] = v
        return d


class DetectorBase(ABC):
    """Abstract base class for event detectors.

    Subclass this to implement rule-based or ML-based detectors.
    All detectors share the same interface.
    """

    @abstractmethod
    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        **params,
    ) -> list[DetectedEvent]:
        """Detect events in a single channel.

        Parameters
        ----------
        recording : EEGRecording
            The loaded EEG recording.
        channel : int
            Channel index to analyze.
        **params
            Detector-specific parameters.

        Returns
        -------
        list[DetectedEvent]
            Detected events sorted by onset time.
        """
        ...

    def detect_all_channels(
        self,
        recording: EEGRecording,
        channels: list[int] | None = None,
        **params,
    ) -> list[DetectedEvent]:
        """Run detection on multiple channels independently.

        Parameters
        ----------
        recording : EEGRecording
            The loaded EEG recording.
        channels : list[int] | None
            Channel indices. None = all channels.
        **params
            Passed to detect().

        Returns
        -------
        list[DetectedEvent]
            All detected events across channels, sorted by onset.
        """
        if channels is None:
            channels = list(range(recording.n_channels))

        all_events = []
        for ch in channels:
            events = self.detect(recording, ch, **params)
            all_events.extend(events)

        return sorted(all_events, key=lambda e: e.onset_sec)


def detect_chunked(
    detector: DetectorBase,
    path: str,
    channels: list[int] | None = None,
    chunk_duration_sec: float = 1800.0,
    overlap_sec: float = 30.0,
    **detect_kwargs,
) -> list[DetectedEvent]:
    """Run detection on a large EDF file in memory-efficient chunks.

    Two-pass approach:
      1. Stream through file computing global percentile baseline.
      2. Detect in chunks with overlap, deduplicate boundary events.

    Parameters
    ----------
    detector : DetectorBase
        The detector instance (e.g. SeizureDetector()).
    path : str
        Path to the EDF file.
    channels : list[int] | None
        Channel indices to process. None = all.
    chunk_duration_sec : float
        Chunk size in seconds (default 30 min).
    overlap_sec : float
        Overlap between adjacent chunks to catch boundary events.
    **detect_kwargs
        Extra keyword arguments passed to detector.detect() (e.g. params).

    Returns
    -------
    list[DetectedEvent]
        All detected events with times relative to file start.
    """
    from eeg_seizure_analyzer.io.edf_reader import (
        read_edf_metadata,
        read_edf_window,
    )
    from eeg_seizure_analyzer.processing.features import compute_zscore_baseline
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

    meta = read_edf_metadata(path, channels)
    total_sec = meta.duration_sec

    # Resolve channels
    if channels is None:
        channels = list(range(len(meta.channel_names)))

    # Extract params if provided
    params = detect_kwargs.get("params", None)
    bandpass_low = getattr(params, "bandpass_low", 1.0)
    bandpass_high = getattr(params, "bandpass_high", 50.0)
    baseline_percentile = getattr(params, "baseline_percentile", 15)
    baseline_rms_window_sec = getattr(params, "baseline_rms_window_sec", 10.0)

    # ── Pass 1: Compute global baseline per channel ──────────────────
    # Stream through in chunks to avoid loading entire file
    rms_windows_per_channel: dict[int, list[float]] = {ch: [] for ch in channels}
    win_samples = int(baseline_rms_window_sec * meta.fs)

    pos = 0.0
    while pos < total_sec:
        dur = min(chunk_duration_sec, total_sec - pos)
        chunk = read_edf_window(path, channels, start_sec=pos, duration_sec=dur)
        for i, ch in enumerate(channels):
            filtered = bandpass_filter(
                chunk.data[i], meta.fs, bandpass_low, bandpass_high
            )
            # Compute RMS in non-overlapping windows
            n_wins = len(filtered) // win_samples
            for w in range(n_wins):
                seg = filtered[w * win_samples : (w + 1) * win_samples]
                rms_windows_per_channel[ch].append(
                    float(np.sqrt(np.mean(seg ** 2)))
                )
        pos += dur

    # Compute percentile baseline from all RMS windows
    baseline_per_channel: dict[int, float] = {}
    for ch in channels:
        rms_arr = np.array(rms_windows_per_channel[ch])
        if len(rms_arr) > 0:
            baseline_per_channel[ch] = max(
                float(np.percentile(rms_arr, baseline_percentile)), 1e-10
            )
        else:
            baseline_per_channel[ch] = 1e-10

    # Free memory
    del rms_windows_per_channel

    # ── Pass 2: Detect in overlapping chunks ─────────────────────────
    all_events: list[DetectedEvent] = []
    pos = 0.0

    while pos < total_sec:
        # Each chunk extends by overlap_sec to catch boundary events
        dur = min(chunk_duration_sec + overlap_sec, total_sec - pos)
        chunk = read_edf_window(path, channels, start_sec=pos, duration_sec=dur)

        for i, ch in enumerate(channels):
            baseline_rms = baseline_per_channel[ch]
            events = detector.detect(
                chunk, i, baseline_rms=baseline_rms, **detect_kwargs
            )

            # Adjust event times to be relative to file start
            for ev in events:
                ev.onset_sec += pos
                ev.offset_sec += pos

                # Only keep events whose onset falls within the non-overlap
                # portion of this chunk (except for the last chunk)
                chunk_end = pos + chunk_duration_sec
                if pos + chunk_duration_sec < total_sec:
                    # Not the last chunk: only keep if onset < chunk_end
                    if ev.onset_sec < chunk_end:
                        all_events.append(ev)
                else:
                    # Last chunk: keep all
                    all_events.append(ev)

        pos += chunk_duration_sec

    # ── Deduplicate overlapping events ───────────────────────────────
    all_events.sort(key=lambda e: (e.channel, e.onset_sec))
    deduped: list[DetectedEvent] = []
    for ev in all_events:
        if deduped and ev.channel == deduped[-1].channel:
            prev = deduped[-1]
            # Events overlap if their time ranges intersect
            if ev.onset_sec < prev.offset_sec:
                # Keep the one with higher confidence
                if ev.confidence > prev.confidence:
                    deduped[-1] = ev
                continue
        deduped.append(ev)

    return sorted(deduped, key=lambda e: e.onset_sec)
