"""Unified data containers for EEG recordings regardless of source format."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Annotation:
    """A single event annotation (manual or detected)."""

    onset_sec: float
    duration_sec: float | None = None
    text: str = ""
    channel: int | None = None  # None means applies to all channels


@dataclass
class RecordInfo:
    """Metadata for a single record/block within a multi-block recording."""

    index: int
    start_sample: int
    n_samples: int
    start_time: datetime | None = None


@dataclass
class EEGRecordingMeta:
    """Lightweight metadata container — no signal data.

    Created by ``scan_edf_channels()`` so the app can show recording info
    and offer channel selection before loading any data.
    """

    fs: float
    channel_names: list[str]
    units: list[str]
    n_samples: int
    duration_sec: float
    start_time: datetime | None = None
    source_path: str = ""
    annotations: list[Annotation] = field(default_factory=list)
    all_channels_info: list[dict] = field(default_factory=list)


@dataclass
class ChannelPairing:
    """Maps each EEG channel to its paired activity channel (if any).

    Created during upload by auto-pairing logic or user assignment.
    """
    eeg_index: int          # index in the EEG recording's data array
    eeg_label: str
    activity_index: int | None = None  # index in the activity recording's data array
    activity_label: str = ""


@dataclass
class ActivityRecording:
    """Low-rate activity/movement signal loaded alongside EEG.

    Stored separately because the sampling rate differs (e.g. 2 Hz vs 2000 Hz).
    """
    data: np.ndarray            # shape (n_channels, n_samples), float32
    fs: float                   # sampling rate (e.g. 2.0 Hz)
    channel_names: list[str]
    units: list[str]
    source_path: str = ""

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs

    def get_channel_data(self, channel: int | str) -> np.ndarray:
        if isinstance(channel, str):
            channel = self.channel_names.index(channel)
        return self.data[channel]

    def get_time_vector(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.fs


@dataclass
class EEGRecording:
    """Unified in-memory representation of an EEG recording.

    Produced by both the .adicht and .edf readers.
    All downstream processing and detection code depends only on this class.
    """

    data: np.ndarray  # shape (n_channels, n_samples), float32
    fs: float  # sampling rate in Hz
    channel_names: list[str]
    units: list[str]
    start_time: datetime | None = None
    annotations: list[Annotation] = field(default_factory=list)
    source_path: str = ""
    records: list[RecordInfo] | None = None  # original block boundaries
    chunk_offset_sec: float = 0.0  # offset in full recording (for chunked loading)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs

    def get_channel_data(self, channel: int | str) -> np.ndarray:
        """Get data for a single channel by index or name."""
        if isinstance(channel, str):
            channel = self.channel_names.index(channel)
        return self.data[channel]

    def get_time_vector(self) -> np.ndarray:
        """Return time axis in seconds."""
        return np.arange(self.n_samples) / self.fs

    def get_window(self, start_sec: float, end_sec: float) -> EEGRecording:
        """Extract a time window as a new EEGRecording."""
        start_idx = max(0, int(start_sec * self.fs))
        end_idx = min(self.n_samples, int(end_sec * self.fs))
        window_annotations = [
            Annotation(
                onset_sec=a.onset_sec - start_sec,
                duration_sec=a.duration_sec,
                text=a.text,
                channel=a.channel,
            )
            for a in self.annotations
            if a.onset_sec >= start_sec
            and a.onset_sec < end_sec
        ]
        return EEGRecording(
            data=self.data[:, start_idx:end_idx],
            fs=self.fs,
            channel_names=self.channel_names,
            units=self.units,
            start_time=self.start_time,
            annotations=window_annotations,
            source_path=self.source_path,
        )
