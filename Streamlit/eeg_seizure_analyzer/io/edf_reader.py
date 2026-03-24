"""Cross-platform EDF/EDF+ reader using pyedflib."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

import numpy as np
import pyedflib

from eeg_seizure_analyzer.io.base import (
    Annotation,
    ActivityRecording,
    ChannelPairing,
    EEGRecording,
    EEGRecordingMeta,
)


def scan_edf_channels(path: str) -> list[dict]:
    """Read channel metadata from an EDF file without loading signal data.

    Parameters
    ----------
    path : str
        Path to the .edf file.

    Returns
    -------
    list[dict]
        One dict per channel with keys: index, label, unit, fs, n_samples.
    """
    f = pyedflib.EdfReader(path)
    try:
        n_channels = f.signals_in_file
        n_samples_all = f.getNSamples()
        channels_info = []
        for i in range(n_channels):
            channels_info.append({
                "index": i,
                "label": f.getLabel(i),
                "unit": f.getPhysicalDimension(i),
                "fs": f.getSampleFrequency(i),
                "n_samples": int(n_samples_all[i]),
            })
    finally:
        f.close()
    return channels_info


def read_edf(
    path: str,
    channels: list[int] | None = None,
) -> EEGRecording:
    """Read an EDF/EDF+ file and return a unified EEGRecording.

    Parameters
    ----------
    path : str
        Path to the .edf file.
    channels : list[int] | None
        Channel indices to load. None loads all channels.

    Returns
    -------
    EEGRecording
    """
    f = pyedflib.EdfReader(path)
    try:
        n_channels_file = f.signals_in_file
        if channels is None:
            channels = list(range(n_channels_file))

        # Read metadata
        channel_names = [f.getLabel(ch) for ch in channels]
        units = [f.getPhysicalDimension(ch) for ch in channels]

        # Verify all selected channels have the same sampling rate
        sample_rates = [f.getSampleFrequency(ch) for ch in channels]
        if len(set(sample_rates)) > 1:
            raise ValueError(
                f"Selected channels have different sampling rates: {sample_rates}. "
                "Select channels with matching rates."
            )
        fs = sample_rates[0]

        # Read data
        n_samples = f.getNSamples()[channels[0]]
        data = np.zeros((len(channels), n_samples), dtype=np.float32)
        for i, ch in enumerate(channels):
            data[i] = f.readSignal(ch).astype(np.float32)

        # Read start time
        start_time = None
        try:
            start_time = f.getStartdatetime()
        except Exception:
            pass

        # Read EDF+ annotations
        annotations = _read_edf_annotations(f)

    finally:
        f.close()

    return EEGRecording(
        data=data,
        fs=fs,
        channel_names=channel_names,
        units=units,
        start_time=start_time,
        annotations=annotations,
        source_path=path,
    )


def read_edf_metadata(
    path: str,
    channels: list[int] | None = None,
) -> EEGRecordingMeta:
    """Read EDF file metadata without loading signal data.

    Returns an ``EEGRecordingMeta`` that has all info needed to display
    recording properties, offer channel selection, and plan chunked loading.
    """
    all_info = scan_edf_channels(path)
    f = pyedflib.EdfReader(path)
    try:
        start_time = None
        try:
            start_time = f.getStartdatetime()
        except Exception:
            pass
        annotations = _read_edf_annotations(f)
    finally:
        f.close()

    if channels is None:
        channels = list(range(len(all_info)))
    selected = [all_info[i] for i in channels]

    # Use first selected channel's properties
    fs = selected[0]["fs"]
    n_samples = selected[0]["n_samples"]

    return EEGRecordingMeta(
        fs=fs,
        channel_names=[ch["label"] for ch in selected],
        units=[ch["unit"] for ch in selected],
        n_samples=n_samples,
        duration_sec=n_samples / fs,
        start_time=start_time,
        source_path=path,
        annotations=annotations,
        all_channels_info=all_info,
    )


def read_edf_window(
    path: str,
    channels: list[int] | None = None,
    start_sec: float = 0.0,
    duration_sec: float = 1800.0,
) -> EEGRecording:
    """Read a specific time window from an EDF file.

    Uses pyedflib's ``readSignal(ch, start, n)`` to avoid loading the
    entire file.  Useful for the viewer and chunk-based processing.

    Parameters
    ----------
    path : path to .edf file
    channels : channel indices to load (None = all same-rate channels)
    start_sec : window start in seconds
    duration_sec : window length in seconds
    """
    f = pyedflib.EdfReader(path)
    try:
        n_channels_file = f.signals_in_file
        if channels is None:
            channels = list(range(n_channels_file))

        channel_names = [f.getLabel(ch) for ch in channels]
        units = [f.getPhysicalDimension(ch) for ch in channels]
        fs = f.getSampleFrequency(channels[0])
        total_samples = int(f.getNSamples()[channels[0]])

        start_sample = max(0, int(start_sec * fs))
        n_samples = min(int(duration_sec * fs), total_samples - start_sample)
        if n_samples <= 0:
            n_samples = 0

        data = np.zeros((len(channels), n_samples), dtype=np.float32)
        for i, ch in enumerate(channels):
            data[i] = f.readSignal(ch, start_sample, n_samples).astype(np.float32)

        start_time = None
        try:
            start_time = f.getStartdatetime()
        except Exception:
            pass

        # Filter annotations to this window
        all_ann = _read_edf_annotations(f)
        end_sec = start_sec + duration_sec
        window_ann = [
            Annotation(
                onset_sec=a.onset_sec - start_sec,
                duration_sec=a.duration_sec,
                text=a.text,
                channel=a.channel,
            )
            for a in all_ann
            if a.onset_sec >= start_sec and a.onset_sec < end_sec
        ]
    finally:
        f.close()

    return EEGRecording(
        data=data,
        fs=fs,
        channel_names=channel_names,
        units=units,
        start_time=start_time,
        annotations=window_ann,
        source_path=path,
        chunk_offset_sec=start_sec,
    )


def read_edf_chunked(
    path: str,
    channels: list[int] | None = None,
    chunk_duration_sec: float = 1800.0,
) -> Iterator[EEGRecording]:
    """Generator yielding chunks of an EDF file as EEGRecording objects.

    Each chunk is ``chunk_duration_sec`` long (default 30 min).  Memory
    usage stays constant regardless of total file length.

    Parameters
    ----------
    path : path to .edf file
    channels : channel indices to load
    chunk_duration_sec : chunk size in seconds
    """
    meta = read_edf_metadata(path, channels)
    total_sec = meta.duration_sec
    pos = 0.0

    while pos < total_sec:
        dur = min(chunk_duration_sec, total_sec - pos)
        yield read_edf_window(path, channels, start_sec=pos, duration_sec=dur)
        pos += dur


def auto_pair_channels(
    channel_info: list[dict],
) -> tuple[list[int], list[int], list[ChannelPairing]]:
    """Auto-pair EEG (Biopot) channels with Activity (Act) channels.

    Looks for common identifiers (e.g. "Ch1") in channel labels containing
    "Biopot" or "Act".  Returns the EEG indices, Activity indices, and
    a list of ChannelPairing objects.

    Parameters
    ----------
    channel_info : list[dict]
        Output from ``scan_edf_channels()`` — each dict has keys:
        index, label, unit, fs, n_samples.

    Returns
    -------
    eeg_indices : list[int]
        File-level indices of EEG channels.
    act_indices : list[int]
        File-level indices of Activity channels.
    pairings : list[ChannelPairing]
        One per EEG channel, with activity_index set if a match was found.
    """
    import re

    eeg_channels = []
    act_channels = []

    for ch in channel_info:
        label = ch["label"].strip()
        label_lower = label.lower()
        if "biopot" in label_lower:
            eeg_channels.append(ch)
        elif "act" in label_lower:
            act_channels.append(ch)

    if not eeg_channels:
        # Fallback: treat all high-rate channels as EEG
        rates = [ch["fs"] for ch in channel_info]
        max_rate = max(rates) if rates else 0
        eeg_channels = [ch for ch in channel_info if ch["fs"] == max_rate]
        act_channels = [ch for ch in channel_info if ch["fs"] != max_rate]

    def _extract_channel_id(label: str) -> str:
        """Extract a channel identifier like 'Ch1' from a label."""
        match = re.search(r'[Cc]h\s*(\d+)', label)
        if match:
            return match.group(1)
        # Try numeric prefix/suffix
        match = re.search(r'(\d+)', label)
        return match.group(1) if match else ""

    # Build lookup: channel_id → act channel info
    act_by_id = {}
    for ch in act_channels:
        ch_id = _extract_channel_id(ch["label"])
        if ch_id:
            act_by_id[ch_id] = ch

    eeg_indices = []
    act_indices_set = set()
    pairings = []

    for i, eeg_ch in enumerate(eeg_channels):
        eeg_indices.append(eeg_ch["index"])
        ch_id = _extract_channel_id(eeg_ch["label"])

        if ch_id and ch_id in act_by_id:
            act_ch = act_by_id[ch_id]
            act_indices_set.add(act_ch["index"])
            pairings.append(ChannelPairing(
                eeg_index=i,  # index within the EEG recording
                eeg_label=eeg_ch["label"],
                activity_index=len(act_indices_set) - 1,  # will be reassigned below
                activity_label=act_ch["label"],
            ))
        else:
            pairings.append(ChannelPairing(
                eeg_index=i,
                eeg_label=eeg_ch["label"],
            ))

    # Build ordered activity indices
    act_indices = sorted(act_indices_set)

    # Reassign activity_index to match position in act_indices list
    act_idx_map = {idx: pos for pos, idx in enumerate(act_indices)}
    for pairing in pairings:
        if pairing.activity_index is not None:
            # Find the original file index from the act channel
            ch_id = _extract_channel_id(pairing.activity_label)
            if ch_id and ch_id in act_by_id:
                file_idx = act_by_id[ch_id]["index"]
                pairing.activity_index = act_idx_map.get(file_idx)

    return eeg_indices, act_indices, pairings


def read_edf_paired(
    path: str,
    eeg_indices: list[int],
    act_indices: list[int],
) -> tuple[EEGRecording, ActivityRecording | None]:
    """Load EEG and activity channels as separate recordings.

    EEG channels (high rate) and activity channels (low rate) are loaded
    independently since they have different sampling rates.

    Parameters
    ----------
    path : path to .edf file
    eeg_indices : file-level indices for EEG channels
    act_indices : file-level indices for activity channels

    Returns
    -------
    (EEGRecording, ActivityRecording | None)
    """
    eeg_rec = read_edf(path, channels=eeg_indices)

    if not act_indices:
        return eeg_rec, None

    f = pyedflib.EdfReader(path)
    try:
        act_names = [f.getLabel(ch) for ch in act_indices]
        act_units = [f.getPhysicalDimension(ch) for ch in act_indices]
        act_fs = f.getSampleFrequency(act_indices[0])
        act_n_samples = int(f.getNSamples()[act_indices[0]])

        act_data = np.zeros((len(act_indices), act_n_samples), dtype=np.float32)
        for i, ch in enumerate(act_indices):
            act_data[i] = f.readSignal(ch).astype(np.float32)
    finally:
        f.close()

    act_rec = ActivityRecording(
        data=act_data,
        fs=act_fs,
        channel_names=act_names,
        units=act_units,
        source_path=path,
    )

    return eeg_rec, act_rec


def _read_edf_annotations(f: pyedflib.EdfReader) -> list[Annotation]:
    """Extract EDF+ annotations from an EdfReader."""
    annotations = []
    try:
        ann_times, ann_durations, ann_texts = f.readAnnotations()
        for onset, duration, text in zip(ann_times, ann_durations, ann_texts):
            dur = float(duration) if duration and float(duration) > 0 else None
            annotations.append(
                Annotation(
                    onset_sec=float(onset),
                    duration_sec=dur,
                    text=str(text),
                )
            )
    except Exception:
        pass
    return annotations
