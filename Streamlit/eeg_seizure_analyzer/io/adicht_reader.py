"""Windows-only reader for LabChart .adicht files using the ADInstruments SDK."""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np

from eeg_seizure_analyzer.io.base import Annotation, EEGRecording, RecordInfo


def read_adicht(
    path: str,
    channels: list[int] | None = None,
) -> EEGRecording:
    """Read a .adicht file and return a unified EEGRecording.

    Concatenates all records into a single continuous array.
    Extracts LabChart comments as Annotation objects.

    Parameters
    ----------
    path : str
        Path to the .adicht file.
    channels : list[int] | None
        Channel indices (0-based) to load. None loads all channels.

    Returns
    -------
    EEGRecording

    Raises
    ------
    ImportError
        If not on Windows or adi-reader is not installed.
    """
    if sys.platform != "win32":
        raise ImportError("adicht reader is only available on Windows.")

    try:
        import adi
    except ImportError:
        raise ImportError(
            "adi-reader package not installed. Install with: pip install adi-reader"
        )

    f = adi.read_file(path)

    if channels is None:
        channels = list(range(f.n_channels))

    # Get channel metadata from first record
    channel_names = [f.channels[ch].name for ch in channels]
    units = [f.channels[ch].units[0] for ch in channels]

    # Verify sampling rates are consistent across records for selected channels
    fs_values = set()
    for ch in channels:
        for rec_fs in f.channels[ch].fs:
            if rec_fs > 0:  # skip empty records
                fs_values.add(rec_fs)
    if len(fs_values) > 1:
        raise ValueError(
            f"Inconsistent sampling rates across records: {fs_values}. "
            "Cannot concatenate records with different rates."
        )
    fs = fs_values.pop()

    # Read and concatenate data across all records
    all_data = []
    record_infos = []
    cumulative_samples = 0

    for rec_idx in range(f.n_records):
        record_id = rec_idx + 1  # ADI SDK uses 1-based indexing

        # Check if this record has data (n_samples > 0 for first selected channel)
        n_samples = f.channels[channels[0]].n_samples[rec_idx]
        if n_samples == 0:
            continue

        # Read data for all selected channels in this record
        rec_data = np.zeros((len(channels), n_samples), dtype=np.float32)
        for i, ch in enumerate(channels):
            rec_data[i] = f.channels[ch].get_data(record_id).astype(np.float32)

        # Track record boundaries
        start_time = None
        try:
            start_time = f.records[rec_idx].record_time.rec_datetime
        except Exception:
            pass

        record_infos.append(
            RecordInfo(
                index=rec_idx,
                start_sample=cumulative_samples,
                n_samples=n_samples,
                start_time=start_time,
            )
        )

        all_data.append(rec_data)
        cumulative_samples += n_samples

    # Concatenate all records
    data = np.concatenate(all_data, axis=1) if all_data else np.zeros((len(channels), 0), dtype=np.float32)

    # Extract annotations from comments
    annotations = _extract_annotations(f, record_infos, fs)

    # Get recording start time from first record
    start_time = None
    if record_infos and record_infos[0].start_time:
        start_time = record_infos[0].start_time

    return EEGRecording(
        data=data,
        fs=fs,
        channel_names=channel_names,
        units=units,
        start_time=start_time,
        annotations=annotations,
        source_path=path,
        records=record_infos,
    )


def _extract_annotations(
    f,
    record_infos: list[RecordInfo],
    fs: float,
) -> list[Annotation]:
    """Convert LabChart comments to Annotation objects.

    Comment times are relative to their record start. We offset them
    to be relative to the start of the concatenated recording.
    """
    annotations = []
    for rec_info in record_infos:
        record = f.records[rec_info.index]
        offset_sec = rec_info.start_sample / fs

        for comment in record.comments:
            annotations.append(
                Annotation(
                    onset_sec=comment.time + offset_sec,
                    duration_sec=None,
                    text=comment.text,
                    channel=comment.channel_ if comment.channel_ > 0 else None,
                )
            )

    return annotations
