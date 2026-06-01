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

    channel_names = [f.channels[ch].name for ch in channels]

    # Per-channel units — LabChart records can carry different units (e.g.
    # channel idle in record 0 leaves units[0] empty, or the user changed
    # range/unit between start/stop blocks). Use the first non-empty value
    # so an idle first record doesn't strip the unit; warn if a channel's
    # units actually drift across non-empty records.
    units: list[str] = []
    for ch in channels:
        per_record_units = list(f.channels[ch].units)
        non_empty = [u for u in per_record_units if u]
        if not non_empty:
            units.append("")
            continue
        units.append(non_empty[0])
        if len(set(non_empty)) > 1:
            print(
                f"[adicht_reader] WARNING: channel '{f.channels[ch].name}' "
                f"has varying units across records: {per_record_units}. "
                f"Using '{non_empty[0]}' for the whole recording — physical "
                "values may not be comparable across record boundaries.",
                file=sys.stderr,
            )

    # Sampling rates must match across non-empty records, otherwise
    # concatenation produces a time-distorted signal. Refuse with a
    # per-channel breakdown so the offending record is easy to spot.
    fs_by_channel = {ch: list(f.channels[ch].fs) for ch in channels}
    non_empty_fs: set[float] = set()
    for ch_fs in fs_by_channel.values():
        for v in ch_fs:
            if v > 0:
                non_empty_fs.add(float(v))
    if not non_empty_fs:
        raise ValueError("No records with samples found in file.")
    if len(non_empty_fs) > 1:
        detail_lines = []
        for ch in channels:
            rates = fs_by_channel[ch]
            if any(r > 0 for r in rates):
                detail_lines.append(
                    f"  channel '{f.channels[ch].name}': "
                    + ", ".join(f"rec{i + 1}={r}Hz" for i, r in enumerate(rates))
                )
        detail = "\n".join(detail_lines)
        raise ValueError(
            "Inconsistent sampling rates across records — cannot concatenate.\n"
            f"Rates seen: {sorted(non_empty_fs)}\n"
            f"Per-channel breakdown:\n{detail}\n"
            "Most likely the LabChart sampling rate was changed between "
            "start/stop blocks. Either split the .adicht into per-record "
            "files before conversion, or re-export with a consistent rate."
        )
    fs = non_empty_fs.pop()

    # Read and concatenate data across all records
    all_data = []
    record_infos = []
    cumulative_samples = 0

    for rec_idx in range(f.n_records):
        record_id = rec_idx + 1  # ADI SDK uses 1-based indexing

        # Per-channel sample counts for this record. Within a single record,
        # channels can independently be empty (paused/disabled in LabChart),
        # so we must check each one — not just channel 0.
        per_ch_n = [int(f.channels[ch].n_samples[rec_idx]) for ch in channels]
        rec_n = max(per_ch_n) if per_ch_n else 0
        if rec_n == 0:
            continue

        # Read data for all selected channels in this record. Empty channels
        # stay as zeros so EDF channel time-alignment is preserved.
        rec_data = np.zeros((len(channels), rec_n), dtype=np.float32)
        for i, ch in enumerate(channels):
            n = per_ch_n[i]
            if n == 0:
                continue
            ch_data = f.channels[ch].get_data(record_id).astype(np.float32)
            rec_data[i, : len(ch_data)] = ch_data

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
                n_samples=rec_n,
                start_time=start_time,
            )
        )

        all_data.append(rec_data)
        cumulative_samples += rec_n

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
