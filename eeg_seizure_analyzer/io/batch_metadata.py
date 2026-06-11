"""Batch metadata — Excel template for associating metadata with EDF files.

The Excel file maps EDF filenames to cohort, group, and animal IDs.
Users prepare this alongside their EDF folder and load it during
batch or live analysis.

Template format (batch_metadata.xlsx) — one row per file. ``cohort``/``group_id``
are file-level defaults; per-channel ``animal_chN`` / ``cohort_chN`` /
``group_chN`` let channels in one file be different animals in different groups
(a blank per-channel tag falls back to the file-level default):

    filename          | cohort   | group_id | animal_ch0 | cohort_ch0 | group_ch0 | animal_ch1 | cohort_ch1 | group_ch1 | ...
    recording_001.edf | Cohort_A |          | Mouse_01   |            | Vehicle   | Mouse_02   |            | Drug_X    |
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_template(
    folder: str,
    edf_files: list[str] | None = None,
    n_channels: int = 8,
) -> str:
    """Generate a batch_metadata.xlsx template in the given folder.

    Parameters
    ----------
    folder : str
        Folder where to write the template.
    edf_files : list[str], optional
        List of EDF file paths. If None, scans folder for .edf files.
    n_channels : int
        Number of animal_ch columns to create.

    Returns
    -------
    str
        Path to the generated template.
    """
    folder_path = Path(folder)

    if edf_files is None:
        edf_files = sorted(
            str(p) for p in folder_path.rglob("*.edf") if p.is_file()
        )

    filenames = [Path(f).name for f in edf_files]

    n = len(filenames)
    data = {
        "filename": filenames,
        # File-level defaults, applied to any channel without a per-channel tag.
        "cohort": [""] * n,
        "group_id": [""] * n,
    }
    # Per channel: animal ID plus optional per-channel cohort/group overrides
    # (channels in one file can be different animals in different groups).
    for i in range(n_channels):
        data[f"animal_ch{i}"] = [""] * n
        data[f"cohort_ch{i}"] = [""] * n
        data[f"group_ch{i}"] = [""] * n

    df = pd.DataFrame(data)

    out_path = folder_path / "batch_metadata.xlsx"
    df.to_excel(str(out_path), index=False)
    return str(out_path)


def load_metadata(excel_path: str) -> dict[str, dict]:
    """Load batch metadata from an Excel file.

    Returns
    -------
    dict[str, dict]
        Mapping of filename → ``{cohort, group_id, channel_ids,
        channel_cohort, channel_group}``, where ``cohort``/``group_id`` are the
        file-level defaults and the ``channel_*`` maps are per channel (a blank
        per-channel tag falls back to the file-level default).
    """
    df = pd.read_excel(excel_path, dtype=str).fillna("")

    result = {}
    for _, row in df.iterrows():
        fname = row.get("filename", "")
        if not fname:
            continue

        file_cohort = row.get("cohort", "")
        file_group = row.get("group_id", "")
        channel_ids, ch_cohort, ch_group = {}, {}, {}
        chans = set()

        for col in df.columns:
            for prefix, store, is_animal in (
                ("animal_ch", channel_ids, True),
                ("cohort_ch", ch_cohort, False),
                ("group_ch", ch_group, False),
            ):
                if col.startswith(prefix):
                    try:
                        ci = int(col[len(prefix):])
                    except ValueError:
                        break
                    val = row.get(col, "")
                    if val:
                        store[ci] = val
                        if is_animal:
                            chans.add(ci)
                    break

        # Resolve per-channel cohort/group, falling back to the file-level
        # default, for every channel with an animal ID or an explicit tag.
        chans |= set(ch_cohort) | set(ch_group)
        channel_cohort = {ci: (ch_cohort.get(ci) or file_cohort)
                          for ci in chans if (ch_cohort.get(ci) or file_cohort)}
        channel_group = {ci: (ch_group.get(ci) or file_group)
                         for ci in chans if (ch_group.get(ci) or file_group)}

        result[fname] = {
            "cohort": file_cohort,
            "group_id": file_group,
            "channel_ids": channel_ids,
            "channel_cohort": channel_cohort,
            "channel_group": channel_group,
        }

    return result


def get_metadata_for_file(
    metadata: dict[str, dict],
    edf_path: str,
) -> dict:
    """Look up metadata for a specific EDF file.

    Matches by filename (not full path).

    Returns
    -------
    dict
        {cohort, group_id, channel_ids} or empty dict if not found.
    """
    fname = Path(edf_path).name
    return metadata.get(fname, {})


# ── Live channel template ────────────────────────────────────────────
# Live mode has no per-file load step and files arrive with unknown names, so
# metadata is keyed by CHANNEL (a live session runs a fixed montage) and applied
# to every incoming file. Columns: channel, animal_id, cohort, group.


def empty_batch_template(n_channels: int = 8) -> "pd.DataFrame":
    """Empty batch-metadata template (headers + one blank row to fill)."""
    data = {"filename": [""], "cohort": [""], "group_id": [""]}
    for i in range(n_channels):
        data[f"animal_ch{i}"] = [""]
        data[f"cohort_ch{i}"] = [""]
        data[f"group_ch{i}"] = [""]
    return pd.DataFrame(data)


def empty_live_template(n_channels: int = 8) -> "pd.DataFrame":
    """Empty live channel template — one row per channel index."""
    return pd.DataFrame({
        "channel": list(range(n_channels)),
        "animal_id": [""] * n_channels,
        "cohort": [""] * n_channels,
        "group": [""] * n_channels,
    })


def load_live_template(path: str) -> dict:
    """Load a live channel template into a ``file_metadata``-shaped dict applied
    to every incoming file by channel index:
    ``{channel_ids, channel_cohort, channel_group}``.
    """
    df = pd.read_excel(path, dtype=str).fillna("")
    channel_ids, ch_cohort, ch_group = {}, {}, {}
    for _, row in df.iterrows():
        try:
            ch = int(float(row.get("channel", "")))
        except (ValueError, TypeError):
            continue
        if row.get("animal_id", ""):
            channel_ids[ch] = row["animal_id"]
        if row.get("cohort", ""):
            ch_cohort[ch] = row["cohort"]
        if row.get("group", ""):
            ch_group[ch] = row["group"]
    return {"channel_ids": channel_ids,
            "channel_cohort": ch_cohort,
            "channel_group": ch_group}
