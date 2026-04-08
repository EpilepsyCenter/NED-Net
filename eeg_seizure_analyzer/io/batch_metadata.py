"""Batch metadata — Excel template for associating metadata with EDF files.

The Excel file maps EDF filenames to cohort, group, and animal IDs.
Users prepare this alongside their EDF folder and load it during
batch or live analysis.

Template format (batch_metadata.xlsx):
    filename          | cohort    | group_id | animal_ch0 | animal_ch1 | ...
    recording_001.edf | Cohort_A  | Vehicle  | Mouse_01   | Mouse_02   |
    recording_002.edf | Cohort_A  | Drug_X   | Mouse_03   | Mouse_04   |
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

    data = {
        "filename": filenames,
        "cohort": [""] * len(filenames),
        "group_id": [""] * len(filenames),
    }
    for i in range(n_channels):
        data[f"animal_ch{i}"] = [""] * len(filenames)

    df = pd.DataFrame(data)

    out_path = folder_path / "batch_metadata.xlsx"
    df.to_excel(str(out_path), index=False)
    return str(out_path)


def load_metadata(excel_path: str) -> dict[str, dict]:
    """Load batch metadata from an Excel file.

    Returns
    -------
    dict[str, dict]
        Mapping of filename → {cohort, group_id, channel_ids: {ch_idx: animal_id}}
    """
    df = pd.read_excel(excel_path, dtype=str).fillna("")

    result = {}
    for _, row in df.iterrows():
        fname = row.get("filename", "")
        if not fname:
            continue

        meta = {
            "cohort": row.get("cohort", ""),
            "group_id": row.get("group_id", ""),
            "channel_ids": {},
        }

        # Collect animal_chN columns
        for col in df.columns:
            if col.startswith("animal_ch"):
                try:
                    ch_idx = int(col.replace("animal_ch", ""))
                    val = row.get(col, "")
                    if val:
                        meta["channel_ids"][ch_idx] = val
                except ValueError:
                    continue

        result[fname] = meta

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
