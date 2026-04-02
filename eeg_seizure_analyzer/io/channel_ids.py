"""Channel-to-Animal-ID mapping persistence.

Stores a mapping of EEG channel index → animal ID next to the EDF file
as ``{filename}_ned_channels.json``.  Loaded automatically on file open
and saved whenever the user edits IDs.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd


def _channel_ids_path(edf_path: str) -> Path:
    """Derive the channel IDs JSON path from an EDF file path."""
    p = Path(edf_path)
    return p.with_suffix("").with_name(p.stem + "_ned_channels.json")


def save_channel_ids(edf_path: str, mapping: dict[int, str]) -> Path:
    """Save channel → animal ID mapping next to the EDF file.

    Parameters
    ----------
    edf_path : path to the EDF file
    mapping : {channel_index: animal_id_string, ...}

    Returns
    -------
    Path of the written file.
    """
    out_path = _channel_ids_path(edf_path)

    payload = {
        "edf_path": str(edf_path),
        "channel_ids": {str(k): v for k, v in mapping.items()},
    }

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".ned_chid_", dir=str(out_path.parent)
    )
    try:
        with os.fdopen(fd, "w") as fp:
            json.dump(payload, fp, indent=2)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return out_path


def load_channel_ids(edf_path: str) -> dict[int, str] | None:
    """Load channel → animal ID mapping for an EDF file.

    Returns
    -------
    dict mapping channel index (int) to animal ID (str), or None if
    no mapping file exists.
    """
    path = _channel_ids_path(edf_path)
    if not path.exists():
        return None
    try:
        with open(path) as fp:
            data = json.load(fp)
        raw = data.get("channel_ids", {})
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return None


def get_animal_id(edf_path: str, channel: int) -> str:
    """Get the animal ID for a specific channel, or empty string."""
    mapping = load_channel_ids(edf_path)
    if mapping is None:
        return ""
    return mapping.get(channel, "")


# ---------------------------------------------------------------------------
# Excel-based bulk channel ID management
# ---------------------------------------------------------------------------

_CHANNEL_COL_RE = re.compile(r"^(?:Ch|Channel\s*)(\d+)$", re.IGNORECASE)


def _normalise_channel_columns(columns: list[str]) -> dict[str, int]:
    """Map DataFrame column names to 0-based channel indices.

    Accepts ``Ch1``, ``Ch2``, … or ``Channel 1``, ``Channel 2``, … etc.
    Column numbers are 1-based (user-facing) and converted to 0-based
    indices internally.  Returns {column_name: channel_index}.
    """
    result: dict[str, int] = {}
    for col in columns:
        m = _CHANNEL_COL_RE.match(str(col).strip())
        if m:
            # 1-based label → 0-based index
            result[col] = int(m.group(1)) - 1
    return result


def read_channel_ids_excel(folder: str) -> dict[str, dict[int, str]]:
    """Read channel-to-animal-ID mappings from an Excel file.

    Parameters
    ----------
    folder : str
        Directory containing ``channel_ids.xlsx``.

    Returns
    -------
    dict mapping EDF filename (without path) to
    ``{channel_index: animal_id}`` for every non-empty cell.
    If the file does not exist, returns an empty dict.
    """
    xlsx_path = Path(folder) / "channel_ids.xlsx"
    if not xlsx_path.exists():
        return {}

    df = pd.read_excel(xlsx_path, dtype=str)

    # Identify the file column (must be called "File")
    if "File" not in df.columns:
        return {}

    col_map = _normalise_channel_columns(list(df.columns))
    if not col_map:
        return {}

    result: dict[str, dict[int, str]] = {}
    for _, row in df.iterrows():
        fname = row.get("File")
        if not isinstance(fname, str) or not fname.strip():
            continue
        fname = fname.strip()
        ch_ids: dict[int, str] = {}
        for col_name, ch_idx in col_map.items():
            val = row.get(col_name)
            if isinstance(val, str) and val.strip():
                ch_ids[ch_idx] = val.strip()
        if ch_ids:
            result[fname] = ch_ids

    return result


def generate_channel_ids_template(
    folder: str, edf_files: list[str]
) -> str:
    """Generate a template ``channel_ids.xlsx`` in *folder*.

    Parameters
    ----------
    folder : str
        Directory where the file will be created.
    edf_files : list[str]
        List of EDF filenames (basenames, without path).

    Returns
    -------
    str – absolute path to the generated file.
    """
    ch_cols = [f"Ch{i}" for i in range(1, 9)]
    data = {"File": edf_files}
    for col in ch_cols:
        data[col] = [""] * len(edf_files)

    df = pd.DataFrame(data)
    out_path = Path(folder) / "channel_ids.xlsx"
    df.to_excel(str(out_path), index=False, engine="openpyxl")
    return str(out_path)


def apply_channel_ids_from_excel(folder: str) -> int:
    """Read ``channel_ids.xlsx`` and save ``_ned_channels.json`` per EDF.

    For each row in the spreadsheet the function locates the corresponding
    EDF file in *folder* (non-recursive) and writes (or overwrites) its
    ``_ned_channels.json`` sidecar via :func:`save_channel_ids`.

    Parameters
    ----------
    folder : str
        Directory containing both ``channel_ids.xlsx`` and the EDF files.

    Returns
    -------
    int – number of EDF files whose channel IDs were updated.
    """
    mappings = read_channel_ids_excel(folder)
    if not mappings:
        return 0

    folder_path = Path(folder)
    updated = 0
    for fname, ch_ids in mappings.items():
        edf_path = folder_path / fname
        if not edf_path.exists():
            continue
        save_channel_ids(str(edf_path), ch_ids)
        updated += 1

    return updated
