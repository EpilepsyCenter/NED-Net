"""Dataset definition storage — scan, save, load training datasets."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DATASETS_DIR = Path.home() / ".eeg_seizure_analyzer" / "datasets"

# Annotation filename suffixes (must match annotation_store.py conventions)
_SEIZURE_SUFFIX = "_ned_annotations.json"
_SPIKE_SUFFIX = "_ned_spike_annotations.json"


def scan_annotation_files(
    folder: str,
    annotation_type: str = "seizure",
) -> list[dict]:
    """Recursively scan *folder* for annotation JSON files.

    Parameters
    ----------
    folder : str
        Root directory to scan.
    annotation_type : str
        ``"seizure"`` or ``"spike"`` — determines which annotation files
        to look for.

    Returns
    -------
    list[dict]
        One dict per annotated file with keys:
        ``edf_path``, ``annotation_path``, ``n_confirmed``, ``n_rejected``,
        ``n_pending``, ``n_total``.
    """
    suffix = _SEIZURE_SUFFIX if annotation_type == "seizure" else _SPIKE_SUFFIX
    results: list[dict] = []

    for dirpath, _dirs, filenames in os.walk(folder):
        for fname in sorted(filenames):
            if not fname.endswith(suffix):
                continue
            ann_path = os.path.join(dirpath, fname)
            # Derive EDF path: strip suffix, add .edf
            stem = fname[: -len(suffix)]
            edf_path = os.path.join(dirpath, stem + ".edf")

            # Read annotation counts
            try:
                with open(ann_path, "r") as fp:
                    data = json.load(fp)
                annotations = data.get("annotations", [])
            except Exception:
                annotations = []

            n_confirmed = sum(
                1 for a in annotations if a.get("label") == "confirmed"
            )
            n_rejected = sum(
                1 for a in annotations if a.get("label") == "rejected"
            )
            n_pending = sum(
                1 for a in annotations if a.get("label", "pending") == "pending"
            )

            results.append(
                {
                    "edf_path": edf_path,
                    "annotation_path": ann_path,
                    "n_confirmed": n_confirmed,
                    "n_rejected": n_rejected,
                    "n_pending": n_pending,
                    "n_total": len(annotations),
                }
            )

    return results


def save_dataset(definition: dict) -> Path:
    """Save a dataset definition to ``~/.eeg_seizure_analyzer/datasets/``.

    Uses atomic write (temp file + rename) to prevent corruption.

    Returns the path of the written file.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    name = definition["name"]
    out_path = DATASETS_DIR / f"{name}.json"

    # Add timestamp
    definition.setdefault("created", datetime.now(timezone.utc).isoformat())
    definition["updated"] = datetime.now(timezone.utc).isoformat()

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".ned_ds_", dir=str(DATASETS_DIR)
    )
    try:
        with os.fdopen(fd, "w") as fp:
            json.dump(definition, fp, indent=2)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return out_path


def load_dataset(name: str) -> dict | None:
    """Load a dataset definition by name.  Returns *None* if not found."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as fp:
            return json.load(fp)
    except Exception:
        return None


def list_datasets() -> list[str]:
    """Return sorted list of available dataset names."""
    if not DATASETS_DIR.exists():
        return []
    return sorted(p.stem for p in DATASETS_DIR.glob("*.json"))


def delete_dataset(name: str) -> bool:
    """Delete a dataset definition.  Returns *True* on success."""
    path = DATASETS_DIR / f"{name}.json"
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
