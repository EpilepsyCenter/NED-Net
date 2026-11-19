"""Model fingerprints for detection provenance.

The database records which model produced each event (``db.detection_runs``).
A model's folder name alone is not enough: folders can be renamed, deleted or
overwritten by a retrain with the same name. The SHA-256 of the weights file
that inference actually loads (``best_model.pt``) pins the exact model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_cache: dict[tuple[str, int, int], str] = {}


def _models_dir() -> Path:
    from eeg_seizure_analyzer.ml.train import MODELS_DIR
    return MODELS_DIR


def weights_sha256(model_name: str | None) -> str | None:
    """SHA-256 of the model's ``best_model.pt``, or None if it can't be read.

    Cached by (path, size, mtime) so batch runs hash each model once.
    """
    if not model_name:
        return None
    path = _models_dir() / model_name / "best_model.pt"
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _cache:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        _cache[key] = h.hexdigest()
    return _cache[key]


def stamp_provenance(events, model_name: str | None,
                     convulsive_model_name: str | None = None,
                     params: dict | None = None) -> None:
    """Write model names, weights hashes and settings into each event's
    ``features`` at detection time, so they survive saving and reloading."""
    stamp = {
        "model_name": model_name,
        "model_sha256": weights_sha256(model_name),
        "convulsive_model_name": convulsive_model_name or None,
        "convulsive_model_sha256": weights_sha256(convulsive_model_name),
        "run_params": dict(params or {}),
    }
    for ev in events:
        ev.features = {**(ev.features or {}), **stamp}


def model_run_id(source: str, model_name: str | None,
                 convulsive_model_name: str | None = None,
                 params: dict | None = None,
                 model_sha256: str | None = None,
                 convulsive_model_sha256: str | None = None) -> int:
    """detection_runs id for an ML detector run (names + weights hashes).

    Hashes stamped at detection time are preferred; otherwise the weights are
    hashed now.
    """
    from eeg_seizure_analyzer import db
    return db.get_or_create_run(
        source,
        model=model_name,
        model_sha256=model_sha256 or weights_sha256(model_name),
        convulsive_model=convulsive_model_name,
        convulsive_model_sha256=(convulsive_model_sha256
                                 or weights_sha256(convulsive_model_name)),
        params=params,
    )


def run_id_from_features(source: str, features: dict | None) -> int:
    """detection_runs id from a stamped event (see :func:`stamp_provenance`).

    Events detected before model tracking existed carry no stamp; their run
    records the source only, with the model left empty (unknown).
    """
    f = features or {}
    if not f.get("model_name"):
        from eeg_seizure_analyzer import db
        return db.get_or_create_run(source)
    return model_run_id(source, f.get("model_name"),
                        f.get("convulsive_model_name"), f.get("run_params"),
                        f.get("model_sha256"), f.get("convulsive_model_sha256"))
