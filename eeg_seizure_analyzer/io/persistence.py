"""Detection persistence: save/load seizure detection results to JSON."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent


def detection_json_path(edf_path: str) -> Path:
    """Derive the detections JSON path from an EDF file path.

    Replaces the file extension with ``_ned_detections.json``.
    """
    p = Path(edf_path)
    return p.with_suffix("").with_name(p.stem + "_ned_detections.json")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types transparently."""

    def default(self, obj):  # noqa: D401
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _sanitize(obj):
    """Recursively convert numpy types so they are JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_detections(
    edf_path: str,
    events: list[DetectedEvent],
    detection_info: dict,
    params_dict: dict,
    detector_name: str = "SpikeTrainSeizureDetector",
    channels: list[int] | None = None,
    animal_id: str = "",
    filter_settings: dict | None = None,
) -> Path:
    """Serialise detection results to a JSON file next to the EDF.

    The write is atomic: data is written to a temporary file in the
    same directory and then renamed, so a crash mid-write cannot
    corrupt the output.

    Returns the path of the written JSON file.
    """
    out_path = detection_json_path(edf_path)

    payload = {
        "edf_path": str(edf_path),
        "detector_name": detector_name,
        "animal_id": animal_id,
        "channels": channels or [],
        "params": _sanitize(params_dict),
        "detection_info": _sanitize(detection_info),
        "events": [_sanitize(e.to_full_dict()) for e in events],
    }
    if filter_settings:
        payload["filter_settings"] = _sanitize(filter_settings)

    # Atomic write: write to temp then rename
    dir_name = str(out_path.parent)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".ned_det_", dir=dir_name
    )
    try:
        with os.fdopen(fd, "w") as fp:
            json.dump(payload, fp, indent=2, cls=_NumpyEncoder)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return out_path


def load_detections(edf_path: str) -> dict | None:
    """Load previously saved detection results for an EDF file.

    Returns ``None`` if no saved detections are found.  Otherwise
    returns a dict with keys ``events`` (list of ``DetectedEvent``),
    ``detection_info``, ``params``, ``detector_name``, ``channels``,
    and ``animal_id``.
    """
    json_path = detection_json_path(edf_path)
    if not json_path.is_file():
        return None

    with open(json_path, "r") as fp:
        raw = json.load(fp)

    events = [DetectedEvent.from_dict(d) for d in raw.get("events", [])]

    # detection_info has integer channel keys but JSON stores them as
    # strings — convert back.
    raw_di = raw.get("detection_info", {})
    detection_info: dict = {}
    for k, v in raw_di.items():
        try:
            detection_info[int(k)] = v
        except (ValueError, TypeError):
            detection_info[k] = v

    return {
        "events": events,
        "detection_info": detection_info,
        "params": raw.get("params", {}),
        "detector_name": raw.get("detector_name", ""),
        "channels": raw.get("channels", []),
        "animal_id": raw.get("animal_id", ""),
        "filter_settings": raw.get("filter_settings", {}),
    }


# ── Interictal spike persistence ─────────────────────────────────────


def spike_detection_json_path(edf_path: str) -> Path:
    """Derive the spike detections JSON path from an EDF file path."""
    p = Path(edf_path)
    return p.with_suffix("").with_name(p.stem + "_ned_spikes.json")


def save_spike_detections(
    edf_path: str,
    events: list[DetectedEvent],
    detection_info: dict,
    params_dict: dict,
    channels: list[int] | None = None,
    filter_settings: dict | None = None,
) -> Path:
    """Serialise interictal spike detection results to JSON."""
    out_path = spike_detection_json_path(edf_path)

    payload = {
        "edf_path": str(edf_path),
        "detector_name": "SpikeDetector",
        "channels": channels or [],
        "params": _sanitize(params_dict),
        "detection_info": _sanitize(detection_info),
        "events": [_sanitize(e.to_full_dict()) for e in events],
    }
    if filter_settings:
        payload["filter_settings"] = _sanitize(filter_settings)

    dir_name = str(out_path.parent)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".ned_sp_", dir=dir_name
    )
    try:
        with os.fdopen(fd, "w") as fp:
            json.dump(payload, fp, indent=2, cls=_NumpyEncoder)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return out_path


def load_spike_detections(edf_path: str) -> dict | None:
    """Load previously saved interictal spike results for an EDF file."""
    json_path = spike_detection_json_path(edf_path)
    if not json_path.is_file():
        return None

    with open(json_path, "r") as fp:
        raw = json.load(fp)

    events = [DetectedEvent.from_dict(d) for d in raw.get("events", [])]

    raw_di = raw.get("detection_info", {})
    detection_info: dict = {}
    for k, v in raw_di.items():
        try:
            detection_info[int(k)] = v
        except (ValueError, TypeError):
            detection_info[k] = v

    return {
        "events": events,
        "detection_info": detection_info,
        "params": raw.get("params", {}),
        "channels": raw.get("channels", []),
        "filter_settings": raw.get("filter_settings", {}),
    }
