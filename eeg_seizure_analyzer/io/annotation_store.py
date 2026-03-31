"""Annotation persistence: save/load human annotations for the ML training pipeline.

This is separate from detection persistence (persistence.py).  Detections are
raw detector output; annotations are human-reviewed labels attached to events.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent

# ---------------------------------------------------------------------------
# Version stamped into every annotation JSON file
# ---------------------------------------------------------------------------
_ANNOTATION_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Numpy-safe serialisation helpers (mirrors persistence.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# AnnotatedEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedEvent:
    """A single annotated EEG event (seizure or spike) for ML training."""

    file_path: str
    animal_id: str
    annotator: str
    onset_sec: float
    offset_sec: float
    channel: int
    label: str  # "confirmed" | "rejected" | "pending"
    source: str  # "detector" | "manual"
    original_onset_sec: float | None = None  # set when boundaries adjusted
    original_offset_sec: float | None = None
    event_type: str = "seizure"
    detector_confidence: float = 0.0
    features: dict = field(default_factory=dict)
    quality_metrics: dict = field(default_factory=dict)
    notes: str = ""
    annotated_at: str = ""  # ISO timestamp, set when label changes
    event_id: int = 0  # stable ID that never changes once assigned

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON persistence."""
        return {
            "file_path": self.file_path,
            "animal_id": self.animal_id,
            "annotator": self.annotator,
            "onset_sec": self.onset_sec,
            "offset_sec": self.offset_sec,
            "channel": self.channel,
            "label": self.label,
            "source": self.source,
            "original_onset_sec": self.original_onset_sec,
            "original_offset_sec": self.original_offset_sec,
            "event_type": self.event_type,
            "detector_confidence": self.detector_confidence,
            "features": dict(self.features),
            "quality_metrics": dict(self.quality_metrics),
            "notes": self.notes,
            "annotated_at": self.annotated_at,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotatedEvent":
        """Reconstruct an ``AnnotatedEvent`` from a dict produced by ``to_dict()``."""
        return cls(
            file_path=str(d.get("file_path", "")),
            animal_id=str(d.get("animal_id", "")),
            annotator=str(d.get("annotator", "")),
            onset_sec=float(d["onset_sec"]),
            offset_sec=float(d["offset_sec"]),
            channel=int(d["channel"]),
            label=str(d.get("label", "pending")),
            source=str(d.get("source", "detector")),
            original_onset_sec=(
                float(d["original_onset_sec"])
                if d.get("original_onset_sec") is not None
                else None
            ),
            original_offset_sec=(
                float(d["original_offset_sec"])
                if d.get("original_offset_sec") is not None
                else None
            ),
            event_type=str(d.get("event_type", "seizure")),
            detector_confidence=float(d.get("detector_confidence", 0.0)),
            features=dict(d.get("features", {})),
            quality_metrics=dict(d.get("quality_metrics", {})),
            notes=str(d.get("notes", "")),
            annotated_at=str(d.get("annotated_at", "")),
            event_id=int(d.get("event_id", 0)),
        )


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def annotation_json_path(edf_path: str) -> Path:
    """Derive the annotations JSON path from an EDF file path.

    Replaces the file extension with ``_ned_annotations.json``.
    """
    p = Path(edf_path)
    return p.with_suffix("").with_name(p.stem + "_ned_annotations.json")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_annotations(
    edf_path: str,
    annotations: list[AnnotatedEvent],
    annotator: str = "",
    animal_id: str = "",
    filter_settings: dict | None = None,
) -> Path:
    """Serialise annotations to a JSON file next to the EDF.

    The write is atomic: data is written to a temporary file in the same
    directory and then renamed, so a crash mid-write cannot corrupt the
    output.

    Returns the path of the written JSON file.
    """
    out_path = annotation_json_path(edf_path)

    payload = {
        "version": _ANNOTATION_FORMAT_VERSION,
        "edf_path": str(edf_path),
        "annotator": annotator,
        "animal_id": animal_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "n_annotations": len(annotations),
        "annotations": [_sanitize(a.to_dict()) for a in annotations],
    }
    if filter_settings:
        payload["filter_settings"] = _sanitize(filter_settings)

    # Atomic write: write to temp then rename
    dir_name = str(out_path.parent)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".ned_ann_", dir=dir_name
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


def load_annotations(edf_path: str) -> list[AnnotatedEvent] | None:
    """Load previously saved annotations for an EDF file.

    Returns ``None`` if no saved annotations are found.
    """
    json_path = annotation_json_path(edf_path)
    if not json_path.is_file():
        return None

    with open(json_path, "r") as fp:
        raw = json.load(fp)

    return [AnnotatedEvent.from_dict(d) for d in raw.get("annotations", [])]


def load_annotations_with_settings(edf_path: str) -> tuple[list[AnnotatedEvent] | None, dict]:
    """Load annotations and filter settings from an EDF annotation file.

    Returns ``(annotations, filter_settings)`` where ``filter_settings``
    may be empty if the file predates that feature.
    """
    json_path = annotation_json_path(edf_path)
    if not json_path.is_file():
        return None, {}

    with open(json_path, "r") as fp:
        raw = json.load(fp)

    annotations = [AnnotatedEvent.from_dict(d) for d in raw.get("annotations", [])]
    filter_settings = raw.get("filter_settings", {})
    return annotations, filter_settings


# ---------------------------------------------------------------------------
# Conversion: DetectedEvent -> AnnotatedEvent
# ---------------------------------------------------------------------------

def detections_to_annotations(
    events: list[DetectedEvent],
    file_path: str,
    animal_id: str = "",
) -> list[AnnotatedEvent]:
    """Convert a list of ``DetectedEvent`` to ``AnnotatedEvent`` with label="pending".

    Preserves features and quality_metrics from the detector output.
    """
    annotations: list[AnnotatedEvent] = []
    for ev in events:
        annotations.append(
            AnnotatedEvent(
                file_path=file_path,
                animal_id=animal_id or ev.animal_id,
                annotator="",
                onset_sec=ev.onset_sec,
                offset_sec=ev.offset_sec,
                channel=ev.channel,
                label="pending",
                source="detector",
                original_onset_sec=None,
                original_offset_sec=None,
                event_type=ev.event_type,
                detector_confidence=ev.confidence,
                features=dict(ev.features),
                quality_metrics=dict(ev.quality_metrics),
                notes="",
                annotated_at="",
                event_id=ev.event_id,
            )
        )
    return annotations


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _events_match(
    a: AnnotatedEvent,
    b: AnnotatedEvent,
    tolerance_sec: float,
) -> bool:
    """Return True if two annotations refer to the same underlying event.

    Match criteria: same channel and onsets within *tolerance_sec* of each
    other.
    """
    if a.channel != b.channel:
        return False
    return abs(a.onset_sec - b.onset_sec) <= tolerance_sec


def merge_annotations(
    existing: list[AnnotatedEvent],
    new_from_detector: list[AnnotatedEvent],
    tolerance_sec: float = 1.0,
) -> list[AnnotatedEvent]:
    """Merge existing annotations with new detector-produced annotations.

    Rules:
    * If an existing annotation matches a new detection (same channel,
      onset within *tolerance_sec*), **keep the existing** annotation's
      label, notes, and boundaries intact.
    * New detections with no match are added with label ``"pending"``.
    * Existing annotations with no matching new detection are kept but
      flagged by appending ``"[orphaned]"`` to their notes (unless
      already flagged).
    """
    merged: list[AnnotatedEvent] = []
    matched_new_indices: set[int] = set()

    for ex in existing:
        found_match = False
        for idx, nd in enumerate(new_from_detector):
            if idx in matched_new_indices:
                continue
            if _events_match(ex, nd, tolerance_sec):
                # Keep existing annotation (human label takes precedence)
                merged.append(ex)
                matched_new_indices.add(idx)
                found_match = True
                break

        if not found_match:
            # Orphaned: no matching detection any more
            orphan = AnnotatedEvent(
                file_path=ex.file_path,
                animal_id=ex.animal_id,
                annotator=ex.annotator,
                onset_sec=ex.onset_sec,
                offset_sec=ex.offset_sec,
                channel=ex.channel,
                label=ex.label,
                source=ex.source,
                original_onset_sec=ex.original_onset_sec,
                original_offset_sec=ex.original_offset_sec,
                event_type=ex.event_type,
                detector_confidence=ex.detector_confidence,
                features=dict(ex.features),
                quality_metrics=dict(ex.quality_metrics),
                notes=(
                    ex.notes
                    if "[orphaned]" in ex.notes
                    else (ex.notes + " [orphaned]").strip()
                ),
                annotated_at=ex.annotated_at,
            )
            merged.append(orphan)

    # Add unmatched new detections as pending
    for idx, nd in enumerate(new_from_detector):
        if idx not in matched_new_indices:
            merged.append(nd)

    # Sort by onset time for consistency
    merged.sort(key=lambda e: (e.channel, e.onset_sec))
    return merged
