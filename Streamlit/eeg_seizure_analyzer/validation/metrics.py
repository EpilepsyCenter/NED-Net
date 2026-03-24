"""Validation metrics: compare detected events against manual annotations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent
from eeg_seizure_analyzer.io.base import Annotation


@dataclass
class ValidationResult:
    """Results from comparing detected events to manual annotations."""

    sensitivity: float  # TP / (TP + FN)
    precision: float  # TP / (TP + FP)
    f1_score: float
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    onset_errors_sec: list[float] = field(default_factory=list)
    offset_errors_sec: list[float] = field(default_factory=list)
    matched_pairs: list[tuple[DetectedEvent, Annotation]] = field(default_factory=list)
    false_positives: list[DetectedEvent] = field(default_factory=list)
    false_negatives: list[Annotation] = field(default_factory=list)

    @property
    def mean_onset_error_sec(self) -> float:
        return float(np.mean(self.onset_errors_sec)) if self.onset_errors_sec else 0.0

    @property
    def mean_offset_error_sec(self) -> float:
        return float(np.mean(self.offset_errors_sec)) if self.offset_errors_sec else 0.0


def validate_detections(
    detected: list[DetectedEvent],
    annotations: list[Annotation],
    overlap_threshold: float = 0.3,
    max_onset_tolerance_sec: float = 10.0,
) -> ValidationResult:
    """Compare detected events against manual annotations.

    Matching strategy:
    - For annotations with duration: use temporal overlap (IoU >= overlap_threshold)
    - For point annotations (no duration): match if a detection contains the annotation
      time point, or if onset is within max_onset_tolerance_sec

    Parameters
    ----------
    detected : list[DetectedEvent]
        Detected events.
    annotations : list[Annotation]
        Manual annotations (ground truth).
    overlap_threshold : float
        Minimum IoU for matching events with duration.
    max_onset_tolerance_sec : float
        Maximum time difference for matching point annotations.

    Returns
    -------
    ValidationResult
    """
    detected_sorted = sorted(detected, key=lambda e: e.onset_sec)
    annotations_sorted = sorted(annotations, key=lambda a: a.onset_sec)

    matched_det = set()
    matched_ann = set()
    matched_pairs = []
    onset_errors = []
    offset_errors = []

    for ann_idx, ann in enumerate(annotations_sorted):
        best_match = None
        best_score = -1.0

        for det_idx, det in enumerate(detected_sorted):
            if det_idx in matched_det:
                continue

            score = _compute_match_score(det, ann, overlap_threshold, max_onset_tolerance_sec)
            if score > best_score:
                best_score = score
                best_match = det_idx

        if best_match is not None and best_score > 0:
            matched_det.add(best_match)
            matched_ann.add(ann_idx)
            det = detected_sorted[best_match]
            matched_pairs.append((det, ann))
            onset_errors.append(det.onset_sec - ann.onset_sec)
            if ann.duration_sec is not None:
                offset_errors.append(det.offset_sec - (ann.onset_sec + ann.duration_sec))

    # Compute metrics
    tp = len(matched_pairs)
    fp_events = [d for i, d in enumerate(detected_sorted) if i not in matched_det]
    fn_annotations = [a for i, a in enumerate(annotations_sorted) if i not in matched_ann]
    fp = len(fp_events)
    fn = len(fn_annotations)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * sensitivity * precision / (sensitivity + precision) if (sensitivity + precision) > 0 else 0.0

    return ValidationResult(
        sensitivity=sensitivity,
        precision=precision,
        f1_score=f1,
        n_true_positives=tp,
        n_false_positives=fp,
        n_false_negatives=fn,
        onset_errors_sec=onset_errors,
        offset_errors_sec=offset_errors,
        matched_pairs=matched_pairs,
        false_positives=fp_events,
        false_negatives=fn_annotations,
    )


def _compute_match_score(
    det: DetectedEvent,
    ann: Annotation,
    overlap_threshold: float,
    max_onset_tolerance_sec: float,
) -> float:
    """Compute match score between a detection and an annotation.

    Returns a score > 0 if they match, 0 otherwise.
    """
    if ann.duration_sec is not None and ann.duration_sec > 0:
        # Duration-based matching: compute IoU
        ann_start = ann.onset_sec
        ann_end = ann.onset_sec + ann.duration_sec
        det_start = det.onset_sec
        det_end = det.offset_sec

        overlap_start = max(ann_start, det_start)
        overlap_end = min(ann_end, det_end)
        overlap = max(0, overlap_end - overlap_start)

        union = (ann_end - ann_start) + (det_end - det_start) - overlap
        iou = overlap / union if union > 0 else 0.0

        return iou if iou >= overlap_threshold else 0.0
    else:
        # Point annotation: check if detection contains the time point
        if det.onset_sec <= ann.onset_sec <= det.offset_sec:
            return 1.0
        # Or if onset is within tolerance
        onset_diff = abs(det.onset_sec - ann.onset_sec)
        if onset_diff <= max_onset_tolerance_sec:
            return 1.0 - (onset_diff / max_onset_tolerance_sec)
        return 0.0
