"""Parse and filter annotations from EEG recordings."""

from __future__ import annotations

import re

from eeg_seizure_analyzer.io.base import Annotation


def find_seizure_annotations(
    annotations: list[Annotation],
    keywords: list[str] | None = None,
) -> list[Annotation]:
    """Filter annotations that likely mark seizure events.

    Parameters
    ----------
    annotations : list[Annotation]
        All annotations from a recording.
    keywords : list[str] | None
        Keywords to match in annotation text (case-insensitive).
        Defaults to common seizure-related terms.

    Returns
    -------
    list[Annotation]
        Filtered annotations matching seizure keywords.
    """
    if keywords is None:
        keywords = [
            "seizure", "sz", "ictal", "onset", "offset",
            "start", "end", "convuls", "tonic", "clonic",
        ]

    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    return [a for a in annotations if pattern.search(a.text)]


def pair_onset_offset_annotations(
    annotations: list[Annotation],
    onset_keywords: list[str] | None = None,
    offset_keywords: list[str] | None = None,
) -> list[Annotation]:
    """Pair onset/offset annotations into duration-based annotations.

    Looks for pairs of annotations marking seizure start and end,
    and returns annotations with onset_sec and duration_sec set.

    Parameters
    ----------
    annotations : list[Annotation]
        Seizure-related annotations.
    onset_keywords : list[str] | None
        Keywords identifying onset markers.
    offset_keywords : list[str] | None
        Keywords identifying offset markers.

    Returns
    -------
    list[Annotation]
        Paired annotations with duration_sec computed.
    """
    if onset_keywords is None:
        onset_keywords = ["onset", "start", "begin", "sz start", "seizure start"]
    if offset_keywords is None:
        offset_keywords = ["offset", "end", "stop", "sz end", "seizure end"]

    onset_pattern = re.compile(
        "|".join(re.escape(k) for k in onset_keywords), re.IGNORECASE
    )
    offset_pattern = re.compile(
        "|".join(re.escape(k) for k in offset_keywords), re.IGNORECASE
    )

    onsets = sorted(
        [a for a in annotations if onset_pattern.search(a.text)],
        key=lambda a: a.onset_sec,
    )
    offsets = sorted(
        [a for a in annotations if offset_pattern.search(a.text)],
        key=lambda a: a.onset_sec,
    )

    paired = []
    offset_idx = 0
    for onset in onsets:
        # Find the next offset after this onset
        while offset_idx < len(offsets) and offsets[offset_idx].onset_sec <= onset.onset_sec:
            offset_idx += 1
        if offset_idx < len(offsets):
            duration = offsets[offset_idx].onset_sec - onset.onset_sec
            paired.append(
                Annotation(
                    onset_sec=onset.onset_sec,
                    duration_sec=duration,
                    text=onset.text,
                    channel=onset.channel,
                )
            )
            offset_idx += 1

    return paired
