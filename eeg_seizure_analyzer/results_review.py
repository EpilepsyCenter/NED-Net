"""Review of the events in a results database (Results > Review events).

The events a detector wrote to a project database are reviewed in the Training
reviewer, on the recordings they came from. Each event is put into its recording's
annotation file as a candidate and linked to its database row; every verdict then
goes to both places:

  * the annotation file, so reviewed events can enter a training set as they are;
  * the database (``db.set_event_review``): review status, and for confirmed events
    the reviewed boundaries and convulsive call. Rejected events are excluded.

Nothing is detected again. Any detector's events can be reviewed.

    locate_recordings(paths, folder)        -> where each recording is on this machine
    prepare_review(events, project, located) -> link events into the annotation files
    sync_annotations(annotations, project, scope) -> verdicts into the database
    progress(event_ids)                     -> counts for the banner

The link is ``features["results_db"] = {project: [event ids]}`` on the annotation.
One annotation can carry several events (two detectors finding the same seizure get
one verdict). An event that overlaps an annotation already confirmed or rejected is
linked to it and takes its verdict; no existing verdict is changed here.

Each annotation file is copied to ``BACKUP_DIR/<project>/`` the first time a review
of that project touches it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from eeg_seizure_analyzer import db
from eeg_seizure_analyzer.io.annotation_store import (
    AnnotatedEvent, annotation_json_path, load_annotations, save_annotations,
)

BACKUP_DIR = Path.home() / ".eeg_seizure_analyzer" / "results_review_backups"
LINK_KEY = "results_db"
ORIGIN = "results_review"

# detector source in the database -> features["detection_method"] of a candidate
_METHOD_OF_SOURCE = {"seizure_unet": "ml_unet", "seizure_bendr": "ml_bendr",
                     "seizure_cnn": "ml_unet"}


# ---------------------------------------------------------------------------
# Session state (state.extra) helpers, shared by the Results and Training pages
# ---------------------------------------------------------------------------

def folder_label(project: str) -> str:
    """What stands in for the project folder while a review is open."""
    return f"Results review — {project}"


def review_open(extra: dict | None) -> bool:
    """A results review is open and its recordings are still the loaded project."""
    if not extra:
        return False
    rr = extra.get("rr_review")
    return bool(rr) and extra.get("project_folder") == rr.get("folder")


def scope_ids(extra: dict | None) -> set[int] | None:
    """Database ids of the events under review, or None when no review is open."""
    if not review_open(extra):
        return None
    return set(extra["rr_review"].get("ids") or [])


def linked_ids(ann: AnnotatedEvent, project: str) -> list[int]:
    link = (ann.features or {}).get(LINK_KEY) or {}
    try:
        return [int(i) for i in (link.get(project) or [])]
    except (TypeError, ValueError, AttributeError):
        return []


def in_scope(ann: AnnotatedEvent, project: str, scope: set[int]) -> bool:
    return any(i in scope for i in linked_ids(ann, project))


# ---------------------------------------------------------------------------
# Where the recordings are
# ---------------------------------------------------------------------------

def locate_recordings(paths, folder: str | None = None) -> dict[str, str | None]:
    """Map each database path to a file on this machine (None when not found).

    A path that exists is used as it is. Otherwise the recording is looked for
    under ``folder`` (searched with its sub-folders): the file sharing the longest
    run of trailing path parts wins, and a tie is not guessed.
    """
    out: dict[str, str | None] = {}
    missing = []
    for p in dict.fromkeys(paths):
        if p and os.path.isfile(p):
            out[p] = p
        else:
            out[p] = None
            if p:
                missing.append(p)
    if not missing or not folder or not os.path.isdir(folder):
        return out
    by_name: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".edf"):
                by_name.setdefault(f, []).append(os.path.join(root, f))
    for p in missing:
        parts = [x for x in p.replace("\\", "/").split("/") if x]
        cands = by_name.get(parts[-1], []) if parts else []
        if len(cands) == 1:
            out[p] = cands[0]
            continue
        best, best_n, tie = None, 0, False
        for c in cands:
            cparts = [x for x in c.replace("\\", "/").split("/") if x]
            n = 0
            while n < min(len(parts), len(cparts)) and parts[-1 - n] == cparts[-1 - n]:
                n += 1
            if n > best_n:
                best, best_n, tie = c, n, False
            elif n == best_n:
                tie = True
        if best is not None and not tie:
            out[p] = best
    return out


# ---------------------------------------------------------------------------
# Events into the annotation files
# ---------------------------------------------------------------------------

def _backup_once(edf_path: str, project: str) -> None:
    src = annotation_json_path(edf_path)
    if not src.is_file():
        return
    folder = BACKUP_DIR / (project or "Default")
    tag = hashlib.sha256(str(src).encode()).hexdigest()[:10]
    if any(folder.glob(f"*_{tag}_*.json")):
        return
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copyfile(src, folder / f"{src.stem}_{tag}_{stamp}.json")


def _file_header(edf_path: str) -> dict:
    """The annotation file's own fields that a save must carry over."""
    p = annotation_json_path(edf_path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return {"annotator": raw.get("annotator", ""), "animal_id": raw.get("animal_id", ""),
            "filter_settings": raw.get("filter_settings") or None}


def _candidate(ev: dict, edf_path: str, event_id: int, project: str) -> AnnotatedEvent:
    label = ev.get("review_label") if ev.get("review_label") in ("confirmed", "rejected") \
        else "pending"
    feats = {
        "detection_method": _METHOD_OF_SOURCE.get(ev.get("source") or "",
                                                  ev.get("source") or ""),
        "detectors": [ev.get("source") or ""],
        "convulsive": ev.get("type") == "convulsive",
        "convulsive_probability": float(ev.get("convulsive_confidence") or 0.0),
        "origin": ORIGIN,
        LINK_KEY: {project: [int(ev["id"])]},
    }
    return AnnotatedEvent(
        file_path=edf_path, animal_id=str(ev.get("animal_id") or ""),
        annotator=str(ev.get("reviewer") or "") if label != "pending" else "",
        onset_sec=float(ev["start_sec"]), offset_sec=float(ev["end_sec"]),
        channel=int(ev["channel"]), label=label, source="detector",
        event_type="seizure", detector_confidence=float(ev.get("cnn_confidence") or 0.0),
        features=feats,
        annotated_at=str(ev.get("reviewed_at") or "") if label != "pending" else "",
        event_id=event_id,
    )


def link_events(annotations: list[AnnotatedEvent], events: list[dict], edf_path: str,
                project: str) -> tuple[list[AnnotatedEvent], dict]:
    """Link one recording's database events to its annotations (in memory).

    Earlier links of this project are dropped first, so ids from a database that was
    since re-analysed cannot point at the wrong event. Returns the new list and
    counts: ``linked`` (to an existing annotation), ``added`` (new candidates),
    ``no_channel`` (events without a recorded channel, which cannot be placed).
    """
    anns = list(annotations)
    for a in anns:
        link = (a.features or {}).get(LINK_KEY)
        if isinstance(link, dict) and project in link:
            link = {k: v for k, v in link.items() if k != project}
            if link:
                a.features[LINK_KEY] = link
            else:
                a.features.pop(LINK_KEY, None)
    counts = {"linked": 0, "added": 0, "no_channel": 0}
    next_id = max((a.event_id for a in anns), default=0) + 1
    for ev in sorted(events, key=lambda e: (e.get("channel") or 0, e.get("start_sec") or 0)):
        if ev.get("channel") is None:
            counts["no_channel"] += 1
            continue
        ch, s, e = int(ev["channel"]), float(ev["start_sec"]), float(ev["end_sec"])
        best, best_ov = None, 0.0
        for a in anns:
            if a.channel != ch or (a.event_type or "seizure") != "seizure":
                continue
            ov = min(a.offset_sec, e) - max(a.onset_sec, s)
            if ov > best_ov:
                best, best_ov = a, ov
        if best is None:
            anns.append(_candidate(ev, edf_path, next_id, project))
            next_id += 1
            counts["added"] += 1
        else:
            if best.features is None:
                best.features = {}
            link = dict(best.features.get(LINK_KEY) or {})
            link[project] = sorted(set(link.get(project) or []) | {int(ev["id"])})
            best.features[LINK_KEY] = link
            counts["linked"] += 1
    anns.sort(key=lambda a: (a.channel, a.onset_sec))
    return anns, counts


def prepare_review(events: list[dict], located: dict[str, str | None],
                   project: str | None = None) -> dict:
    """Put the events into their recordings' annotation files and link them.

    ``events``: database rows (``db.get_events``) of the active project;
    ``located``: ``locate_recordings`` of their paths. Events ticked Exclude by hand
    and never reviewed are left out. Verdicts already in the annotation files go to
    the database. Returns the review: ``project``, ``ids`` (events under review),
    ``files`` (recordings, in order), and counts for the user (``not_found``,
    ``no_channel``, ``skipped_excluded``, ``added``, ``linked``).
    """
    project = project or db.get_active_project()
    by_path: dict[str, list[dict]] = {}
    out = {"project": project, "folder": folder_label(project), "ids": [], "files": [],
           "not_found": 0, "not_found_files": [], "no_channel": 0,
           "skipped_excluded": 0, "added": 0, "linked": 0}
    for ev in events:
        if (ev.get("category") or "seizure") != "seizure":
            continue
        if ev.get("excluded") and not ev.get("review_label"):
            out["skipped_excluded"] += 1
            continue
        by_path.setdefault(ev.get("path") or "", []).append(ev)
    for path in sorted(by_path, key=lambda p: (min(e.get("date") or "" for e in by_path[p]), p)):
        evs = by_path[path]
        local = located.get(path)
        if not local:
            out["not_found"] += len(evs)
            out["not_found_files"].append(path)
            continue
        placed = [int(e["id"]) for e in evs if e.get("channel") is not None]
        out["no_channel"] += len(evs) - len(placed)
        if not placed:          # nothing to review here: leave the file alone
            continue
        _backup_once(local, project)
        head = _file_header(local)
        anns, counts = link_events(load_annotations(local) or [], evs, local, project)
        save_annotations(local, anns, annotator=head.get("annotator", ""),
                         animal_id=head.get("animal_id", ""),
                         filter_settings=head.get("filter_settings"))
        out["added"] += counts["added"]
        out["linked"] += counts["linked"]
        out["ids"] += placed
        out["files"].append(local)
        sync_annotations(anns, project, set(placed))
    return out


# ---------------------------------------------------------------------------
# Verdicts into the database
# ---------------------------------------------------------------------------

def sync_annotations(annotations: list[AnnotatedEvent], project: str,
                     scope: set[int] | None = None, reviewer: str = "") -> int:
    """Write the verdicts of linked annotations to the database. Returns rows changed.

    Only writes when ``project`` is the active project, and only to events in
    ``scope`` (all linked events when None).
    """
    if project != db.get_active_project():
        return 0
    n = 0
    for a in annotations:
        ids = linked_ids(a, project)
        if scope is not None:
            ids = [i for i in ids if i in scope]
        if not ids:
            continue
        feats = a.features or {}
        conv = bool(feats["convulsive"]) if "convulsive" in feats else None
        for i in ids:
            n += bool(db.set_event_review(
                i, a.label, reviewer=a.annotator or reviewer,
                start_sec=a.onset_sec, end_sec=a.offset_sec, convulsive=conv))
    return n


def sync_files(files, project: str, scope: set[int] | None = None) -> int:
    """``sync_annotations`` from the annotation files on disk (used when a review closes)."""
    n = 0
    for f in files:
        try:
            anns = load_annotations(f) or []
        except (OSError, ValueError):
            continue
        n += sync_annotations(anns, project, scope)
    return n


def progress(event_ids) -> dict:
    """Counts of the events under review: total, confirmed, rejected, pending."""
    rows = db.get_events_by_id(event_ids)
    c = sum(r.get("review_label") == "confirmed" for r in rows)
    r = sum(r.get("review_label") == "rejected" for r in rows)
    return {"total": len(rows), "confirmed": c, "rejected": r,
            "pending": len(rows) - c - r}
