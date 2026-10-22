"""Blind review of a model's detections (plan, build, score, merge).

What a model's detections really are can't be read off validation scores: in
our tests most "false positives" were seizures nobody had annotated. A blind review
measures it: the model runs once over recordings it never trained on, a random
sample of its never-reviewed detections is drawn per score band, and the user
reviews them without seeing the model's confidence or the band.

    build_review(model, recordings, out_dir)  -> review id (folder + key written)
    review_progress(review_id)                -> (items, reviewed)
    score_review(review_id)                   -> per-band results (also saved)
    merge_review(review_id, apply=False)      -> what merging would change; apply=True writes

Storage
  * Review folder (user-chosen): recordings linked (symbolic link, else hard link,
    else copy), channel maps copied, blinded annotation files (method
    "blind_review", confidence 0, no convulsive flag), and `ned_review.json` with
    only the review id. Nothing in it reveals the model's scores.
  * Key and settings: ~/.eeg_seizure_analyzer/reviews/<id>/ (review.json,
    key.json); results and merge backups go there too.
The original recordings and their annotation files are only read, except by
merge_review(apply=True), which backs each file up first.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

REVIEWS_DIR = Path.home() / ".eeg_seizure_analyzer" / "reviews"
MANIFEST = "ned_review.json"
TOL_SEC = 5.0
N_PER_BAND = 40


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _review_dir(review_id: str) -> Path:
    return REVIEWS_DIR / review_id


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _overlaps(a0, a1, b0, b1, tol=0.0) -> bool:
    return a0 < b1 + tol and a1 > b0 - tol


def _link_or_copy(src: Path, dst: Path) -> str:
    """Symbolic link, else hard link (same disk), else copy. Returns the method."""
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------

def model_metadata(model_name: str) -> dict:
    from eeg_seizure_analyzer.ml.train import MODELS_DIR
    return _read_json(MODELS_DIR / model_name / "metadata.json", {}) or {}


def held_out_recordings(model_name: str) -> list[tuple[str, str]] | None:
    """(animal, EDF path) pairs the model never trained on, or None if the model
    predates saving them (2026-10-21)."""
    pairs = model_metadata(model_name).get("val_pairs")
    if not pairs:
        return None
    return [(str(a), str(p)) for a, p in pairs]


def operating_point(model_name: str) -> dict:
    """Detection settings the app uses for this model."""
    md = model_metadata(model_name)
    op = md.get("operating_point") or {}
    thr = op.get("threshold")
    if thr is None:
        thr = (md.get("best_metrics") or {}).get("best_threshold", 0.5)
    return {"threshold": float(thr),
            "boundary": float(op.get("boundary", 0.1)),
            "min_duration_sec": 5.0, "merge_gap_sec": 2.0}


def default_bands(model_name: str, n_per_band: int = N_PER_BAND) -> list[dict]:
    """At or above the operating threshold (what the app reports), and from half
    the threshold up to it (what the threshold leaves out)."""
    t = operating_point(model_name)["threshold"]
    return [{"lo": round(t, 4), "hi": 1.01, "n": n_per_band},
            {"lo": round(t / 2, 4), "hi": round(t, 4), "n": n_per_band}]


# ---------------------------------------------------------------------------
# Reviews on disk
# ---------------------------------------------------------------------------

def list_reviews() -> list[dict]:
    out = []
    if REVIEWS_DIR.exists():
        for d in sorted(REVIEWS_DIR.iterdir()):
            r = _read_json(d / "review.json")
            if r:
                out.append(r)
    return sorted(out, key=lambda r: r.get("created", ""), reverse=True)


def load_review(review_id: str) -> dict | None:
    return _read_json(_review_dir(review_id) / "review.json")


def review_id_for_folder(folder: str) -> str | None:
    m = _read_json(Path(folder) / MANIFEST)
    return (m or {}).get("review_id")


def _verdicts(review: dict) -> dict[int, str]:
    """item id -> label, read from the review folder's annotation files."""
    out = {}
    for f in Path(review["folder"]).glob("*_ned_annotations.json"):
        for a in (_read_json(f, {}) or {}).get("annotations", []):
            it = (a.get("features") or {}).get("review_item")
            if it is not None:
                out[int(it)] = a.get("label", "pending")
    return out


def review_progress(review_id: str) -> tuple[int, int]:
    review = load_review(review_id)
    if not review:
        return 0, 0
    v = _verdicts(review)
    return len(v), sum(1 for x in v.values() if x in ("confirmed", "rejected"))


def _prior_labels(edf_stems: set[str], exclude_id: str | None = None) -> list[dict]:
    """Verdicts from earlier blind reviews of the same recordings."""
    rows = []
    for r in list_reviews():
        if r.get("review_id") == exclude_id:
            continue
        key = _read_json(_review_dir(r["review_id"]) / "key.json", []) or []
        v = _verdicts(r) if Path(r.get("folder", "")).exists() else {}
        for k in key:
            lab = v.get(int(k["item"]))
            if k["stem"] in edf_stems and lab in ("confirmed", "rejected"):
                rows.append({**k, "label": lab})
    return rows


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_review(
    model_name: str,
    recordings: list[tuple[str, str]],
    out_dir: str,
    bands: list[dict] | None = None,
    seed: int | None = None,
    progress_callback=None,
    stop_flag=None,
) -> str:
    """Run the model over ``recordings`` [(animal, EDF path)], sample never-reviewed
    detections per band, and write the review folder and key. Returns the id.

    Only the listed animals' channels are scored (other channels of the same
    recording may belong to animals the model trained on)."""
    from eeg_seizure_analyzer.io.annotation_store import load_annotations
    from eeg_seizure_analyzer.io.channel_ids import load_channel_ids
    from eeg_seizure_analyzer.ml.predict import predict_seizures

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty: choose an empty or new folder")
    bands = bands or default_bands(model_name)
    op = operating_point(model_name)
    lowest = min(b["lo"] for b in bands)
    review_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    rng = random.Random(seed if seed is not None else uuid.uuid4().int)

    by_edf: dict[str, set[str]] = {}
    for animal, edf in recordings:
        by_edf.setdefault(str(edf), set()).add(str(animal))

    # 1. Detect once per recording at the lowest band threshold; keep peaks.
    dets = []      # one dict per detection on a held-out channel
    reviewed = {}  # stem -> confirmed/rejected annotations (channel, on, off, label)
    n_files = len(by_edf)
    for i, (edf, animals) in enumerate(sorted(by_edf.items())):
        if stop_flag is not None and stop_flag():
            raise RuntimeError("stopped")
        if progress_callback:
            progress_callback(i, n_files, f"Detecting in {Path(edf).name}")
        ids = load_channel_ids(edf) or {}
        chans = sorted(ch for ch, a in ids.items() if str(a) in animals)
        if not chans:
            continue
        stem = Path(edf).stem
        events = predict_seizures(
            edf, model_name, channels=chans, threshold=lowest,
            boundary_threshold=op["boundary"], min_duration_sec=op["min_duration_sec"],
            merge_gap_sec=op["merge_gap_sec"])
        for ev in events:
            peak = float((ev.features or {}).get("peak_probability", ev.confidence))
            dets.append({"edf": edf, "stem": stem, "channel": int(ev.channel),
                         "animal": str(ids.get(ev.channel, "")),
                         "onset": float(ev.onset_sec), "offset": float(ev.offset_sec),
                         "peak": peak})
        reviewed[stem] = [(int(a.channel), a.onset_sec, a.offset_sec, a.label)
                          for a in (load_annotations(edf) or [])
                          if a.label in ("confirmed", "rejected")
                          and int(a.channel) in chans]
    if progress_callback:
        progress_callback(n_files, n_files, "Sampling")

    # 2. Per band: never-reviewed detections, earlier verdicts reused, sample.
    prior = _prior_labels({d["stem"] for d in dets})
    strata, picks = [], []
    for b in bands:
        lo, hi, n = float(b["lo"]), float(b["hi"]), int(b["n"])
        in_band = [d for d in dets if lo <= d["peak"] < hi]
        hits = fps = 0
        never = []
        for d in in_band:
            ov = [lab for ch, a0, a1, lab in reviewed.get(d["stem"], [])
                  if ch == d["channel"] and _overlaps(d["onset"], d["offset"], a0, a1, TOL_SEC)]
            if "confirmed" in ov:
                hits += 1
            elif ov:
                fps += 1
            else:
                never.append(d)
        known = {"confirmed": 0, "rejected": 0}
        pool = []
        for d in never:
            q = [p["label"] for p in prior if p["stem"] == d["stem"] and p["channel"] == d["channel"]
                 and _overlaps(d["onset"], d["offset"], p["onset"], p["offset"])]
            if q:
                known["confirmed" if "confirmed" in q else "rejected"] += 1
            else:
                pool.append(d)
        take = rng.sample(pool, min(n, len(pool)))
        band = f"[{lo:g}, {min(hi, 1):g}{']' if hi > 1 else ')'}"
        strata.append({"band": band, "lo": lo, "hi": hi, "detections": len(in_band),
                       "annotated_seizure": hits, "annotated_not_seizure": fps,
                       "never_reviewed": len(never), "known_confirmed": known["confirmed"],
                       "known_rejected": known["rejected"], "pool": len(pool),
                       "sampled": len(take)})
        picks += [{**d, "band": band} for d in take]

    # 3. Review folder: one item per event (overlapping picks merged), blinded.
    rng.shuffle(picks)
    picks.sort(key=lambda d: (d["stem"], d["channel"], d["onset"]))
    items, last = [], None
    for d in picks:
        if last and (d["stem"], d["channel"]) == (last["stem"], last["channel"]) \
                and d["onset"] < last["offset"]:
            last["offset"] = max(last["offset"], d["offset"])
            last["members"].append(d)
            continue
        last = {"stem": d["stem"], "edf": d["edf"], "channel": d["channel"],
                "animal": d["animal"], "onset": d["onset"], "offset": d["offset"],
                "members": [d]}
        items.append(last)
    order = list(range(1, len(items) + 1))
    rng.shuffle(order)
    for it, num in zip(items, order):
        it["item"] = num

    out.mkdir(parents=True, exist_ok=True)
    link_methods = set()
    from eeg_seizure_analyzer.io.annotation_store import AnnotatedEvent, save_annotations
    for stem in sorted({it["stem"] for it in items}):
        src = Path(next(it["edf"] for it in items if it["stem"] == stem))
        link_methods.add(_link_or_copy(Path(os.path.realpath(src)), out / src.name))
        ch = src.with_name(src.stem + "_ned_channels.json")
        if ch.exists():
            shutil.copyfile(ch, out / ch.name)
        anns = []
        for eid, it in enumerate(sorted((x for x in items if x["stem"] == stem),
                                        key=lambda x: (x["channel"], x["onset"])), start=1):
            anns.append(AnnotatedEvent(
                file_path=str(out / src.name), animal_id=it["animal"], annotator="",
                onset_sec=round(it["onset"], 3), offset_sec=round(it["offset"], 3),
                channel=it["channel"], label="pending", source="detector",
                event_type="seizure", detector_confidence=0.0,
                features={"detection_method": "blind_review", "review_item": it["item"]},
                event_id=eid))
        save_annotations(str(out / src.name), anns)
    _write_json(out / MANIFEST, {"review_id": review_id, "created": _now(),
                                 "note": "Blind review folder made by NED-Net."})

    key = [{"item": it["item"], "edf": it["edf"], "stem": it["stem"],
            "channel": it["channel"], "animal": it["animal"], "onset": it["onset"],
            "offset": it["offset"],
            "members": [{"band": m["band"], "peak": m["peak"], "onset": m["onset"],
                         "offset": m["offset"]} for m in it["members"]]}
           for it in items]
    rd = _review_dir(review_id)
    _write_json(rd / "key.json", key)
    review = {"review_id": review_id, "created": _now(), "model": model_name,
              "folder": str(out), "recordings": [[a, e] for a, e in recordings],
              "bands": bands, "operating_point": op, "strata": strata,
              "n_items": len(items), "n_detections": len(dets),
              "link_method": sorted(link_methods), "merged": None}
    _write_json(rd / "review.json", review)
    return review_id


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def score_review(review_id: str) -> dict:
    """Per band: share of seizures among sampled never-reviewed detections (95%
    Wilson interval) and the estimated seizures among all of them; overall share of
    detections that are seizures at or above the operating threshold."""
    review = load_review(review_id)
    if not review:
        raise ValueError(f"no review {review_id}")
    key = _read_json(_review_dir(review_id) / "key.json", []) or []
    v = _verdicts(review)
    rows = []
    for st in review["strata"]:
        labs = [v.get(int(k["item"]), "pending") for k in key
                if any(m["band"] == st["band"] for m in k["members"])]
        done = [x for x in labs if x in ("confirmed", "rejected")]
        k_conf = sum(1 for x in done if x == "confirmed")
        n = len(done)
        lo, hi = wilson(k_conf, n)
        share = k_conf / n if n else float("nan")
        est = st["known_confirmed"] + (share * st["pool"] if n else float("nan"))
        seizures = st["annotated_seizure"] + est
        rows.append({**st, "reviewed": n, "pending": len(labs) - n,
                     "sample_seizures": k_conf, "sample_share": share,
                     "ci_low": lo, "ci_high": hi,
                     "est_seizures_never_reviewed": est,
                     "est_share_of_detections": (seizures / st["detections"]
                                                 if st["detections"] else float("nan"))})
    thr = review["operating_point"]["threshold"]
    top = [r for r in rows if r["lo"] >= thr]
    det = sum(r["detections"] for r in top)
    sz = sum(r["annotated_seizure"] + r["est_seizures_never_reviewed"] for r in top)
    result = {"review_id": review_id, "model": review["model"], "scored": _now(),
              "threshold": thr, "bands": rows,
              "at_threshold": {"detections": det,
                               "est_seizures": sz,
                               "est_share": sz / det if det else float("nan")}}
    _write_json(_review_dir(review_id) / "results.json", result)
    folder = Path(review["folder"])
    if folder.exists():
        _write_json(folder / "ned_review_results.json", {
            k: result[k] for k in ("review_id", "model", "scored", "threshold", "at_threshold")}
            | {"bands": [{k: r[k] for k in ("band", "detections", "reviewed",
                                            "sample_seizures", "sample_share", "ci_low",
                                            "ci_high", "est_share_of_detections")}
                         for r in rows]})
    return result


# ---------------------------------------------------------------------------
# Merge verdicts into the real annotations
# ---------------------------------------------------------------------------

def merge_review(review_id: str, apply: bool = False, annotator: str = "") -> dict:
    """Add the review's verdicts to the original recordings' annotation files.

    A verdict updates a matching *pending* candidate (same channel, overlapping) or
    is added as a new annotation; anything already confirmed or rejected is never
    changed. With apply=False nothing is written: returns what would change. With
    apply=True each file is backed up under the review's data folder first."""
    from eeg_seizure_analyzer.io.annotation_store import (
        AnnotatedEvent, load_annotations, save_annotations)
    review = load_review(review_id)
    if not review:
        raise ValueError(f"no review {review_id}")
    key = _read_json(_review_dir(review_id) / "key.json", []) or []
    v = _verdicts(review)
    by_edf: dict[str, list[dict]] = {}
    for k in key:
        lab = v.get(int(k["item"]))
        if lab in ("confirmed", "rejected"):
            by_edf.setdefault(k["edf"], []).append({**k, "label": lab})
    plan = {"files": [], "added": 0, "updated": 0, "skipped_reviewed": 0}
    backup = _review_dir(review_id) / "merge_backup"
    for edf, verdicts in sorted(by_edf.items()):
        anns = load_annotations(edf) or []
        added = updated = skipped = 0
        next_id = max((a.event_id for a in anns), default=0) + 1
        for k in verdicts:
            match = [a for a in anns if int(a.channel) == int(k["channel"])
                     and a.event_type == "seizure"
                     and _overlaps(a.onset_sec, a.offset_sec, k["onset"], k["offset"])]
            if any(a.label in ("confirmed", "rejected") for a in match):
                skipped += 1
                continue
            feats = {"origin": "blind_review", "review_id": review_id,
                     "review_model": review["model"]}
            if match:
                a = match[0]
                a.label = k["label"]
                a.annotator = annotator or a.annotator
                a.annotated_at = _now()
                a.features = {**(a.features or {}), **feats}
                updated += 1
            else:
                anns.append(AnnotatedEvent(
                    file_path=str(edf), animal_id=k.get("animal", ""), annotator=annotator,
                    onset_sec=round(k["onset"], 3), offset_sec=round(k["offset"], 3),
                    channel=int(k["channel"]), label=k["label"], source="detector",
                    event_type="seizure", features={"detection_method": "ml_unet", **feats},
                    annotated_at=_now(), event_id=next_id))
                next_id += 1
                added += 1
        plan["files"].append({"edf": edf, "added": added, "updated": updated,
                              "skipped_reviewed": skipped})
        plan["added"] += added
        plan["updated"] += updated
        plan["skipped_reviewed"] += skipped
        if apply and (added or updated):
            from eeg_seizure_analyzer.io.annotation_store import annotation_json_path
            src = annotation_json_path(edf)
            if src.exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, backup / src.name)
            save_annotations(edf, sorted(anns, key=lambda a: (a.channel, a.onset_sec)))
    if apply:
        review["merged"] = {"at": _now(), **{k: plan[k] for k in ("added", "updated",
                                                                   "skipped_reviewed")}}
        _write_json(_review_dir(review_id) / "review.json", review)
    return plan
