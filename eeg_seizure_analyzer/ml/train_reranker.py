"""Train + apply the seizure event re-ranker (Stage-3 precision layer).

The re-ranker is a tabular HistGradientBoosting classifier that scores a
*candidate* seizure event as real-vs-false from detector-agnostic signal
features ([[reranker_features]]). It is a learned confidence/filter layer: it
generates no candidates of its own, only re-scores whatever a classical detector
OR the U-Net proposed, so the user can apply it to either front end.

NED-Net ships the *training capability*, not a model — this mirrors the U-Net /
convulsive trainers (same ``train_*_model`` signature, saved under MODELS_DIR
with a ``metadata.json``) so the existing UI training machinery drives it.

Design notes baked in (validated on SV2A, see commit history):
- Labels = the confirmed/rejected annotation decision. Negatives are real
  detector false positives (U-Net + classical both present in typical data).
- Features are computed on DETECTOR boundaries for both classes (the annotator
  only re-draws borders on confirmed events; using stored borders would leak).
- Signal is standardised to TARGET_FS so train/inference features match.
- Per-animal split (no animal leakage); HistGBT handles NaN context natively.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import joblib
import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent
from eeg_seizure_analyzer.io.annotation_store import load_annotations
from eeg_seizure_analyzer.io.dataset_store import scan_annotation_files
from eeg_seizure_analyzer.io.edf_reader import read_edf_window, scan_edf_channels
from eeg_seizure_analyzer.ml.dataset import _downsample
from eeg_seizure_analyzer.ml.reranker_features import (
    FEATURE_NAMES, extract_event_features, features_to_row,
)
from eeg_seizure_analyzer.ml.train import MODELS_DIR

TARGET_FS = 250  # standardise every window to the app's working rate


# ── feature-table construction ───────────────────────────────────────────────

def _detector_bounds(a) -> tuple[float, float]:
    """Boundaries as the DETECTOR proposed them (pre human adjustment)."""
    on = a.original_onset_sec if a.original_onset_sec is not None else a.onset_sec
    off = a.original_offset_sec if a.original_offset_sec is not None else a.offset_sec
    return on, off


def _read_eeg_at_target_fs(edf_path: str):
    """Read the EEG (high-rate) channels of an EDF, decimated to TARGET_FS.

    Returns a lightweight recording (``fs``/``data``/``n_samples``) — the only
    attributes the feature extractor touches — or None on failure.
    """
    ch_info = scan_edf_channels(edf_path)
    max_fs = max(c["fs"] for c in ch_info)
    eeg_idx = [c["index"] for c in ch_info if c["fs"] == max_fs]  # EEG = high-rate group
    dur = min(c["n_samples"] / c["fs"] for c in ch_info if c["index"] in eeg_idx)
    rec = read_edf_window(edf_path, channels=eeg_idx, start_sec=0.0, duration_sec=dur - 0.5)
    data = rec.data
    if abs(rec.fs - TARGET_FS) > 1e-6:
        data = _downsample(data, rec.fs, TARGET_FS)
    return SimpleNamespace(fs=float(TARGET_FS), data=data, n_samples=data.shape[1])


def build_reranker_table(dataset_def: dict, progress_callback=None,
                         exclude_animals=()) -> dict:
    """Scan a dataset's annotations and compute the re-ranker feature table.

    ``exclude_animals`` (animal IDs) are dropped from the table entirely, so the
    fit and its cross-validated metrics never see them — a clean hold-out.

    Returns a dict with X, y, groups, conv, det_conf, method, feature_names.
    """
    exclude = set(exclude_animals or ())
    files = dataset_def.get("files")
    if not files:
        files = scan_annotation_files(dataset_def["folder"], annotation_type="seizure")

    X, y, groups, conv, det_conf, method = [], [], [], [], [], []
    n_files = len(files)
    t0 = time.time()
    for fi, f in enumerate(files):
        edf = f.get("edf_path") or f.get("file_path")
        if not edf:
            continue
        anns = load_annotations(edf)
        if not anns:
            continue
        labeled = [a for a in anns
                   if a.label in ("confirmed", "rejected") and a.event_type == "seizure"
                   and (a.animal_id or edf) not in exclude]
        if not labeled:
            continue
        try:
            rec = _read_eeg_at_target_fs(edf)
        except Exception as e:  # noqa: BLE001 — one bad EDF shouldn't kill training
            print(f"  ! skip {edf}: {e}", file=sys.stderr)
            continue
        evs = []
        for a in labeled:
            on, off = _detector_bounds(a)
            evs.append(DetectedEvent(onset_sec=on, offset_sec=off,
                                     duration_sec=off - on, channel=a.channel,
                                     event_type="seizure"))
        for a, ev in zip(labeled, evs):
            if ev.channel >= rec.data.shape[0]:
                continue
            feats = extract_event_features(rec, ev, all_events=evs)
            X.append(features_to_row(feats))
            y.append(1 if a.label == "confirmed" else 0)
            cv = (a.features or {}).get("convulsive", None)
            conv.append(1.0 if cv is True else (0.0 if cv is False else np.nan))
            groups.append(a.animal_id or edf)
            det_conf.append(a.detector_confidence)
            method.append((a.features or {}).get("detection_method", "?"))
        if progress_callback and (fi == 0 or (fi + 1) % 5 == 0 or fi + 1 == n_files):
            progress_callback({"stage": "build", "files_done": fi + 1,
                               "n_files": n_files, "events": len(y),
                               "elapsed_sec": time.time() - t0})
    return {
        "X": np.asarray(X, float), "y": np.asarray(y, int),
        "groups": np.asarray(groups), "conv": np.asarray(conv, float),
        "det_conf": np.asarray(det_conf, float), "method": np.asarray(method),
        "feature_names": list(FEATURE_NAMES),
    }


# ── metrics ──────────────────────────────────────────────────────────────────

def _new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42)


def _oof_metrics(X, y, groups) -> dict:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    n_splits = min(5, len(set(groups)))
    if n_splits < 2 or len(set(y)) < 2:
        return {}
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = _new_model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof)
    n_neg = int((y == 0).sum())
    out = {"roc_auc": float(roc_auc_score(y, oof)),
           "avg_precision": float(average_precision_score(y, oof)),
           "n_events": int(len(y)), "n_pos": int(y.sum()), "n_neg": n_neg,
           "n_animals": int(len(set(groups)))}
    for target in (0.95, 0.90):
        idxs = [k for k in range(len(thr)) if rec[k] >= target]
        if not idxs:
            continue
        k = max(idxs, key=lambda k: prec[k])
        t = float(thr[k])
        tag = f"{int(target*100)}"
        out[f"threshold_at_recall_{tag}"] = t
        out[f"precision_at_recall_{tag}"] = float(prec[k])
        out[f"fp_removed_at_recall_{tag}"] = float((oof[y == 0] < t).sum() / max(n_neg, 1))
    # UI-compatible aliases so existing progress/listing code renders something
    out["event_f1"] = out["avg_precision"]
    out["best_event_f1"] = out["avg_precision"]
    out["best_threshold"] = out.get("threshold_at_recall_90", 0.5)
    return out


# ── public training entrypoint (mirrors train_*_model) ───────────────────────

def train_reranker_model(
    dataset_def: dict,
    dataset_config=None,
    train_config=None,
    model_name: str | None = None,
    progress_callback=None,
    stop_check_fn=None,
) -> dict:
    """Train an event re-ranker from a dataset definition and save it.

    Returns dict with model_path, best_metrics, n_params(=0), history, stopped_by_user.
    """
    if model_name is None:
        model_name = dataset_def.get("name", "reranker")

    if progress_callback:
        progress_callback({"stage": "build", "files_done": 0,
                           "n_files": len(dataset_def.get("files") or []), "events": 0})
    exclude = tuple(getattr(dataset_config, "exclude_animals", ()) or ())
    tbl = build_reranker_table(dataset_def, progress_callback=progress_callback,
                               exclude_animals=exclude)
    X, y, groups = tbl["X"], tbl["y"], tbl["groups"]
    if len(y) < 50 or len(set(y)) < 2 or len(set(groups)) < 2:
        raise ValueError(
            f"Re-ranker needs ≥2 classes over ≥2 animals; got {len(y)} events, "
            f"{int(y.sum())} confirmed / {int((y==0).sum())} rejected, "
            f"{len(set(groups))} animals.")

    if stop_check_fn and stop_check_fn():
        return {"stopped_by_user": True}

    metrics = _oof_metrics(X, y, groups)
    # Final model fit on ALL data (what gets shipped/applied).
    model = _new_model().fit(X, y)

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": tbl["feature_names"]},
                model_dir / "model.pkl")
    metadata = {
        "model_name": model_name,
        "architecture": "reranker",
        "dataset_name": dataset_def.get("name", ""),
        "dataset_folder": dataset_def.get("folder", ""),
        "created": datetime.now(timezone.utc).isoformat(),
        "target_fs": TARGET_FS,
        "feature_names": tbl["feature_names"],
        "n_params": 0,
        "best_metrics": metrics,
        "method_counts": {m: int((tbl["method"] == m).sum())
                          for m in sorted(set(tbl["method"].tolist()))},
    }
    with open(model_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    final = {"epoch": 1, "train_loss": 0.0, "val_loss": 0.0,
             "val_metrics": metrics, "best_epoch": 1}
    if progress_callback:
        progress_callback(final)
    # Return the full train_*_model contract the UI worker expects (epochs are
    # not a re-ranker concept, so they collapse to a single "epoch").
    return {"model_path": str(model_dir), "model_name": model_name,
            "best_metrics": metrics, "best_val_loss": 0.0, "best_epoch": 1,
            "n_params": 0, "history": [final], "stopped_by_user": False}


# ── inference (apply) ────────────────────────────────────────────────────────

def load_reranker(model_name: str):
    """Load a trained re-ranker. Returns (sklearn_model, feature_names, metadata)."""
    model_dir = MODELS_DIR / model_name
    with open(model_dir / "metadata.json") as fh:
        meta = json.load(fh)
    bundle = joblib.load(model_dir / "model.pkl")
    return bundle["model"], bundle["feature_names"], meta


def apply_reranker(events: list[DetectedEvent], recording, model_name: str,
                   all_events: list[DetectedEvent] | None = None) -> list[DetectedEvent]:
    """Score each event's P(real) and write it to ``event.confidence``.

    ``recording`` must be at ~TARGET_FS (the app's working rate). The previous
    heuristic confidence is preserved under ``quality_metrics['heuristic_confidence']``.
    """
    model, feat_names, _ = load_reranker(model_name)
    ctx = all_events if all_events is not None else events
    for ev in events:
        feats = extract_event_features(recording, ev, all_events=ctx)
        row = np.asarray([[float(feats.get(k, np.nan)) for k in feat_names]], float)
        p = float(model.predict_proba(row)[0, 1])
        ev.quality_metrics = dict(ev.quality_metrics or {})
        ev.quality_metrics["heuristic_confidence"] = ev.confidence
        ev.quality_metrics["reranker_score"] = p
        ev.confidence = p
    return events


# ── headless CLI (testing / docs) ────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train an event re-ranker headlessly.")
    p.add_argument("--data-dir", required=True,
                   help="folder of EDFs + *_ned_annotations.json")
    p.add_argument("--model-name", default="reranker_v1")
    args = p.parse_args(argv)
    files = scan_annotation_files(args.data_dir, annotation_type="seizure")
    files = [f for f in files if f.get("n_confirmed") or f.get("n_rejected")]
    if not files:
        print(f"ERROR: no annotations under {args.data_dir}", file=sys.stderr)
        return 1
    dataset_def = {"name": args.model_name, "folder": args.data_dir, "files": files}
    t0 = time.time()
    res = train_reranker_model(dataset_def, model_name=args.model_name,
                               progress_callback=lambda d: None)
    m = res["best_metrics"]
    print(f"\nTrained in {time.time()-t0:.0f}s → {res['model_path']}")
    print(f"  events {m.get('n_events')}  pos {m.get('n_pos')}  neg {m.get('n_neg')}  "
          f"animals {m.get('n_animals')}")
    print(f"  ROC-AUC {m.get('roc_auc'):.3f}  AP {m.get('avg_precision'):.3f}")
    for tag in ("95", "90"):
        if f"fp_removed_at_recall_{tag}" in m:
            print(f"  @recall {tag}%: precision {m[f'precision_at_recall_{tag}']:.3f}, "
                  f"FP removed {m[f'fp_removed_at_recall_{tag}']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
