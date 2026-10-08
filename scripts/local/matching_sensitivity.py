#!/usr/bin/env python3
"""How much of the measured recall is an artefact of a permissive match rule?

`validate_frozen_unet_vs_mir.py` scores a detection as catching a ground-truth
seizure on *any* interval overlap with 5 s of slack on each side. That is the
standard loose criterion and it is defensible, but it is permissive in three
ways that matter in this cohort:

  * no minimum overlap -- a 1 s graze scores like a perfect match, and Mir's
    annotations include blocks (his `False` rows are median 68.9 s; several
    confirmed events run 172-239 s and are chained seizures), so touching the
    edge of a long block counts as a hit;
  * not one-to-one -- a ground-truth event counts as caught if ANY detection
    overlaps it, so a model that fires 792 times on a channel (animal 449385)
    buys its hits as cheaply as one that fires twice;
  * +-5 s is ~45% of the median convulsive duration (22 s), and exceeds the
    shortest confirmed event (3.4 s) entirely.

All three inflate recall, so the loose figure is an UPPER BOUND -- safe for the
claim "the frozen model fails here". It is not safe for frozen-vs-retrained
comparisons, because it favours whichever model fires more, which is precisely
the axis the two differ on. This script reports recall under five rules so the
comparison can be stated with the sensitivity beside it.

Usage:
    python scripts/local/matching_sensitivity.py \
        --db ~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db
    python scripts/local/matching_sensitivity.py --db A.db --db B.db --labels frozen A3
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_frozen_unet_vs_mir import load  # noqa: E402

TOL = 5.0


def _pairs(gt_iv: np.ndarray, det_iv: np.ndarray):
    """Intersection / union / IoU matrices for every (gt, det) pair."""
    gs, ge = gt_iv[:, 0][:, None], gt_iv[:, 1][:, None]
    ds, de = det_iv[:, 0][None, :], det_iv[:, 1][None, :]
    inter = np.clip(np.minimum(ge, de) - np.maximum(gs, ds), 0, None)
    union = (ge - gs) + (de - ds) - inter
    return inter, np.where(union > 0, inter / union, 0.0)


def score(gt: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Recall under five match rules, plus the IoU distribution of loose hits."""
    det = {k: g[["start_sec", "end_sec"]].to_numpy()
           for k, g in ev.groupby(["stem", "channel"])}
    hits = {k: 0 for k in ("tol5", "strict", "iou20", "cover50", "one2one")}
    n_gt = 0
    matched_det = 0            # one-to-one: detections consumed by a true event
    ious: list[float] = []

    for (stem, ch), g in gt.groupby(["stem", "ch0"]):
        gt_iv = g[["start_s", "end_s"]].to_numpy()
        n_gt += len(gt_iv)
        det_iv = det.get((stem, ch))
        if det_iv is None or not len(det_iv):
            continue

        inter, iou = _pairs(gt_iv, det_iv)
        gt_dur = (gt_iv[:, 1] - gt_iv[:, 0])[:, None]

        # Loose rule, as currently used: overlap with TOL slack on both sides.
        loose = ((gt_iv[:, 0][:, None] < det_iv[:, 1][None, :] + TOL)
                 & (gt_iv[:, 1][:, None] > det_iv[:, 0][None, :] - TOL))
        hits["tol5"] += int(loose.any(axis=1).sum())
        hits["strict"] += int((inter > 0).any(axis=1).sum())
        hits["iou20"] += int((iou >= 0.20).any(axis=1).sum())
        # Did a detection actually cover half the annotated seizure?
        hits["cover50"] += int((inter >= 0.5 * np.maximum(gt_dur, 1e-9)).any(axis=1).sum())
        ious.extend(iou[loose.any(axis=1)].max(axis=1).tolist())

        # One-to-one: greedy by overlap, each detection spent at most once.
        cand = [(inter[i, j], i, j) for i in range(len(gt_iv))
                for j in range(len(det_iv)) if inter[i, j] > 0]
        cand.sort(reverse=True)
        used_g, used_d = set(), set()
        for _, i, j in cand:
            if i in used_g or j in used_d:
                continue
            used_g.add(i); used_d.add(j)
        hits["one2one"] += len(used_g)
        matched_det += len(used_d)

    return {"n_gt": n_gt, "n_det": len(ev), "hits": hits,
            "matched_det": matched_det,
            "median_iou": float(np.median(ious)) if ious else float("nan")}


RULES = [
    ("tol5",    "any overlap, +-5 s   (as reported)"),
    ("strict",  "any overlap, no slack"),
    ("iou20",   "IoU >= 0.20"),
    ("cover50", "detection covers >=50% of the seizure"),
    ("one2one", "any overlap, one-to-one (greedy)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--ground-truth", default="~/ground_truth/mir_ramgdnf_annotations.csv")
    ap.add_argument("--convulsive-only", action="store_true", default=True)
    ap.add_argument("--all-seizures", dest="convulsive_only", action="store_false")
    ap.add_argument("--conf", type=float, default=None, help="min cnn_confidence")
    ap.add_argument("--batch", default=None, help="restrict to one batch digit, e.g. 3")
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(d).replace(".db", "") for d in a.db]

    out = {}
    for db, lab in zip(a.db, labels):
        gt, ev = load(db, os.path.expanduser(a.ground_truth))
        sz = gt[gt.label == "Seizure"]
        if a.convulsive_only:
            sz = sz[sz.candidate_type == "convulsive"]
        if a.conf is not None:
            ev = ev[ev.cnn_confidence >= a.conf]
        if a.batch:
            sz = sz[sz.batch.astype(str).str.endswith(a.batch)]
            ev = ev[ev.batch == a.batch]
        # Like-for-like: only recordings Mir actually reviewed.
        ev = ev[ev.stem.isin(set(gt.stem))]
        out[lab] = score(sz, ev)

    ref = out[labels[0]]
    print(f"\nground-truth events: {ref['n_gt']}"
          f"{'  (convulsive only)' if a.convulsive_only else ''}"
          f"{f'  batch {a.batch}' if a.batch else ''}"
          f"{f'  conf >= {a.conf}' if a.conf is not None else ''}\n")
    w = max(len(l) for l in labels)
    print("rule".ljust(40) + "".join(l.rjust(w + 10) for l in labels))
    for key, desc in RULES:
        row = desc.ljust(40)
        for lab in labels:
            r = out[lab]
            row += f"{r['hits'][key]:4d}/{r['n_gt']} = {100*r['hits'][key]/max(r['n_gt'],1):5.1f}%".rjust(w + 10)
        print(row)
    print()
    for lab in labels:
        r = out[lab]
        loose, strict = r["hits"]["tol5"], r["hits"]["one2one"]
        print(f"{lab}: {r['n_det']} detections on reviewed recordings; "
              f"median IoU of loose hits {r['median_iou']:.2f}; "
              f"{r['matched_det']} detections consumed by a true event "
              f"({100*r['matched_det']/max(r['n_det'],1):.2f}% of output); "
              f"loose/one-to-one inflation {loose/max(strict,1):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
