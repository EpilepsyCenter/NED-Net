#!/usr/bin/env python3
"""Select a (threshold / boundary) operating point on the held-IN batches.

Every Phase-2 number before this was measured through a hand-chosen operating
point: the threshold came from the model's own validation, but the hysteresis
BOUNDARY was guessed, and boundary choice has swung coverage from 0.01% to 72%
of recorded time. So the comparisons were partly measuring threshold guesses.

Scored on Batch 1 + Batch 2 only -- held IN for every arm, never the test fold.
Absolute numbers are therefore optimistic (Mir's labels there were seen in
training); that is acceptable for *selecting* an operating point, which is what
validation data is for, but the winner may be mis-tuned for unseen data and that
must be stated. B1+B2 hold only ~69 retained convulsive seizures, so the ranking
is coarse: enough to separate 60% coverage from 20%, not adjacent thresholds.

Reports, per operating point:
  * convulsive recall vs Mir's labels (the reference is convulsive-only)
  * precision against the candidates he actually adjudicated
  * fraction of recordings fired on -- coverage matters as much as recall,
    because the defect is per-recording SILENCE
  * median IoU and matched detection duration -- the boundary is what controls
    event extension, and the frozen model truncates badly (9 s detections inside
    37 s seizures, 2026-10-08), so this says whether that is a threshold
    artefact or a sensitivity limit

Usage:
    python scripts/local/operating_point_table.py \
        --db ~/.eeg_seizure_analyzer/projects/armA3_tune_t0.5_b0.1.db \
        --db ~/.eeg_seizure_analyzer/projects/armA3_tune_t0.7_b0.3.db \
        --db ~/.eeg_seizure_analyzer/projects/armA3_tune_t0.8_b0.5.db \
        --db ~/.eeg_seizure_analyzer/projects/armA3_tune_t0.9_b0.5.db
"""
from __future__ import annotations
import argparse, os, re, sqlite3, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_frozen_unet_vs_mir import load, recall, precision  # noqa: E402

# Retained set: drop Batch 4 entirely, B2's noisy UI ch3/ch4 (code ch2/ch3) and
# the dead electrode 449382 (B1 UI ch2). Defined in EXCLUSION_CRITERIA.md.
EXCLUDE_CH = {"Batch_1": {1}, "Batch_2": {2, 3}}


def _label(db: str) -> str:
    m = re.search(r"t([\d.]+)_b([\d.]+)", os.path.basename(db))
    return f"thr {m.group(1)} / bnd {m.group(2)}" if m else os.path.basename(db)


def _retain(df: pd.DataFrame, ch_col: str, batch_col: str) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for b, chs in EXCLUDE_CH.items():
        hit = df[batch_col].astype(str).str.endswith(b[-1]) & df[ch_col].isin(chs)
        keep &= ~hit
    return df[keep]


def quality(gt: pd.DataFrame, ev: pd.DataFrame) -> tuple[float, float, int]:
    """Median IoU and median duration of detections matched to a true seizure."""
    det = {k: g[["start_sec", "end_sec"]].to_numpy()
           for k, g in ev.groupby(["stem", "channel"])}
    ious, durs = [], []
    for (stem, ch), g in gt.groupby(["stem", "ch0"]):
        a = det.get((stem, ch))
        if a is None:
            continue
        for s, e in g[["start_s", "end_s"]].to_numpy():
            inter = np.clip(np.minimum(e, a[:, 1]) - np.maximum(s, a[:, 0]), 0, None)
            if not (inter > 0).any():
                continue
            j = int(inter.argmax())
            union = (e - s) + (a[j, 1] - a[j, 0]) - inter[j]
            ious.append(inter[j] / union if union > 0 else 0.0)
            durs.append(a[j, 1] - a[j, 0])
    return (float(np.median(ious)) if ious else float("nan"),
            float(np.median(durs)) if durs else float("nan"), len(ious))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--ground-truth", default="~/ground_truth/mir_ramgdnf_annotations.csv")
    ap.add_argument("--batches", default="12", help="batch digits to score, e.g. 12 or 3")
    ap.add_argument("--conv-only", action="store_true", default=True,
                    help="ground truth: convulsive only (Mir's reference is convulsive)")
    ap.add_argument("--det-conv-only", action="store_true",
                    help="also require the DETECTION to be labelled convulsive by "
                         "Stage 2 -- i.e. score the full cascade rather than Stage 1 "
                         "alone. Stage 2 only labels and never filters, so without this "
                         "the table measures the U-Net on its own.")
    a = ap.parse_args()
    bset = set(a.batches)

    rows = []
    for db in a.db:
        gt, ev = load(os.path.expanduser(db), os.path.expanduser(a.ground_truth))
        gt = gt[gt.batch.astype(str).str[-1].isin(bset)]
        gt = _retain(gt, "ch0", "batch")
        ev = ev[ev.batch.isin(bset)]
        ev = _retain(ev, "channel", "batch")
        # Like-for-like: only recordings Mir reviewed.
        reviewed = set(gt.stem)
        ev = ev[ev.stem.isin(reviewed)]
        if a.det_conv_only:
            ev = ev[ev.type == "convulsive"]

        sz = gt[gt.label == "Seizure"]
        if a.conv_only:
            sz = sz[sz.candidate_type == "convulsive"]
        hits = recall(ev, sz)
        tp, fp = precision(ev, gt)
        iou, mdur, n_m = quality(sz, ev)
        rows.append({
            "operating point": _label(db),
            "det": len(ev),
            "recall": f"{hits}/{len(sz)} = {100*hits/max(len(sz),1):.1f}%",
            "precision": f"{100*tp/(tp+fp):.0f}% ({tp}/{tp+fp})" if tp + fp else "n/a",
            "fired on": f"{100*ev.stem.nunique()/max(len(reviewed),1):.0f}% of rec",
            "med IoU": f"{iou:.2f}",
            "med det dur": f"{mdur:.0f} s",
        })
    df = pd.DataFrame(rows)
    print(f"\nBatches {a.batches}, retained channels, "
          f"{'convulsive' if a.conv_only else 'all'} ground truth; detections: "
          f"{'Stage-2 convulsive only (full cascade)' if a.det_conv_only else 'ALL (Stage 1 alone)'}\n")
    print(df.to_string(index=False))
    print("\nHeld-IN data: absolute values are optimistic. Use for RANKING only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
