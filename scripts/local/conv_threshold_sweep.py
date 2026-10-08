#!/usr/bin/env python3
"""Sweep the Stage-2 (convulsive) threshold WITHOUT re-running detection.

`events.convulsive_confidence` is stored per event, and Stage 2 only labels --
it never filters (`_apply_convulsive_classifier`, predict.py:242). So the whole
CONV_THRESHOLD sweep is arithmetic on an existing project DB: no SLURM job, no
re-detection. Re-deriving `type` from the stored probability reproduces exactly
what a re-run at that threshold would have produced.

Finding this was written for (2026-10-08, B1+B2 retained, convulsive GT):
  * frozen UNetv2 + Convulsive_v4LUNARC: Stage 2 earns its place -- precision
    33% -> 61% at 0.45, 81% at 0.80.
  * retrained A3 + conv_armA_holdB3: Stage 2 is pure loss at 0.45 -- it drops 6
    of 32 correct detections for ZERO precision gain (11% -> 11%).
  * cause: convulsive_confidence p50 is 0.12 for the frozen classifier and 0.66
    for the retrained one. conv_armA_holdB3 took 48% of its positive class from
    Mir's convulsive events, and build_convulsive_window_specs draws no
    negatives from rejected rows -- its negatives are only non-convulsive
    *confirmed* seizures -- so the balance tilted convulsive and it largely
    stopped discriminating. The 0.45 inherited from the frozen pipeline is
    mis-set for it.

So: re-tune CONV_THRESHOLD after every Stage-2 retrain, and never inherit it.

Usage:
    python scripts/local/conv_threshold_sweep.py \
        --db ~/.eeg_seizure_analyzer/projects/armA3_tune_t0.9_b0.5.db --batches 12
"""
from __future__ import annotations
import argparse, os, sqlite3, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_frozen_unet_vs_mir import load, recall, precision  # noqa: E402
from operating_point_table import _retain  # noqa: E402

THRS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--ground-truth", default="~/ground_truth/mir_ramgdnf_annotations.csv")
    ap.add_argument("--batches", default="12")
    a = ap.parse_args()
    bset = set(a.batches)

    for db in a.db:
        db = os.path.expanduser(db)
        gt, ev = load(db, os.path.expanduser(a.ground_truth))
        con = sqlite3.connect(db)
        cc = pd.read_sql("SELECT c.path, e.start_sec, e.channel, e.convulsive_confidence "
                         "FROM events e JOIN chunks c ON c.id = e.chunk_id", con)
        cc["stem"] = cc.path.str.rsplit("/", n=1).str[-1].str.replace(".edf", "", regex=False)
        ev = ev.merge(cc[["stem", "start_sec", "channel", "convulsive_confidence"]],
                      on=["stem", "start_sec", "channel"], how="left")

        gt = _retain(gt[gt.batch.astype(str).str[-1].isin(bset)], "ch0", "batch")
        ev = _retain(ev[ev.batch.isin(bset)], "channel", "batch")
        reviewed = set(gt.stem)
        ev = ev[ev.stem.isin(reviewed)]
        sz = gt[(gt.label == "Seizure") & (gt.candidate_type == "convulsive")]

        q = ev.convulsive_confidence.dropna()
        print(f"\n=== {os.path.basename(db)} ===  {len(ev)} detections, "
              f"{len(sz)} convulsive ground-truth events")
        print(f"convulsive_confidence: p10={np.percentile(q,10):.2f} "
              f"p50={np.percentile(q,50):.2f} p90={np.percentile(q,90):.2f}")
        rows = []
        for thr in THRS:
            sub = ev[ev.convulsive_confidence >= thr] if thr > 0 else ev
            h = recall(sub, sz)
            tp, fp = precision(sub, gt)
            rows.append({"conv_thr": "none (Stage 1)" if thr == 0 else f"{thr:.2f}",
                         "det": len(sub),
                         "recall": f"{h}/{len(sz)} = {100*h/max(len(sz),1):.1f}%",
                         "precision": f"{100*tp/(tp+fp):.0f}% ({tp}/{tp+fp})" if tp+fp else "n/a",
                         "fired on": f"{100*sub.stem.nunique()/max(len(reviewed),1):.0f}%"})
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
