#!/usr/bin/env python3
"""Pick a model's operating point by a rule fixed BEFORE the results are seen.

The pre-registered rule (RUN_LOG 2026-10-09, "_pr sweeps"):

  * Search the joint grid: every sweep DB (one Stage-1 threshold/boundary each)
    x every Stage-2 threshold in conv_threshold_sweep.THRS. Stage 2 only labels,
    so its threshold is arithmetic on the stored `convulsive_confidence`. 0 means
    Stage 2 off, which is a legitimate configuration.
  * Score in CASCADE scope, against Mir's convulsive reference: retained channels,
    recordings he reviewed, on the --select-batches. Recall is over his convulsive
    seizures; precision is over the candidates he adjudicated (the same functions
    as operating_point_table.py and conv_threshold_sweep.py).
  * Choose the maximum convulsive-event F1. Ties go to the higher Stage-1
    threshold, then the higher boundary, then the higher Stage-2 threshold.
  * Coverage ("fired on"), IoU and duration are REPORTED, never selected on.

Why F1, in cascade scope: the paper's claim is that a lab gets a usable detector
on its own data after annotating it. "Usable" needs both recall and precision,
and the reference is convulsive-only, so only the cascade output is comparable
with it. Choosing Stage 1 and Stage 2 jointly follows the paper's own pitfall
that operating points do not transfer between models. Precision here is
against Mir's adjudicated candidates only; true precision comes from the
precision sample (draw_precision_sample.py).

The winner is then reported on --report-batches (in-sample models: the same
batches; A3: the held-out Batch 3, read from the same sweep DBs), next to the
frozen pipeline at its production point, on the same subset.

Usage:
    python scripts/local/select_operating_point.py \
        --db ~/.eeg_seizure_analyzer/projects/temporal_pr_sweep_t0.5_b0.1.db [...x6] \
        --select-batches 123 --report-batches 123 --out review/opsel_temporal_pr.csv
    # A3: select on held-in B1+B2, report on held-out B3
    python scripts/local/select_operating_point.py --db ...A3 x6... \
        --select-batches 12 --report-batches 3
"""
from __future__ import annotations
import argparse, os, re, sqlite3, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_frozen_unet_vs_mir import load, recall, precision  # noqa: E402
from operating_point_table import _retain, quality  # noqa: E402
from conv_threshold_sweep import THRS  # noqa: E402

FROZEN_DB = "~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db"
FROZEN_POINT = (0.5, 0.1, 0.45)   # UNetv2 thr / bnd + Convulsive_v4LUNARC thr


def _point(db: str) -> tuple[float, float]:
    m = re.search(r"_t([\d.]+)_b([\d.]+)\.db$", os.path.basename(db))
    if not m:
        raise SystemExit(f"cannot read t<thr>_b<bnd> from {db}")
    return float(m.group(1)), float(m.group(2))


def _load(db: str, gt_csv: str):
    db = os.path.expanduser(db)
    gt, ev = load(db, os.path.expanduser(gt_csv))
    con = sqlite3.connect(db)
    cc = pd.read_sql("SELECT c.path, e.start_sec, e.channel, e.convulsive_confidence "
                     "FROM events e JOIN chunks c ON c.id = e.chunk_id", con)
    cc["stem"] = cc.path.str.rsplit("/", n=1).str[-1].str.replace(".edf", "", regex=False)
    ev = ev.merge(cc[["stem", "start_sec", "channel", "convulsive_confidence"]],
                  on=["stem", "start_sec", "channel"], how="left")
    return gt, ev


def _score(gt, ev, batches: str, conv_thr: float) -> dict:
    bset = set(batches)
    gt = _retain(gt[gt.batch.astype(str).str[-1].isin(bset)], "ch0", "batch")
    ev = _retain(ev[ev.batch.isin(bset)], "channel", "batch")
    reviewed = set(gt.stem)
    ev = ev[ev.stem.isin(reviewed)]
    if conv_thr > 0:
        ev = ev[ev.convulsive_confidence >= conv_thr]
    sz = gt[(gt.label == "Seizure") & (gt.candidate_type == "convulsive")]
    h = recall(ev, sz)
    tp, fp = precision(ev, gt)
    r = h / max(len(sz), 1)
    p = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou, mdur, _ = quality(sz, ev)
    return {"det": len(ev), "hits": h, "n_gt": len(sz), "recall": r,
            "tp": tp, "fp": fp, "precision": p, "f1": f1,
            "fired_on": ev.stem.nunique() / max(len(reviewed), 1),
            "med_iou": iou, "med_dur_s": mdur}


def _fmt(d: dict) -> str:
    return (f"det {d['det']:>6}  recall {d['hits']}/{d['n_gt']} = {100*d['recall']:.1f}%  "
            f"precision {100*d['precision']:.0f}% ({d['tp']}/{d['tp']+d['fp']})  "
            f"F1 {d['f1']:.3f}  fired on {100*d['fired_on']:.0f}%  "
            f"IoU {d['med_iou']:.2f}  dur {d['med_dur_s']:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True,
                    help="one sweep DB per Stage-1 point; name must end _t<thr>_b<bnd>.db")
    ap.add_argument("--ground-truth", default="~/ground_truth/mir_ramgdnf_annotations.csv")
    ap.add_argument("--select-batches", required=True, help="e.g. 123, or 12 for A3")
    ap.add_argument("--report-batches", required=True, help="e.g. 123, or 3 for A3")
    ap.add_argument("--frozen-db", default=FROZEN_DB)
    ap.add_argument("--out", help="write the full joint-grid table (CSV) here")
    a = ap.parse_args()

    loaded = {db: _load(db, a.ground_truth) for db in a.db}
    rows = []
    for db, (gt, ev) in loaded.items():
        thr, bnd = _point(db)
        for ct in THRS:
            s = _score(gt, ev, a.select_batches, ct)
            rows.append({"thr": thr, "bnd": bnd, "conv_thr": ct, "db": db, **s})
    grid = pd.DataFrame(rows)
    grid = grid.sort_values(["f1", "thr", "bnd", "conv_thr"],
                            ascending=False, kind="mergesort")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        grid.drop(columns="db").to_csv(a.out, index=False)

    print(f"\nJoint grid on batches {a.select_batches} (cascade scope, top 10 of {len(grid)}):")
    cols = ["thr", "bnd", "conv_thr", "det", "recall", "precision", "f1", "fired_on",
            "med_iou"]
    print(grid[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    win = grid.iloc[0]
    print(f"\nWINNER (max F1, ties -> higher thr/bnd/conv): thr {win.thr} / bnd {win.bnd} "
          f"/ conv {win.conv_thr}")
    gt, ev = loaded[win.db]
    rep = _score(gt, ev, a.report_batches, win.conv_thr)
    print(f"  on report batches {a.report_batches}:  {_fmt(rep)}")

    fgt, fev = _load(a.frozen_db, a.ground_truth)
    fr = _score(fgt, fev, a.report_batches, FROZEN_POINT[2])
    print(f"  frozen @ {FROZEN_POINT[0]}/{FROZEN_POINT[1]}/{FROZEN_POINT[2]}, "
          f"same batches:  {_fmt(fr)}")
    if fr["recall"]:
        print(f"  recall ratio retrained/frozen: {rep['recall']/fr['recall']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
