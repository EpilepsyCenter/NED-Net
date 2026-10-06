#!/usr/bin/env python3
"""Phase-1: score the frozen NED-Net U-Net against Mir's independent annotations.

Zero new annotation required -- this is downstream arithmetic on two event lists.
Ground truth is read from the consolidated CSV (see consolidate_mir_annotations.py);
it is never written as sidecars, so it cannot leak into training.

Alignment facts established 2026-10-02 (docs/methods_paper/VERIFICATION_LOG):
  * Mir channel k  ->  NED-Net channel k-1  (9.1% vs 1.4%/0.2% for k/k+1)
  * time bases agree (median gap to nearest detection -2.2 s, no systematic shift)
  * Batch 4 session_name carries a '.edf' suffix the others lack -- strip it, or
    192 of the 430 confirmed seizures silently drop out of the denominator.

Usage:
    python scripts/local/validate_frozen_unet_vs_mir.py \
        --db ~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db \
        --ground-truth ~/ground_truth/mir_ramgdnf_annotations.csv
"""
from __future__ import annotations
import argparse, os, sqlite3
import numpy as np
import pandas as pd

TOL = 5.0  # seconds of slack when matching a detection to a ground-truth event


def load(db: str, gt_csv: str, drop_out_of_range: bool = True):
    gt = pd.read_csv(gt_csv)
    gt["stem"] = gt.session_name.str.replace(r"\.edf$", "", regex=True)
    gt["ch0"] = gt.channel - 1
    con = sqlite3.connect(os.path.expanduser(db))
    ev = pd.read_sql(
        "SELECT c.path, e.start_sec, e.end_sec, e.channel, e.type, e.cnn_confidence "
        "FROM events e JOIN chunks c ON c.id = e.chunk_id", con)
    ev["stem"] = ev.path.str.rsplit("/", n=1).str[-1].str.replace(".edf", "", regex=False)
    ev["batch"] = ev.path.str.extract(r"Batch_(\d)_Recordings")[0]

    # 242 ground-truth rows (2.0%), including 38 of 430 confirmed seizures, carry
    # timestamps past the end of their EDF -- recordings are 90 min and tile the
    # 21-day protocol continuously (2,014 h x 8 ch = 16,114 animal-hours, no file
    # over 92 min), so these are a misalignment in the source export, not longer
    # files. They can never match a detection, so leaving them in the denominator
    # counts them as misses and understates recall.
    if drop_out_of_range:
        dur = pd.read_sql("SELECT path, chunk_start_sec, chunk_end_sec FROM chunks", con)
        dur["stem"] = dur.path.str.rsplit("/", n=1).str[-1].str.replace(".edf", "", regex=False)
        dur["dur"] = dur.chunk_end_sec - dur.chunk_start_sec
        gt = gt.merge(dur[["stem", "dur"]], on="stem", how="left")
        n_before = len(gt)
        gt = gt[(gt.dur.isna()) | (gt.end_s <= gt.dur)].copy()
        if len(gt) != n_before:
            print(f"dropped {n_before - len(gt)} ground-truth rows with timestamps "
                  f"outside their EDF\n")
    return gt, ev


def _index(ev: pd.DataFrame) -> dict:
    return {k: g[["start_sec", "end_sec"]].to_numpy()
            for k, g in ev.groupby(["stem", "channel"])}


def recall(ev: pd.DataFrame, sz: pd.DataFrame, tol: float = TOL) -> int:
    idx = _index(ev)
    return sum(
        1 for r in sz.itertuples()
        if (a := idx.get((r.stem, r.ch0))) is not None
        and np.any((a[:, 0] < r.end_s + tol) & (a[:, 1] > r.start_s - tol)))


def precision(ev: pd.DataFrame, gt: pd.DataFrame, tol: float = TOL) -> tuple[int, int]:
    """TP/FP among detections that overlap a candidate Mir actually adjudicated.

    Detections overlapping nothing he reviewed are UNKNOWN, not false positives --
    his candidate pool is not exhaustive. So this is precision on a subset.
    """
    rev = {k: (g[["start_s", "end_s"]].to_numpy(), g["label"].to_numpy())
           for k, g in gt.groupby(["stem", "ch0"])}
    tp = fp = 0
    for r in ev.itertuples():
        hit = rev.get((r.stem, r.channel))
        if hit is None:
            continue
        a, lab = hit
        m = (a[:, 0] < r.end_sec + tol) & (a[:, 1] > r.start_sec - tol)
        if not m.any():
            continue
        tp += bool((lab[m] == "Seizure").any())
        fp += not bool((lab[m] == "Seizure").any())
    return tp, fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db")
    ap.add_argument("--ground-truth", default="~/ground_truth/mir_ramgdnf_annotations.csv")
    a = ap.parse_args()
    gt, ev = load(a.db, os.path.expanduser(a.ground_truth))
    sz = gt[gt.label == "Seizure"]
    cnv = sz[sz.candidate_type == "convulsive"]

    print(f"detections {len(ev)} | ground-truth seizures {len(sz)} "
          f"({len(cnv)} convulsive) over {gt.stem.nunique()} recordings\n")

    print("--- operating-point sweep (recall vs Mir's convulsive set) ---")
    for name, sub in [
            ("all events", ev),
            ("conf >= 0.5", ev[ev.cnn_confidence >= 0.5]),
            ("convulsive only", ev[ev.type == "convulsive"]),
            ("convulsive + conf >= 0.5", ev[(ev.type == "convulsive") & (ev.cnn_confidence >= 0.5)])]:
        h = recall(sub, cnv)
        tp, fp = precision(sub, gt)
        p = f"{100*tp/(tp+fp):.1f}% ({tp}/{tp+fp})" if tp + fp else "n/a"
        print(f"  {name:26s} n={len(sub):6d}  recall {100*h/len(cnv):5.1f}%  precision {p}")

    print("\n--- by batch, restricted to Mir-annotated recordings (like-for-like) ---")
    ann = ev[ev.stem.isin(set(gt.stem))]
    for b in sorted(x for x in gt.batch.dropna().unique()):
        s = sz[sz.batch == b]
        sub = ann[ann.batch == b[-1]]
        nf = gt[gt.batch == b].stem.nunique()
        h = recall(sub, s)
        print(f"  {b}: {nf:4d} files | {len(sub):6d} det ({len(sub)/nf:5.1f}/file) | "
              f"{len(s):4d} seizures | recall {100*h/len(s):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
