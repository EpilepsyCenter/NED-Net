#!/usr/bin/env python3
"""Consolidate Mir's 69 per-day RAM_GDNF annotation workbooks into one CSV.

The workbooks live on the LU research share under
``RAM_GDNF_2025/**/<name>_Annotations/Day_N.xlsx`` (candidate-review schema, one row
per reviewed candidate). This writes a single tidy CSV outside any tree that a
NED-Net trainer scans -- the ground truth is a SCORING reference, never training
data (see docs/methods_paper/NEDNet_validation_HANDOFF.md section 12).

Usage:
    python scripts/local/consolidate_mir_annotations.py \
        --share '/Volumes/research/LU26D1055-epicenter/Data/KAHA recordings/RAM_GDNF_2025' \
        --out ~/ground_truth/mir_ramgdnf_annotations.csv
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = sorted(p for p in args.share.rglob("Day_*.xlsx")
                   if not p.name.startswith("~$"))
    if not files:
        print(f"no Day_*.xlsx under {args.share}", file=sys.stderr)
        return 1

    frames = []
    for p in files:
        df = pd.read_excel(p)
        rel = p.relative_to(args.share)
        df["source_workbook"] = str(rel)
        # Batch / week from the path, e.g. Batch_3_Recordings/Week_1/...
        m = re.search(r"Batch_(\d)_Recordings", str(rel))
        df["batch"] = f"Batch_{m.group(1)}" if m else None
        m = re.search(r"Week_(\d)", str(rel))
        df["week"] = int(m.group(1)) if m else None
        df["day_file"] = p.stem
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(files)} workbooks -> {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
