#!/usr/bin/env python3
"""Probe every `annotations.xlsx` under a root: confirm it matches the
converter's expected format, count rows/labels, and check that each file's
session_names resolve to an EDF sitting in the same folder.

Usage:
    python scripts/probe_kaha_annotations.py <root-dir> [glob]

    <root-dir>  directory to scan recursively
    [glob]      filename pattern (default: annotations.xlsx)
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import openpyxl

EXPECTED = {"session_name", "channel", "start_s", "end_s", "label", "candidate_type"}

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
pattern = sys.argv[2] if len(sys.argv) > 2 else "annotations.xlsx"
ann_files = sorted(root.rglob(pattern))
print(f"{len(ann_files)} {pattern} files under {root}\n")

grand = collections.Counter()
tot_sessions = tot_resolved = 0
for f in sorted(ann_files):
    try:
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c is not None else "" for c in next(it)]
        data = [r for r in it if r and any(c is not None for c in r)]
    except Exception as exc:
        print(f"  ERROR {f}: {exc}")
        continue

    hset = set(header)
    fmt_ok = EXPECTED.issubset(hset)
    ci = {name: header.index(name) for name in EXPECTED if name in hset}
    labels = collections.Counter(str(r[ci["label"]]).strip() for r in data) if fmt_ok else {}
    sessions = {str(r[ci["session_name"]]) for r in data} if fmt_ok else set()

    # Resolve session_name -> EDF in the same folder.
    folder = f.parent
    resolved = sum(1 for s in sessions if (folder / f"{s}.edf").exists())
    tot_sessions += len(sessions)
    tot_resolved += resolved
    for k, v in labels.items():
        grand[k] += v

    try:
        rel = f.relative_to(root)
    except ValueError:
        rel = f
    flag = "OK " if fmt_ok else "FMT?"
    seiz = labels.get("Seizure", 0)
    print(
        f"{flag} {str(rel.parent):70} rows={len(data):4} seiz={seiz:3} "
        f"sess={len(sessions):3} edf_ok={resolved:3} "
        f"{'' if fmt_ok else 'cols='+str(header)}"
    )

print(f"\nGrand label totals: {dict(grand)}")
print(f"Sessions: {tot_sessions} total, {tot_resolved} resolve to an EDF in-folder")
