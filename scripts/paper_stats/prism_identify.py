#!/usr/bin/env python
"""Identify what every seizure table in the Prism file actually contains.

For each data table it takes the numeric cells row-by-row (and column-by-column)
and searches the candidate series from prism_series.py for an exact match. What
comes out is a map of table -> definition (which weeks, which denominator,
which metric, which group block), derived from the numbers rather than assumed.

Tables that match nothing are listed as unidentified rather than guessed at.

    python scripts/paper_stats/prism_identify.py [--prism FILE]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import tempfile
import zipfile

import prism_series

TOL = 5e-4
FLOOR = 0.01


def _num(cell):
    cell = (cell or "").strip()
    if cell in ("", "-"):
        return None
    try:
        return float(cell.replace(",", "."))
    except ValueError:
        return None


def match_block(vals, series, nE, nS):
    """Find (name, group) whose values equal `vals`. group in {egfp, sv2a, all}."""
    n = len(vals)
    for name, full in series.items():
        if name.startswith("_"):
            continue
        for gname, cand0 in (("egfp", full[:nE]), ("sv2a", full[nE:]), ("all", full)):
            # An animal with no events of that type is BLANK in the file but NaN
            # in the series, so also try the NaN-dropped form; without this every
            # duration table fails on length alone.
            for cand in (cand0, [c for c in cand0 if c == c]):
                if len(cand) != n:
                    continue
                ok = True
                for v, c in zip(vals, cand):
                    if c != c:                  # NaN in db, number in file
                        ok = False; break
                    ref = FLOOR if (abs(c) < 1e-12 and abs(v - FLOOR) < 1e-9) else c
                    if abs(v - ref) > TOL:
                        ok = False; break
                if ok:
                    return name, gname
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prism", default="Seizure_data.prism")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    series = prism_series.build(0.0)
    nE, nS = prism_series.group_slices(series)
    print(f"n = {nE} EGFP / {nS} SV2A   candidate series: "
          f"{len([k for k in series if not k.startswith('_')])}\n")

    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(args.prism) as z:
        z.extractall(tmp)

    titles = {}
    for sj in glob.glob(os.path.join(tmp, "data/sheets/*/sheet.json")):
        d = json.load(open(sj))
        if d.get("@class") == "DataSheet" and "ANALYSIS_VIEW" not in (d.get("flags") or []):
            tu = (d.get("table") or {}).get("uid")
            if tu:
                titles[tu] = d.get("title", "")

    found, unknown = {}, []
    for tu, title in sorted(titles.items(), key=lambda kv: kv[1]):
        p = os.path.join(tmp, "data/tables", tu, "data.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        rows = list(csv.reader(open(p)))
        hits = []

        # row-wise: each row may hold 1-2 blocks
        for ri, row in enumerate(rows):
            cells = [(ci, _num(c)) for ci, c in enumerate(row)]
            nums = [(ci, v) for ci, v in cells if v is not None]
            if not nums:
                continue
            for split in (len(nums), nE, nS):
                if split <= 0 or split > len(nums):
                    continue
                m1 = match_block([v for _c, v in nums[:split]], series, nE, nS)
                if not m1:
                    continue
                rest = nums[split:]
                m2 = match_block([v for _c, v in rest], series, nE, nS) if rest else None
                if rest and not m2:
                    continue
                hits.append((f"row{ri}", m1, m2, split))
                break

        # column-wise
        if not hits:
            width = max(len(r) for r in rows)
            for cj in range(width):
                col = [(ri, _num(rows[ri][cj]))
                       for ri in range(len(rows)) if cj < len(rows[ri])]
                nums = [(ri, v) for ri, v in col if v is not None]
                if not nums:
                    continue
                m = match_block([v for _r, v in nums], series, nE, nS)
                if m:
                    hits.append((f"col{cj}", m, None, len(nums)))

        if hits:
            found[tu] = (title, hits)
            print(f"{tu[:8]}  {title[:44]:46s}")
            for where, m1, m2, split in hits:
                s = f"    {where:6s} {m1[0]}/{m1[1]}"
                if m2:
                    s += f"  +  {m2[0]}/{m2[1]}"
                print(s)
        else:
            unknown.append((tu, title, len(rows)))

    print(f"\n=== identified {len(found)} tables; {len(unknown)} unmatched:")
    for tu, title, nr in unknown:
        print(f"  {tu[:8]}  {title[:50]:52s} ({nr} rows)")

    if args.json_out:
        json.dump({tu: {"title": t, "hits": [[w, m1, m2, sp] for w, m1, m2, sp in h]}
                   for tu, (t, h) in found.items()}, open(args.json_out, "w"), indent=1)
        print(f"\nWrote {args.json_out}")
    shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
