#!/usr/bin/env python
"""Rewrite the Prism figure file's data tables at a detector-confidence cut.

A .prism file is a zip of per-table ``data/tables/<uuid>/data.csv``, so the
numbers can be replaced in place, leaving every graph, layout and analysis
attached to those tables untouched.

Verified provenance of the existing numbers (exact match, all non-zero cells):
    DB       lunarc_detect_wk1-6_final.db
    baseline weeks 1-3      LEV  weeks 4-5   <- NOT 4-6, which is what
                                                stats_dig.py/paper_stats.py use
    rate     events / (recording_hours / 24)
    exclude  x, 355676, 372837, 30           -> n = 7 EGFP / 13 SV2A
    order    animal ID sorted as text, within group; zeros floored per table

SAFETY: this is verify-then-replace. For every table it first reproduces the
CURRENT file contents from the database at cut 0. If a single cell disagrees,
that table is left alone and reported — so a layout this script has misread can
never be silently overwritten with plausible-looking wrong numbers.

    python scripts/paper_stats/prism_apply_conf_cut.py --check      # verify only
    python scripts/paper_stats/prism_apply_conf_cut.py --cut 0.5 --go
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

PROJECTS = os.path.expanduser("~/.eeg_seizure_analyzer/projects")
DB = "lunarc_detect_wk1-6_final.db"
EXCLUDE = {"x", "355676", "372837", "30"}
BASE_WEEKS, LEV_WEEKS = {1, 2, 3}, {4, 5}
TOL = 5e-4


def per_animal(cut: float, weeks: set[int]):
    """-> {group: [(animal, conv_per_24h, nonconv_per_24h)]}, animals sorted."""
    conn = sqlite3.connect(os.path.join(PROJECTS, DB))
    conn.row_factory = sqlite3.Row
    wk = {}
    for r in conn.execute("SELECT id, path FROM chunks"):
        m = re.search(r"Week(\d+)-Day", r["path"])
        wk[r["id"]] = int(m.group(1)) if m else None
    hrs = collections.defaultdict(float)
    for r in conn.execute("SELECT chunk_id ci, animal_id a, valid_sec s FROM file_animals"):
        if wk.get(r["ci"]) in weeks:
            hrs[r["a"]] += (r["s"] or 0) / 3600
    cnt = collections.defaultdict(collections.Counter)
    for r in conn.execute("SELECT chunk_id ci, animal_id a, type t, cnn_confidence cf "
                          "FROM events WHERE excluded=0"):
        if (r["cf"] or 0) >= cut and wk.get(r["ci"]) in weeks:
            cnt[r["a"]][r["t"]] += 1
    grp = {r[0]: r[1] for r in conn.execute(
        "SELECT DISTINCT animal_id, group_id FROM file_animals")}
    conn.close()
    out = collections.defaultdict(list)
    for a in sorted(hrs):
        if a in EXCLUDE or not hrs[a] or a not in grp:
            continue
        d = hrs[a] / 24
        out[grp[a]].append((a, cnt[a]["convulsive"] / d,
                            cnt[a]["non_convulsive"] / d))
    return out


def _fmt(v: float, floor: bool) -> str:
    """Match the file's own formatting: 4 dp, trailing zeros stripped."""
    if floor and abs(v) < 1e-12:
        return "0.01"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s or "0"


def _num(cell: str):
    cell = (cell or "").strip()
    if cell in ("", "-"):
        return None
    try:
        return float(cell.replace(",", "."))
    except ValueError:
        return None


def build_series(vals, kind: str):
    """kind 'conv'|'non' -> (egfp list, sv2a list) in file order."""
    i = 1 if kind == "conv" else 2
    return ([v[i] for v in vals["Control"]], [v[i] for v in vals["SV2A"]])


def rewrite_table(rows, layout, old, new):
    """Replace numeric cells per `layout`; returns (new_rows, n_checked, errors)."""
    errs = []
    out = [list(r) for r in rows]
    for (ri, ci), (series, idx) in layout.items():
        cur = _num(rows[ri][ci])
        want_old = old[series][idx]
        floor = cur is not None and abs(cur - 0.01) < 1e-9 and abs(want_old) < 1e-12
        if cur is None:
            errs.append(f"r{ri}c{ci}: file empty, expected {want_old:.4f}")
            continue
        ref = 0.01 if floor else want_old
        if abs(cur - ref) > TOL:
            errs.append(f"r{ri}c{ci}: file {cur} != db {ref:.4f}")
            continue
        out[ri][ci] = _fmt(new[series][idx], floor)
    return out, len(layout), errs


def layout_wide(rows, series_names, blocks):
    """Tables shaped: label, <block1 values>, blanks, <block2 values>...

    Assigns each numeric cell in row order to the next expected series entry,
    which is why the verify pass matters — a misread layout shows up as a
    mismatch rather than a silent shift.
    """
    layout = {}
    for ri, (sname_pair) in enumerate(series_names):
        cells = [ci for ci in range(len(rows[ri])) if _num(rows[ri][ci]) is not None]
        pos = 0
        for sname, n in zip(sname_pair, blocks):
            for k in range(n):
                if pos >= len(cells):
                    return None
                layout[(ri, cells[pos])] = (sname, k)
                pos += 1
        if pos != len(cells):
            return None
    return layout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prism", default="Seizure_data.prism")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--go", action="store_true", help="write the modified copy")
    args = ap.parse_args()

    if not os.path.exists(args.prism):
        print(f"not found: {args.prism}", file=sys.stderr)
        return 1
    out_path = args.out or args.prism.replace(".prism", f"_conf{args.cut:g}.prism")

    b0, l0 = per_animal(0.0, BASE_WEEKS), per_animal(0.0, LEV_WEEKS)
    b1, l1 = per_animal(args.cut, BASE_WEEKS), per_animal(args.cut, LEV_WEEKS)
    old, new = {}, {}
    for tag, v in (("base", b0), ("lev", l0)):
        for kind in ("conv", "non"):
            e, s = build_series(v, kind)
            old[f"{tag}_{kind}_egfp"], old[f"{tag}_{kind}_sv2a"] = e, s
    for tag, v in (("base", b1), ("lev", l1)):
        for kind in ("conv", "non"):
            e, s = build_series(v, kind)
            new[f"{tag}_{kind}_egfp"], new[f"{tag}_{kind}_sv2a"] = e, s
    nE, nS = len(old["base_conv_egfp"]), len(old["base_conv_sv2a"])
    print(f"n = {nE} EGFP / {nS} SV2A   baseline=wk{sorted(BASE_WEEKS)} "
          f"LEV=wk{sorted(LEV_WEEKS)}   cut={args.cut}\n")

    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(args.prism) as z:
        z.extractall(tmp)

    # Candidate layouts, tried in order; the verify pass decides which applies.
    CANDIDATES = [
        # rows = [Convulsive, Non-convulsive]; blocks = baseline EGFP + SV2A
        ([("base_conv_egfp", "base_conv_sv2a"),
          ("base_non_egfp", "base_non_sv2a")], (nE, nS)),
        # rows = [EGFP, SV2A]; blocks = baseline then LEV, one metric per table
        ([("base_conv_egfp", "lev_conv_egfp"),
          ("base_conv_sv2a", "lev_conv_sv2a")], (nE, nE)),
        ([("base_non_egfp", "lev_non_egfp"),
          ("base_non_sv2a", "lev_non_sv2a")], (nE, nE)),
    ]

    # Column-oriented: EGFP baseline, EGFP LEV, SV2A baseline, SV2A LEV.
    COLUMN_CANDIDATES = [
        ["base_non_egfp", "lev_non_egfp", "base_non_sv2a", "lev_non_sv2a"],
        ["base_conv_egfp", "lev_conv_egfp", "base_conv_sv2a", "lev_conv_sv2a"],
    ]

    changed, skipped = [], []
    for tdir in sorted(os.listdir(os.path.join(tmp, "data", "tables"))):
        csv_path = os.path.join(tmp, "data", "tables", tdir, "data.csv")
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            continue
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        flat = " ".join(",".join(r) for r in rows)
        if "67.5415" not in flat and "171.8045" not in flat:
            continue

        done = False
        # Column-oriented tables: one series per column, animals down the rows.
        for cols in COLUMN_CANDIDATES:
            lay = {}
            ok = True
            for cj, sname in enumerate(cols):
                cells = [ri for ri in range(len(rows))
                         if cj < len(rows[ri]) and _num(rows[ri][cj]) is not None]
                if len(cells) != len(old[sname]):
                    ok = False
                    break
                for k, ri in enumerate(cells):
                    lay[(ri, cj)] = (sname, k)
            if not ok:
                continue
            newrows, n, errs = rewrite_table(rows, lay, old, new)
            if errs:
                continue
            if args.go:
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f).writerows(newrows)
            changed.append((tdir, n, cols))
            done = True
            break
        if done:
            continue

        for names, blocks in CANDIDATES:
            if len(rows) < len(names):
                continue
            # SV2A rows can be longer than EGFP rows; size blocks per row.
            per_row = []
            for ri, pair in enumerate(names):
                per_row.append(tuple(len(old[s]) for s in pair))
            lay = {}
            ok = True
            for ri, pair in enumerate(names):
                cells = [ci for ci in range(len(rows[ri]))
                         if _num(rows[ri][ci]) is not None]
                pos = 0
                for sname in pair:
                    for k in range(len(old[sname])):
                        if pos >= len(cells):
                            ok = False; break
                        lay[(ri, cells[pos])] = (sname, k); pos += 1
                    if not ok: break
                if not ok or pos != len(cells):
                    ok = False; break
            if not ok:
                continue
            newrows, n, errs = rewrite_table(rows, lay, old, new)
            if errs:
                continue
            if args.go:
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f).writerows(newrows)
            changed.append((tdir, n, [s for p in names for s in p]))
            done = True
            break
        if not done:
            skipped.append(tdir)

    for tdir, n, names in changed:
        print(f"  {'rewrote' if args.go else 'verified'} {tdir[:8]}  {n:3d} cells  "
              f"{' + '.join(sorted(set(x.rsplit('_',1)[0] for x in names)))}")
    for tdir in skipped:
        print(f"  SKIPPED {tdir[:8]} — layout not recognised or values disagree")

    if args.go and changed:
        if os.path.exists(out_path):
            os.remove(out_path)
        # Deflate, matching the original: -0 (store) tripled the file size.
        subprocess.run(["zip", "-r", "-X", "-q",
                        os.path.abspath(out_path), "."], cwd=tmp, check=True)
        print(f"\nWrote {out_path}")
        print("Open it in Prism and check the graphs before trusting it — the "
              "per-table .dt sidecars are not regenerated.")
    elif not args.go:
        print(f"\nCheck only. {len(changed)} table(s) reproduce exactly and would "
              f"be rewritten; {len(skipped)} skipped. Re-run with --go.")
    shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
