#!/usr/bin/env python
"""Recompute every seizure table in the Prism file at a confidence cut.

Fills each table with the correct values for what its TITLE says it is, rather
than reproducing however it was originally made (several tables were built by
hand in Excel from an older export, so they cannot be reverse-engineered — and
do not need to be). Every panel therefore ends up on one detection run, one
definition, one cut.

Definitions used throughout (confirmed with Marco):
    baseline  weeks 1-3 (Day 1-21)      LEV  weeks 4-5 (Day 22-35), week 6 excluded
    rates     events / (recording_hours / 24)      durations  mean duration_sec
    free days % of RECORDED days with no convulsive seizure
    exclude   x, 355676, 372837, 30  ->  n = 7 EGFP / 13 SV2A
    order     animal ID sorted as text, EGFP block then SV2A block

Layout is preserved cell-for-cell: block boundaries are read off the existing
grid (columns that hold data, separated by empty spacer columns), so graphs stay
bound to the same cells. Where a value is undefined at the new cut — an animal
with no events of that type has no mean duration — the cell is blanked, and
where the file used a 0.01 log-axis floor that floor is kept.

    python scripts/paper_stats/prism_rewrite_all.py --cut 0.5 --go
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import zipfile

PROJECTS = os.path.expanduser("~/.eeg_seizure_analyzer/projects")
DB = "lunarc_detect_wk1-6_final.db"
EXCLUDE = {"x", "355676", "372837", "30"}
BASE_W, LEV_W = {1, 2, 3}, {4, 5}
NAN = float("nan")


# ---------------------------------------------------------------- data layer
class Data:
    def __init__(self, cut: float):
        conn = sqlite3.connect(os.path.join(PROJECTS, DB))
        conn.row_factory = sqlite3.Row
        self.wk, self.day, self.date = {}, {}, {}
        for r in conn.execute("SELECT id, path, date FROM chunks"):
            m = re.search(r"Week(\d+)-Day(\d+)", r["path"])
            self.wk[r["id"]] = int(m.group(1)) if m else None
            self.day[r["id"]] = int(m.group(2)) if m else None
            self.date[r["id"]] = r["date"]
        self.grp = {r[0]: r[1] for r in conn.execute(
            "SELECT DISTINCT animal_id, group_id FROM file_animals")}
        self.fa = [(r["chunk_id"], r["animal_id"], r["valid_sec"] or 0)
                   for r in conn.execute(
                       "SELECT chunk_id, animal_id, valid_sec FROM file_animals")]
        self.ev = [(r["chunk_id"], r["animal_id"], r["type"], r["duration_sec"])
                   for r in conn.execute(
                       "SELECT chunk_id, animal_id, type, duration_sec, cnn_confidence "
                       "FROM events WHERE excluded=0 AND cnn_confidence >= ?", (cut,))]
        conn.close()
        animals = sorted({a for _c, a, _s in self.fa
                          if a not in EXCLUDE and self.grp.get(a) in ("Control", "SV2A")})
        self.egfp = [a for a in animals if self.grp[a] == "Control"]
        self.sv2a = [a for a in animals if self.grp[a] == "SV2A"]

    def _sel(self, weeks=None, days=None):
        return lambda ci: (self.wk.get(ci) in weeks if weeks
                           else self.day.get(ci) == days)

    def _agg(self, sel):
        hrs = collections.defaultdict(float)
        rdays = collections.defaultdict(set)
        for ci, a, s in self.fa:
            if sel(ci):
                hrs[a] += s / 3600
                rdays[a].add(self.date[ci])
        cnt = collections.defaultdict(collections.Counter)
        durs = collections.defaultdict(list)
        cdays = collections.defaultdict(set)
        for ci, a, t, d in self.ev:
            if sel(ci):
                cnt[a][t] += 1
                durs[(a, t)].append(d)
                durs[(a, "all")].append(d)
                if t == "convulsive":
                    cdays[a].add(self.date[ci])
        return hrs, rdays, cnt, durs, cdays

    def rate(self, group, kind, weeks=None, days=None):
        """events per 24 h of recording."""
        hrs, _rd, cnt, _du, _cd = self._agg(self._sel(weeks, days))
        key = {"conv": ["convulsive"], "non": ["non_convulsive"],
               "all": ["convulsive", "non_convulsive"]}[kind]
        out = []
        for a in group:
            d = hrs[a] / 24
            out.append(sum(cnt[a][k] for k in key) / d if d else NAN)
        return out

    def dur(self, group, kind, weeks=None, days=None):
        _h, _rd, _c, durs, _cd = self._agg(self._sel(weeks, days))
        k = {"conv": "convulsive", "non": "non_convulsive", "all": "all"}[kind]
        return [statistics.mean(durs[(a, k)]) if durs[(a, k)] else NAN
                for a in group]

    def convfree(self, group, weeks=None, days=None):
        _h, rdays, _c, _du, cdays = self._agg(self._sel(weeks, days))
        return [100.0 * (len(rdays[a]) - len(cdays[a])) / len(rdays[a])
                if rdays[a] else NAN for a in group]


# --------------------------------------------------------------- csv helpers
def _num(c):
    c = (c or "").strip()
    if c in ("", "-"):
        return None
    try:
        return float(c.replace(",", "."))
    except ValueError:
        return None


def blocks_of(rows, skip_first_col_numeric=False):
    """Column runs that hold data, separated by all-empty spacer columns."""
    width = max(len(r) for r in rows)
    has = []
    for c in range(width):
        any_num = any(c < len(r) and _num(r[c]) is not None for r in rows)
        has.append(any_num)
    if skip_first_col_numeric and has:
        has[0] = False
    runs, start = [], None
    for c, h in enumerate(has):
        if h and start is None:
            start = c
        elif not h and start is not None:
            runs.append((start, c - 1))
            start = None
    if start is not None:
        runs.append((start, width - 1))
    return runs


def fmt(v, floor: float):
    if v != v:
        return ""
    if floor and abs(v) < 1e-12:
        return f"{floor:g}"
    return f"{v:.4f}".rstrip("0").rstrip(".") or "0"


def write_cells(rows, plan, floors):
    """plan: {(row, col): value}; returns a new grid."""
    out = [list(r) for r in rows]
    width = max(len(r) for r in rows)
    for r in out:
        r.extend([""] * (width - len(r)))
    for (ri, ci), v in plan.items():
        if ri < len(out) and ci < len(out[ri]):
            out[ri][ci] = fmt(v, floors)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prism", default="Seizure_data.prism")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()
    out_path = args.out or args.prism.replace(".prism", f"_conf{args.cut:g}.prism")

    D = Data(args.cut)
    E, S = D.egfp, D.sv2a
    print(f"n = {len(E)} EGFP / {len(S)} SV2A   baseline=wk1-3  LEV=wk4-5  "
          f"cut={args.cut}\n")

    # title -> (kind, spec).  'rows2' = 2 blocks of animals across the row.
    def R(kind, weeks):
        return lambda g: D.rate(g, kind, weeks=weeks)

    def U(kind, weeks):
        return lambda g: D.dur(g, kind, weeks=weeks)

    def F(weeks):
        return lambda g: D.convfree(g, weeks=weeks)

    REG = {
        # rows = metric, blocks = [EGFP, SV2A]
        "Seizures/day":                 ("rowsGG", [R("conv", BASE_W), R("non", BASE_W)], 0.01),
        "Seizures/day linearSeizures/day": ("rowsGG", [R("conv", BASE_W), R("non", BASE_W)], 0.01),
        "Duration":                     ("rowsGG", [U("conv", BASE_W | LEV_W | {6}), U("non", BASE_W | LEV_W | {6})], 0),
        "LEV Duration":                 ("rowsGG", [U("conv", LEV_W), U("non", LEV_W)], 0),
        "LEV Seizures/day":             ("rowsGG", [R("conv", LEV_W), R("non", LEV_W)], 0),
        # rows = group, blocks = [baseline, LEV]
        "Convulsive pre-post LEV":      ("rowsPP", [R("conv", BASE_W), R("conv", LEV_W)], 0),
        "Non-Convulsive pre-post LEV":  ("rowsPP", [R("non", BASE_W), R("non", LEV_W)], 0),
        "Convulsive Duration pre-post LEV":     ("rowsPP", [U("conv", BASE_W), U("conv", LEV_W)], 0),
        "Non-Convulsive Duration pre-post LEV": ("rowsPP", [U("non", BASE_W), U("non", LEV_W)], 0),
        "Convulsive free days pre-post LEV":    ("rowsPP", [F(BASE_W), F(LEV_W)], 0),
        # 4 columns: EGFP base, EGFP LEV, SV2A base, SV2A LEV
        "Convulsive pre-post LEV for stats":        ("cols4", [R("conv", BASE_W), R("conv", LEV_W)], 0),
        "Non-Convulsive pre-post LEV for stats":    ("cols4", [R("non", BASE_W), R("non", LEV_W)], 0),
        # NOTE: this one interleaves by GROUP (conv EGFP, conv SV2A, non EGFP,
        # non SV2A), unlike the pre-post tables which interleave by phase.
        "LEV Seizures/day for stats":               ("cols4G", [R("conv", LEV_W), R("non", LEV_W)], 0),
        "Convulsive Duration pre-post LEV for stats":     ("cols4", [U("conv", BASE_W), U("conv", LEV_W)], 0),
        "Non-Convulsive Duration pre-post LEV for stats": ("cols4", [U("non", BASE_W), U("non", LEV_W)], 0),
        "Convulsive free days pre-post LEV for stats":    ("cols4", [F(BASE_W), F(LEV_W)], 0),
        # 2 columns: EGFP, SV2A
        "Convulsive free days baseline":  ("cols2", [F(BASE_W)], 0),
        # rows = week number in column 0; all seizures per 24 h. Zeros floored
        # at 0.1 here, not 0.01 — this table uses its own log-axis floor.
        "Per week seizures":  ("weekrows", None, 0.1),
        # rows = day number in column 0; all seizures per HOUR (not per 24 h).
        "Events/animal-hour ALL DAYS":  ("dayrows", None, 0),
        # rows = group, cols = [increase, decrease] week 1 -> week 3.
        # Animals with no seizures in either week count as "did not increase",
        # i.e. decrease, so the two columns still sum to n.
        "All seizures increase/decrease":    ("incdec", "n", 0),
        "All seizures increase/decrease %":  ("incdec", "pct", 0),
    }

    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(args.prism) as z:
        z.extractall(tmp)
    titles = {}
    for sj in glob.glob(os.path.join(tmp, "data/sheets/*/sheet.json")):
        d = json.load(open(sj))
        if d.get("@class") == "DataSheet" and "ANALYSIS_VIEW" not in (d.get("flags") or []):
            tu = (d.get("table") or {}).get("uid")
            if tu:
                titles.setdefault(d.get("title", ""), tu)

    done, missing = [], []
    for title, (kind, fns, floor) in REG.items():
        tu = titles.get(title)
        if not tu:
            missing.append(f"{title} (no such sheet)")
            continue
        p = os.path.join(tmp, "data/tables", tu, "data.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            missing.append(f"{title} (empty table)")
            continue
        rows = list(csv.reader(open(p)))
        plan = {}

        if kind in ("rowsGG", "rowsPP"):
            # Regular layout: column 0 is the row label, then two equal-width
            # blocks. Inferring blocks from where data happens to sit breaks on
            # duration tables, where animals with no events leave internal gaps
            # that fragment the runs.
            width = max(len(r) for r in rows)
            b1, b2 = 1, 1 + (width - 1) // 2
            if b2 - b1 < max(len(E), len(S)):
                missing.append(f"{title} (block width {b2-b1} too narrow)")
                continue
            for ri, fn in enumerate(fns):
                if ri >= len(rows):
                    break
                if kind == "rowsGG":          # blocks = EGFP, SV2A
                    for k, v in enumerate(fn(E)):
                        plan[(ri, b1 + k)] = v
                    for k, v in enumerate(fn(S)):
                        plan[(ri, b2 + k)] = v
                else:                          # rows = groups, blocks = phases
                    grp = E if ri == 0 else S
                    for k, v in enumerate(fns[0](grp)):
                        plan[(ri, b1 + k)] = v
                    for k, v in enumerate(fns[1](grp)):
                        plan[(ri, b2 + k)] = v
                if kind == "rowsPP":
                    break                      # both rows handled in one pass
            if kind == "rowsPP":
                for ri, grp in ((0, E), (1, S)):
                    if ri >= len(rows):
                        continue
                    for k, v in enumerate(fns[0](grp)):
                        plan[(ri, b1 + k)] = v
                    for k, v in enumerate(fns[1](grp)):
                        plan[(ri, b2 + k)] = v

        elif kind in ("cols4", "cols4G"):
            a, b = fns
            cols = ([a(E), a(S), b(E), b(S)] if kind == "cols4G"
                    else [a(E), b(E), a(S), b(S)])
            for cj, vals in enumerate(cols):
                for k, v in enumerate(vals):
                    plan[(k, cj)] = v

        elif kind in ("weekrows", "dayrows"):
            width = max(len(r) for r in rows)
            b1, b2 = 1, 1 + (width - 1) // 2
            for ri, row in enumerate(rows):
                idx = _num(row[0]) if row else None
                if idx is None:
                    continue
                idx = int(idx)
                if kind == "weekrows":
                    if not 1 <= idx <= 6:
                        continue
                    ve, vs = (D.rate(E, "all", weeks={idx}),
                              D.rate(S, "all", weeks={idx}))
                else:
                    if not 1 <= idx <= 35:      # weeks 1-5; week 6 excluded
                        continue
                    ve = [v / 24 for v in D.rate(E, "all", days=idx)]
                    vs = [v / 24 for v in D.rate(S, "all", days=idx)]
                for k, v in enumerate(ve):
                    plan[(ri, b1 + k)] = v
                for k, v in enumerate(vs):
                    plan[(ri, b2 + k)] = v

        elif kind == "incdec":
            for ri, grp in ((0, E), (1, S)):
                w1 = D.rate(grp, "all", weeks={1})
                w3 = D.rate(grp, "all", weeks={3})
                inc = sum(1 for x, y in zip(w1, w3) if y > x)
                dec = len(grp) - inc
                if fns == "pct":
                    vals = [round(100.0 * inc / len(grp)),
                            round(100.0 * dec / len(grp))]
                else:
                    vals = [inc, dec]
                plan[(ri, 1)] = float(vals[0])
                plan[(ri, 2)] = float(vals[1])

        elif kind == "cols2":
            for cj, vals in enumerate([fns[0](E), fns[0](S)]):
                for k, v in enumerate(vals):
                    plan[(k, cj)] = v

        need_rows = max(r for r, _c in plan) + 1
        grid = [list(r) for r in rows]
        while len(grid) < need_rows:
            grid.append([""] * (max(len(x) for x in grid) if grid else 1))
        newrows = write_cells(grid, plan, floor)
        if args.go:
            with open(p, "w", newline="") as f:
                csv.writer(f).writerows(newrows)
        done.append((title, len(plan)))

    for t, n in done:
        print(f"  {'wrote' if args.go else 'would write'} {n:3d} cells   {t}")
    if missing:
        print("\n  NOT DONE:")
        for m in missing:
            print(f"    {m}")

    if args.go:
        if os.path.exists(out_path):
            os.remove(out_path)
        subprocess.run(["zip", "-r", "-X", "-q", os.path.abspath(out_path), "."],
                       cwd=tmp, check=True)
        print(f"\nWrote {out_path}")
    shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
