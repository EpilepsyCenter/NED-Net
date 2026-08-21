#!/usr/bin/env python
"""Does the SV2A result survive a 0.5 detector-confidence cut?

The manuscript figures were built with NO confidence cut (verified: the
figure's per-animal rates reproduce exactly from lunarc_detect_wk1-6_final.db
at min_conf = 0, weeks 1-3, excluding x/355676/372837/30). This re-runs every
headline comparison at min_conf = 0 and 0.5 side by side and flags anything
that changes.

Same recipe as stats_dig.py + paper_stats.py, so "still holds" means the same
thing it did in the manuscript:
  * animal is the unit of analysis; events filtered on excluded = 0
  * denominator = distinct recording DAYS per animal per phase
  * baseline = weeks 1-3, levetiracetam = weeks 4-6 (per-batch relative)
  * between groups: Mann-Whitney U, two-tailed
  * within animal, baseline vs LEV: Wilcoxon signed-rank

    python scripts/paper_stats/conf_sensitivity.py
    python scripts/paper_stats/conf_sensitivity.py --cut 0.35 --db other.db
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sqlite3
import statistics
import sys

import numpy as np
from scipy import stats

PROJECTS = os.path.expanduser("~/.eeg_seizure_analyzer/projects")
DEFAULT_DB = "lunarc_detect_wk1-6_final.db"
EXCLUDE = {"30", "355676", "372837", "x"}
ALPHA = 0.05


def per_animal(db: str, cut: float):
    """-> rows of per-animal, per-phase metrics at a given confidence cut."""
    conn = sqlite3.connect(os.path.join(PROJECTS, db))
    conn.row_factory = sqlite3.Row

    week = {}
    for r in conn.execute("SELECT id, path, date FROM chunks"):
        m = re.search(r"Week(\d+)-Day(\d+)", r["path"])
        week[r["id"]] = (int(m.group(1)) if m else None, r["date"])

    grp = {}
    for r in conn.execute("SELECT DISTINCT animal_id, group_id FROM file_animals"):
        g = (r["group_id"] or "").strip().lower()
        if g in ("control", "sv2a") and r["animal_id"] not in EXCLUDE:
            grp[r["animal_id"]] = g

    # Recording days per animal per phase, and per animal per WEEK for the
    # development-over-weeks view. Independent of events, so animals with no
    # seizures still contribute a denominator (and count as seizure-free).
    days = collections.defaultdict(set)
    wk_days = collections.defaultdict(set)
    for r in conn.execute("SELECT animal_id a, chunk_id ci FROM file_animals"):
        w, d = week.get(r["ci"], (None, None))
        if w is None or r["a"] not in grp:
            continue
        days[(r["a"], "base" if w <= 3 else "lev")].add(d)
        wk_days[(r["a"], w)].add(d)

    counts = collections.defaultdict(collections.Counter)
    wk_counts = collections.defaultdict(collections.Counter)
    durs = collections.defaultdict(list)
    conv_days = collections.defaultdict(set)
    any_days = collections.defaultdict(set)
    for r in conn.execute("SELECT animal_id a, chunk_id ci, type, duration_sec, "
                          "cnn_confidence FROM events WHERE excluded=0"):
        if (r["cnn_confidence"] or 0) < cut:
            continue
        w, d = week.get(r["ci"], (None, None))
        if w is None or r["a"] not in grp:
            continue
        ph = "base" if w <= 3 else "lev"
        counts[(r["a"], ph)][r["type"]] += 1
        wk_counts[(r["a"], w)][r["type"]] += 1
        durs[(r["a"], ph, r["type"])].append(r["duration_sec"])
        any_days[(r["a"], ph)].add(d)
        if r["type"] == "convulsive":
            conv_days[(r["a"], ph)].add(d)
    conn.close()

    rows = []
    for a in sorted(grp, key=lambda x: (grp[x], x)):
        for ph in ("base", "lev"):
            nd = len(days[(a, ph)])
            if nd == 0:
                continue
            nc = counts[(a, ph)]["convulsive"]
            nn = counts[(a, ph)]["non_convulsive"]
            cdur = (statistics.mean(durs[(a, ph, "convulsive")])
                    if durs[(a, ph, "convulsive")] else float("nan"))
            rows.append(dict(
                a=a, g=grp[a], ph=ph, nd=nd, conv=nc, non=nn,
                cpd=nc / nd, npd=nn / nd, tpd=(nc + nn) / nd,
                csf=100.0 * (nd - len(conv_days[(a, ph)])) / nd,
                asf=100.0 * (nd - len(any_days[(a, ph)])) / nd,
                cd=cdur))
    weekly = {}
    for (a, w), ds in wk_days.items():
        if a in grp and ds:
            c = wk_counts[(a, w)]
            weekly[(a, w)] = ((c["convulsive"] + c["non_convulsive"]) / len(ds),
                              grp[a])
    return rows, weekly


def _col(rows, group, phase, key):
    v = np.array([r[key] for r in rows if r["g"] == group and r["ph"] == phase], float)
    return v[~np.isnan(v)]


def _mw(rows, phase, key):
    a, b = _col(rows, "control", phase, key), _col(rows, "sv2a", phase, key)
    if len(a) < 1 or len(b) < 1:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(a)), float(np.median(b)),
            float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))


def _wilcoxon(rows, group, key):
    base = {r["a"]: r[key] for r in rows if r["g"] == group and r["ph"] == "base"}
    lev = {r["a"]: r[key] for r in rows if r["g"] == group and r["ph"] == "lev"}
    ids = [a for a in base if a in lev
           and not (np.isnan(base[a]) or np.isnan(lev[a]))]
    x = np.array([base[a] for a in ids]); y = np.array([lev[a] for a in ids])
    if len(ids) < 3 or np.all(x == y):
        return float("nan"), float("nan"), float("nan"), len(ids)
    return (float(np.median(x)), float(np.median(y)),
            float(stats.wilcoxon(x, y).pvalue), len(ids))


def _flag(p0, p1):
    """Did significance change between the two cuts?"""
    if np.isnan(p0) or np.isnan(p1):
        return "  n/a"
    s0, s1 = p0 < ALPHA, p1 < ALPHA
    if s0 and not s1:
        return "  ** LOST"
    if s1 and not s0:
        return "  ** GAINED"
    return ""


MEASURES = [("cpd", "convulsive seizures/day"),
            ("npd", "non-convulsive seizures/day"),
            ("tpd", "all seizures/day"),
            ("csf", "convulsive seizure-free days %"),
            ("asf", "any-seizure-free days %"),
            ("cd", "mean convulsive duration (s)")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cut", type=float, default=0.5)
    args = ap.parse_args()

    r0, w0 = per_animal(args.db, 0.0)
    r1, w1 = per_animal(args.db, args.cut)
    n_c = len({r["a"] for r in r0 if r["g"] == "control"})
    n_s = len({r["a"] for r in r0 if r["g"] == "sv2a"})
    print(f"DB: {args.db}   excluding {sorted(EXCLUDE)}   "
          f"n = {n_c} EGFP / {n_s} SV2A")

    # ---- how much the cut removes, and whether it removes it evenly ----
    print(f"\n=== Events retained at conf >= {args.cut} ===")
    print(f"{'phase':8}{'group':10}{'conv 0':>9}{'conv cut':>10}{'kept':>8}"
          f"{'nonconv 0':>11}{'nonconv cut':>13}{'kept':>8}")
    for ph in ("base", "lev"):
        for g in ("control", "sv2a"):
            c0 = sum(r["conv"] for r in r0 if r["g"] == g and r["ph"] == ph)
            c1 = sum(r["conv"] for r in r1 if r["g"] == g and r["ph"] == ph)
            n0 = sum(r["non"] for r in r0 if r["g"] == g and r["ph"] == ph)
            n1 = sum(r["non"] for r in r1 if r["g"] == g and r["ph"] == ph)
            print(f"{ph:8}{g:10}{c0:>9}{c1:>10}{100*c1/c0 if c0 else 0:>7.0f}%"
                  f"{n0:>11}{n1:>13}{100*n1/n0 if n0 else 0:>7.0f}%")

    # ---- between groups, each phase ----
    for ph, title in (("base", "BASELINE (weeks 1-3)"),
                      ("lev", "LEVETIRACETAM (weeks 4-6)")):
        print(f"\n=== {title} — between groups (Mann-Whitney U, two-tailed) ===")
        print(f"{'measure':34}{'EGFP':>9}{'SV2A':>9}{'P(conf0)':>11}"
              f"{'EGFP':>9}{'SV2A':>9}{f'P(>={args.cut})':>11}")
        for key, label in MEASURES:
            a0, b0, p0 = _mw(r0, ph, key)
            a1, b1, p1 = _mw(r1, ph, key)
            print(f"{label:34}{a0:>9.2f}{b0:>9.2f}{p0:>11.4g}"
                  f"{a1:>9.2f}{b1:>9.2f}{p1:>11.4g}{_flag(p0, p1)}")

    # ---- within animal, baseline vs LEV ----
    print("\n=== BASELINE vs LEVETIRACETAM — within animal (Wilcoxon) ===")
    print(f"{'measure':30}{'group':9}{'base':>8}{'LEV':>8}{'P(conf0)':>11}"
          f"{'base':>8}{'LEV':>8}{f'P(>={args.cut})':>11}")
    for key, label in MEASURES[:5]:
        for g in ("control", "sv2a"):
            x0, y0, p0, n = _wilcoxon(r0, g, key)
            x1, y1, p1, _ = _wilcoxon(r1, g, key)
            print(f"{label:30}{g:9}{x0:>8.2f}{y0:>8.2f}{p0:>11.4g}"
                  f"{x1:>8.2f}{y1:>8.2f}{p1:>11.4g}{_flag(p0, p1)}  n={n}")

    # ---- development over weeks ----
    print("\n=== Development over weeks (median all-seizures/day per group) ===")
    print(f"{'week':6}{'EGFP conf0':>12}{'EGFP cut':>11}{'SV2A conf0':>13}{'SV2A cut':>11}")
    for w in range(1, 7):
        cells = []
        for wk in (w0, w1):
            for g in ("control", "sv2a"):
                v = [r for (a, ww), (r, gg) in wk.items() if ww == w and gg == g]
                cells.append(np.median(v) if v else float("nan"))
        print(f"{w:<6}{cells[0]:>12.2f}{cells[2]:>11.2f}{cells[1]:>13.2f}{cells[3]:>11.2f}")

    print(f"\n** = significance changed across the cut (alpha={ALPHA}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
