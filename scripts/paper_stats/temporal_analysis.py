#!/usr/bin/env python
"""Diel distribution and temporal clustering of seizures, EGFP vs SV2A.

Two analyses that use the time stamps continuous video-EEG already provides,
and need no new recordings:

  1. DIEL — do seizures favour the light or dark phase, and does SV2A change
     that? Per animal: proportion of seizures in the dark phase, plus a
     Rayleigh test of the 24 h distribution for departure from uniformity.
  2. CLUSTERING — inter-seizure intervals, their coefficient of variation
     (CV, and CV2 which is insensitive to slow drift in rate), the fraction of
     seizures following another within a short window, and the longest
     seizure-free run per animal.

Both are computed per animal, with the animal as the unit of analysis, matching
the rest of the manuscript.

Timing precision: hour_of_day is derived from each recording's start time, so
event times are exact to the hour of file start; within-file intervals are
exact. Intervals spanning files carry up to ~1 h of jitter, which is
immaterial at the hours-to-days scale these metrics describe.

    python scripts/paper_stats/temporal_analysis.py   # lights on 08:00-20:00
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism_rewrite_all import Data, PROJECTS, DB, EXCLUDE   # noqa: E402

BASE = {1, 2, 3}


def load_events(cut, weeks, types=None):
    """-> {animal: [(time_hours, hour_of_day)]}, sorted."""
    import re
    import sqlite3
    conn = sqlite3.connect(os.path.join(PROJECTS, DB))
    conn.row_factory = sqlite3.Row
    wk = {}
    for r in conn.execute("SELECT id, path FROM chunks"):
        m = re.search(r"Week(\d+)-Day", r["path"])
        wk[r["id"]] = int(m.group(1)) if m else None
    out = collections.defaultdict(list)
    for r in conn.execute(
            "SELECT e.animal_id a, e.chunk_id ci, e.date d, e.start_sec s, "
            "e.hour_of_day h, e.type t FROM events e "
            "WHERE e.excluded=0 AND e.cnn_confidence >= ?", (cut,)):
        if wk.get(r["ci"]) not in weeks or r["a"] in EXCLUDE:
            continue
        if types and r["t"] not in types:
            continue
        if r["h"] is None or not r["d"]:
            continue
        day = dt.date.fromisoformat(r["d"]).toordinal()
        # hour_of_day already folds in the file start hour; add the within-hour
        # offset so ordering inside a recording is exact.
        t_h = day * 24 + r["h"] + (r["s"] % 3600) / 3600.0
        out[r["a"]].append((t_h, int(r["h"])))
    conn.close()
    for a in out:
        out[a].sort()
    return out


def rayleigh(hours):
    """Circular test for a 24 h rhythm. -> (mean hour, vector length R, p)."""
    ang = np.array(hours, float) * 2 * np.pi / 24
    n = len(ang)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    C, S = np.cos(ang).sum(), np.sin(ang).sum()
    R = np.hypot(C, S) / n
    mu = (np.arctan2(S, C) % (2 * np.pi)) * 24 / (2 * np.pi)
    Z = n * R * R
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n))     # Zar approximation
    return float(mu), float(R), float(min(1.0, p))


def cv2(intervals):
    """Mean of 2|I(i+1)-I(i)|/(I(i+1)+I(i)) — insensitive to slow rate drift."""
    v = np.asarray(intervals, float)
    if len(v) < 3:
        return float("nan")
    a, b = v[:-1], v[1:]
    ok = (a + b) > 0
    return float(np.mean(2 * np.abs(b - a)[ok] / (a + b)[ok]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--lights-on", type=int, default=8,
                    help="hour lights come ON (12:12 cycle assumed)")
    ap.add_argument("--phase", choices=("base", "lev", "all"), default="base")
    ap.add_argument("--type", choices=("all", "convulsive", "non_convulsive"),
                    default="all")
    ap.add_argument("--cluster-window", type=float, default=6.0,
                    help="hours; a seizure within this of the previous one "
                         "counts as clustered")
    ap.add_argument("--min-events", type=int, default=5,
                    help="animals with fewer events are reported but excluded "
                         "from the statistics")
    args = ap.parse_args()

    weeks = {"base": BASE, "lev": {4, 5}, "all": {1, 2, 3, 4, 5}}[args.phase]
    types = None if args.type == "all" else {args.type}
    D = Data(args.cut)
    E, S = D.egfp, D.sv2a
    ev = load_events(args.cut, weeks, types)
    lights_off = (args.lights_on + 12) % 24

    def is_dark(h):
        return not (args.lights_on <= h < lights_off)

    hrs, rdays, _c, _d, _cd = D._agg(D._sel(weeks=weeks))

    print(f"phase={args.phase} (weeks {sorted(weeks)})  type={args.type}  "
          f"cut={args.cut}")
    print(f"light phase {args.lights_on:02d}:00-{lights_off:02d}:00, "
          f"dark {lights_off:02d}:00-{args.lights_on:02d}:00\n")

    rows = []
    for grp, animals in (("EGFP", E), ("SV2A", S)):
        for a in animals:
            evs = ev.get(a, [])
            n = len(evs)
            times = [t for t, _h in evs]
            hod = [h for _t, h in evs]
            dark = sum(1 for h in hod if is_dark(h))
            iv = np.diff(times) if n > 1 else np.array([])
            mu, R, p_ray = rayleigh(hod) if n >= 3 else (np.nan,) * 3
            clustered = (float(np.mean(iv <= args.cluster_window))
                         if len(iv) else np.nan)
            # longest gap without a seizure, bounded by recorded days
            longest = float(np.max(iv) / 24) if len(iv) else np.nan
            rows.append(dict(g=grp, a=a, n=n,
                             pct_dark=100.0 * dark / n if n else np.nan,
                             mu=mu, R=R, p_ray=p_ray,
                             median_isi=float(np.median(iv) / 24) if len(iv) else np.nan,
                             cv=float(np.std(iv, ddof=1) / np.mean(iv))
                             if len(iv) > 1 and np.mean(iv) else np.nan,
                             cv2=cv2(iv), clustered=clustered, longest=longest,
                             free_days=None))

    print(f"{'grp':6}{'animal':9}{'n':>5}{'%dark':>7}{'peak h':>8}{'R':>6}"
          f"{'p_ray':>9}{'medISI d':>10}{'CV':>7}{'CV2':>7}"
          f"{'%clust':>8}{'maxfree d':>10}")
    for r in rows:
        f = lambda v, d=2: ("  -  " if v is None or v != v else f"{v:.{d}f}")
        print(f"{r['g']:6}{r['a']:9}{r['n']:>5}{f(r['pct_dark'],0):>7}"
              f"{f(r['mu'],1):>8}{f(r['R']):>6}{f(r['p_ray'],4):>9}"
              f"{f(r['median_isi']):>10}{f(r['cv']):>7}{f(r['cv2']):>7}"
              f"{f(100*r['clustered'] if r['clustered']==r['clustered'] else np.nan,0):>8}"
              f"{f(r['longest']):>10}")

    def col(g, k):
        return np.array([r[k] for r in rows
                         if r["g"] == g and r["n"] >= args.min_events
                         and r[k] == r[k]], float)

    print(f"\n=== group comparisons (animals with >= {args.min_events} events; "
          "Mann-Whitney U, two-tailed)")
    print(f"{'measure':34}{'EGFP':>16}{'SV2A':>16}{'P':>10}")
    for k, lab in (("pct_dark", "% seizures in dark phase"),
                   ("median_isi", "median inter-seizure interval (d)"),
                   ("cv", "CV of inter-seizure intervals"),
                   ("cv2", "CV2 (drift-insensitive)"),
                   ("clustered", f"fraction within {args.cluster_window:g} h"),
                   ("longest", "longest seizure-free run (d)")):
        a, b = col("EGFP", k), col("SV2A", k)
        if len(a) < 2 or len(b) < 2:
            print(f"{lab:34}{'n too small':>16}{'':>16}{'':>10}")
            continue
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        print(f"{lab:34}{np.median(a):>10.2f} (n={len(a)})"
              f"{np.median(b):>10.2f} (n={len(b)}){p:>10.4g} {star}")

    print("\n=== is the dark-phase proportion different from 50%? "
          "(Wilcoxon signed-rank vs 50)")
    for g in ("EGFP", "SV2A"):
        v = col(g, "pct_dark")
        if len(v) >= 3:
            p = stats.wilcoxon(v - 50).pvalue
            print(f"  {g}: median {np.median(v):.1f}%  n={len(v)}  P={p:.4g}"
                  + ("  *" if p < .05 else ""))

    print("\n=== pooled 24 h profile (events per hour, all animals)")
    prof = collections.Counter()
    for grp, animals in (("EGFP", E), ("SV2A", S)):
        for a in animals:
            for _t, h in ev.get(a, []):
                prof[(grp, h)] += 1
    for grp in ("EGFP", "SV2A"):
        line = " ".join(f"{prof[(grp, h)]:>4d}" for h in range(24))
        tot = sum(prof[(grp, h)] for h in range(24))
        dark = sum(prof[(grp, h)] for h in range(24) if is_dark(h))
        mu, R, p = rayleigh([h for a in (E if grp == "EGFP" else S)
                             for _t, h in ev.get(a, [])])
        print(f"  {grp} h00-23: {line}")
        print(f"       total {tot}, dark {100*dark/tot:.1f}%  "
              f"Rayleigh peak {mu:.1f} h, R={R:.3f}, P={p:.3g}"
              + ("  *" if p < .05 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
