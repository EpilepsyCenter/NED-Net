#!/usr/bin/env python
"""Week-by-week interictal-spike trend for the AD (5xFAD) cohort.

Rates are per animal per recording hour, with the denominator taken from
file_animals.valid_sec rather than assumed: files are ~1.5 h but the last of
each day is short, and two files failed detection, so raw counts per week are
not comparable across weeks. Recording time differs by week; rates do not.

    python scripts/local/ad_spike_trend.py
    python scripts/local/ad_spike_trend.py --by-day     # daily detail too
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import defaultdict

DEFAULT_DB = os.path.expanduser("~/.eeg_seizure_analyzer/projects/ad_spikes.db")
_WEEK_RE = re.compile(r"/(Week_\d+)/")
_DAY_RE = re.compile(r"/(W\d+_D\d+)/")


def load(db_path: str):
    """-> (spikes[(week,day,animal)], hours[(week,day,animal)])"""
    conn = sqlite3.connect(db_path)
    try:
        # Recording time per animal per file — the denominator, independent of
        # whether that animal had any events.
        hours: dict[tuple, float] = defaultdict(float)
        for path, animal, sec in conn.execute(
                "SELECT c.path, f.animal_id, f.valid_sec "
                "FROM file_animals f JOIN chunks c ON f.chunk_id = c.id"):
            key = _key(path, animal)
            if key:
                hours[key] += (sec or 0) / 3600.0

        spikes: dict[tuple, int] = defaultdict(int)
        for path, animal, n in conn.execute(
                "SELECT c.path, e.animal_id, COUNT(*) "
                "FROM events e JOIN chunks c ON e.chunk_id = c.id "
                "WHERE e.type = 'interictal_spike' "
                "GROUP BY c.path, e.animal_id"):
            key = _key(path, animal)
            if key:
                spikes[key] += n
    finally:
        conn.close()
    return spikes, hours


def _key(path: str, animal: str):
    w = _WEEK_RE.search(path)
    d = _DAY_RE.search(path)
    if not (w and d and animal):
        return None
    return (w.group(1), d.group(1), animal)


def _agg(d: dict, idx: tuple[int, ...]) -> dict:
    """Sum a {(week,day,animal): value} dict over the chosen key positions."""
    out = defaultdict(float)
    for key, val in d.items():
        out[tuple(key[i] for i in idx)] += val
    return out


def _rate(n: float, h: float) -> float:
    return n / h if h else 0.0


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    if len(xs) < 3:
        return None
    r = spearmanr(xs, ys)
    return float(r.statistic), float(r.pvalue)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--by-day", action="store_true", help="also print daily rates")
    ap.add_argument("--csv-outdir", help="also write Prism-ready CSVs here: "
                    "animals as rows, weeks/days as columns")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    spikes, hours = load(args.db)
    if not hours:
        print("No file_animals rows — nothing to aggregate.", file=sys.stderr)
        return 1

    weeks = sorted({k[0] for k in hours})
    animals = sorted({k[2] for k in hours})

    wa_sp, wa_h = _agg(spikes, (0, 2)), _agg(hours, (0, 2))
    w_sp, w_h = _agg(spikes, (0,)), _agg(hours, (0,))

    print(f"\nDB: {args.db}")
    print(f"{sum(spikes.values()):,.0f} spikes | "
          f"{sum(hours.values()) / len(animals):.1f} h per animal | "
          f"animals {', '.join(animals)}\n")

    # ---- Week x animal, as rate per hour ----
    print("Spikes per hour, by week")
    print(f"  {'animal':<8}" + "".join(f"{w:>10}" for w in weeks) + f"{'W1->W3':>10}")
    for a in animals:
        rates = [_rate(wa_sp.get((w, a), 0), wa_h.get((w, a), 0)) for w in weeks]
        chg = (f"{(rates[-1] / rates[0] - 1) * 100:+.0f}%"
               if rates and rates[0] else "-")
        print(f"  {a:<8}" + "".join(f"{r:>10.1f}" for r in rates) + f"{chg:>10}")

    cohort = [_rate(w_sp.get((w,), 0), w_h.get((w,), 0)) for w in weeks]
    chg = f"{(cohort[-1] / cohort[0] - 1) * 100:+.0f}%" if cohort and cohort[0] else "-"
    print(f"  {'ALL':<8}" + "".join(f"{r:>10.1f}" for r in cohort) + f"{chg:>10}")

    print("\n  recording hours per animal")
    print(f"  {'':<8}" + "".join(f"{wa_h.get((w, animals[0]), 0):>10.1f}"
                                 for w in weeks))

    # ---- Per-animal daily trend ----
    da_sp, da_h = _agg(spikes, (0, 1, 2)), _agg(hours, (0, 1, 2))
    days = sorted({(k[0], k[1]) for k in hours},
                  key=lambda t: (t[0], int(t[1].split("_D")[1])))

    print("\nDaily trend across all 21 days (Spearman rho vs day index)")
    for a in animals:
        xs, ys = [], []
        for i, (w, d) in enumerate(days):
            h = da_h.get((w, d, a), 0)
            if h:
                xs.append(i)
                ys.append(_rate(da_sp.get((w, d, a), 0), h))
        s = spearman(xs, ys)
        if s is None:
            print(f"  animal {a}: (scipy unavailable)")
        else:
            rho, p = s
            mark = "*" if p < 0.05 else " "
            print(f"  animal {a}: rho={rho:+.2f}  p={p:.3f} {mark}  "
                  f"(n={len(xs)} days)")
    print("  * p<0.05, uncorrected; 4 animals tested, so treat a single "
          "starred result cautiously.")

    if args.by_day:
        print("\nSpikes per hour, by day")
        print(f"  {'day':<10}" + "".join(f"{'animal ' + a:>11}" for a in animals))
        for w, d in days:
            row = "".join(f"{_rate(da_sp.get((w, d, a), 0), da_h.get((w, d, a), 0)):>11.1f}"
                          for a in animals)
            print(f"  {d:<10}{row}")

    if args.csv_outdir:
        import csv
        os.makedirs(args.csv_outdir, exist_ok=True)

        # One row per animal, one column per week — the layout Prism wants for a
        # grouped/repeated-measures plot (each animal is a matched subject).
        p = os.path.join(args.csv_outdir, "spike_rate_by_week.csv")
        with open(p, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["animal"] + weeks)
            for a in animals:
                w_.writerow([a] + [f"{_rate(wa_sp.get((w, a), 0), wa_h.get((w, a), 0)):.3f}"
                                   for w in weeks])
        # Same shape by day, for a time-course plot.
        p2 = os.path.join(args.csv_outdir, "spike_rate_by_day.csv")
        with open(p2, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["day", "day_number"] + [f"animal_{a}" for a in animals])
            for i, (w, d) in enumerate(days, start=1):
                w_.writerow([d, i] + [f"{_rate(da_sp.get((w, d, a), 0), da_h.get((w, d, a), 0)):.3f}"
                                      for a in animals])
        # Denominators, so the rates can be audited or re-derived.
        p3 = os.path.join(args.csv_outdir, "recording_hours_by_week.csv")
        with open(p3, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["animal"] + weeks)
            for a in animals:
                w_.writerow([a] + [f"{wa_h.get((w, a), 0):.2f}" for w in weeks])
        print(f"\nWrote {p}\n      {p2}\n      {p3}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
