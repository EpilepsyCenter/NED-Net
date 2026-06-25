#!/usr/bin/env python
"""Compare two batch-detection project DBs (e.g. old U-Net-only vs the new
lower-threshold + hysteresis + re-ranker run).

The new run uses a lower detection threshold, so it has MORE raw events — many
of them marginal candidates the re-ranker scores near 0. The fair comparison is
therefore "new run AFTER a confidence cut" vs the old run: pass --min-confidence
to apply the re-ranker operating point (the new run stores P(real) in
cnn_confidence). Reports totals, convulsive/non-convulsive split, mean duration,
and per-animal event counts.

    python scripts/local/compare_detect_runs.py \
        --old ~/.eeg_seizure_analyzer/projects/lunarc_detect_wk1-3.db \
        --new ~/.eeg_seizure_analyzer/projects/lunarc_detect_wk1-3_v2.db \
        --min-confidence 0.5
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict


def _summarise(db_path: str, min_conf: float) -> dict:
    db_path = os.path.abspath(os.path.expanduser(db_path))
    c = sqlite3.connect(db_path)
    # Only non-excluded events; apply the confidence cut (no-op at 0.0).
    rows = c.execute(
        "SELECT animal_id, type, duration_sec, cnn_confidence "
        "FROM events WHERE COALESCE(excluded,0)=0 "
        "AND COALESCE(cnn_confidence,1.0) >= ?", (min_conf,)
    ).fetchall()
    n_files = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    c.close()

    per_animal: dict[str, int] = defaultdict(int)
    n_conv = n_nonconv = 0
    total_dur = 0.0
    for animal_id, etype, dur, _conf in rows:
        per_animal[animal_id or "(none)"] += 1
        if etype == "convulsive":
            n_conv += 1
        else:
            n_nonconv += 1
        total_dur += dur or 0.0
    n = len(rows)
    return {
        "db": db_path, "n_files": n_files, "n_events": n,
        "n_conv": n_conv, "n_nonconv": n_nonconv,
        "mean_dur": (total_dur / n) if n else 0.0,
        "per_animal": dict(per_animal),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="baseline DB (U-Net only)")
    ap.add_argument("--new", required=True, help="new-workflow DB")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="keep only events with cnn_confidence >= this. For the "
                         "new (re-ranked) DB this is the re-ranker operating "
                         "point; the old DB's confidences are raw CNN means.")
    ap.add_argument("--old-min-confidence", type=float, default=None,
                    help="separate cut for the OLD DB (default: 0.0, since its "
                         "confidences aren't re-ranker scores)")
    args = ap.parse_args()

    old = _summarise(args.old, args.old_min_confidence or 0.0)
    new = _summarise(args.new, args.min_confidence)

    def line(label, a, b):
        print(f"  {label:24s} {a:>12} {b:>12}")

    print("=" * 52)
    print(f"{'':24s} {'OLD':>12} {'NEW':>12}")
    print(f"  (new cut: cnn_confidence >= {args.min_confidence})")
    print("-" * 52)
    line("files", old["n_files"], new["n_files"])
    line("events (total)", old["n_events"], new["n_events"])
    line("  convulsive", old["n_conv"], new["n_conv"])
    line("  non-convulsive", old["n_nonconv"], new["n_nonconv"])
    line("mean duration (s)", f"{old['mean_dur']:.1f}", f"{new['mean_dur']:.1f}")
    line("animals with events", len(old["per_animal"]), len(new["per_animal"]))
    print("-" * 52)

    print("Per-animal event counts (old -> new):")
    animals = sorted(set(old["per_animal"]) | set(new["per_animal"]))
    for a in animals:
        o = old["per_animal"].get(a, 0)
        n = new["per_animal"].get(a, 0)
        flag = "  <-- changed" if o != n else ""
        print(f"  {a:20s} {o:>6} -> {n:<6}{flag}")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
