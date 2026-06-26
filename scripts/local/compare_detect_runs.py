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


def _summarise(db_path: str, min_conf: float, keep_convulsive: bool = False,
               path_like: str | None = None) -> dict:
    db_path = os.path.abspath(os.path.expanduser(db_path))
    c = sqlite3.connect(db_path)
    # Only non-excluded events; apply the confidence cut (no-op at 0.0). With
    # keep_convulsive, the cut applies to non-convulsive only — convulsive events
    # are trusted via the cascade classifier, not the (non-conv) re-ranker.
    # path_like restricts to files whose path contains the substring (e.g.
    # "Week4" / a SQL LIKE pattern) so one combined DB can be sliced by week.
    conv_clause = "type='convulsive' OR " if keep_convulsive else ""
    join = "JOIN chunks ch ON e.chunk_id=ch.id "
    # GLOB so character classes work: 'Week[123]' = wk1-3, 'Week[456]' = wk4-6.
    glob = f"*{path_like}*" if path_like else None
    where_path = "AND ch.path GLOB ? " if glob else ""
    params = [min_conf] + ([glob] if glob else [])
    rows = c.execute(
        "SELECT e.animal_id, e.type, e.duration_sec, e.cnn_confidence "
        f"FROM events e {join}WHERE COALESCE(e.excluded,0)=0 "
        f"AND ({conv_clause}COALESCE(e.cnn_confidence,1.0) >= ?) {where_path}",
        params).fetchall()
    nf_where = "WHERE path GLOB ?" if glob else ""
    n_files = c.execute(f"SELECT COUNT(*) FROM chunks {nf_where}",
                        ([glob] if glob else [])).fetchone()[0]
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
    ap.add_argument("--keep-convulsive", action="store_true",
                    help="never drop convulsive events on the confidence cut "
                         "(re-ranker is a non-convulsive layer; convulsive are "
                         "trusted via the cascade). Applies to the NEW DB.")
    ap.add_argument("--path-like", default=None,
                    help="restrict BOTH DBs to files whose path contains this "
                         "(e.g. 'Week4' or a LIKE pattern) — slice one combined "
                         "DB by week. Repeat per week for in/out-of-sample splits.")
    args = ap.parse_args()

    # The old DB's confidences are raw CNN, so the convulsive carve-out is moot
    # there; apply keep_convulsive only to the (re-ranked) new DB.
    old = _summarise(args.old, args.old_min_confidence or 0.0, path_like=args.path_like)
    new = _summarise(args.new, args.min_confidence,
                     keep_convulsive=args.keep_convulsive, path_like=args.path_like)

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
