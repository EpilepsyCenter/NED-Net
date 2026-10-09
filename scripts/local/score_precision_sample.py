#!/usr/bin/env python3
"""Score a reviewed precision spot-check sample (see draw_precision_sample.py).

Reads the review tree's sidecars after every sampled event has been adjudicated
in the UI and reports:

  * **overall precision** of the retrained model's own output -- the number that
    turns "fires more" into "finds more", and the only one comparable to the
    source model's 93.6%;
  * **convulsive precision** on the subset stage 2 labelled convulsive -- the
    honest test of the stage-2 fix on real output, as opposed to validation
    windows;
  * precision per **animal** (the unit of variability: per-animal recall spans
    0-58%) and per **confidence tercile** (confidence has been monotonically
    informative in every earlier sample);
  * Wilson 95% intervals, since several strata will be small.

Refuses to score while events remain `pending`: a partial review biases the
estimate toward whatever the reviewer found easy to judge.

    python scripts/local/score_precision_sample.py --tree ~/review_sample_temporal
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
from collections import defaultdict

import pandas as pd


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval — correct at the 0/n and n/n ends, unlike normal-approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score anyway with events still pending (biases the estimate)")
    a = ap.parse_args()
    tree = os.path.expanduser(a.tree)

    rows = []
    for p in sorted(glob.glob(os.path.join(tree, "*_ned_annotations.json"))):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"unreadable: {p} ({e})", file=sys.stderr); continue
        for x in d.get("annotations", []):
            f = x.get("features") or {}
            rows.append({
                "animal": str(x.get("animal_id") or ""),
                "channel": x.get("channel"),
                "label": x.get("label"),
                "conf": float(x.get("detector_confidence") or 0.0),
                "stage2_convulsive": bool(f.get("convulsive")),
                "conv_prob": float(f.get("convulsive_probability") or 0.0),
                "db_event_id": f.get("sample_db_event_id"),
                "seed": f.get("precision_sample_seed"),
            })
    if not rows:
        print(f"no sidecars found in {tree}", file=sys.stderr); return 1
    df = pd.DataFrame(rows)

    n_pend = (df.label == "pending").sum()
    seeds = sorted({s for s in df.seed.dropna().unique()})
    print(f"{len(df)} sampled events, seed(s) {seeds}, {df.animal.nunique()} animals")
    if n_pend:
        print(f"\n{n_pend} of {len(df)} events still PENDING.", file=sys.stderr)
        if not a.allow_partial:
            print("Refusing to score: a partial review biases the estimate toward the\n"
                  "events that were easy to judge. Finish the queue, or pass\n"
                  "--allow-partial and label the result as partial.", file=sys.stderr)
            return 2

    adj = df[df.label.isin(["confirmed", "rejected"])]
    def line(name, sub):
        n = len(sub); k = int((sub.label == "confirmed").sum())
        if n == 0:
            return f"  {name:28} —"
        lo, hi = wilson(k, n)
        return (f"  {name:28} {100*k/n:5.1f}%  ({k}/{n})  "
                f"95% CI {100*lo:.0f}-{100*hi:.0f}%")

    print("\n=== PRECISION ===")
    print(line("overall", adj))
    print(line("stage 2 said convulsive", adj[adj.stage2_convulsive]))
    print(line("stage 2 said non-convulsive", adj[~adj.stage2_convulsive]))

    print("\n=== by confidence tercile (of the sampled events) ===")
    if len(adj) >= 3:
        q = adj.conf.quantile([1/3, 2/3]).tolist()
        for lab, sub in (("low", adj[adj.conf < q[0]]),
                         ("mid", adj[(adj.conf >= q[0]) & (adj.conf < q[1])]),
                         ("high", adj[adj.conf >= q[1]])):
            print(line(f"{lab} (conf {sub.conf.min():.2f}-{sub.conf.max():.2f})"
                       if len(sub) else lab, sub))

    print("\n=== by animal (the unit of variability) ===")
    for an, sub in sorted(adj.groupby("animal"), key=lambda kv: -len(kv[1])):
        print(line(an, sub))

    print("\nNOTE: this is precision on the model's OWN output — it says nothing about\n"
          "recall. Recall needs a reference independent of the detector (the exhaustive\n"
          "review), not this sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
