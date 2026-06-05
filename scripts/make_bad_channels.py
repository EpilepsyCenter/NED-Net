#!/usr/bin/env python3
"""Generate a ``bad_channels.json`` for BENDR pre-training from per-batch rules.

``bendr_pretrain.py --bad-channels`` takes a JSON map of *EDF basename* ->
*list of 0-based channel indices to exclude for that file*. Our exclusion rule,
however, is **per recording batch**, not per file:

    Batch 1 : exclude electrode 1            -> index 0
    Batch 2 : exclude electrodes 6 and 8     -> indices 5, 7
    Batch 3 : exclude none

(The lab numbers electrodes **1-based**; the model indexes channels **0-based**,
so electrode N maps to index N-1. The conversion is baked into BATCH_EXCLUDE
below as indices already — edit it in index space.)

This script walks the EDF tree, figures out each file's batch from the
``batch N`` folder it lives under (matching the consolidated
``SV2A_2024/batch {1,2,3}/Week*-Day*/`` layout, at any nesting depth), and
writes the per-file JSON the trainer expects.

It is deliberately path-driven (no pyedflib / no EDF reads) so it runs anywhere
the files are visible — on the cluster after rsync, or locally on the Z: share.

Usage (on the cluster, after the data has been transferred):

    # LUNARC
    python scripts/make_bad_channels.py \
        --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data

    # Arrhenius
    python scripts/make_bad_channels.py \
        --data-dir "$PROJECT_STORAGE/edf_data"

By default the JSON is written to ``<data-dir>/../bad_channels.json`` — the same
path the pretrain/resume scripts pass to ``--bad-channels``. Override with
``--out``.

When the next cohort arrives, add its batch number + indices to BATCH_EXCLUDE
and re-run; the trainer scripts already point at the output path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# EDIT HERE when exclusions change or a new batch arrives.
# Keys are batch numbers; values are 0-BASED channel indices to drop.
# (Electrode label N, 1-based, -> index N-1.)
# ---------------------------------------------------------------------------
BATCH_EXCLUDE: dict[int, list[int]] = {
    1: [0],      # electrode 1
    2: [5, 7],   # electrodes 6 and 8
    3: [],       # none
}

# Matches a path segment like "batch 1", "Batch_2", "batch3" -> group = number.
_BATCH_RE = re.compile(r"^batch[ _-]?0*([0-9]+)$", re.IGNORECASE)


def batch_of(path: str) -> int | None:
    """Return the batch number from the first ``batch N`` segment in *path*."""
    for part in os.path.normpath(path).split(os.sep):
        m = _BATCH_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="Root of the EDF tree (searched recursively)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <data-dir>/../bad_channels.json)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_path = args.out or os.path.join(os.path.dirname(data_dir), "bad_channels.json")

    if not os.path.isdir(data_dir):
        print(f"ERROR: data-dir not found: {data_dir}", file=sys.stderr)
        return 1

    # basename -> set of exclusion tuples seen (to catch cross-batch collisions)
    seen: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    per_batch_files: dict[int, int] = defaultdict(int)
    unbatched: list[str] = []
    unknown_batch: dict[int, int] = defaultdict(int)
    bad_map: dict[str, list[int]] = {}

    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith(".edf"):
                continue
            full = os.path.join(root, name)
            b = batch_of(full)
            if b is None:
                unbatched.append(full)
                continue
            if b not in BATCH_EXCLUDE:
                unknown_batch[b] += 1
                continue
            per_batch_files[b] += 1
            excl = sorted(BATCH_EXCLUDE[b])
            seen[name].add(tuple(excl))
            if excl:  # unlisted files default to all channels — omit the empties
                bad_map[name] = excl

    # --- hard-fail conditions ------------------------------------------------
    collisions = {n: lists for n, lists in seen.items() if len(lists) > 1}
    if collisions:
        print("ERROR: same basename appears under batches with DIFFERENT "
              "exclusions — the JSON keys on basename, so this is ambiguous:",
              file=sys.stderr)
        for n, lists in sorted(collisions.items())[:20]:
            print(f"  {n}: {sorted(lists)}", file=sys.stderr)
        return 2
    if unbatched:
        print(f"ERROR: {len(unbatched)} EDF(s) are not under any 'batch N' "
              f"folder — cannot assign a rule. First few:", file=sys.stderr)
        for p in unbatched[:10]:
            print(f"  {p}", file=sys.stderr)
        return 3

    # --- write + report ------------------------------------------------------
    with open(out_path, "w") as f:
        json.dump(bad_map, f, indent=2, sort_keys=True)

    total = sum(per_batch_files.values())
    print(f"Scanned {total} EDF(s) under {data_dir}")
    for b in sorted(per_batch_files):
        idx = sorted(BATCH_EXCLUDE[b])
        print(f"  batch {b}: {per_batch_files[b]:5d} file(s)  exclude indices {idx or 'none'}")
    if unknown_batch:
        print("  NOTE: files found under batches with no rule in BATCH_EXCLUDE "
              "(left untouched = all channels):", file=sys.stderr)
        for b in sorted(unknown_batch):
            print(f"    batch {b}: {unknown_batch[b]} file(s)", file=sys.stderr)
    print(f"\nWrote {len(bad_map)} file entr(ies) (batch-3 / no-exclusion files "
          f"omitted on purpose) to:\n  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
