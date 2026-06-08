#!/usr/bin/env python3
"""Generate a ``bad_channels.json`` for BENDR pre-training from per-batch rules.

``bendr_pretrain.py --bad-channels`` takes a JSON map of *EDF basename* ->
*list of 0-based channel indices to exclude for that file*. Our exclusion rule,
however, is **per recording batch within each cohort**, not per file. The BENDR
staging tree holds one folder per cohort, each with its own ``batch {1,2,3}``::

    edf_data/
      SV2A_2024/     batch {1,2,3}/Week*-Day*/*.edf
      RAM_GDNF_2025/ batch {1,2,3}/W*-D*/*.edf

Batch numbers repeat across cohorts (SV2A batch 1 != GDNF batch 1) and the bad
channels differ, so the rules are keyed on **(cohort, batch)** — see
COHORT_EXCLUDE below.

(The lab numbers electrodes **1-based**; the model indexes channels **0-based**,
so electrode N maps to index N-1. The conversion is baked into COHORT_EXCLUDE
as indices already — edit it in index space. The lab's plain-text source of
truth is ``edf_data_for_bendr/bad channels.txt``.)

This script walks the EDF tree, figures out each file's cohort (from the cohort
folder it lives under) and batch (from the ``batch N`` folder, at any nesting
depth), and writes the per-file JSON the trainer expects.

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

When the next cohort or batch arrives, add its entry to COHORT_EXCLUDE and
re-run; the trainer scripts already point at the output path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# EDIT HERE when exclusions change, a batch's bad channels change, or a new
# cohort/batch arrives. Keyed on cohort folder name, then batch number; values
# are 0-BASED channel indices to drop. (Electrode label N, 1-based, -> index
# N-1.) Source of truth: edf_data_for_bendr/bad channels.txt.
# ---------------------------------------------------------------------------
COHORT_EXCLUDE: dict[str, dict[int, list[int]]] = {
    "SV2A_2024": {
        1: [0, 2, 7],   # electrodes 1, 3, 8
        2: [4, 5],      # electrodes 5, 6
        3: [0, 2, 5],   # electrodes 1, 3, 6
    },
    "RAM_GDNF_2025": {
        1: [1, 4],      # electrodes 2, 5
        2: [2, 3, 6],   # electrodes 3, 4, 7
        3: [],          # all ok
    },
}

# Matches a path segment like "batch 1", "Batch_2", "batch3" -> group = number.
_BATCH_RE = re.compile(r"^batch[ _-]?0*([0-9]+)$", re.IGNORECASE)


def cohort_of(path: str) -> str | None:
    """Return the cohort name from the first segment matching a COHORT_EXCLUDE key."""
    for part in os.path.normpath(path).split(os.sep):
        if part in COHORT_EXCLUDE:
            return part
    return None


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

    # basename -> set of exclusion tuples seen (to catch collisions across
    # cohorts/batches: same filename, different rule = ambiguous JSON key)
    seen: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    per_group_files: dict[tuple[str, int], int] = defaultdict(int)
    uncohorted: list[str] = []
    unbatched: list[str] = []
    unknown_batch: dict[tuple[str, int], int] = defaultdict(int)
    bad_map: dict[str, list[int]] = {}

    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith(".edf"):
                continue
            full = os.path.join(root, name)
            cohort = cohort_of(full)
            if cohort is None:
                uncohorted.append(full)
                continue
            b = batch_of(full)
            if b is None:
                unbatched.append(full)
                continue
            rules = COHORT_EXCLUDE[cohort]
            if b not in rules:
                unknown_batch[(cohort, b)] += 1
                continue
            per_group_files[(cohort, b)] += 1
            excl = sorted(rules[b])
            seen[name].add(tuple(excl))
            if excl:  # unlisted files default to all channels — omit the empties
                bad_map[name] = excl

    # --- hard-fail conditions ------------------------------------------------
    collisions = {n: lists for n, lists in seen.items() if len(lists) > 1}
    if collisions:
        print("ERROR: same basename appears under groups with DIFFERENT "
              "exclusions — the JSON keys on basename, so this is ambiguous:",
              file=sys.stderr)
        for n, lists in sorted(collisions.items())[:20]:
            print(f"  {n}: {sorted(lists)}", file=sys.stderr)
        return 2
    if uncohorted:
        print(f"ERROR: {len(uncohorted)} EDF(s) are not under any known cohort "
              f"folder ({', '.join(sorted(COHORT_EXCLUDE))}) — cannot assign a "
              f"rule. First few:", file=sys.stderr)
        for p in uncohorted[:10]:
            print(f"  {p}", file=sys.stderr)
        return 4
    if unbatched:
        print(f"ERROR: {len(unbatched)} EDF(s) are not under any 'batch N' "
              f"folder — cannot assign a rule. First few:", file=sys.stderr)
        for p in unbatched[:10]:
            print(f"  {p}", file=sys.stderr)
        return 3

    # --- write + report ------------------------------------------------------
    with open(out_path, "w") as f:
        json.dump(bad_map, f, indent=2, sort_keys=True)

    total = sum(per_group_files.values())
    print(f"Scanned {total} EDF(s) under {data_dir}")
    for cohort in sorted({c for c, _ in per_group_files}):
        for (c, b) in sorted(per_group_files):
            if c != cohort:
                continue
            idx = sorted(COHORT_EXCLUDE[c][b])
            print(f"  {c} / batch {b}: {per_group_files[(c, b)]:5d} file(s)  "
                  f"exclude indices {idx or 'none'}")
    if unknown_batch:
        print("  NOTE: files found under cohort/batch with no rule in "
              "COHORT_EXCLUDE (left untouched = all channels):", file=sys.stderr)
        for (c, b) in sorted(unknown_batch):
            print(f"    {c} / batch {b}: {unknown_batch[(c, b)]} file(s)", file=sys.stderr)
    print(f"\nWrote {len(bad_map)} file entr(ies) (no-exclusion files omitted "
          f"on purpose) to:\n  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
