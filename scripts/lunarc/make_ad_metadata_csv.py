#!/usr/bin/env python
"""Generate the batch-metadata CSV for the AD (5xFAD) cohort.

Why this is needed: analysis.process_spike_chunk_classical only detects on EEG
channels that carry an animal ID (analysis.py, "No animal-ID-mapped EEG
channels"), and the AD EDF headers are generic — 'Ch1 Biopot' .. 'Ch8 Biopot'
plus 8 'Ch<N> Act' activity channels, no IDs anywhere. Without this CSV every
file errors out.

The montage comes from detect_5xFAD_IED_ISI.m, which sits next to the
recordings and states it outright:

    % Each selected EDF channel is a DIFFERENT animal:
    %   Ch1, Ch4, Ch6, Ch7

Those are Marco's 1-based channel numbers, i.e. code indices 0, 3, 5, 6 (the
usual chN -> ch(N-1) offset). The other four Biopot channels are unmapped on
purpose: leaving them blank means the detector skips them, which matches what
the MATLAB analysis did and keeps empty channels out of the results.

The montage is assumed constant across Week_1..Week_3 — the MATLAB script
applies the same four channels to any EDF it is pointed at.

Usage (one row per EDF, keyed on basename):

    python scripts/lunarc/make_ad_metadata_csv.py \
        --edf-dir /lunarc/nobackup/projects/lu2026-2-60/ad_edf_data \
        --out scripts/lunarc/ad_metadata.csv

Override the IDs / genotypes once they are known — no need to edit this file:

    --animal-map '0=1234,3=1235,5=1236,6=1237' \
    --group-map  '0=5xFAD,3=5xFAD,5=WT,6=WT'
"""
from __future__ import annotations

import argparse
import csv
import os

N_CH = 8

# code channel index -> animal ID. Defaults follow the MATLAB output folder
# names (Animal_Ch1 etc.) so the two analyses can be compared row for row.
DEFAULT_ANIMALS = {0: "Animal_Ch1", 3: "Animal_Ch4", 5: "Animal_Ch6", 6: "Animal_Ch7"}
# code channel index -> genotype/group. Blank until confirmed: the recordings
# carry no genotype anywhere, and guessing one would silently mislabel the stats.
DEFAULT_GROUPS: dict[int, str] = {}
DEFAULT_COHORT = "AD_5xFAD"


def _parse_map(spec: str | None, what: str) -> dict[int, str]:
    """'0=abc,3=def' -> {0: 'abc', 3: 'def'}."""
    if not spec:
        return {}
    out: dict[int, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--{what}: expected 'ch=value', got {item!r}")
        k, v = item.split("=", 1)
        idx = int(k.strip())
        if not 0 <= idx < N_CH:
            raise SystemExit(f"--{what}: channel {idx} out of range 0..{N_CH - 1}")
        out[idx] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf-dir", help="scan this folder recursively for *.edf")
    ap.add_argument("--file-list", help="text file with one EDF path per line, so "
                    "the CSV can be built without the EDFs present")
    ap.add_argument("--out", default="ad_metadata.csv")
    ap.add_argument("--cohort", default=DEFAULT_COHORT)
    ap.add_argument("--animal-map", help="'0=id,3=id,...' overriding the defaults "
                    "(replaces them wholesale, so list every mapped channel)")
    ap.add_argument("--group-map", help="'0=5xFAD,3=WT,...' per-channel genotype")
    args = ap.parse_args()

    if args.file_list:
        with open(args.file_list) as f:
            edfs = [ln.strip() for ln in f
                    if ln.strip() and ln.strip().lower().endswith(".edf")]
    elif args.edf_dir:
        edfs = []
        for root, _dirs, names in os.walk(args.edf_dir):
            edfs += [os.path.join(root, n) for n in names
                     if n.lower().endswith(".edf")]
    else:
        raise SystemExit("give --edf-dir or --file-list")

    animals = _parse_map(args.animal_map, "animal-map") or dict(DEFAULT_ANIMALS)
    groups = _parse_map(args.group_map, "group-map") or dict(DEFAULT_GROUPS)

    # Basenames are the key the loader matches on, so a collision would make one
    # file silently adopt another's row. Day prefixes make them unique here.
    names = sorted({os.path.basename(p) for p in edfs})
    if len(names) != len(edfs):
        print(f"!! {len(edfs) - len(names)} duplicate basenames — check the tree")

    header = ["filename", "cohort", "group_id"]
    for i in range(N_CH):
        header += [f"animal_ch{i}", f"cohort_ch{i}", f"group_ch{i}"]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name in names:
            row = [name, args.cohort, ""]
            for i in range(N_CH):
                row += [animals.get(i, ""), "", groups.get(i, "")]
            w.writerow(row)

    mapped = ", ".join(f"ch{i}={animals[i]}" for i in sorted(animals))
    print(f"Wrote {args.out}: {len(names)} files")
    print(f"  mapped channels: {mapped}")
    print(f"  unmapped (skipped by the detector): "
          f"{[i for i in range(N_CH) if i not in animals]}")
    if not groups:
        print("  NOTE: no genotype/group set — events land with animal IDs but "
              "no group. Re-run with --group-map to set them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
