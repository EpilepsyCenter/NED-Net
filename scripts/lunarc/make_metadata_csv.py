#!/usr/bin/env python
"""Generate a batch-metadata CSV for the whole SV2A run.

One row per EDF, in the format the Analysis tab / detect_batch.py read
(eeg_seizure_analyzer.io.batch_metadata): filename, cohort, group_id,
animal_chN, cohort_chN, group_chN.

The SV2A montage is fixed per batch (channel -> animal / cohort / group) — one
combination for each of the 3 batches — so a file's mapping is decided purely by
which batch folder it sits in. Sidecars (*_ned_channels.json / *_ned_meta.json)
still override this CSV at detection time, so the CSV only has to be right for
files that lack sidecars; any file with its own (correct) sidecar is handled by
that sidecar regardless.

Run it where ALL the EDFs live (i.e. on LUNARC) so the CSV covers the full set:

    python scripts/lunarc/make_metadata_csv.py \
        --edf-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data \
        --out batch_metadata.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re

N_CH = 8

# The single channel -> animal / group mapping for each batch (cohort is uniform
# per batch). Derived from the sidecars; batch 2 confirmed unified after the
# Day08 fix (ch1=372829, ch6=372838).
BATCH_DEFAULTS = {
    "1": {
        "cohort": "batch 1",
        "animals": ["355676", "355679", "355668", "355670",
                    "355675", "355673", "355669", "355677"],
        "groups":  ["Control", "Control", "SV2A", "SV2A",
                    "Control", "SV2A", "SV2A", "SV2A"],
    },
    "2": {
        "cohort": "batch 2",
        "animals": ["372830", "372829", "372828", "372832",
                    "372833", "372837", "372838", "x"],
        "groups":  ["SV2A", "SV2A", "SV2A", "SV2A",
                    "Control", "Control", "Control", "x"],
    },
    "3": {
        "cohort": "batch 3",
        "animals": ["30", "29", "31", "33", "32", "25", "24", "26"],
        "groups":  ["SV2A", "SV2A", "Control", "Control",
                    "SV2A", "Control", "SV2A", "SV2A"],
    },
}

_BATCH_RE = re.compile(r"batch\s*([123])", re.IGNORECASE)


def _batch_of(path: str) -> str | None:
    m = _BATCH_RE.search(path)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf-dir", help="scan this folder recursively for *.edf")
    ap.add_argument("--file-list", help="text file with one EDF path per line "
                    "(e.g. `find <edf_dir> -name '*.edf'` from LUNARC) — lets you "
                    "build the full CSV from a path list, no EDFs needed")
    ap.add_argument("--out", default="batch_metadata.csv")
    args = ap.parse_args()

    if args.file_list:
        with open(args.file_list) as f:
            edfs = [ln.strip() for ln in f
                    if ln.strip() and ln.strip().lower().endswith(".edf")]
    elif args.edf_dir:
        edfs = sorted(glob.glob(os.path.join(args.edf_dir, "**", "*.edf"),
                                recursive=True))
    else:
        ap.error("give --edf-dir or --file-list")
    if not edfs:
        print("No EDF paths found.")
        return 1

    # The CSV is keyed by basename (so is the app); warn on any collisions.
    name_counts = collections.Counter(os.path.basename(e) for e in edfs)
    dups = [n for n, c in name_counts.items() if c > 1]
    if dups:
        print(f"  [!] {len(dups)} duplicate basenames across batches — a "
              f"basename-keyed CSV can't disambiguate these; e.g. {dups[:3]}")

    rows = []
    skipped = []
    per_batch = collections.Counter()
    for edf in edfs:
        batch = _batch_of(edf)
        if batch is None:
            skipped.append(edf)
            continue
        m = BATCH_DEFAULTS[batch]
        row = {"filename": os.path.basename(edf), "cohort": m["cohort"],
               "group_id": ""}
        for i in range(N_CH):
            row[f"animal_ch{i}"] = m["animals"][i]
            row[f"cohort_ch{i}"] = ""          # uniform -> file-level cohort
            row[f"group_ch{i}"] = m["groups"][i]
        rows.append(row)
        per_batch[batch] += 1

    header = ["filename", "cohort", "group_id"]
    for i in range(N_CH):
        header += [f"animal_ch{i}", f"cohort_ch{i}", f"group_ch{i}"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    for b in ("1", "2", "3"):
        print(f"  batch {b}: {per_batch[b]} files")
    if skipped:
        print(f"  [!] {len(skipped)} EDFs not under a 'batch N' folder (skipped); "
              f"e.g. {os.path.relpath(skipped[0], args.edf_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
