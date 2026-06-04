#!/usr/bin/env python3
"""Verify NED-Net can ingest the converted Mir annotations end-to-end.

Runs the real planner (build_window_specs) over every EDF that has a
`*_ned_annotations.json` beside it, then pulls a few actual windows through
SeizureDataset to confirm tensors + masks come out and that positive windows
really contain seizure.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from eeg_seizure_analyzer.ml.dataset import (
    DatasetConfig,
    SeizureDataset,
    build_window_specs,
)

EDF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"Z:\LU26D1055-epicenter\Data\KAHA recordings\RAM_GDNF_2025"
    r"\Batch_1_Recordings\Week_1\EDF and Dat Files\W1_D1"
)

jsons = sorted(EDF_DIR.glob("*_ned_annotations.json"))
files = []
for j in jsons:
    stem = j.name[: -len("_ned_annotations.json")]
    edf = j.with_name(stem + ".edf")
    if edf.exists():
        files.append({"edf_path": str(edf)})
    else:
        print(f"MISSING EDF for {j.name}", file=sys.stderr)

print(f"Found {len(jsons)} annotation files, {len(files)} with matching EDF.")

cfg = DatasetConfig(include_activity=False, augment=False)
specs = build_window_specs({"files": files}, cfg)

pos = [s for s in specs if s.is_positive]
neg = [s for s in specs if not s.is_positive]
print(f"\nPlanned {len(specs)} windows: {len(pos)} positive, {len(neg)} negative.")

per_ch = Counter((Path(s.edf_path).stem, s.eeg_channel) for s in pos)
print("\nPositive windows by (session, channel index):")
for (sess, ch), n in sorted(per_ch.items()):
    print(f"  ch{ch}  {sess}: {n}")

# Pull a few real windows through the Dataset.
ds = SeizureDataset(specs, cfg, augment=False)
print(f"\nSeizureDataset length: {len(ds)}")

# Check one positive and one negative end-to-end.
to_check = []
if pos:
    to_check.append(("positive", specs.index(pos[0])))
if neg:
    to_check.append(("negative", specs.index(neg[0])))

for kind, idx in to_check:
    eeg, mask, meta = ds[idx]
    sz_frac = float(mask[0].float().mean()) if mask.ndim == 2 else float(mask.float().mean())
    print(
        f"\n[{kind}] idx={idx} eeg={tuple(eeg.shape)} mask={tuple(mask.shape)} "
        f"seizure_frac={sz_frac:.3f} ch={meta.get('eeg_channel')} "
        f"animal={meta.get('animal_id')}"
    )

print("\nOK — loader ingested the converted annotations.")
