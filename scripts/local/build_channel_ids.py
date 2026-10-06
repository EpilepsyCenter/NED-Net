#!/usr/bin/env python3
"""Write ``*_ned_channels.json`` (channel -> animal ID) beside every cohort EDF.

Without these the trainer falls back to ``f"ch{index}"`` as the animal ID
(``ml/dataset.py:294``), which pools every recording's ch0 into one "animal" --
only 8 groups instead of 32. Two consequences, both silent:

  * ``--exclude-animals 459657 ...`` matches NOTHING, so a leave-one-batch-out
    fold does not actually hold anything out and trains on its own test set;
  * the train/val split by animal is meaningless.

The annotation sidecar's own ``animal_id`` field is NOT used for grouping, so
this is a separate file per EDF.

Channel indices are 0-based and match the ``animal_ch0..animal_ch7`` columns of
RAM_GDNF_2025_batch_metadata.csv, the DB's ``events.channel``, and Mir's
``channel - 1``. Verify after upload by re-running ``train_unet.py --analyze``:
it must report **32** animal groups, not 8.

Usage:
    python scripts/local/build_channel_ids.py --out-dir ~/staged_channel_ids
    rsync -av --include='*/' --include='*_ned_channels.json' --exclude='*' \
        ~/staged_channel_ids/ \
        cosmos:/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025/
"""
from __future__ import annotations
import argparse, json, os, re, sqlite3
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="scripts/lunarc/RAM_GDNF_2025_batch_metadata.csv")
    ap.add_argument("--db", default="~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db",
                    help="source of the real LUNARC EDF paths")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.metadata)
    m["stem"] = m.filename.str.replace(".edf", "", regex=False)
    acols = [f"animal_ch{i}" for i in range(8)]

    con = sqlite3.connect(os.path.expanduser(a.db))
    paths = [r[0] for r in con.execute("SELECT path FROM chunks")]
    by_stem = {Path(p).stem: p for p in paths}

    out_root = Path(os.path.expanduser(a.out_dir))
    n = 0
    missing: list[str] = []
    animals: set[str] = set()
    for r in m.itertuples():
        path = by_stem.get(r.stem)
        if path is None:
            missing.append(r.stem)
            continue
        mapping = {}
        for i, c in enumerate(acols):
            v = getattr(r, c)
            if pd.notna(v):
                mapping[str(i)] = str(int(v))
                animals.add(str(int(v)))
        if not mapping:
            continue
        rel = re.sub(r"^.*?/RAM_GDNF_2025/", "", path)
        dest = (out_root / rel).with_name(Path(path).stem + "_ned_channels.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(
            {"edf_path": path, "channel_ids": mapping}, indent=2))
        n += 1

    print(f"wrote {n} channel-id files under {out_root}")
    print(f"distinct animals: {len(animals)}  -> {sorted(animals)[:4]} ... {sorted(animals)[-2:]}")
    if missing:
        print(f"metadata rows with no matching processed EDF: {len(missing)} "
              f"(e.g. {missing[:3]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
