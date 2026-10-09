#!/usr/bin/env python3
"""Draw the pre-registered precision spot-check sample and build a review tree.

Measures what nothing else can: **precision of a RETRAINED model**. Every other
precision figure in this project is scored against the external convulsive-only
candidate reference, which structurally cannot credit a non-convulsive detection
-- and most retrained output is non-convulsive. So "fires more" is established
and "finds more" is not.

Protocol, fixed in PAPER_OUTLINE.md R6 BEFORE any sample was drawn:
  * random, stratified by **animal** (not batch -- within-batch recall spread
    exceeds between-batch) and by **confidence tercile**, terciles computed over
    the retained set at the chosen operating point;
  * seed recorded; ~150-200 events; **every drawn event adjudicated**, no
    skipping hard cases;
  * one pass yields overall precision AND convulsive precision on the subset
    stage 2 called convulsive -- the only honest test of the stage-2 fix on real
    output rather than validation windows.

SAFETY: this never touches the live sidecars. It builds a separate tree of
EDF **symlinks** plus sidecars containing ONLY the sampled events as `pending`,
so the review queue in the UI *is* the sample -- reviewing extras would bias the
estimate -- and the live tree keeps Mir's labels and all existing review work.

Run on LUNARC (needs the EDFs and the project DB):
    python scripts/lunarc/draw_precision_sample.py \
        --db ~/.eeg_seizure_analyzer/projects/armA3_tune_t0.9_b0.5.db \
        --out ~/review_sample_armA3 --n 180 --seed 17
Then open --out as the data folder in NED-Net, review every event, and run
    scripts/local/score_precision_sample.py --tree <out>
"""
from __future__ import annotations
import argparse, json, os, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd

# Retained set (EXCLUSION_CRITERIA.md): no Batch 4, no B2 UI ch3/ch4 (code 2/3),
# no animal 449382 (dead electrode, B1 UI ch2 = code ch1).
EXCLUDE_CH = {"1": {1}, "2": {2, 3}}
KEEP_BATCHES = {"1", "2", "3"}
_FMT = "1.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, help="review tree to create (symlinks + sidecars)")
    ap.add_argument("--cohort-key",
                    default=str(Path(__file__).with_name("RAM_GDNF_2025_cohort_key.csv")))
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, required=True,
                    help="record this in RUN_LOG.md BEFORE opening any event")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(os.path.expanduser(a.db))
    ev = pd.read_sql(
        "SELECT e.id, c.path, e.start_sec, e.end_sec, e.channel, e.type, "
        "       e.cnn_confidence, e.convulsive_confidence, e.animal_id "
        "FROM events e JOIN chunks c ON c.id = e.chunk_id", con)
    if ev.empty:
        print("no events in DB", file=sys.stderr); return 1
    ev["batch"] = ev.path.str.extract(r"Batch_(\d)_Recordings")[0]
    ev = ev[ev.batch.isin(KEEP_BATCHES)]
    keep = pd.Series(True, index=ev.index)
    for b, chs in EXCLUDE_CH.items():
        keep &= ~((ev.batch == b) & (ev.channel.isin(chs)))
    ev = ev[keep & (ev.cnn_confidence >= a.min_confidence)].copy()

    # Animal IDs from the cohort key (its `channel` is 1-based UI; code = -1).
    key = pd.read_csv(a.cohort_key)
    key = key[key.channel != "(none)"].copy()
    key["channel"] = key.channel.astype(int) - 1
    key["batch"] = key.batch.str[-1]
    amap = key.set_index(["batch", "channel"]).animal_id.astype(str).to_dict()
    ev["animal"] = [amap.get((b, c), f"ch{c}") for b, c in zip(ev.batch, ev.channel)]
    ev = ev[ev.animal != "449382"]

    # Confidence terciles over the retained set, so strata are defined by the
    # population being sampled rather than per animal.
    qs = ev.cnn_confidence.quantile([1/3, 2/3]).tolist()
    ev["tercile"] = np.digitize(ev.cnn_confidence, qs)  # 0 low, 1 mid, 2 high

    print(f"retained population: {len(ev)} detections, {ev.animal.nunique()} animals")
    print(f"confidence terciles at {qs[0]:.3f} / {qs[1]:.3f}")

    # Proportional allocation across (animal, tercile), at least 1 where non-empty.
    rng = np.random.default_rng(a.seed)
    groups = list(ev.groupby(["animal", "tercile"]))
    share = {k: len(g) / len(ev) for k, g in groups}
    picks = []
    for (k, g) in groups:
        want = max(1, int(round(a.n * share[k])))
        want = min(want, len(g))
        idx = rng.choice(g.index.to_numpy(), size=want, replace=False)
        picks.append(g.loc[idx])
    samp = pd.concat(picks).sort_values(["path", "channel", "start_sec"])
    print(f"\nsampled {len(samp)} events over {samp.path.nunique()} recordings")
    print(samp.groupby(["animal", "tercile"]).size().unstack(fill_value=0).to_string())
    print(f"\nby stage-2 label: {samp.type.value_counts().to_dict()}")

    out = Path(os.path.expanduser(a.out))
    man = out / "SAMPLE_MANIFEST.csv"
    if a.dry_run:
        print(f"\n[dry run] would write {out} and {man}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for path, g in samp.groupby("path"):
        edf = Path(path)
        link = out / edf.name
        if not link.exists():
            try:
                link.symlink_to(edf)
            except OSError as e:
                print(f"  symlink failed for {edf.name}: {e}", file=sys.stderr)
                continue
        # carry the channel->animal map so the UI shows animal IDs
        ch_src = edf.parent / (edf.stem + "_ned_channels.json")
        ch_dst = out / (edf.stem + "_ned_channels.json")
        if ch_src.exists() and not ch_dst.exists():
            ch_dst.write_text(ch_src.read_text())

        anns = []
        for n, r in enumerate(g.sort_values(["channel", "start_sec"]).itertuples(), 1):
            anns.append({
                "event_id": n,                       # MUST be int (from_dict casts)
                "file_path": str(edf),
                "animal_id": str(r.animal),
                "annotator": "",
                "onset_sec": float(r.start_sec),
                "offset_sec": float(r.end_sec),
                "channel": int(r.channel),
                "label": "pending",
                "source": "detector",
                "event_type": "seizure",
                "detector_confidence": float(min(max(r.cnn_confidence or 0.0, 0.0), 1.0)),
                "features": {
                    "detection_method": "ml_unet",
                    "detectors": ["seizure_unet"],
                    "convulsive": str(r.type or "").lower() == "convulsive",
                    "convulsive_probability": float(r.convulsive_confidence or 0.0),
                    "mean_probability": float(r.cnn_confidence or 0.0),
                    "sample_db_event_id": int(r.id),   # ties back to the manifest
                    "precision_sample_seed": int(a.seed),
                },
            })
        (out / (edf.stem + "_ned_annotations.json")).write_text(json.dumps({
            "version": _FMT, "edf_path": str(edf), "annotator": "", "animal_id": "",
            "saved_at": pd.Timestamp.utcnow().isoformat(),
            "n_annotations": len(anns), "annotations": anns,
        }, indent=2))
        n_files += 1

    samp.assign(seed=a.seed).to_csv(man, index=False)
    print(f"\nwrote {n_files} sidecars + symlinks to {out}")
    print(f"manifest: {man}")
    print("\nOpen that folder in NED-Net, review EVERY event (no skipping), then:")
    print(f"  python scripts/local/score_precision_sample.py --tree {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
