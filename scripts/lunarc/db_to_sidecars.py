#!/usr/bin/env python3
"""Convert a NED-Net project DB into per-EDF ``*_ned_annotations.json`` sidecars
with ``label="pending"``, turning a batch detection run into a UI review queue.

Run ON LUNARC in the ``bendr`` env -- sidecars are written beside each EDF.

What the UI can filter on (eeg_seizure_analyzer/dash_app/pages/training.py):
  * min/max confidence -> ``detector_confidence``      <- populated here
  * min/max duration   -> onset/offset                 <- populated here
  * min/max local BL   -> quality_metrics['local_baseline_ratio']   NOT AVAILABLE
  * min/max Amp (xBL)  -> features['max_amplitude_x_baseline']      NOT AVAILABLE
The last two are classical-detector features and are not stored in the events
table, so they cannot be reconstructed from the DB. Leave those two filters at 0
during review: ``min_amp > 0`` DROPS every annotation lacking the field, i.e. all
U-Net events disappear from the view. Triage by confidence and duration instead.

Pending annotations are ignored by the trainers (ml/dataset.py uses only
confirmed/rejected), so writing these is safe with respect to training.

Usage (subset to one batch, highest-confidence first):
    python scripts/lunarc/db_to_sidecars.py \
        --db ~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db \
        --path-include 'Batch_4_Recordings' --min-confidence 0.5 --dry-run
"""
from __future__ import annotations
import argparse, os, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--path-include", default=None,
                    help="regex; only EDF paths matching it get a sidecar")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--type", choices=["convulsive", "non_convulsive"], default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite existing sidecars (default: skip)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from eeg_seizure_analyzer.io.annotation_store import AnnotatedEvent, save_annotations

    con = sqlite3.connect(os.path.expanduser(a.db))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT c.path, e.start_sec, e.end_sec, e.channel, e.type, e.subtype, "
        "       e.cnn_confidence, e.convulsive_confidence, e.movement_flag, e.animal_id "
        "FROM events e JOIN chunks c ON c.id = e.chunk_id "
        "WHERE e.cnn_confidence >= ? "
        + ("AND e.type = ? " if a.type else "")
        + "ORDER BY c.path, e.channel, e.start_sec",
        (a.min_confidence, a.type) if a.type else (a.min_confidence,)).fetchall()

    pat = re.compile(a.path_include) if a.path_include else None
    by_file: dict[str, list] = defaultdict(list)
    for r in rows:
        if pat and not pat.search(r["path"]):
            continue
        by_file[r["path"]].append(r)

    n_written = n_skipped = n_events = n_missing = 0
    for path, evs in sorted(by_file.items()):
        edf = Path(path)
        if not edf.exists():
            n_missing += 1
            continue
        out = edf.with_name(edf.stem + "_ned_annotations.json")
        if out.exists() and not a.overwrite:
            n_skipped += 1
            continue
        anns = []
        for i, e in enumerate(evs):
            conv = str(e["type"] or "").strip().lower() == "convulsive"
            anns.append(AnnotatedEvent(
                file_path=str(edf),
                animal_id=str(e["animal_id"] or ""),
                onset_sec=float(e["start_sec"]),
                offset_sec=float(e["end_sec"]),
                channel=int(e["channel"]),
                label="pending",
                source="detector",
                event_type="seizure",
                # drives the UI's confidence filter and sorting -- without this,
                # min_conf > 0 would silently hide everything.
                detector_confidence=float(e["cnn_confidence"] or 0.0),
                event_id=i + 1,   # int: from_dict() does int(d["event_id"])
                features={
                    "detection_method": "ml_unet",
                    "convulsive": conv,
                    "convulsive_probability": float(e["convulsive_confidence"] or 0.0),
                    "mean_probability": float(e["cnn_confidence"] or 0.0),
                    "seizure_subtype": e["subtype"] or ("convulsive" if conv else "seizure"),
                    "movement_flag": bool(e["movement_flag"]),
                },
            ))
        n_events += len(anns)
        if a.dry_run:
            n_written += 1
            continue
        save_annotations(str(edf), anns)
        n_written += 1

    verb = "would write" if a.dry_run else "wrote"
    print(f"{verb} {n_written} sidecars | skipped existing: {n_skipped} | "
          f"events: {n_events} | EDF not found: {n_missing}")
    if n_skipped:
        print("  (use --overwrite to replace existing sidecars)")
    if a.dry_run:
        print("  DRY RUN -- nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
