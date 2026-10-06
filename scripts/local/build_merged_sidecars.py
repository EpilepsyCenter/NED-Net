#!/usr/bin/env python3
"""Build merged ``*_ned_annotations.json`` sidecars from TWO sources:

  * Mir's adjudicated rows (consolidated CSV)  -> ``confirmed`` / ``rejected``
  * the U-Net detection DB                     -> ``pending`` (the review queue)

On 578 recordings both apply and share one filename, so writing either alone would
destroy the other. This merges them, de-duplicating by temporal overlap and recording
in ``features["detectors"]`` which source(s) proposed each event.

Output goes to a LOCAL mirror tree (never straight onto the cluster), so it can be
inspected before upload:

    <out-dir>/<path under RAM_GDNF_2025>/<stem>_ned_annotations.json

then placed with:

    rsync -av <out-dir>/ cosmos:/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025/

Alignment facts (see docs/methods_paper/VERIFICATION_LOG_20261001.md):
  * Mir channel ``k`` -> NED-Net channel ``k-1``  (confirmed at a 6x margin)
  * Batch 4's ``session_name`` carries a ``.edf`` suffix the others lack -- strip it,
    or 192 of the 430 confirmed seizures silently vanish
  * ``Normal`` rows are dropped, matching scripts/import_mir_annotations.py; background
    negatives are auto-sampled by the trainer

Usage:
    python scripts/local/build_merged_sidecars.py --batch 3 --out-dir ~/staged_sidecars
"""
from __future__ import annotations
import argparse, os, re, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eeg_seizure_analyzer.io.annotation_store import (        # noqa: E402
    AnnotatedEvent, _ANNOTATION_FORMAT_VERSION, _sanitize,
)

LABEL_MAP = {"Seizure": "confirmed", "False": "rejected"}      # 'Normal' -> dropped


def _num(row, field: str) -> float | None:
    """Numeric feature off a namedtuple row, or None when absent/NaN."""
    v = getattr(row, field, None)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f      # drop NaN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="~/ground_truth/mir_ramgdnf_annotations.csv")
    ap.add_argument("--db", default="~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", default=None, help="restrict to one batch, e.g. 3")
    ap.add_argument("--animals", nargs="*", default=None, metavar="ID",
                    help="restrict the U-Net review queue to these animal IDs "
                         "(Mir's labels for the same files are still written)")
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="seconds of slack when deciding a U-Net event duplicates a Mir event")
    ap.add_argument("--unet-min-confidence", type=float, default=0.0)
    ap.add_argument("--no-unet", action="store_true",
                    help="write only Mir's labels (training sidecars, no review queue)")
    a = ap.parse_args()

    gt = pd.read_csv(os.path.expanduser(a.ground_truth))
    gt["stem"] = gt.session_name.str.replace(r"\.edf$", "", regex=True)
    gt["ch0"] = gt.channel - 1
    gt = gt[gt.label.isin(LABEL_MAP)]

    con = sqlite3.connect(os.path.expanduser(a.db))
    con.row_factory = sqlite3.Row
    chunks = {r["path"]: r["id"] for r in con.execute("SELECT id, path FROM chunks")}
    ev = pd.read_sql(
        "SELECT c.path, e.start_sec, e.end_sec, e.channel, e.type, "
        "       e.cnn_confidence, e.convulsive_confidence, e.animal_id "
        "FROM events e JOIN chunks c ON c.id = e.chunk_id "
        "WHERE e.cnn_confidence >= ?", con, params=(a.unet_min_confidence,))
    if a.no_unet:
        ev = ev.iloc[0:0]
    if a.animals:
        ev = ev[ev.animal_id.astype(str).isin({str(x) for x in a.animals})]
    ev["stem"] = ev.path.str.rsplit("/", n=1).str[-1].str.replace(".edf", "", regex=False)

    paths = sorted(chunks)
    if a.batch:
        paths = [p for p in paths if f"Batch_{a.batch}_Recordings" in p]
    gt_by, ev_by = dict(tuple(gt.groupby("stem"))), dict(tuple(ev.groupby("stem")))

    out_root = Path(os.path.expanduser(a.out_dir))
    n_files = n_conf = n_rej = n_pend = n_dup = 0

    for path in paths:
        stem = Path(path).stem
        g, e = gt_by.get(stem), ev_by.get(stem)
        if g is None and e is None:
            continue

        anns: list[AnnotatedEvent] = []
        claimed: list[tuple[int, float, float, int]] = []   # (channel, on, off, ann index)

        if g is not None:
            for r in g.itertuples():
                conv = r.candidate_type == "convulsive"
                anns.append(AnnotatedEvent(
                    file_path=path, animal_id="", annotator="mir",
                    onset_sec=float(r.start_s), offset_sec=float(r.end_s),
                    channel=int(r.ch0), label=LABEL_MAP[r.label],
                    source="detector", event_type="seizure",
                    # detector_confidence MUST be a 0-1 probability: the UI's
                    # confidence inputs are dcc.Input(min=0, max=1) and an
                    # out-of-domain value breaks the Training tab render.
                    # Mir's candidate_peak_score is an autocorrelation peak score
                    # (hundreds to hundreds of thousands), not a probability, and
                    # is not comparable to the U-Net's cnn_confidence -- it is kept
                    # in features instead. These events are human-adjudicated, so
                    # a detector confidence is not meaningful for them.
                    detector_confidence=0.0,
                    features={"detection_method": "mir_candidate",
                              "detectors": ["mir"],
                              "convulsive": bool(conv),
                              "candidate_type": r.candidate_type,
                              "provenance": "mir_video",
                              "candidate_peak_score": _num(r, "candidate_peak_score"),
                              "candidate_mean_score": _num(r, "candidate_mean_score"),
                              "candidate_spike_freq": _num(r, "candidate_spike_freq"),
                              "candidate_spike_count": _num(r, "candidate_spike_count"),
                              "candidate_mean_low": _num(r, "candidate_mean_low"),
                              "candidate_mean_gamma": _num(r, "candidate_mean_gamma"),
                              "candidate_mean_broad": _num(r, "candidate_mean_broad")},
                ))
                claimed.append((int(r.ch0), float(r.start_s), float(r.end_s), len(anns) - 1))
                n_conf += LABEL_MAP[r.label] == "confirmed"
                n_rej += LABEL_MAP[r.label] == "rejected"

        if e is not None:
            for i, r in enumerate(e.itertuples()):
                hit = next((idx for ch, on, off, idx in claimed
                            if ch == int(r.channel)
                            and on < r.end_sec + a.tolerance
                            and off > r.start_sec - a.tolerance), None)
                if hit is not None:
                    # already adjudicated by Mir -- keep his label, note the U-Net agreed
                    d = anns[hit].features.setdefault("detectors", [])
                    if "seizure_unet" not in d:
                        d.append("seizure_unet")
                    n_dup += 1
                    continue
                conv = str(r.type or "").strip().lower() == "convulsive"
                anns.append(AnnotatedEvent(
                    file_path=path, animal_id=str(r.animal_id or ""), annotator="",
                    onset_sec=float(r.start_sec), offset_sec=float(r.end_sec),
                    channel=int(r.channel), label="pending",
                    source="detector", event_type="seizure",
                    detector_confidence=float(r.cnn_confidence or 0.0),
                    features={"detection_method": "ml_unet",
                              "detectors": ["seizure_unet"],
                              "convulsive": conv,
                              "convulsive_probability": float(r.convulsive_confidence or 0.0),
                              "mean_probability": float(r.cnn_confidence or 0.0)},
                ))
                n_pend += 1

        anns.sort(key=lambda x: (x.channel, x.onset_sec))
        # event_id is an int in this schema -- annotation_store.from_dict() does
        # int(d["event_id"]) and a string id makes the sidecar unloadable in the UI.
        for n, x in enumerate(anns, start=1):
            x.event_id = n
        rel = re.sub(r"^.*?/RAM_GDNF_2025/", "", path)
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _ANNOTATION_FORMAT_VERSION,
            "edf_path": path,
            "annotator": "",
            "animal_id": "",
            "saved_at": pd.Timestamp.utcnow().isoformat(),
            "n_annotations": len(anns),
            "annotations": [_sanitize(x.to_dict()) for x in anns],
        }
        import json
        dest.with_name(Path(path).stem + "_ned_annotations.json").write_text(
            json.dumps(payload, indent=2))
        n_files += 1

    print(f"wrote {n_files} sidecars under {out_root}")
    print(f"  confirmed {n_conf} | rejected {n_rej} | pending (U-Net) {n_pend} "
          f"| U-Net events merged into a Mir event {n_dup}")
    print("\nupload with:\n  rsync -av --include='*/' --include='*_ned_annotations.json' "
          f"--exclude='*' {out_root}/ \\\n"
          "    cosmos:/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
