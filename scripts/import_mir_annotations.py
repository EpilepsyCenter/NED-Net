#!/usr/bin/env python3
"""Convert a colleague's Excel candidate-annotation export into NED-Net's
``<edf_stem>_ned_annotations.json`` files for direct training.

Source format (one row per reviewed candidate event), columns::

    session_idx, session_name, channel, start_s, end_s, duration_s,
    label, candidate_idx, candidate_type,
    candidate_spike_freq, candidate_peak_score, created_at

Label / type mapping to NED-Net (see notes/annotation_import.md and
eeg_seizure_analyzer/ml/dataset.py):

    label 'Seizure' -> {"label": "confirmed", "event_type": "seizure"}   (positive)
    label 'False'   -> {"label": "rejected",  "event_type": "seizure"}   (hard negative)
    label 'Normal'  -> dropped (rely on auto-sampled background negatives)

    candidate_type 'convulsive' -> features.convulsive = True
    candidate_type 'behavior'   -> features.convulsive = False

Two alignment gotchas this script handles:

1. CHANNEL INDEX. NED-Net matches each annotation's integer ``channel`` against
   the *file-level* EEG channel index it auto-detects for the exact EDF it reads
   (``auto_pair_channels``). The colleague numbers EEG channels 1..N. By default
   we assume his channel ``k`` is the k-th EEG channel, i.e. NED-Net index
   ``k - 1`` (8 contiguous Biopot channels). If you pass ``--edf-dir`` pointing
   at the real EDFs (e.g. the LU Research share once mounted), the script scans
   each EDF with NED-Net's own pairing logic and maps ``k -> eeg_idx[k-1]``,
   which is correct even when EEG/activity channels are interleaved.

2. FILE LINK. The JSON must be named ``<edf_stem>_ned_annotations.json`` and sit
   next to the EDF. We assume ``edf_stem == session_name`` unless an EDF with a
   matching stem is found under ``--edf-dir``.

Usage::

    # Offline (no EDFs): writes JSONs to ./ned_annotations_out/, channel = k-1
    python scripts/import_mir_annotations.py annotationsMir.xlsx -o ned_annotations_out

    # With EDFs mounted: resolve true channel indices and drop JSON next to EDF
    python scripts/import_mir_annotations.py annotationsMir.xlsx --edf-dir /mnt/lu-share/recordings --inplace
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

# Source label -> NED-Net (label, keep?) ; None means drop the row.
_LABEL_MAP = {
    "Seizure": ("confirmed", True),
    "False": ("rejected", True),
    "Normal": (None, False),  # dropped per project decision
}


def _load_rows(xlsx_path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    out = []
    for raw in rows[1:]:
        if raw is None or all(c is None for c in raw):
            continue
        out.append(dict(zip(header, raw)))
    return out


def _resolve_channel_maps(
    sessions: set[str], edf_dir: Path | None
) -> dict[str, tuple[Path | None, dict[int, int]]]:
    """For each session return (edf_path_or_None, {excel_channel -> ned_index}).

    Without an EDF we fall back to ``k -> k - 1``. With one we map the k-th EEG
    channel to its true file-level index via NED-Net's auto_pair_channels.
    """
    result: dict[str, tuple[Path | None, dict[int, int]]] = {}
    eeg_idx_cache: dict[Path, list[int]] = {}

    scan = None
    pair = None
    if edf_dir is not None:
        try:
            from eeg_seizure_analyzer.io.edf_reader import (
                auto_pair_channels,
                scan_edf_channels,
            )

            scan, pair = scan_edf_channels, auto_pair_channels
        except Exception as exc:  # pragma: no cover - import/runtime guard
            print(
                f"WARNING: could not import NED-Net EDF reader ({exc}); "
                "falling back to channel = k-1 for all sessions.",
                file=sys.stderr,
            )
            edf_dir = None

    for session in sorted(sessions):
        edf_path = None
        eeg_idx: list[int] | None = None
        if edf_dir is not None:
            cand = edf_dir / f"{session}.edf"
            if not cand.exists():
                matches = list(edf_dir.glob(f"{session}.*"))
                cand = matches[0] if matches else None
            if cand and cand.exists():
                edf_path = cand
                if cand not in eeg_idx_cache:
                    try:
                        eeg_idx_cache[cand], _, _ = pair(scan(str(cand)))  # type: ignore[misc]
                    except Exception as exc:
                        print(
                            f"WARNING: failed to scan {cand.name} ({exc}); "
                            "using channel = k-1 for this session.",
                            file=sys.stderr,
                        )
                        eeg_idx_cache[cand] = []
                eeg_idx = eeg_idx_cache[cand] or None
            else:
                print(
                    f"WARNING: no EDF found for session '{session}' under "
                    f"{edf_dir}; using channel = k-1.",
                    file=sys.stderr,
                )

        def _map(k: int, _eeg_idx=eeg_idx) -> int:
            if _eeg_idx is not None and 1 <= k <= len(_eeg_idx):
                return _eeg_idx[k - 1]
            return k - 1

        # Build the map lazily over channels actually seen later; here just
        # stash the mapping function via a dict comprehension over 1..8 (and a
        # bit of headroom) so downstream code stays simple.
        chan_map = {k: _map(k) for k in range(1, 33)}
        result[session] = (edf_path, chan_map)

    return result


def convert(
    xlsx_path: Path,
    out_dir: Path | None,
    edf_dir: Path | None,
    inplace: bool,
) -> None:
    rows = _load_rows(xlsx_path)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_session[str(r["session_name"])].append(r)

    chan_maps = _resolve_channel_maps(set(by_session), edf_dir)

    stats = defaultdict(int)
    for session, srows in sorted(by_session.items()):
        edf_path, chan_map = chan_maps[session]
        annotations = []
        for r in srows:
            src_label = str(r["label"]).strip()
            mapped = _LABEL_MAP.get(src_label)
            if mapped is None:
                stats[f"unknown_label:{src_label}"] += 1
                continue
            ned_label, keep = mapped
            if not keep:
                stats["dropped_normal"] += 1
                continue

            excel_ch = int(r["channel"])
            ned_ch = chan_map.get(excel_ch, excel_ch - 1)
            convulsive = str(r.get("candidate_type", "")).strip().lower() == "convulsive"

            ann = {
                "channel": ned_ch,
                "event_type": "seizure",
                "label": ned_label,
                "onset_sec": float(r["start_s"]),
                "offset_sec": float(r["end_s"]),
                "features": {"convulsive": convulsive},
                # provenance (ignored by the loader, handy for auditing)
                "source": {
                    "session_name": session,
                    "excel_channel": excel_ch,
                    "candidate_idx": r.get("candidate_idx"),
                    "candidate_type": r.get("candidate_type"),
                    "src_label": src_label,
                },
            }
            annotations.append(ann)
            stats[f"out_{ned_label}"] += 1

        if not annotations:
            continue

        # Decide where the JSON goes.
        if inplace and edf_path is not None:
            out_path = edf_path.with_name(edf_path.stem + "_ned_annotations.json")
        else:
            target_dir = out_dir or Path("ned_annotations_out")
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"{session}_ned_annotations.json"

        out_path.write_text(
            json.dumps({"annotations": annotations}, indent=2), encoding="utf-8"
        )
        n_pos = sum(1 for a in annotations if a["label"] == "confirmed")
        n_neg = len(annotations) - n_pos
        print(f"{out_path}  ({n_pos} confirmed, {n_neg} rejected)")

    print("\nSummary:", dict(stats), file=sys.stderr)


def batch(root: Path) -> None:
    """Find every `annotations.xlsx` under *root* and convert each in place.

    All EDFs share the same channel convention (8 ``ChN Biopot`` at idx 0-7 +
    8 ``ChN Act``), so channel ``k`` maps to index ``k-1`` without scanning —
    and ``session_name`` equals the EDF stem, so writing
    ``<session_name>_ned_annotations.json`` into the file's own folder lands it
    right next to the EDF.
    """
    ann_files = sorted(root.rglob("annotations.xlsx"))
    print(f"Found {len(ann_files)} annotations.xlsx under {root}\n")
    for f in ann_files:
        print(f"=== {f.parent} ===")
        convert(f, out_dir=f.parent, edf_dir=None, inplace=False)
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xlsx", type=Path, nargs="?", help="colleague's annotation .xlsx export")
    p.add_argument(
        "--batch-root", type=Path, default=None,
        help="instead of a single xlsx, recursively find every annotations.xlsx "
             "under this root and convert each in place next to its EDFs",
    )
    p.add_argument(
        "-o", "--out-dir", type=Path, default=None,
        help="directory for the generated JSONs (default: ./ned_annotations_out)",
    )
    p.add_argument(
        "--edf-dir", type=Path, default=None,
        help="directory of real EDFs; scanned to resolve true channel indices "
             "and confirm stems (e.g. the LU Research share once mounted)",
    )
    p.add_argument(
        "--inplace", action="store_true",
        help="write each JSON next to its matched EDF (requires --edf-dir)",
    )
    args = p.parse_args(argv)

    if args.batch_root is not None:
        if not args.batch_root.is_dir():
            p.error(f"not a directory: {args.batch_root}")
        batch(args.batch_root)
        return 0

    if args.xlsx is None:
        p.error("provide an xlsx file or --batch-root")
    if args.inplace and args.edf_dir is None:
        p.error("--inplace requires --edf-dir")
    if not args.xlsx.exists():
        p.error(f"file not found: {args.xlsx}")

    convert(args.xlsx, args.out_dir, args.edf_dir, args.inplace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
