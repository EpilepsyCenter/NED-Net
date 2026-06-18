#!/usr/bin/env python
"""Headless, parallel batch CLASSICAL interictal-spike detection -> project DB.

The spike twin of detect_batch.py. Runs the rule-based SpikeDetector (NO trained
model) over a folder of EDFs, fully in parallel across CPU cores, and writes the
results into a project SQLite DB as ``source="spike_classical"`` /
``category="spike"`` events — exactly the same schema the GUI "Add to DB" button
and the Results Spikes view read. Optionally reads a batch-metadata CSV
(filename -> cohort / group / per-channel animal IDs), identical to the seizure
sweep.

Why per-worker DBs + merge: db.py opens WAL but sets no busy_timeout, so dozens
of processes writing one DB file would hit "database is locked". Each worker
writes its OWN part-DB (keyed by pid); when all files are done the part-DBs are
merged into the final project DB. Schema columns are read at runtime so the
merge tracks any db.py migrations.

NOTE: the merge is destructive *per file* — it replaces any existing record for
a given EDF path. Run a spike sweep into its OWN project DB (one detector family
per DB), not on top of a seizure DB, unless you intend to overwrite those files.

Resumable: files already in the target DB are skipped unless --overwrite.

Example
-------
    python scripts/lunarc/detect_spikes_batch.py \
        --edf-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data \
        --db-path ~/.eeg_seizure_analyzer/projects/sv2a_spikes.db \
        --zscore 4.0 --metadata-csv batch_metadata.csv --workers 48
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys
import time
from multiprocessing import Pool

# Set per-worker in the pool initializer (each worker process is isolated).
_CFG: dict | None = None


def _discover(edf_dir: str, recursive: bool) -> list[str]:
    pat = "**/*.edf" if recursive else "*.edf"
    return sorted(
        os.path.abspath(p)
        for p in glob.glob(os.path.join(edf_dir, pat), recursive=recursive)
    )


def _init_worker(cfg: dict, tmpdir: str) -> None:
    """Each worker gets its own part-DB. Classical detection is numpy/scipy —
    no torch — so there's nothing to single-thread here."""
    global _CFG
    from eeg_seizure_analyzer import db
    db.init_db(os.path.join(tmpdir, f"part_{os.getpid()}.db"))
    _CFG = cfg


def _process_one(edf_path: str):
    from eeg_seizure_analyzer import analysis
    cfg = _CFG
    meta = cfg["metadata"].get(os.path.basename(edf_path))
    try:
        r = analysis.process_spike_chunk_classical(
            edf_path=edf_path,
            params=cfg["params"],
            mode="batch",
            file_metadata=meta,
            min_confidence=cfg["min_confidence"],
            min_local_snr=cfg["min_local_snr"],
            min_amplitude_x_baseline=cfg["min_xbaseline"],
        )
        return edf_path, int(r.get("n_events", 0)), None
    except Exception as e:  # one bad file must not abort the sweep
        return edf_path, 0, repr(e)


# --------------------------------------------------------------------------
# Merge part-DBs into the final project DB (FK-remapping, schema-introspected)
# --------------------------------------------------------------------------

def _cols(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _merge_part(final_db: str, part_db: str) -> None:
    # Fold the part-DB's WAL into its main file so ATTACH sees every row.
    c = sqlite3.connect(part_db)
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()

    fin = sqlite3.connect(final_db)
    fin.execute("ATTACH ? AS p", (part_db,))
    chunk_cols = [c for c in _cols(fin, "chunks") if c != "id"]
    path_i = chunk_cols.index("path")
    idmap: dict[int, int] = {}
    rows = fin.execute(
        f"SELECT id,{','.join(chunk_cols)} FROM p.chunks").fetchall()
    for row in rows:
        old_id, vals = row[0], list(row[1:])
        # Idempotent: drop any existing record for this file path first.
        for (eid,) in fin.execute(
                "SELECT id FROM chunks WHERE path=?", (vals[path_i],)).fetchall():
            for t in ("events", "chunk_summary", "file_animals"):
                fin.execute(f"DELETE FROM {t} WHERE chunk_id=?", (eid,))
            fin.execute("DELETE FROM chunks WHERE id=?", (eid,))
        ph = ",".join(["?"] * len(chunk_cols))
        cur = fin.execute(
            f"INSERT INTO chunks ({','.join(chunk_cols)}) VALUES ({ph})", vals)
        idmap[old_id] = cur.lastrowid
    for tbl in ("events", "chunk_summary", "file_animals"):
        cols = [c for c in _cols(fin, tbl) if c != "id"]
        ci = cols.index("chunk_id")
        ph = ",".join(["?"] * len(cols))
        for row in fin.execute(f"SELECT {','.join(cols)} FROM p.{tbl}").fetchall():
            vals = list(row)
            new_id = idmap.get(vals[ci])
            if new_id is None:
                continue
            vals[ci] = new_id
            fin.execute(f"INSERT INTO {tbl} ({','.join(cols)}) VALUES ({ph})", vals)
    ascols = _cols(fin, "animal_status")
    ph = ",".join(["?"] * len(ascols))
    for row in fin.execute(
            f"SELECT {','.join(ascols)} FROM p.animal_status").fetchall():
        fin.execute(
            f"INSERT OR IGNORE INTO animal_status ({','.join(ascols)}) "
            f"VALUES ({ph})", row)
    fin.commit()
    fin.execute("DETACH p")
    fin.close()


def _build_params(args):
    """Construct SpikeDetectionParams from CLI args (defaults match config.py)."""
    from eeg_seizure_analyzer.config import SpikeDetectionParams
    return SpikeDetectionParams(
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        amplitude_threshold_zscore=args.zscore,
        spike_min_amplitude_uv=args.min_amplitude_uv,
        spike_prominence_x_baseline=args.prominence,
        max_duration_ms=args.max_duration_ms,
        min_duration_ms=args.min_duration_ms,
        refractory_ms=args.refractory_ms,
        baseline_method=args.baseline_method,
        baseline_percentile=args.baseline_percentile,
        isolation_window_sec=args.isolation_window_sec,
        isolation_max_neighbours=args.isolation_max_neighbours,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--db-path", required=True, help="final project DB to write")
    ap.add_argument("--metadata-csv", default=None,
                    help="batch metadata CSV/XLSX (filename -> cohort/group/animal)")
    # Classical spike detector knobs (defaults == config.SpikeDetectionParams).
    ap.add_argument("--zscore", type=float, default=4.0,
                    help="amplitude threshold as mean + z*std (the GUI's z=4-5)")
    ap.add_argument("--bandpass-low", type=float, default=10.0)
    ap.add_argument("--bandpass-high", type=float, default=70.0)
    ap.add_argument("--min-amplitude-uv", type=float, default=0.0,
                    help="absolute amplitude floor (uV); 0 = disabled")
    ap.add_argument("--prominence", type=float, default=1.5,
                    help="prominence relative to baseline")
    ap.add_argument("--max-duration-ms", type=float, default=70.0)
    ap.add_argument("--min-duration-ms", type=float, default=2.0)
    ap.add_argument("--refractory-ms", type=float, default=200.0)
    ap.add_argument("--baseline-method", default="percentile",
                    choices=("percentile", "rolling", "first_n"))
    ap.add_argument("--baseline-percentile", type=int, default=15)
    ap.add_argument("--isolation-window-sec", type=float, default=2.0,
                    help="window to count neighbours for burst rejection")
    ap.add_argument("--isolation-max-neighbours", type=int, default=6,
                    help="max spikes in window before the spike is rejected")
    # Post-detection quality filters (same as the Spikes-tab filters; 0 = off).
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="keep spikes with composite confidence >= this (GUI: 0.7)")
    ap.add_argument("--min-local-snr", type=float, default=0.0,
                    help="keep spikes with local SNR >= this (GUI: 10)")
    ap.add_argument("--min-xbaseline", type=float, default=0.0,
                    help="keep spikes with amplitude/baseline >= this (GUI: 15)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--path-include", default=None,
                    help="only process EDFs whose full path matches this regex "
                         "(e.g. 'Week[123]-' for the first 3 weeks, or a "
                         "day-subsample pattern)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-detect files already in the DB")
    ap.add_argument("--tmpdir", default=os.environ.get("SNIC_TMP", "/tmp"),
                    help="node-local scratch for per-worker part-DBs")
    args = ap.parse_args()

    from eeg_seizure_analyzer import db
    from eeg_seizure_analyzer.io.batch_metadata import load_metadata

    files = _discover(args.edf_dir, not args.no_recursive)
    if not files:
        print(f"No EDFs under {args.edf_dir}", file=sys.stderr)
        return 1
    if args.path_include:
        rx = re.compile(args.path_include)
        n_before = len(files)
        files = [f for f in files if rx.search(f)]
        print(f"Path filter '{args.path_include}': {len(files)}/{n_before} EDFs kept.",
              flush=True)
        if not files:
            print(f"No EDFs match --path-include '{args.path_include}'", file=sys.stderr)
            return 1

    db.init_db(os.path.abspath(os.path.expanduser(args.db_path)))
    done = set() if args.overwrite else db.get_processed_paths()
    todo = [f for f in files if f not in done]
    print(f"{len(files)} EDFs found; {len(files) - len(todo)} already done; "
          f"{len(todo)} to process on {args.workers} workers.", flush=True)
    print(f"Detector: z={args.zscore} bp={args.bandpass_low}-{args.bandpass_high}Hz "
          f"prom={args.prominence} iso={args.isolation_max_neighbours}/"
          f"{args.isolation_window_sec}s", flush=True)
    print(f"Filters:  confidence>={args.min_confidence} "
          f"local_snr>={args.min_local_snr} x_baseline>={args.min_xbaseline} "
          f"(0 = off)", flush=True)
    if not todo:
        print("Nothing to do.")
        return 0

    metadata = load_metadata(args.metadata_csv) if args.metadata_csv else {}
    if args.metadata_csv:
        print(f"Loaded metadata for {len(metadata)} files from {args.metadata_csv}")

    cfg = dict(params=_build_params(args), metadata=metadata,
               min_confidence=args.min_confidence,
               min_local_snr=args.min_local_snr,
               min_xbaseline=args.min_xbaseline)
    work_tmp = os.path.join(args.tmpdir, f"nedspk_{os.getpid()}")
    os.makedirs(work_tmp, exist_ok=True)

    t0 = time.time()
    n_events = n_err = n_done = 0
    with Pool(args.workers, initializer=_init_worker,
              initargs=(cfg, work_tmp)) as pool:
        for path, ne, err in pool.imap_unordered(_process_one, todo, chunksize=1):
            n_done += 1
            if err:
                n_err += 1
                print(f"  [ERR] {os.path.basename(path)}: {err}", file=sys.stderr)
            else:
                n_events += ne
            if n_done % 25 == 0 or n_done == len(todo):
                print(f"  {n_done}/{len(todo)} files | {n_events} spikes | "
                      f"{n_err} errors | {time.time() - t0:.0f}s", flush=True)

    db_final = os.path.abspath(os.path.expanduser(args.db_path))
    parts = sorted(glob.glob(os.path.join(work_tmp, "part_*.db")))
    print(f"Merging {len(parts)} part-DBs -> {db_final} ...", flush=True)
    for part in parts:
        _merge_part(db_final, part)
        os.remove(part)
    try:
        os.rmdir(work_tmp)
    except OSError:
        pass

    print(f"Done. {n_done} files, {n_events} spikes, {n_err} errors, "
          f"{time.time() - t0:.0f}s. DB: {db_final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
