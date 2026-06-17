#!/usr/bin/env python
"""Headless, parallel batch seizure detection -> project SQLite DB.

Runs the SAME pipeline as the Analysis tab (predict_seizures + classify +
convulsive cascade) over a folder of EDFs, fully in parallel across CPU cores,
and writes the results into a project database (NOT sidecar JSONs). Optionally
reads a batch-metadata CSV (filename -> cohort / group / per-channel animal IDs),
exactly like the UI's metadata browse.

Why per-worker DBs + merge: db.py opens its connection in WAL mode but sets no
busy_timeout, so dozens of processes writing one DB file would hit "database is
locked". Instead each worker writes its OWN part-DB (keyed by pid); when all
files are done the part-DBs are merged into the final project DB. Schema columns
are read at runtime (PRAGMA table_info) so the merge tracks any db.py migrations.

Resumable: files already in the target DB are skipped unless --overwrite.

Example
-------
    python scripts/lunarc/detect_batch.py \
        --edf-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data \
        --db-path ~/.eeg_seizure_analyzer/projects/lunarc_detect.db \
        --model unet_kaha_v1 --convulsive-model conv_kaha_v1 \
        --metadata-csv batch_metadata.csv --workers 48
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
    """Each worker gets its own DB file and single-threaded torch."""
    global _CFG
    import torch
    torch.set_num_threads(1)  # 48 workers must not each spawn 48 threads
    from eeg_seizure_analyzer import db
    db.init_db(os.path.join(tmpdir, f"part_{os.getpid()}.db"))
    _CFG = cfg


def _process_one(edf_path: str):
    from eeg_seizure_analyzer import analysis
    from eeg_seizure_analyzer.analysis import ClassificationParams
    cfg = _CFG
    meta = cfg["metadata"].get(os.path.basename(edf_path))
    try:
        r = analysis.process_chunk(
            edf_path=edf_path,
            model_name=cfg["model"],
            confidence_threshold=cfg["threshold"],
            convulsive_threshold=cfg["conv_threshold"],
            min_duration_sec=cfg["min_duration"],
            merge_gap_sec=cfg["merge_gap"],
            mode="batch",
            classification_params=ClassificationParams(),
            file_metadata=meta,
            overwrite=True,  # part-DB is always fresh; never skip here
            convulsive_model_name=cfg["conv_model"],
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--db-path", required=True, help="final project DB to write")
    ap.add_argument("--model", required=True, help="seizure detector model name")
    ap.add_argument("--convulsive-model", default=None,
                    help="convulsive classifier model name (cascade stage 2)")
    ap.add_argument("--metadata-csv", default=None,
                    help="batch metadata CSV/XLSX (filename -> cohort/group/animal)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--conv-threshold", type=float, default=0.5)
    ap.add_argument("--min-duration", type=float, default=5.0)
    ap.add_argument("--merge-gap", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--path-include", default=None,
                    help="only process EDFs whose full path matches this regex "
                         "(e.g. 'Week[123]-' for the first 3 weeks)")
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
    if not todo:
        print("Nothing to do.")
        return 0

    metadata = load_metadata(args.metadata_csv) if args.metadata_csv else {}
    if args.metadata_csv:
        print(f"Loaded metadata for {len(metadata)} files from {args.metadata_csv}")

    cfg = dict(
        model=args.model, conv_model=args.convulsive_model, metadata=metadata,
        threshold=args.threshold, conv_threshold=args.conv_threshold,
        min_duration=args.min_duration, merge_gap=args.merge_gap,
    )
    work_tmp = os.path.join(args.tmpdir, f"neddet_{os.getpid()}")
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
                print(f"  {n_done}/{len(todo)} files | {n_events} events | "
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

    print(f"Done. {n_done} files, {n_events} events, {n_err} errors, "
          f"{time.time() - t0:.0f}s. DB: {db_final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
