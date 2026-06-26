#!/usr/bin/env python
"""Split a project detection DB into a new DB containing only the files whose
path matches a GLOB (e.g. one batch of weeks).

Use it to carve a combined run (lunarc_detect_wk1-6_v2.db) into the per-period
DBs your analysis expects:

    python scripts/local/split_project_db.py --src …wk1-6_v2.db \
        --dest …wk1-3_v2.db --path-glob '*Week[123]-*'
    python scripts/local/split_project_db.py --src …wk1-6_v2.db \
        --dest …wk4-6_v2.db --path-glob '*Week[456]-*'

Copies the matching chunks and their dependent rows (events, chunk_summary,
file_animals) with foreign keys remapped, plus animal_status — the same
schema-introspected FK remap the batch merge uses, so it tracks db.py
migrations. The destination is created fresh.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys


def _cols(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def split(src: str, dest: str, path_glob: str) -> None:
    src = os.path.abspath(os.path.expanduser(src))
    dest = os.path.abspath(os.path.expanduser(dest))
    if os.path.exists(dest):
        print(f"ERROR: dest already exists: {dest}", file=sys.stderr)
        raise SystemExit(2)

    # Create the destination with the current schema, then copy into it.
    from eeg_seizure_analyzer import db
    db.init_db(dest)

    fin = sqlite3.connect(dest)
    fin.execute("ATTACH ? AS s", (src,))

    chunk_cols = [c for c in _cols(fin, "chunks") if c != "id"]
    sel = fin.execute(
        f"SELECT id,{','.join(chunk_cols)} FROM s.chunks WHERE path GLOB ?",
        (path_glob,)).fetchall()
    idmap: dict[int, int] = {}
    ph = ",".join(["?"] * len(chunk_cols))
    for row in sel:
        old_id, vals = row[0], list(row[1:])
        cur = fin.execute(
            f"INSERT INTO chunks ({','.join(chunk_cols)}) VALUES ({ph})", vals)
        idmap[old_id] = cur.lastrowid

    for tbl in ("events", "chunk_summary", "file_animals"):
        cols = [c for c in _cols(fin, tbl) if c != "id"]
        ci = cols.index("chunk_id")
        ph = ",".join(["?"] * len(cols))
        for row in fin.execute(f"SELECT {','.join(cols)} FROM s.{tbl}").fetchall():
            vals = list(row)
            new_id = idmap.get(vals[ci])
            if new_id is None:        # row belongs to a chunk we didn't copy
                continue
            vals[ci] = new_id
            fin.execute(f"INSERT INTO {tbl} ({','.join(cols)}) VALUES ({ph})", vals)

    ascols = _cols(fin, "animal_status")
    ph = ",".join(["?"] * len(ascols))
    for row in fin.execute(f"SELECT {','.join(ascols)} FROM s.animal_status").fetchall():
        fin.execute(
            f"INSERT OR IGNORE INTO animal_status ({','.join(ascols)}) VALUES ({ph})",
            row)

    fin.commit()
    n_chunks = fin.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_events = fin.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    fin.execute("DETACH s")
    fin.close()
    print(f"{path_glob}: {n_chunks} files, {n_events} events -> {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="combined project DB to split")
    ap.add_argument("--dest", required=True, help="new DB to create (must not exist)")
    ap.add_argument("--path-glob", required=True,
                    help="SQLite GLOB on chunk path, e.g. '*Week[123]-*'")
    args = ap.parse_args()
    split(args.src, args.dest, args.path_glob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
