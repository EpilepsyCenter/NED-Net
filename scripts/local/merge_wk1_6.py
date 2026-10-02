#!/usr/bin/env python3
"""Combine the weeks-1-3 and weeks-4-6 SV2A project DBs into wk1-6 DBs.

Copies each base (wk1-3) DB to a new wk1-6 name, then appends the wk4-6 DB
using the same FK-remapping merge that detect_batch.py uses for part-DBs
(schema-introspected; chunk ids reassigned; idempotent per file path, so a
re-run won't duplicate). Originals are left untouched as backups.
"""
import os
import shutil
import sqlite3
import sys

PROJ = os.path.expanduser("~/.eeg_seizure_analyzer/projects")

# (base wk1-3, add wk4-6, output wk1-6)
PAIRS = [
    ("SV2A_UNet_wk1-3.db", "lunarc_detect_wk4-6.db", "SV2A_UNet_wk1-6.db"),
    ("sv2a_spikes.db",     "sv2a_spikes_wk4-6.db",   "sv2a_spikes_wk1-6.db"),
]


def _cols(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _merge_part(final_db, part_db):
    # Fold the part-DB's WAL into its main file so ATTACH sees every row.
    c = sqlite3.connect(part_db)
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()

    fin = sqlite3.connect(final_db)
    fin.execute("ATTACH ? AS p", (part_db,))
    chunk_cols = [c for c in _cols(fin, "chunks") if c != "id"]
    path_i = chunk_cols.index("path")
    idmap = {}
    rows = fin.execute(f"SELECT id,{','.join(chunk_cols)} FROM p.chunks").fetchall()
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


def _count(db):
    c = sqlite3.connect(db)
    e = c.execute("SELECT count(*) FROM events").fetchone()[0]
    k = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
    c.close()
    return e, k


def main():
    for base, add, out in PAIRS:
        bp, ap, op = (os.path.join(PROJ, f) for f in (base, add, out))
        if not (os.path.exists(bp) and os.path.exists(ap)):
            print(f"!! skip {out}: missing {base} or {add}")
            continue
        # Clean any stale output + its WAL/SHM sidecars.
        for ext in ("", "-wal", "-shm"):
            p = op + ext
            if os.path.exists(p):
                os.remove(p)
        # Snapshot the base (fold its WAL first), then append the wk4-6 DB.
        c = sqlite3.connect(bp)
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
        shutil.copy2(bp, op)
        be, bk = _count(op)
        ae, ak = _count(ap)
        _merge_part(op, ap)
        oe, ok = _count(op)
        print(f"{out}: events {be}+{ae} -> {oe} (exp {be+ae}), "
              f"chunks {bk}+{ak} -> {ok} (exp {bk+ak})  "
              f"{'OK' if oe == be+ae and ok == bk+ak else 'MISMATCH!'}")


if __name__ == "__main__":
    sys.exit(main())
