"""SQLite database module for analysis results.

All database reads and writes go through this module.
No other part of the app should interact with SQLite directly.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

# Default database path
_DEFAULT_DB_DIR = Path.home() / ".eeg_seizure_analyzer"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "analysis.db"

# Thread-local connections
_local = threading.local()

_db_path: Path = _DEFAULT_DB_PATH


def init_db(db_path: str | Path | None = None) -> None:
    """Create database and tables if they do not exist."""
    global _db_path
    if db_path is not None:
        _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id              INTEGER PRIMARY KEY,
            path            TEXT UNIQUE,
            cohort          TEXT,
            group_id        TEXT,
            date            TEXT,
            chunk_start_sec REAL,
            chunk_end_sec   REAL,
            processed_at    TEXT,
            processing_sec  REAL,
            status          TEXT,
            mode            TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY,
            chunk_id        INTEGER REFERENCES chunks(id),
            animal_id       TEXT,
            date            TEXT,
            start_sec       REAL,
            end_sec         REAL,
            duration_sec    REAL,
            type            TEXT,
            subtype         TEXT,
            cnn_confidence  REAL,
            convulsive_confidence REAL,
            movement_flag   BOOLEAN,
            recording_day   INTEGER,
            hour_of_day     INTEGER,
            source          TEXT DEFAULT 'seizure_cnn'
        );

        CREATE TABLE IF NOT EXISTS chunk_summary (
            chunk_id            INTEGER REFERENCES chunks(id),
            animal_id           TEXT,
            n_convulsive        INTEGER,
            n_nonconvulsive     INTEGER,
            n_flagged           INTEGER,
            total_duration_sec  REAL
        );

        CREATE INDEX IF NOT EXISTS idx_events_chunk ON events(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_events_animal ON events(animal_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
        CREATE INDEX IF NOT EXISTS idx_chunk_summary_chunk ON chunk_summary(chunk_id);
    """)
    conn.commit()

    # Migration: add source column to existing databases
    try:
        conn.execute("SELECT source FROM events LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            "ALTER TABLE events ADD COLUMN source TEXT DEFAULT 'seizure_cnn'"
        )
        conn.commit()

    # Create source index (after migration ensures column exists)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)"
    )
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def get_processed_paths() -> set[str]:
    """Return set of all EDF paths already in chunks table."""
    conn = _get_conn()
    rows = conn.execute("SELECT path FROM chunks WHERE status = 'ok'").fetchall()
    return {r["path"] for r in rows}


def write_chunk(path: str, meta: dict, mode: str) -> int:
    """Insert or replace chunk record, return chunk_id.

    If the path already exists, the old chunk and its events/summaries
    are replaced (for re-processing).
    """
    conn = _get_conn()

    # Delete existing data for this path (re-process)
    existing = conn.execute(
        "SELECT id FROM chunks WHERE path = ?", (str(path),)
    ).fetchone()
    if existing:
        chunk_id = existing["id"]
        conn.execute("DELETE FROM events WHERE chunk_id = ?", (chunk_id,))
        conn.execute("DELETE FROM chunk_summary WHERE chunk_id = ?", (chunk_id,))
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    cursor = conn.execute(
        """INSERT INTO chunks (path, cohort, group_id, date,
           chunk_start_sec, chunk_end_sec, processed_at,
           processing_sec, status, mode)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(path),
            meta.get("cohort", ""),
            meta.get("group_id", ""),
            meta.get("date", ""),
            meta.get("chunk_start_sec", 0),
            meta.get("chunk_end_sec", 0),
            meta.get("processed_at", datetime.now(timezone.utc).isoformat()),
            meta.get("processing_sec", 0),
            meta.get("status", "ok"),
            mode,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def write_events(chunk_id: int, events: list[dict], source: str = "seizure_cnn") -> None:
    """Insert list of event dicts for a chunk."""
    conn = _get_conn()
    for ev in events:
        conn.execute(
            """INSERT INTO events (chunk_id, animal_id, date, start_sec,
               end_sec, duration_sec, type, subtype, cnn_confidence,
               convulsive_confidence, movement_flag, recording_day, hour_of_day,
               source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk_id,
                ev.get("animal_id", ""),
                ev.get("date", ""),
                ev.get("start_sec", 0),
                ev.get("end_sec", 0),
                ev.get("duration_sec", 0),
                ev.get("type", "non_convulsive"),
                ev.get("subtype"),
                ev.get("cnn_confidence", 0),
                ev.get("convulsive_confidence", 0),
                ev.get("movement_flag", False),
                ev.get("recording_day"),
                ev.get("hour_of_day"),
                ev.get("source", source),
            ),
        )
    conn.commit()


def write_summary(chunk_id: int, animal_id: str, summary: dict) -> None:
    """Insert pre-computed summary for a chunk/animal."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO chunk_summary (chunk_id, animal_id,
           n_convulsive, n_nonconvulsive, n_flagged, total_duration_sec)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            animal_id,
            summary.get("n_convulsive", 0),
            summary.get("n_nonconvulsive", 0),
            summary.get("n_flagged", 0),
            summary.get("total_duration_sec", 0),
        ),
    )
    conn.commit()


def update_chunk_timing(chunk_id: int, processing_sec: float) -> None:
    """Update processing time for a chunk."""
    conn = _get_conn()
    conn.execute(
        "UPDATE chunks SET processing_sec = ? WHERE id = ?",
        (processing_sec, chunk_id),
    )
    conn.commit()


def mark_chunk_error(chunk_id: int, error_msg: str = "") -> None:
    """Mark a chunk as errored."""
    conn = _get_conn()
    conn.execute(
        "UPDATE chunks SET status = 'error' WHERE id = ?", (chunk_id,)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_summary(
    cohort: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    animal_id: str | None = None,
    mode: str | None = None,
    min_confidence: float | None = None,
    event_type: str | None = None,
    source: str | None = None,
) -> dict:
    """Query summary statistics with optional filters.

    Returns dict with total counts and breakdowns.
    """
    conn = _get_conn()

    # Build WHERE clause
    conditions = ["c.status = 'ok'"]
    params: list = []

    if cohort:
        conditions.append("c.cohort = ?")
        params.append(cohort)
    if date_start:
        conditions.append("c.date >= ?")
        params.append(date_start)
    if date_end:
        conditions.append("c.date <= ?")
        params.append(date_end)
    if mode:
        conditions.append("c.mode = ?")
        params.append(mode)

    where = " AND ".join(conditions)

    # Event-level filters
    ev_conditions = []
    ev_params: list = []
    if animal_id:
        ev_conditions.append("e.animal_id = ?")
        ev_params.append(animal_id)
    if min_confidence is not None:
        ev_conditions.append("e.cnn_confidence >= ?")
        ev_params.append(min_confidence)
    if event_type:
        ev_conditions.append("e.type = ?")
        ev_params.append(event_type)
    if source:
        ev_conditions.append("e.source = ?")
        ev_params.append(source)

    ev_where = (" AND " + " AND ".join(ev_conditions)) if ev_conditions else ""

    # Files processed
    n_files = conn.execute(
        f"SELECT COUNT(DISTINCT c.id) FROM chunks c WHERE {where}", params
    ).fetchone()[0]

    # Animals
    animals = conn.execute(
        f"""SELECT DISTINCT e.animal_id FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where}""",
        params + ev_params,
    ).fetchall()
    animal_list = [r[0] for r in animals if r[0]]

    # Event counts
    total = conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where}""",
        params + ev_params,
    ).fetchone()[0]

    n_conv = conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where} AND e.type = 'convulsive'""",
        params + ev_params,
    ).fetchone()[0]

    n_nonconv = conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where} AND e.type = 'non_convulsive'""",
        params + ev_params,
    ).fetchone()[0]

    n_hvsw = conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where} AND e.subtype = 'HVSW'""",
        params + ev_params,
    ).fetchone()[0]

    n_hpd = conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where} AND e.subtype = 'HPD'""",
        params + ev_params,
    ).fetchone()[0]

    n_flagged = conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where} AND e.movement_flag = 1""",
        params + ev_params,
    ).fetchone()[0]

    return {
        "n_files": n_files,
        "animals": animal_list,
        "n_animals": len(animal_list),
        "total_events": total,
        "n_convulsive": n_conv,
        "n_nonconvulsive": n_nonconv,
        "n_hvsw": n_hvsw,
        "n_hpd": n_hpd,
        "n_flagged": n_flagged,
    }


def get_events(
    cohort: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    animal_id: str | None = None,
    mode: str | None = None,
    min_confidence: float | None = None,
    event_type: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """Query events with optional filters. Returns list of dicts."""
    conn = _get_conn()

    conditions = ["c.status = 'ok'"]
    params: list = []

    if cohort:
        conditions.append("c.cohort = ?")
        params.append(cohort)
    if date_start:
        conditions.append("c.date >= ?")
        params.append(date_start)
    if date_end:
        conditions.append("c.date <= ?")
        params.append(date_end)
    if mode:
        conditions.append("c.mode = ?")
        params.append(mode)
    if animal_id:
        conditions.append("e.animal_id = ?")
        params.append(animal_id)
    if min_confidence is not None:
        conditions.append("e.cnn_confidence >= ?")
        params.append(min_confidence)
    if event_type:
        conditions.append("e.type = ?")
        params.append(event_type)
    if source:
        conditions.append("e.source = ?")
        params.append(source)

    where = " AND ".join(conditions)

    rows = conn.execute(
        f"""SELECT e.*, c.path, c.cohort, c.group_id, c.mode, c.date as chunk_date
            FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}
            ORDER BY c.date, e.start_sec""",
        params,
    ).fetchall()

    return [dict(r) for r in rows]


def get_chunk_status() -> list[dict]:
    """Return processing status for all chunks, ordered by processing time."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, path, status, mode, processed_at, processing_sec,
                  chunk_start_sec, chunk_end_sec, date
           FROM chunks
           ORDER BY processed_at DESC
           LIMIT 100"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_animals() -> list[str]:
    """Return sorted list of all unique animal IDs."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT animal_id FROM events WHERE animal_id != '' ORDER BY animal_id"
    ).fetchall()
    return [r[0] for r in rows]


def get_all_files() -> list[dict]:
    """Return list of all processed files with mode and date."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, path, mode, date, cohort, group_id, processed_at
           FROM chunks WHERE status = 'ok'
           ORDER BY processed_at DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_date_range() -> tuple[str, str]:
    """Return (min_date, max_date) from chunks."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT MIN(date), MAX(date) FROM chunks WHERE status = 'ok' AND date != ''"
    ).fetchone()
    return (row[0] or "", row[1] or "")


def get_daily_burden(
    animal_id: str | None = None,
    min_confidence: float | None = None,
    source: str | None = None,
) -> list[dict]:
    """Return daily event counts grouped by date and type."""
    conn = _get_conn()
    conditions = ["c.status = 'ok'"]
    params: list = []
    if animal_id:
        conditions.append("e.animal_id = ?")
        params.append(animal_id)
    if min_confidence is not None:
        conditions.append("e.cnn_confidence >= ?")
        params.append(min_confidence)
    if source:
        conditions.append("e.source = ?")
        params.append(source)

    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""SELECT c.date, e.type, COUNT(*) as n_events,
                   SUM(e.duration_sec) as total_duration
            FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}
            GROUP BY c.date, e.type
            ORDER BY c.date""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def get_circadian(
    animal_id: str | None = None,
    min_confidence: float | None = None,
    source: str | None = None,
) -> list[dict]:
    """Return hourly event counts for circadian analysis."""
    conn = _get_conn()
    conditions = ["c.status = 'ok'"]
    params: list = []
    if animal_id:
        conditions.append("e.animal_id = ?")
        params.append(animal_id)
    if min_confidence is not None:
        conditions.append("e.cnn_confidence >= ?")
        params.append(min_confidence)
    if source:
        conditions.append("e.source = ?")
        params.append(source)

    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""SELECT e.hour_of_day, e.type, COUNT(*) as n_events
            FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where} AND e.hour_of_day IS NOT NULL
            GROUP BY e.hour_of_day, e.type
            ORDER BY e.hour_of_day""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]
