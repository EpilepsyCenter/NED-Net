"""SQLite database module for analysis results.

All database reads and writes go through this module.
No other part of the app should interact with SQLite directly.
"""

from __future__ import annotations

import re
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

        -- Which animals were recorded in each file and for how long, captured
        -- at analysis time from the channel→animal map. Independent of events,
        -- so per-animal recording time is exact even for animals with zero
        -- detected events (the denominator for per-hour rates / coverage).
        CREATE TABLE IF NOT EXISTS file_animals (
            chunk_id   INTEGER REFERENCES chunks(id),
            animal_id  TEXT,
            valid_sec  REAL,
            cohort     TEXT,
            group_id   TEXT
        );

        -- Per-animal review status: exclude an animal from aggregations, or
        -- censor it after a date (e.g. died / dropped mid-experiment). Keyed by
        -- animal_id within the project DB.
        CREATE TABLE IF NOT EXISTS animal_status (
            animal_id            TEXT PRIMARY KEY,
            excluded             INTEGER DEFAULT 0,
            valid_until          TEXT,
            notes                TEXT,
            recording_start_date TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_events_chunk ON events(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_events_animal ON events(animal_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
        CREATE INDEX IF NOT EXISTS idx_chunk_summary_chunk ON chunk_summary(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_file_animals_chunk ON file_animals(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_file_animals_animal ON file_animals(animal_id);
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

    # Migration: per-event exclude flag. Excluded events are dropped from
    # summaries/plots/exports but kept in the Results table so they can be
    # toggled back.
    try:
        conn.execute("SELECT excluded FROM events LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE events ADD COLUMN excluded INTEGER DEFAULT 0")
        conn.commit()

    # Migration: high-level category (seizure | spike) kept separate from the
    # specific detector in `source`, so classical detections show under the
    # same Seizures/Spikes split as ML ones. Backfill from existing sources.
    try:
        conn.execute("SELECT category FROM events LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE events ADD COLUMN category TEXT")
        conn.execute(
            "UPDATE events SET category = 'spike' "
            "WHERE category IS NULL AND source LIKE '%spike%'"
        )
        conn.execute("UPDATE events SET category = 'seizure' WHERE category IS NULL")
        conn.commit()

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_excluded ON events(excluded)"
    )
    conn.commit()

    # Migration: per-animal recording_start_date (day-1 reference for the
    # longitudinal view, so cohorts with different calendar starts align).
    try:
        conn.execute("SELECT recording_start_date FROM animal_status LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            "ALTER TABLE animal_status ADD COLUMN recording_start_date TEXT"
        )
        conn.commit()

    # Migration: per-event cohort/group. Cohort/group are per channel, so a
    # chunk's single cohort/group_id can't represent them. Backfill existing
    # events from their chunk so prior data stays filterable.
    for col in ("cohort", "group_id"):
        try:
            conn.execute(f"SELECT {col} FROM events LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
            conn.execute(
                f"UPDATE events SET {col} = "
                f"(SELECT c.{col} FROM chunks c WHERE c.id = events.chunk_id) "
                f"WHERE {col} IS NULL"
            )
            conn.commit()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local connection, reconnecting if the active DB changed.

    The active project can be switched at runtime (``set_active_project``), so
    a cached connection may point at a stale file. Reopen when the thread's
    connection was made against a different path.
    """
    cur = getattr(_local, "conn", None)
    if cur is None or getattr(_local, "conn_path", None) != str(_db_path):
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        _local.conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn_path = str(_db_path)
    return _local.conn


# ---------------------------------------------------------------------------
# Project databases
# ---------------------------------------------------------------------------
# Each project is a separate SQLite file, switchable at runtime so users can
# keep experiments apart and still add cohorts to an existing one later. The
# "Default" project is the legacy ~/.eeg_seizure_analyzer/analysis.db; named
# projects live in ~/.eeg_seizure_analyzer/projects/<name>.db. The active
# project is app-wide (Analysis writes and Results reads the same file) and is
# remembered across restarts via a small pointer file.

_PROJECTS_DIR = _DEFAULT_DB_DIR / "projects"
_ACTIVE_PTR = _DEFAULT_DB_DIR / "active_project.txt"
_DEFAULT_PROJECT = "Default"
_active_project: str = _DEFAULT_PROJECT


def _sanitize_project_name(name: str) -> str:
    """Filesystem-safe project name (alphanumerics, space, dash, underscore)."""
    name = re.sub(r"[^A-Za-z0-9 _-]", "", (name or "").strip())
    return re.sub(r"\s+", " ", name).strip()


def project_path(name: str) -> Path:
    """Resolve a project name to its .db file path."""
    safe = _sanitize_project_name(name)
    if not safe or safe == _DEFAULT_PROJECT:
        return _DEFAULT_DB_PATH
    return _PROJECTS_DIR / f"{safe}.db"


def list_projects() -> list[str]:
    """Known project names: Default plus every projects/*.db, sorted."""
    names = [_DEFAULT_PROJECT]
    try:
        if _PROJECTS_DIR.exists():
            names += [p.stem for p in sorted(_PROJECTS_DIR.glob("*.db"))]
    except Exception:
        pass
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_active_project() -> str:
    """Name of the currently active project."""
    return _active_project


def set_active_project(name: str) -> str:
    """Switch the active project DB (creating its schema if needed) and persist
    the choice. Thread connections pick up the change lazily via ``_get_conn``.
    Returns the active project name.
    """
    global _active_project
    path = project_path(name)
    target = (
        _DEFAULT_PROJECT if path == _DEFAULT_DB_PATH
        else _sanitize_project_name(name)
    )
    # Idempotent: two callbacks can react to the same switch; skip the
    # re-init / pointer write when this project is already active.
    if target == _active_project and str(path) == str(_db_path):
        return _active_project
    init_db(path)
    _active_project = target
    try:
        _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
        _ACTIVE_PTR.write_text(_active_project)
    except Exception:
        pass
    return _active_project


def create_project(name: str) -> str:
    """Create a new, empty project DB and make it active.

    Raises ValueError if the name is blank, reserved, or already taken.
    Returns the new active project name.
    """
    safe = _sanitize_project_name(name)
    if not safe:
        raise ValueError("Enter a project name.")
    if safe == _DEFAULT_PROJECT:
        raise ValueError(f"'{_DEFAULT_PROJECT}' is a reserved name.")
    if project_path(safe).exists():
        raise ValueError(f"Project '{safe}' already exists.")
    return set_active_project(safe)


def delete_project(name: str) -> None:
    """Delete a project's database file (and WAL/SHM sidecars).

    The Default project cannot be deleted. If the deleted project is active,
    switch to Default first so cached connections reconnect off the removed
    file. Raises ValueError on Default or a missing project.
    """
    safe = _sanitize_project_name(name)
    if not safe or safe == _DEFAULT_PROJECT:
        raise ValueError("The Default project cannot be deleted.")
    path = project_path(safe)
    if not path.exists():
        raise ValueError(f"Project '{safe}' not found.")
    if _active_project == safe:
        set_active_project(_DEFAULT_PROJECT)
    for q in (path, path.with_name(path.name + "-wal"),
              path.with_name(path.name + "-shm")):
        try:
            q.unlink()
        except OSError:
            pass


def _restore_active_project() -> None:
    """On import, point ``_db_path`` at the persisted active project, if any."""
    global _active_project, _db_path
    try:
        name = _ACTIVE_PTR.read_text().strip()
    except Exception:
        return
    if not name:
        return
    _db_path = project_path(name)
    _active_project = (
        _DEFAULT_PROJECT if _db_path == _DEFAULT_DB_PATH else name
    )


_restore_active_project()


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
        conn.execute("DELETE FROM file_animals WHERE chunk_id = ?", (chunk_id,))
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


def write_events(chunk_id: int, events: list[dict], source: str = "seizure_cnn",
                 category: str | None = None) -> None:
    """Insert list of event dicts for a chunk.

    ``source`` is the specific detector (seizure_cnn, spike_cnn, autocorrelation,
    spectral_band, spike_train, ensemble, ...). ``category`` is the high-level
    seizure/spike bucket used by the Results Seizures/Spikes split; when omitted
    it is derived from the source.
    """
    conn = _get_conn()
    for ev in events:
        ev_source = ev.get("source", source)
        ev_cat = (ev.get("category") or category
                  or ("spike" if "spike" in ev_source else "seizure"))
        conn.execute(
            """INSERT INTO events (chunk_id, animal_id, date, start_sec,
               end_sec, duration_sec, type, subtype, cnn_confidence,
               convulsive_confidence, movement_flag, recording_day, hour_of_day,
               source, category, cohort, group_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                ev_source,
                ev_cat,
                ev.get("cohort", ""),
                ev.get("group_id", ""),
            ),
        )
    conn.commit()


def set_event_excluded(event_id: int, excluded: bool) -> None:
    """Mark a single event excluded (1) or included (0) in the active DB."""
    conn = _get_conn()
    conn.execute(
        "UPDATE events SET excluded = ? WHERE id = ?",
        (1 if excluded else 0, int(event_id)),
    )
    conn.commit()


def write_file_animals(chunk_id: int, animals: dict | None) -> None:
    """Replace the per-animal observation rows for a chunk.

    ``animals`` maps animal_id -> ``{"valid_sec", "cohort", "group_id"}``.
    Records which animals were recorded in this file and for how long.
    """
    conn = _get_conn()
    conn.execute("DELETE FROM file_animals WHERE chunk_id = ?", (chunk_id,))
    for aid, info in (animals or {}).items():
        if not aid:
            continue
        info = info or {}
        conn.execute(
            """INSERT INTO file_animals (chunk_id, animal_id, valid_sec,
               cohort, group_id) VALUES (?, ?, ?, ?, ?)""",
            (chunk_id, aid, float(info.get("valid_sec") or 0),
             info.get("cohort", "") or "", info.get("group_id", "") or ""),
        )
    conn.commit()


def add_manual_events(edf_path: str, event_dicts: list[dict], source: str,
                      category: str = "seizure", mode: str = "manual",
                      recording_sec: float | None = None,
                      file_animals: dict | None = None) -> int:
    """Add classical/manual detection events to the active project DB.

    Non-destructive across detectors: finds or creates the file's chunk WITHOUT
    deleting it (so ML or other-detector events for the same file survive), then
    replaces only this ``source``'s events for the file (replace-on-re-add).

    ``recording_sec`` is the file's full recording length (seconds), used as the
    denominator for per-hour rates in Results. It is stored as the chunk's
    duration when the chunk is first created here, and backfilled if an existing
    chunk has no duration yet — but never overwrites a real value (e.g. one set
    by the ML analysis path). Returns the number of events written.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, chunk_end_sec FROM chunks WHERE path = ?", (str(edf_path),)
    ).fetchone()
    if row:
        chunk_id = row["id"]
        if recording_sec and not (row["chunk_end_sec"] or 0):
            conn.execute(
                "UPDATE chunks SET chunk_end_sec = ? WHERE id = ?",
                (float(recording_sec), chunk_id),
            )
    else:
        date = event_dicts[0].get("date", "") if event_dicts else ""
        cur = conn.execute(
            """INSERT INTO chunks (path, cohort, group_id, date, chunk_start_sec,
               chunk_end_sec, processed_at, processing_sec, status, mode)
               VALUES (?, '', '', ?, 0, ?, ?, 0, 'ok', ?)""",
            (str(edf_path), date, float(recording_sec or 0),
             datetime.now(timezone.utc).isoformat(), mode),
        )
        chunk_id = cur.lastrowid
    # Replace only this detector's events for the file (keep other sources).
    conn.execute(
        "DELETE FROM events WHERE chunk_id = ? AND source = ?",
        (chunk_id, source),
    )
    conn.commit()
    write_events(chunk_id, event_dicts, source=source, category=category)

    # Record per-animal observation, but only if this file has none yet — don't
    # clobber a mapping written by the ML analysis path. Fall back to the event
    # animals when no explicit mapping is supplied.
    if file_animals is None and recording_sec:
        file_animals = {}
        for ev in event_dicts:
            aid = ev.get("animal_id")
            if aid:
                file_animals.setdefault(aid, {
                    "valid_sec": recording_sec,
                    "cohort": ev.get("cohort", ""),
                    "group_id": ev.get("group_id", ""),
                })
    if file_animals:
        has = conn.execute(
            "SELECT COUNT(*) FROM file_animals WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()[0]
        if not has:
            write_file_animals(chunk_id, file_animals)
    return len(event_dicts)


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
    category: str | None = None,
    group_id: str | None = None,
) -> dict:
    """Query summary statistics with optional filters.

    Returns dict with total counts and breakdowns.
    """
    conn = _get_conn()

    # Build WHERE clause
    conditions = ["c.status = 'ok'"]
    params: list = []

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

    # Event-level filters (cohort/group are per-event)
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
    if category:
        ev_conditions.append("e.category = ?")
        ev_params.append(category)
    if cohort:
        ev_conditions.append("e.cohort = ?")
        ev_params.append(cohort)
    if group_id:
        ev_conditions.append("e.group_id = ?")
        ev_params.append(group_id)
    # Excluded events never count toward summaries / plots.
    ev_conditions.append("e.excluded = 0")

    ev_where = (" AND " + " AND ".join(ev_conditions)) if ev_conditions else ""

    # Files with matching events (honours the event-level filters)
    n_files = conn.execute(
        f"""SELECT COUNT(DISTINCT e.chunk_id) FROM events e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE {where}{ev_where}""",
        params + ev_params,
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
    category: str | None = None,
    group_id: str | None = None,
) -> list[dict]:
    """Query events with optional filters. Returns list of dicts.

    Note: excluded events ARE returned (with ``excluded=1``) so the Results
    table can show and un-exclude them; only the aggregate queries drop them.
    """
    conn = _get_conn()

    conditions = ["c.status = 'ok'"]
    params: list = []

    if cohort:
        conditions.append("e.cohort = ?")
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
    if category:
        conditions.append("e.category = ?")
        params.append(category)
    if group_id:
        conditions.append("e.group_id = ?")
        params.append(group_id)

    where = " AND ".join(conditions)

    rows = conn.execute(
        f"""SELECT e.*, c.path, c.cohort as chunk_cohort,
                   c.group_id as chunk_group, c.mode, c.date as chunk_date
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


def get_all_cohorts() -> list[str]:
    """Return sorted list of distinct non-empty per-event cohorts."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT cohort FROM events "
        "WHERE cohort IS NOT NULL AND cohort != '' ORDER BY cohort"
    ).fetchall()
    return [r[0] for r in rows]


def get_all_groups() -> list[str]:
    """Return sorted list of distinct non-empty per-event groups."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT group_id FROM events "
        "WHERE group_id IS NOT NULL AND group_id != '' ORDER BY group_id"
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
    category: str | None = None,
    cohort: str | None = None,
    group_id: str | None = None,
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
    if category:
        conditions.append("e.category = ?")
        params.append(category)
    if cohort:
        conditions.append("e.cohort = ?")
        params.append(cohort)
    if group_id:
        conditions.append("e.group_id = ?")
        params.append(group_id)
    conditions.append("e.excluded = 0")

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
    category: str | None = None,
    cohort: str | None = None,
    group_id: str | None = None,
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
    if category:
        conditions.append("e.category = ?")
        params.append(category)
    if cohort:
        conditions.append("e.cohort = ?")
        params.append(cohort)
    if group_id:
        conditions.append("e.group_id = ?")
        params.append(group_id)
    conditions.append("e.excluded = 0")

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


def get_file_animals(
    date_start: str | None = None,
    date_end: str | None = None,
    animal_id: str | None = None,
    mode: str | None = None,
    cohort: str | None = None,
    group_id: str | None = None,
) -> list[dict]:
    """Per-(file, animal) observation rows with the file's date.

    Used to compute exact per-animal/group recording time (the denominator for
    rates and coverage). Honours the same file-level filters as the event
    queries, but NOT event-level ones (confidence/type/detector) — an animal's
    observation time does not depend on which events you are looking at.
    """
    conn = _get_conn()
    conditions = ["c.status = 'ok'"]
    params: list = []
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
        conditions.append("fa.animal_id = ?")
        params.append(animal_id)
    if cohort:
        conditions.append("fa.cohort = ?")
        params.append(cohort)
    if group_id:
        conditions.append("fa.group_id = ?")
        params.append(group_id)
    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""SELECT fa.chunk_id, fa.animal_id, fa.valid_sec,
                   fa.cohort, fa.group_id, c.date as date, c.mode, c.path
            FROM file_animals fa
            JOIN chunks c ON fa.chunk_id = c.id
            WHERE {where}""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def get_animal_status() -> dict:
    """Return {animal_id: {excluded, valid_until, notes, recording_start_date}}
    for the active DB."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT animal_id, excluded, valid_until, notes, recording_start_date "
        "FROM animal_status"
    ).fetchall()
    return {
        r["animal_id"]: {
            "excluded": bool(r["excluded"]),
            "valid_until": r["valid_until"] or "",
            "notes": r["notes"] or "",
            "recording_start_date": r["recording_start_date"] or "",
        }
        for r in rows
    }


def set_animal_status(animal_id: str, excluded: bool | None = None,
                      valid_until: str | None = None,
                      notes: str | None = None,
                      recording_start_date: str | None = None) -> None:
    """Upsert per-animal status. Only the arguments passed (non-None) change;
    string fields set to an empty string clear that field."""
    if not animal_id:
        return
    conn = _get_conn()
    cur = conn.execute(
        "SELECT excluded, valid_until, notes, recording_start_date "
        "FROM animal_status WHERE animal_id = ?",
        (animal_id,),
    ).fetchone()
    if cur:
        ex = cur["excluded"] if excluded is None else (1 if excluded else 0)
        vu = cur["valid_until"] if valid_until is None else (valid_until or None)
        nt = cur["notes"] if notes is None else (notes or None)
        rs = (cur["recording_start_date"] if recording_start_date is None
              else (recording_start_date or None))
        conn.execute(
            "UPDATE animal_status SET excluded = ?, valid_until = ?, notes = ?, "
            "recording_start_date = ? WHERE animal_id = ?",
            (ex, vu, nt, rs, animal_id),
        )
    else:
        conn.execute(
            "INSERT INTO animal_status (animal_id, excluded, valid_until, notes, "
            "recording_start_date) VALUES (?, ?, ?, ?, ?)",
            (animal_id, 1 if excluded else 0, valid_until or None,
             notes or None, recording_start_date or None),
        )
    conn.commit()
