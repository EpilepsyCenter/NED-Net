"""Recently opened recordings and folders, listed on the start screen.

Stored in ``~/.eeg_seizure_analyzer/recent.json``, most recent first and at most
``MAX_RECENT`` entries: ``{"path", "kind": "file" | "folder", "opened_at"}``.
Writes are best effort: a failure never blocks opening a file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

MAX_RECENT = 10


def _store() -> Path:
    return Path.home() / ".eeg_seizure_analyzer" / "recent.json"


def list_recent() -> list[dict]:
    """Recent entries, most recent first, each with ``exists`` set."""
    try:
        items = json.loads(_store().read_text())
    except Exception:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        p = it.get("path")
        if not p:
            continue
        kind = it.get("kind", "file")
        exists = os.path.isdir(p) if kind == "folder" else os.path.isfile(p)
        out.append({**it, "kind": kind, "exists": exists})
    return out


def add_recent(path: str, kind: str = "file") -> None:
    """Put ``path`` at the top of the list (moving it there if already listed)."""
    try:
        path = os.path.abspath(path)
        items = [it for it in list_recent() if it.get("path") != path]
        items.insert(0, {"path": path, "kind": kind,
                         "opened_at": datetime.now().isoformat(timespec="minutes")})
        f = _store()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            [{k: it[k] for k in ("path", "kind", "opened_at") if k in it}
             for it in items[:MAX_RECENT]], indent=1))
        os.replace(tmp, f)
    except Exception:
        pass
