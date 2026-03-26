"""Server-side session state manager.

Stores large objects (recordings, detection results, numpy arrays)
server-side, keyed by a session UUID that lives in a lightweight
dcc.Store on the client.  This avoids serialising megabytes of
data through Dash callbacks.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """All mutable state for one browser session."""

    # Recording data
    recording: Any = None
    activity_recordings: dict = field(default_factory=dict)
    channel_pairings: list = field(default_factory=list)
    all_channels_info: list = field(default_factory=list)

    # Detection results
    seizure_events: list = field(default_factory=list)
    spike_events: list = field(default_factory=list)
    detected_events: list = field(default_factory=list)

    # Per-channel detection metadata (baseline, threshold, spike positions)
    st_detection_info: dict = field(default_factory=dict)
    sp_detection_info: dict = field(default_factory=dict)

    # Blinding
    blinding_on: bool = True
    blinding_log: list = field(default_factory=list)

    # User defaults
    user_defaults: dict = field(default_factory=dict)

    # Arbitrary key-value store for extensibility
    extra: dict = field(default_factory=dict)


# ── Global session store ──────────────────────────────────────────────

_sessions: dict[str, SessionState] = {}


def create_session() -> str:
    """Create a new session and return its UUID."""
    sid = str(uuid.uuid4())
    _sessions[sid] = SessionState()
    return sid


def get_session(session_id: str | None) -> SessionState:
    """Get session state, creating if needed."""
    if session_id is None or session_id not in _sessions:
        session_id = create_session()
    return _sessions[session_id]


def get(session_id: str | None, key: str, default: Any = None) -> Any:
    """Get a value from session state."""
    state = get_session(session_id)
    if hasattr(state, key):
        return getattr(state, key)
    return state.extra.get(key, default)


def put(session_id: str | None, key: str, value: Any):
    """Set a value in session state."""
    state = get_session(session_id)
    if hasattr(state, key):
        setattr(state, key, value)
    else:
        state.extra[key] = value


def clear_detections(session_id: str | None, which: str = "all"):
    """Clear detection results.  which: 'seizures', 'spikes', or 'all'."""
    state = get_session(session_id)
    if which in ("seizures", "all"):
        state.seizure_events = []
        state.st_detection_info = {}
    if which in ("spikes", "all"):
        state.spike_events = []
        state.sp_detection_info = {}
    state.detected_events = state.seizure_events + state.spike_events
