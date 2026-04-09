"""NED-Net Dash application entry point.

Run with:  python -m eeg_seizure_analyzer.dash_app.main
"""

from __future__ import annotations

from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    blinding_badge,
    section_header,
    sidebar_divider,
)
from eeg_seizure_analyzer.io.persistence import detection_json_path, spike_detection_json_path
from eeg_seizure_analyzer.io.annotation_store import annotation_json_path

# ── Import tab modules ────────────────────────────────────────────────

from eeg_seizure_analyzer.dash_app.pages import upload, viewer, seizures, spikes, training, training_spikes, tools, adicht_converter, ml_datasets, ml_detection, ml_results, analysis as analysis_page, results as results_page

# ── Initialise SQLite database ────────────────────────────────────────
from eeg_seizure_analyzer import db as _db
_db.init_db()

# ── App setup ─────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder="assets",
    title="NED-Net",
    update_title="NED-Net | Loading...",
)

server = app.server

# ── Video streaming route (range-request support) ────────────────────

import os
from flask import request, Response, send_file

@server.route("/video/<path:session_id>")
def serve_video(session_id):
    """Serve the associated MP4 video file with HTTP range request support."""
    state = server_state.get_session(session_id)
    if state is None or state.recording is None:
        return Response("No recording loaded", status=404)

    video_path = state.extra.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        return Response("No video found", status=404)

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("Range")

    if range_header:
        # Parse range: "bytes=start-end"
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else min(start + 2 * 1024 * 1024, file_size - 1)
        end = min(end, file_size - 1)
        length = end - start + 1

        with open(video_path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return Response(data, status=206, headers=headers)

    # Full file (shouldn't happen with video, but handle it)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


# ── Clientside callback for slider/input sync ─────────────────────────
# Single callback with both as Input; uses dash_clientside.callback_context
# to determine which triggered and return the value to the other.

from dash import clientside_callback, MATCH, ALL

clientside_callback(
    """
    function(sliderVal, inputVal) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) return [sliderVal, sliderVal, '  ' + sliderVal];
        const trigger = ctx.triggered[0].prop_id;
        if (trigger.includes('param-slider')) {
            return [sliderVal, sliderVal, '  ' + sliderVal];
        }
        return [inputVal, inputVal, '  ' + inputVal];
    }
    """,
    Output({"type": "param-slider", "key": MATCH}, "value"),
    Output({"type": "param-input", "key": MATCH}, "value"),
    Output({"type": "param-display", "key": MATCH}, "children"),
    Input({"type": "param-slider", "key": MATCH}, "value"),
    Input({"type": "param-input", "key": MATCH}, "value"),
)

# ── Clientside callbacks for video sync ───────────────────────────────

# Viewer: auto-seek video when EEG start position changes (arrows, input)
clientside_callback(
    """
    function(startSec) {
        var video = document.getElementById('viewer-video-player');
        if (video && startSec != null) {
            video.currentTime = startSec;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("viewer-video-sync", "n_clicks"),
    Input("viewer-start-input", "value"),
    prevent_initial_call=True,
)


# ── Sidebar layout ────────────────────────────────────────────────────


def _sidebar():
    return html.Div(
        id="sidebar",
        children=[
            # Brand
            html.Div(
                id="sidebar-brand",
                children=[
                    html.H4("NED-Net"),
                    html.Div("Neural Event Detection", className="subtitle"),
                ],
            ),
            # Sidebar body
            html.Div(
                id="sidebar-content",
                children=[
                    # ── Blinding (hidden for now) ─────────────────
                    html.Div(
                        style={"display": "none"},
                        children=[
                            section_header("BLINDING"),
                            html.Div(id="blinding-badge-container",
                                     children=[blinding_badge(False)]),
                            dbc.Switch(
                                id="blinding-toggle",
                                label="Blinding OFF",
                                value=False,
                                className="mt-2",
                                style={"fontSize": "0.82rem"},
                            ),
                        ],
                    ),

                    # ── File info ─────────────────────────────────
                    section_header("RECORDING"),
                    html.Div(
                        id="sidebar-file-info",
                        children=[
                            html.Div(
                                "No file loaded",
                                className="file-info",
                                style={"opacity": "0.5"},
                            ),
                        ],
                    ),

                    sidebar_divider(),

                    # ── Analysis status ────────────────────────────
                    section_header("STATUS"),
                    html.Div(id="sidebar-analysis-status"),

                    # Hidden placeholder for channel selector (moved to Viewer/Browse)
                    html.Div(id="sidebar-channel-selector", style={"display": "none"}),
                ],
            ),
            # Footer
            html.Div(
                id="sidebar-footer",
                children=[
                    html.Span("NED-Net v0.1"),
                    html.Span(" \u00B7 ", style={"opacity": "0.4"}),
                    html.Span("EEG Analysis Platform"),
                ],
            ),
        ],
    )


# ── Tab bar ───────────────────────────────────────────────────────────

# Top-level tabs visible in the tab bar
TOP_TAB_DEFS = [
    ("upload", "Load"),
    ("viewer", "Viewer"),
    ("detection", "Detection"),              # parent — has subtabs
    ("training_grp", "Training"),             # parent — has subtabs
    ("ml_grp", "Dataset / Model"),            # parent — has subtabs
    ("analysis", "Analysis"),                 # unified CNN detection (single/batch/live)
    ("tools_grp", "Tools"),                   # parent — has subtabs
    ("results", "Results"),
    ("settings", "Settings"),
]

# Subtabs for parent tab groups
DETECTION_SUBTABS = [
    ("seizures", "Seizure"),
    ("spikes", "Interictal Spikes"),
]
TRAINING_SUBTABS = [
    ("training", "Seizure"),
    ("training_spikes", "Interictal Spikes"),
]
ML_SUBTABS = [
    ("ml_datasets", "Dataset"),
]
TOOLS_SUBTABS = [
    ("video_converter", "Video Converter"),
    ("adicht_converter", "ADICHT → EDF"),
]

# All routable tab IDs (for render_tab and state)
ALL_TAB_IDS = (
    ["upload", "viewer", "analysis"]
    + [tid for tid, _ in DETECTION_SUBTABS]
    + [tid for tid, _ in TRAINING_SUBTABS]
    + [tid for tid, _ in ML_SUBTABS]
    + [tid for tid, _ in TOOLS_SUBTABS]
    + ["results", "settings"]
)

# Legacy TAB_DEFS kept for any remaining references
TAB_DEFS = [(tid, label) for tid, label in TOP_TAB_DEFS]


def _tab_bar():
    # Split tabs: main tabs on the left, Tools & Settings pushed right
    _LEFT_TABS = [t for t in TOP_TAB_DEFS if t[0] not in ("tools_grp", "settings")]
    _RIGHT_TABS = [t for t in TOP_TAB_DEFS if t[0] in ("tools_grp", "settings")]
    nav_items = [
        dbc.NavLink(label, id=f"tab-{tid}", active=tid == "upload", n_clicks=0)
        for tid, label in _LEFT_TABS
    ]
    nav_items.append(html.Div(style={"flex": "1"}))  # spacer
    nav_items.extend(
        dbc.NavLink(label, id=f"tab-{tid}", active=False, n_clicks=0)
        for tid, label in _RIGHT_TABS
    )
    return html.Div(
        id="tab-bar",
        children=[
            # Main tab row
            dbc.Nav(
                nav_items,
                pills=False,
                className="nav-tabs",
                style={"display": "flex", "flexWrap": "nowrap"},
            ),
            # Subtab row (shown when Detection or Training is active)
            html.Div(
                id="subtab-bar",
                style={"display": "none"},
                children=[
                    dbc.Nav(
                        id="subtab-nav",
                        pills=False,
                        className="nav-tabs subtab-nav",
                    ),
                ],
            ),
        ],
    )


# ── Main layout ───────────────────────────────────────────────────────

app.layout = html.Div(
    id="app-container",
    children=[
        # Session store (lightweight client-side UUID)
        dcc.Store(id="session-id", storage_type="session"),
        # Active tab store
        dcc.Store(id="active-tab", data="upload"),
        # Refresh counter — increment to force tab content re-render
        dcc.Store(id="tab-refresh", data=0),
        # Blinding confirmation modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Disable Blinding?")),
                dbc.ModalBody(
                    "Turning off blinding will reveal animal IDs, treatment "
                    "groups, and other identifying information. This action "
                    "will be logged. Are you sure?"
                ),
                dbc.ModalFooter([
                    dbc.Button(
                        "Keep Blinded",
                        id="blinding-cancel",
                        className="btn-ned-secondary",
                    ),
                    dbc.Button(
                        "Disable Blinding",
                        id="blinding-confirm",
                        className="btn-ned-danger",
                    ),
                ]),
            ],
            id="blinding-modal",
            is_open=False,
            centered=True,
        ),

        # Store for selected channels (list of indices)
        dcc.Store(id="selected-channels", data=None),
        # Auto-save store for all param-input values (survives tab switches)
        dcc.Store(id="store-all-params", data={}),
        # Auto-save store for seizure non-slider components
        dcc.Store(id="store-sz-extras", data={}),
        # Auto-save store for spike non-slider components
        dcc.Store(id="store-sp-extras", data={}),
        # Subtab click relay
        dcc.Store(id="subtab-click", data=""),

        # Sidebar
        _sidebar(),

        # Main content area
        html.Div(
            id="main-content",
            children=[
                _tab_bar(),
                # File selector bar (shown when multiple files loaded)
                html.Div(
                    id="file-selector-bar",
                    style={"display": "none"},
                ),
                html.Div(id="tab-content"),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("session-id", "data"),
    Input("session-id", "data"),
)
def init_session(sid):
    """Ensure a session UUID exists."""
    if sid:
        return sid
    return server_state.create_session()


# ── Tab switching ─────────────────────────────────────────────────────


# Map parent tabs to their default subtab
_PARENT_DEFAULT_SUBTAB = {
    "detection": "seizures",
    "training_grp": "training",
    "ml_grp": "ml_datasets",
    "tools_grp": "video_converter",
}
# Map subtab IDs back to their parent
_SUBTAB_TO_PARENT = {}
for _st_id, _ in DETECTION_SUBTABS:
    _SUBTAB_TO_PARENT[_st_id] = "detection"
for _st_id, _ in TRAINING_SUBTABS:
    _SUBTAB_TO_PARENT[_st_id] = "training_grp"
for _st_id, _ in ML_SUBTABS:
    _SUBTAB_TO_PARENT[_st_id] = "ml_grp"
for _st_id, _ in TOOLS_SUBTABS:
    _SUBTAB_TO_PARENT[_st_id] = "tools_grp"


@callback(
    Output("active-tab", "data"),
    *[Output(f"tab-{tid}", "active") for tid, _ in TOP_TAB_DEFS],
    Output("subtab-bar", "style"),
    Output("subtab-nav", "children"),
    *[Input(f"tab-{tid}", "n_clicks") for tid, _ in TOP_TAB_DEFS],
    Input("subtab-click", "data"),
    State("active-tab", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def switch_tab(*args):
    """Handle tab click — update active tab store, nav link states, and subtabs."""
    n_top = len(TOP_TAB_DEFS)
    subtab_click = args[n_top]     # Input subtab-click
    current = args[n_top + 1]      # State active-tab
    sid = args[n_top + 2]          # State session-id

    triggered = ctx.triggered_id
    if triggered is None:
        parent = _SUBTAB_TO_PARENT.get(current)
        active_flags = tuple(
            tid == (parent or current) for tid, _ in TOP_TAB_DEFS
        )
        subtab_style, subtab_children = _build_subtab_bar(current)
        return (current,) + active_flags + (subtab_style, subtab_children)

    # Subtab click
    if triggered == "subtab-click" and subtab_click:
        new_tab = subtab_click
        parent = _SUBTAB_TO_PARENT.get(new_tab, new_tab)
        # Remember last subtab for this parent
        state = server_state.get_session(sid) if sid else None
        if state:
            state.extra[f"last_subtab_{parent}"] = new_tab
        active_flags = tuple(tid == parent for tid, _ in TOP_TAB_DEFS)
        subtab_style, subtab_children = _build_subtab_bar(new_tab)
        return (new_tab,) + active_flags + (subtab_style, subtab_children)

    # Top-level tab click
    new_tab = triggered.replace("tab-", "")
    # If it's a parent tab, route to last-used subtab or default
    if new_tab in _PARENT_DEFAULT_SUBTAB:
        state = server_state.get_session(sid) if sid else None
        last = (state.extra.get(f"last_subtab_{new_tab}")
                if state else None)
        actual_tab = last or _PARENT_DEFAULT_SUBTAB[new_tab]
        active_flags = tuple(tid == new_tab for tid, _ in TOP_TAB_DEFS)
        subtab_style, subtab_children = _build_subtab_bar(actual_tab)
        return (actual_tab,) + active_flags + (subtab_style, subtab_children)

    # Simple (non-parent) tab
    active_flags = tuple(tid == new_tab for tid, _ in TOP_TAB_DEFS)
    return (new_tab,) + active_flags + ({"display": "none"}, [])


def _build_subtab_bar(active_subtab: str):
    """Return (style, children) for the subtab bar."""
    parent = _SUBTAB_TO_PARENT.get(active_subtab)
    if parent == "detection":
        subtabs = DETECTION_SUBTABS
    elif parent == "training_grp":
        subtabs = TRAINING_SUBTABS
    elif parent == "ml_grp":
        subtabs = ML_SUBTABS
    elif parent == "tools_grp":
        subtabs = TOOLS_SUBTABS
    else:
        return {"display": "none"}, []

    children = [
        dbc.NavLink(
            label,
            id={"type": "subtab-link", "index": tid},
            active=tid == active_subtab,
            n_clicks=0,
        )
        for tid, label in subtabs
    ]
    return {"display": "block"}, children


# Relay subtab NavLink clicks into the subtab-click store
@callback(
    Output("subtab-click", "data"),
    Input({"type": "subtab-link", "index": ALL}, "n_clicks"),
    State({"type": "subtab-link", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def relay_subtab_click(all_clicks, all_ids):
    """When a subtab link is clicked, write its ID to the subtab-click store."""
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered["index"]
    return no_update


@callback(
    Output("tab-content", "children"),
    Input("active-tab", "data"),
    Input("tab-refresh", "data"),
    State("session-id", "data"),
)
def render_tab(active_tab, _refresh, sid):
    """Render the content for the active tab."""
    if active_tab == "upload":
        return upload.layout(sid)
    elif active_tab == "viewer":
        return viewer.layout(sid)
    elif active_tab == "seizures":
        return seizures.layout(sid)
    elif active_tab == "spikes":
        return spikes.layout(sid)
    elif active_tab == "training":
        return training.layout(sid)
    elif active_tab == "training_spikes":
        return training_spikes.layout(sid)
    elif active_tab == "video_converter":
        return tools.layout(sid)
    elif active_tab == "adicht_converter":
        return adicht_converter.layout(sid)
    elif active_tab == "ml_datasets":
        return ml_datasets.layout(sid)
    elif active_tab == "analysis":
        return analysis_page.layout(sid)
    elif active_tab == "ml_detection":
        return ml_detection.layout(sid)
    elif active_tab == "ml_results":
        return ml_results.layout(sid)
    elif active_tab == "results":
        return results_page.layout(sid)
    elif active_tab == "settings":
        return _placeholder_tab(
            "Settings",
            "Application settings (dark mode, batch processing, defaults) "
            "will be configured here.",
        )
    return html.Div("Unknown tab")


def _placeholder_tab(title: str, description: str):
    return html.Div(
        id="tab-content-inner",
        style={"padding": "24px"},
        children=[
            html.H4(title, style={"marginBottom": "16px"}),
            html.Div(
                className="empty-state",
                children=[
                    html.Div("\u2692", className="empty-icon"),
                    html.Div("Coming Soon", className="empty-title"),
                    html.Div(description, className="empty-text"),
                ],
            ),
        ],
    )


# ── Blinding toggle with confirmation ─────────────────────────────────


@callback(
    Output("blinding-modal", "is_open"),
    Output("blinding-toggle", "value"),
    Input("blinding-toggle", "value"),
    Input("blinding-confirm", "n_clicks"),
    Input("blinding-cancel", "n_clicks"),
    State("blinding-modal", "is_open"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_blinding(toggle_val, confirm_clicks, cancel_clicks, modal_open, sid):
    """Intercept blinding toggle — require confirmation to unblind."""
    trigger = ctx.triggered_id

    if trigger == "blinding-toggle":
        if not toggle_val:
            # User is trying to turn OFF blinding -> show modal, revert toggle
            return True, True
        else:
            # Turning ON is always allowed
            state = server_state.get_session(sid)
            state.blinding_on = True
            return False, True

    if trigger == "blinding-confirm":
        # User confirmed unblinding
        state = server_state.get_session(sid)
        state.blinding_on = False
        from datetime import datetime
        state.blinding_log.append({
            "action": "unblinded",
            "timestamp": datetime.now().isoformat(),
        })
        return False, False

    if trigger == "blinding-cancel":
        # User cancelled — keep blinding on
        return False, True

    return no_update, no_update


@callback(
    Output("blinding-badge-container", "children"),
    Input("blinding-toggle", "value"),
)
def update_blinding_badge(is_on):
    """Update the blinding badge in the sidebar."""
    return blinding_badge(is_on)


# ── Sidebar file info & channel selector ─────────────────────────────


@callback(
    Output("sidebar-file-info", "children"),
    Output("sidebar-analysis-status", "children"),
    Output("sidebar-channel-selector", "children"),
    Output("selected-channels", "data"),
    Input("tab-refresh", "data"),
    Input("active-tab", "data"),
    State("session-id", "data"),
    State("selected-channels", "data"),
)
def update_sidebar_info(_refresh, _tab, sid, current_selected):
    """Update sidebar recording info, analysis status, and channel toggles."""
    state = server_state.get_session(sid)
    rec = state.recording

    _muted = {"fontSize": "0.75rem", "color": "#8b949e"}
    _dim = {"fontSize": "0.75rem", "color": "#484f58"}

    if rec is None:
        return (
            html.Div("No file loaded", className="file-info",
                     style={"opacity": "0.5"}),
            html.Div("—", style=_dim),
            html.Div(),
            None,
        )

    # File info
    import os
    fname = os.path.basename(rec.source_path) if rec.source_path else "Unknown"
    dur_h = rec.duration_sec / 3600

    # Build channel list with pairings
    pairings = getattr(state, "channel_pairings", None) or []
    act_rec = state.activity_recordings.get("paired") if hasattr(state, "activity_recordings") else None
    channel_items = []
    paired_eeg_indices = set()

    if pairings:
        for p in pairings:
            eeg_label = rec.channel_names[p.eeg_index] if p.eeg_index < len(rec.channel_names) else f"Ch{p.eeg_index}"
            paired_eeg_indices.add(p.eeg_index)
            if p.activity_index is not None:
                act_label = p.activity_label or f"Act{p.activity_index}"
                channel_items.append(html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "4px",
                           "padding": "1px 0"},
                    children=[
                        html.Span(eeg_label,
                                  style={"fontSize": "0.72rem", "color": "#58a6ff",
                                         "fontWeight": "500"}),
                        html.Span("\u2194",
                                  style={"fontSize": "0.7rem", "color": "#484f58"}),
                        html.Span(act_label,
                                  style={"fontSize": "0.72rem", "color": "#3fb950"}),
                    ],
                ))
            else:
                channel_items.append(html.Div(
                    eeg_label,
                    style={"fontSize": "0.72rem", "color": "#58a6ff", "padding": "1px 0"},
                ))

    # Add any EEG channels not in pairings
    for i in range(rec.n_channels):
        if i not in paired_eeg_indices:
            ch_name = rec.channel_names[i] if i < len(rec.channel_names) else f"Ch{i}"
            channel_items.append(html.Div(
                ch_name,
                style={"fontSize": "0.72rem", "color": "#8b949e", "padding": "1px 0"},
            ))

    file_info = html.Div([
        html.Div(fname, className="file-info",
                 style={"fontWeight": "600", "fontSize": "0.82rem",
                        "wordBreak": "break-all"}),
        html.Div(
            f"{rec.n_channels} ch \u00B7 {rec.fs:.0f} Hz \u00B7 "
            f"{rec.duration_sec:.0f}s ({dur_h:.1f}h)",
            className="file-info",
            style={"fontSize": "0.78rem", "color": "#8b949e", "marginTop": "4px"},
        ),
        html.Div(
            channel_items,
            style={"marginTop": "6px", "display": "flex", "flexDirection": "column",
                   "gap": "1px"},
        ) if channel_items else html.Div(),
    ])

    # ── Analysis status ───────────────────────────────────────────────
    status_items = []

    # Detection file
    has_det_file = False
    n_det = 0
    if rec.source_path:
        det_path = detection_json_path(rec.source_path)
        has_det_file = det_path.is_file()
    # Also count in-memory detections
    if state.seizure_events:
        n_det = len(state.seizure_events)
    elif state.detected_events:
        n_det = len([e for e in state.detected_events if e.event_type == "seizure"])

    if has_det_file or n_det > 0:
        det_icon = "\u2705" if has_det_file else "\u26A0\uFE0F"
        det_label = f"{det_icon} Detections: {n_det} seizures"
        if has_det_file:
            det_label += " (saved)"
        status_items.append(html.Div(det_label, style=_muted))
    else:
        status_items.append(html.Div("\u2B55 No detections yet", style=_dim))

    # Interictal spikes
    has_sp_file = False
    n_sp = 0
    if rec.source_path:
        sp_path = spike_detection_json_path(rec.source_path)
        has_sp_file = sp_path.is_file()
    if state.spike_events:
        n_sp = len(state.spike_events)

    if has_sp_file or n_sp > 0:
        sp_icon = "\u2705" if has_sp_file else "\u26A0\uFE0F"
        sp_label = f"{sp_icon} IS detections: {n_sp} spikes"
        if has_sp_file:
            sp_label += " (saved)"
        status_items.append(html.Div(sp_label, style=_muted))
    else:
        status_items.append(html.Div("\u2B55 No IS detections yet", style=_dim))

    # Annotation file
    has_ann_file = False
    ann_counts = {"confirmed": 0, "rejected": 0, "pending": 0, "manual": 0}
    if rec.source_path:
        ann_path = annotation_json_path(rec.source_path)
        has_ann_file = ann_path.is_file()
    # Count from in-memory annotations
    tr_anns = state.extra.get("tr_annotations", [])
    if tr_anns:
        for a in tr_anns:
            lbl = a.get("label", "pending") if isinstance(a, dict) else getattr(a, "label", "pending")
            src = a.get("source", "") if isinstance(a, dict) else getattr(a, "source", "")
            if src == "manual":
                ann_counts["manual"] += 1
            elif lbl in ann_counts:
                ann_counts[lbl] += 1
        total_ann = sum(ann_counts.values())
        reviewed = ann_counts["confirmed"] + ann_counts["rejected"]
        pct = int(100 * reviewed / total_ann) if total_ann else 0
        ann_icon = "\u2705" if pct == 100 else "\U0001F4DD"
        status_items.append(html.Div(
            f"{ann_icon} Annotations: {reviewed}/{total_ann} reviewed ({pct}%)",
            style=_muted,
        ))
        # Breakdown
        parts = []
        if ann_counts["confirmed"]:
            parts.append(f'{ann_counts["confirmed"]}✓')
        if ann_counts["rejected"]:
            parts.append(f'{ann_counts["rejected"]}✗')
        if ann_counts["pending"]:
            parts.append(f'{ann_counts["pending"]}?')
        if ann_counts["manual"]:
            parts.append(f'{ann_counts["manual"]}+')
        if parts:
            status_items.append(html.Div(
                "  ".join(parts),
                style={"fontSize": "0.72rem", "color": "#6e7681",
                       "marginLeft": "20px"},
            ))
    elif has_ann_file:
        status_items.append(html.Div("\U0001F4DD Annotations: saved to disk", style=_muted))
    else:
        status_items.append(html.Div("\u2B55 No annotations yet", style=_dim))

    # Video file
    video_path = state.extra.get("video_path")
    if video_path:
        vname = os.path.basename(video_path)
        status_items.append(html.Div(f"\u25B6 Video: {vname}", style=_muted))
    else:
        status_items.append(html.Div("\u2B55 No video file", style=_dim))

    # Hidden placeholder — "Recall Detection Params" moved inline to detection pages
    status_items.append(html.Div(
        dbc.Button(
            id="sidebar-recall-det-params",
            style={"display": "none"},
        ),
    ))

    # Flash message (set by recall_detection_params, consumed once)
    flash = state.extra.pop("sidebar_flash", None)
    if flash:
        status_items.append(html.Div(
            flash,
            style={"fontSize": "0.75rem", "color": "#3fb950", "marginTop": "4px"},
        ))

    analysis_status = html.Div(
        status_items,
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )

    # Channel selector is now in Viewer/Browse tabs — keep hidden placeholder
    all_indices = list(range(rec.n_channels))
    if current_selected is None:
        selected = all_indices
    else:
        selected = current_selected

    # Hidden checklist — still needed so the callback IDs exist
    channel_selector = html.Div([
        dbc.Checklist(
            id="sidebar-channel-checks",
            options=[{"label": rec.channel_names[i], "value": i}
                     for i in range(rec.n_channels)],
            value=selected,
            style={"display": "none"},
        ),
        html.A(id="ch-select-all", style={"display": "none"}),
        html.A(id="ch-select-none", style={"display": "none"}),
    ], style={"display": "none"})

    return file_info, analysis_status, channel_selector, selected


@callback(
    Output("selected-channels", "data", allow_duplicate=True),
    Input("sidebar-channel-checks", "value"),
    prevent_initial_call=True,
)
def on_channel_toggle(checked):
    """Store selected channels when user toggles checkboxes."""
    return checked


# ── File selector bar (multi-file project) ─────────────────────────


@callback(
    Output("file-selector-bar", "children"),
    Output("file-selector-bar", "style"),
    Input("tab-refresh", "data"),
    State("session-id", "data"),
)
def update_file_selector_bar(_refresh, sid):
    """Show file selector bar when a multi-file project is loaded."""
    state = server_state.get_session(sid)
    project_files = state.extra.get("project_files")
    if not project_files or len(project_files) < 2:
        return [], {"display": "none"}

    active_idx = state.extra.get("project_active_idx", 0)
    n_files = len(project_files)
    folder = state.extra.get("project_folder", "")

    # Build dropdown options with live status icons
    options = []
    for i, pf in enumerate(project_files):
        fname = pf["filename"]
        edf_path = pf["edf_path"]
        stem = os.path.splitext(fname)[0]
        d = os.path.dirname(edf_path)

        # Check live status from disk
        has_det = os.path.isfile(os.path.join(d, stem + "_ned_detections.json"))
        has_ann = os.path.isfile(os.path.join(d, stem + "_ned_annotations.json"))

        parts = []
        if has_det:
            parts.append("\u2705")
        if has_ann:
            parts.append("\U0001F4DD")
        status = " ".join(parts)
        label = f"{fname}  {status}" if status else fname
        options.append({"label": label, "value": i})

    bar = html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "12px",
            "padding": "6px 16px",
            "backgroundColor": "#161b22",
            "borderBottom": "1px solid #30363d",
            "fontSize": "0.82rem",
        },
        children=[
            html.Span(
                f"\U0001F4C1 {n_files} files",
                style={"color": "#8b949e", "whiteSpace": "nowrap"},
            ),
            dcc.Dropdown(
                id="file-bar-dropdown",
                options=options,
                value=active_idx,
                clearable=False,
                style={"width": "360px", "fontSize": "0.82rem"},
            ),
            html.Span(
                os.path.basename(folder),
                style={"color": "#484f58", "fontSize": "0.75rem",
                       "marginLeft": "auto"},
            ),
        ],
    )

    return [bar], {"display": "block"}


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("file-bar-dropdown", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def file_bar_switch(file_idx, sid, refresh):
    """Switch file from the file selector bar dropdown."""
    if file_idx is None:
        return no_update

    state = server_state.get_session(sid)
    project_files = state.extra.get("project_files", [])
    if not project_files:
        return no_update

    file_idx = int(file_idx)
    if file_idx == state.extra.get("project_active_idx"):
        return no_update

    if file_idx < 0 or file_idx >= len(project_files):
        return no_update

    entry = project_files[file_idx]

    # Clear current recording state
    state.recording = None
    state.all_channels_info = []
    state.activity_recordings = {}
    state.channel_pairings = []
    state.seizure_events = []
    state.spike_events = []
    state.detected_events = []
    state.st_detection_info = {}
    state.sp_detection_info = {}
    state.extra.pop("tr_annotations", None)
    state.extra.pop("tr_current_idx", None)
    state.extra.pop("sz_selected_event_key", None)
    state.extra.pop("video_path", None)

    # Load the new file
    from eeg_seizure_analyzer.dash_app.pages.upload import _load_edf_into_state
    try:
        _load_edf_into_state(state, entry["edf_path"])
    except Exception:
        import traceback
        traceback.print_exc()
        return no_update

    state.extra["project_active_idx"] = file_idx

    return (refresh or 0) + 1


@callback(
    Output("sidebar-channel-checks", "value"),
    Input("ch-select-all", "n_clicks"),
    Input("ch-select-none", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_select_all_none(all_clicks, none_clicks, sid):
    """Handle Select All / None links."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update
    if trigger == "ch-select-all":
        return list(range(state.recording.n_channels))
    if trigger == "ch-select-none":
        return []
    return no_update


# ── Recall detection params ───────────────────────────────────────────


@callback(
    Output("sidebar-analysis-status", "children", allow_duplicate=True),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("sidebar-recall-det-params", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def recall_detection_params(n_clicks, sid, refresh):
    """Load saved detection parameters and filters from disk for both Seizure and IS tabs."""
    if not n_clicks:
        return no_update, no_update
    state = server_state.get_session(sid)
    rec = state.recording
    if rec is None or not rec.source_path:
        return html.Div(
            "⚠️ No file loaded.",
            style={"fontSize": "0.75rem", "color": "#d29922", "marginTop": "4px"},
        ), no_update

    from eeg_seizure_analyzer.io.persistence import load_detections, load_spike_detections

    parts = []

    # ── Seizure detections ──────────────────────────────────────
    result = load_detections(rec.source_path)
    n_sz_params = 0
    n_sz_filters = 0
    if result is not None:
        saved_params = result.get("params", {})
        if saved_params:
            state.extra["sz_param_overrides"] = dict(saved_params)
            state.extra["sz_params"] = dict(saved_params)
            if "sz-bl-method" in saved_params:
                state.extra["sz_bl_method"] = saved_params["sz-bl-method"]
            if "sz-bnd-method" in saved_params:
                state.extra["sz_bnd_method"] = saved_params["sz-bnd-method"]
            n_sz_params = len(saved_params)

        saved_channels = result.get("channels", [])
        if saved_channels:
            state.extra["sz_selected_channels"] = saved_channels

        fs = result.get("filter_settings", {})
        if fs:
            filter_on = fs.get("filter_enabled", True)
            filter_vals = fs.get("filter_values", {})
            state.extra["sz_filter_enabled"] = filter_on
            if filter_vals:
                state.extra["sz_filter_values"] = filter_vals
            state.extra["tr_filter_on"] = filter_on
            if filter_vals:
                state.extra["tr_min_conf"] = filter_vals.get("min_conf", 0)
                state.extra["tr_min_dur"] = filter_vals.get("min_dur", 0)
                state.extra["tr_min_lbl"] = filter_vals.get("min_lbl", 0)
                state.extra["tr_max_conf"] = filter_vals.get("max_conf", None)
                state.extra["tr_max_dur"] = filter_vals.get("max_dur", None)
                state.extra["tr_max_lbl"] = filter_vals.get("max_lbl", None)
            n_sz_filters = len(filter_vals)

        parts.append(f"Sz: {n_sz_params} params"
                     f"{f' + {n_sz_filters} filters' if n_sz_filters else ''}")

    # ── Interictal spike detections ─────────────────────────────
    sp_result = load_spike_detections(rec.source_path)
    n_sp_params = 0
    n_sp_filters = 0
    if sp_result is not None:
        sp_saved_params = sp_result.get("params", {})
        if sp_saved_params:
            state.extra["sp_param_overrides"] = dict(sp_saved_params)
            state.extra["sp_params"] = dict(sp_saved_params)
            if "sp-bl-method" in sp_saved_params:
                state.extra["sp_bl_method"] = sp_saved_params["sp-bl-method"]
            n_sp_params = len(sp_saved_params)

        sp_channels = sp_result.get("channels", [])
        if sp_channels:
            state.extra["sp_selected_channels"] = sp_channels

        sp_fs = sp_result.get("filter_settings", {})
        if sp_fs:
            sp_filter_on = sp_fs.get("filter_enabled", True)
            sp_filter_vals = sp_fs.get("filter_values", {})
            sp_filter_vals.pop("channel", None)  # channel filter not persisted
            state.extra["sp_filter_enabled"] = sp_filter_on
            if sp_filter_vals:
                state.extra["sp_filter_values"] = sp_filter_vals
            n_sp_filters = len(sp_filter_vals)

        parts.append(f"IS: {n_sp_params} params"
                     f"{f' + {n_sp_filters} filters' if n_sp_filters else ''}")

    if not parts:
        return html.Div(
            "⚠️ No saved detections found on disk.",
            style={"fontSize": "0.75rem", "color": "#d29922", "marginTop": "4px"},
        ), no_update

    state.extra["sidebar_flash"] = f"✅ Recalled: {'; '.join(parts)}."
    return no_update, (refresh or 0) + 1


# ── Auto-save ALL param-input values on change ──────────────────────
# Uses ALL pattern (not MATCH) so it doesn't conflict with the
# clientside slider/input sync callback.  Fires whenever ANY
# param-input component changes, saves to server state by prefix.


@callback(
    Output("store-all-params", "data"),
    Input({"type": "param-input", "key": ALL}, "value"),
    State({"type": "param-input", "key": ALL}, "id"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def auto_save_params(all_values, all_ids, sid):
    """Auto-save all param-input values to server state on any change."""
    if not all_ids or not sid:
        return no_update
    state = server_state.get_session(sid)
    sz_params = {}
    sp_params = {}
    for id_dict, val in zip(all_ids, all_values):
        key = id_dict["key"]
        if val is not None:
            if key.startswith("sz-"):
                sz_params[key] = val
            elif key.startswith("sp-"):
                sp_params[key] = val
    if sz_params:
        existing_sz = state.extra.get("sz_params", {})
        existing_sz.update(sz_params)
        state.extra["sz_params"] = existing_sz
    if sp_params:
        existing_sp = state.extra.get("sp_params", {})
        existing_sp.update(sp_params)
        state.extra["sp_params"] = existing_sp
    return {"sz": sz_params, "sp": sp_params}


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050)
