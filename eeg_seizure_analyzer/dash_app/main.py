"""NED-Net Dash application entry point.

Run with:  python -m eeg_seizure_analyzer.dash_app.main
"""

from __future__ import annotations

from eeg_seizure_analyzer import __version__
from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    blinding_badge,
    section_header,
    set_plotly_theme,
    sidebar_divider,
)
from eeg_seizure_analyzer.io.persistence import detection_json_path, spike_detection_json_path
from eeg_seizure_analyzer.io.annotation_store import annotation_json_path

# ── Import tab modules ────────────────────────────────────────────────

from eeg_seizure_analyzer.dash_app.pages import upload, viewer, seizures, spikes, training, training_spikes, training_background, tools, adicht_converter, ml_datasets, analysis as analysis_page, results as results_page
from eeg_seizure_analyzer.dash_app.pages import blind_review

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
    update_title=None,
)

server = app.server

# ── Vendored front-end libraries (outside assets/, which Dash would inject
# into every page). The manual page loads marked from here, so Help also
# works offline. marked v12.0.2 (MIT), sha256 15fabce5…a894.
import os
from flask import send_from_directory

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


@server.route("/vendor/<path:filename>")
def _vendor_file(filename):
    return send_from_directory(_VENDOR_DIR, filename)


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

# Viewer: "Sync to EEG" jumps the video to the current Start time (the button
# had no click handler; only the automatic seek above existed).
clientside_callback(
    """
    function(n, startSec) {
        var video = document.getElementById('viewer-video-player');
        if (n && video && startSec != null) {
            video.currentTime = startSec;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("viewer-video-sync", "title"),
    Input("viewer-video-sync", "n_clicks"),
    State("viewer-start-input", "value"),
    prevent_initial_call=True,
)


# ── Navigation ────────────────────────────────────────────────────────

# Sidebar sections: (group label, [(tab id, label, icon name)]).
# Icons are CSS masks defined in assets/style.css (.icon-<name>).
NAV_GROUPS = [
    ("Recording", [
        ("upload", "Load", "folder"),
        ("viewer", "Viewer", "wave"),
    ]),
    ("Detection", [
        ("seizures", "Seizures", "bolt"),
        ("spikes", "Interictal Spikes", "spike"),
    ]),
    ("Training", [
        ("training", "Seizure Annotation", "tag"),
        ("training_spikes", "Spike Annotation", "check"),
        ("training_background", "Background", "stripes"),
    ]),
    ("Model", [
        ("ml_datasets", "Dataset & Model", "database"),
        ("blind_review", "Blind Review", "eye"),
        ("analysis", "Batch Analysis", "layers"),
        ("results", "Results", "chart"),
    ]),
    ("Tools", [
        ("video_converter", "Video Converter", "film"),
        ("adicht_converter", "ADICHT → EDF", "convert"),
    ]),
]

# Every routable tab, in sidebar order
NAV_IDS = [tid for _, items in NAV_GROUPS for tid, _, _ in items]
ALL_TAB_IDS = list(NAV_IDS)


def _nav_item(tid: str, label: str, icon: str, active: bool = False):
    return html.Button(
        [
            html.Span(className=f"nav-icon icon-{icon}"),
            html.Span(label, className="nav-label"),
        ],
        id=f"section-{tid}",
        n_clicks=0,
        className="nav-item active" if active else "nav-item",
    )


# ── Sidebar layout ────────────────────────────────────────────────────


def _sidebar():
    nav_children = []
    for group, items in NAV_GROUPS:
        nav_children.append(html.Div(group, className="nav-group-label"))
        nav_children.extend(
            _nav_item(tid, label, icon, active=(tid == "upload"))
            for tid, label, icon in items
        )

    return html.Div(
        id="sidebar",
        children=[
            # Brand (same height as the toolbar beside it)
            html.Div(
                id="sidebar-brand",
                children=[html.Button("NED-Net", id="brand-home",
                                      className="brand-name",
                                      title="Start page")],
            ),
            html.Nav(nav_children, id="sidebar-nav",
                     **{"aria-label": "Sections"}),

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
                    ),
                ],
            ),
            # Hidden placeholder for channel selector (moved to Viewer/Browse)
            html.Div(id="sidebar-channel-selector", style={"display": "none"}),

            # Footer
            html.Div(
                id="sidebar-footer",
                children=[
                    dbc.Switch(
                        id="theme-toggle",
                        label="Dark mode",
                        value=False,
                        className="theme-switch",
                    ),
                    html.Div(
                        className="sidebar-version",
                        children=[
                            html.Span(f"NED-Net {__version__}"),
                            html.A("Help", href="/assets/manual.html",
                                   target="_blank", className="footer-link"),
                        ],
                    ),
                ],
            ),
        ],
    )


# ── Toolbar and status bar ────────────────────────────────────────────


def _title_block(title: str, subtitle: str, tooltip: str | None = None):
    """Toolbar title: recording name over a one-line summary."""
    return [
        html.Div(title, className="toolbar-title"),
        html.Div(subtitle, className="toolbar-subtitle", title=tooltip),
    ]


def _toolbar():
    return html.Div(
        id="app-toolbar",
        children=[
            html.Div(
                id="sidebar-file-info",
                className="toolbar-title-block",
                children=_title_block("NED-Net", "No recording loaded"),
            ),
            html.Div(className="toolbar-spacer"),
            # File selector (shown when several files are loaded)
            html.Div(
                id="file-selector-bar",
                className="file-selector-bar",
                style={"display": "none"},
            ),
            # Show / hide the settings panel of two-pane pages
            # (handled in assets/app_shell.js; hidden on one-pane pages)
            html.Button(
                html.Span(className="btn-icon icon-panel-right"),
                id="panel-toggle",
                className="toolbar-icon-btn active",
                title="Hide the settings panel",
            ),
        ],
    )


def _status_bar():
    return html.Div(
        id="status-bar",
        children=[html.Div(id="sidebar-analysis-status", className="status-items")],
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
        # Theme store (persists across tab switches)
        dcc.Store(id="theme-store", data="light"),

        # Sidebar
        _sidebar(),

        # Main content area
        html.Div(
            id="main-content",
            children=[
                _toolbar(),
                html.Div(id="tab-content"),
            ],
        ),

        # Status bar along the bottom
        _status_bar(),
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


# ── Theme toggle ─────────────────────────────────────────────────────

# Clientside: flip data-theme on <html> immediately (no server round-trip)
app.clientside_callback(
    """
    function(isDark) {
        var theme = isDark ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", theme);
        return theme;
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle", "value"),
)


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("theme-store", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_theme_change(theme, refresh):
    """Re-render tab content when theme changes so Plotly figures update."""
    set_plotly_theme(theme or "light")  # the app starts light
    return (refresh or 0) + 1


# ── Tab switching ─────────────────────────────────────────────────────


@callback(
    Output("active-tab", "data"),
    *[Output(f"section-{tid}", "className") for tid in NAV_IDS],
    *[Input(f"section-{tid}", "n_clicks") for tid in NAV_IDS],
    prevent_initial_call=True,
)
def switch_tab(*_clicks):
    """Sidebar click: make that section active and highlight it."""
    triggered = ctx.triggered_id
    if not isinstance(triggered, str) or not triggered.startswith("section-"):
        return (no_update,) * (1 + len(NAV_IDS))
    new_tab = triggered[len("section-"):]
    classes = tuple(
        "nav-item active" if tid == new_tab else "nav-item" for tid in NAV_IDS
    )
    return (new_tab,) + classes



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
    elif active_tab == "training_background":
        return training_background.layout(sid)
    elif active_tab == "video_converter":
        return tools.layout(sid)
    elif active_tab == "adicht_converter":
        return adicht_converter.layout(sid)
    elif active_tab == "ml_datasets":
        return ml_datasets.layout(sid)
    elif active_tab == "blind_review":
        return blind_review.layout(sid)
    elif active_tab == "analysis":
        return analysis_page.layout(sid)
    elif active_tab == "results":
        return results_page.layout(sid)
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


def _status(text, kind: str = "none", tooltip: str | None = None):
    """One status-bar item: coloured dot + short text.

    kind: "ok" (saved / complete), "warn" (not saved), "info", "none" (absent).
    """
    return html.Span(
        [html.Span(className=f"status-dot {kind}"), html.Span(text)],
        className=f"status-item {kind}",
        title=tooltip,
    )


def _fmt_duration(sec: float) -> str:
    """5400 -> '5400 s (1 h 30 min)'."""
    total_min = int(round(sec / 60))
    h, m = divmod(total_min, 60)
    human = f"{h} h {m} min" if h else f"{m} min"
    return f"{sec:.0f} s ({human})"


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
    """Update the toolbar title (recording) and the status bar."""
    state = server_state.get_session(sid)
    rec = state.recording

    if rec is None:
        return (
            _title_block("NED-Net", "No recording loaded"),
            [_status("Ready", "none")],
            html.Div(),
            None,
        )

    # ── Toolbar title: file name + one-line summary ──────────────────
    import os
    fname = os.path.basename(rec.source_path) if rec.source_path else "Unknown"

    # Channel pairings go in the subtitle's tooltip
    pairings = getattr(state, "channel_pairings", None) or []
    pair_lines = []
    for p in pairings:
        eeg_label = (rec.channel_names[p.eeg_index]
                     if p.eeg_index < len(rec.channel_names) else f"Ch{p.eeg_index}")
        if p.activity_index is not None:
            act_label = p.activity_label or f"Act{p.activity_index}"
            pair_lines.append(f"{eeg_label} ↔ {act_label}")
        else:
            pair_lines.append(eeg_label)

    n_act = sum(1 for p in pairings if p.activity_index is not None)
    summary = (f"{rec.n_channels} ch · {rec.fs:.0f} Hz · "
               f"{_fmt_duration(rec.duration_sec)}")
    if n_act:
        summary += f" · {n_act} activity channel{'s' if n_act != 1 else ''}"
    file_info = _title_block(fname, summary,
                             tooltip="\n".join(pair_lines) or None)

    # ── Status bar ────────────────────────────────────────────────────
    status_items = []

    # Seizure detections
    has_det_file = False
    n_det = 0
    seizure_list = []
    if rec.source_path:
        has_det_file = detection_json_path(rec.source_path).is_file()
    if state.seizure_events:
        seizure_list = state.seizure_events
        n_det = len(seizure_list)
    elif state.detected_events:
        seizure_list = [e for e in state.detected_events if e.event_type == "seizure"]
        n_det = len(seizure_list)

    if has_det_file or n_det > 0:
        _mlabels = {
            "spike_train": "Spike-Train",
            "spectral_band": "Spectral Band",
            "autocorrelation": "Autocorrelation",
            "ensemble": "Ensemble",
            "ml_unet": "ML U-Net",
            "line_length_energy": "LL / Energy",
        }
        method_counts: dict[str, int] = {}
        for ev in seizure_list:
            feat = ev.features if hasattr(ev, "features") else {}
            m = (feat or {}).get("detection_method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1
        breakdown = ", ".join(
            f"{_mlabels.get(m, m)} {c}" for m, c in sorted(method_counts.items())
            if m != "unknown"
        )
        text = f"{n_det} seizure{'s' if n_det != 1 else ''}"
        text += " saved" if has_det_file else " not saved"
        status_items.append(_status(text, "ok" if has_det_file else "warn",
                                    tooltip=breakdown or None))
    else:
        status_items.append(_status("No seizure detections", "none"))

    # Interictal spikes
    has_sp_file = False
    n_sp = len(state.spike_events) if state.spike_events else 0
    if rec.source_path:
        has_sp_file = spike_detection_json_path(rec.source_path).is_file()
    if has_sp_file or n_sp > 0:
        text = f"{n_sp} IS spike{'s' if n_sp != 1 else ''}"
        text += " saved" if has_sp_file else " not saved"
        status_items.append(_status(text, "ok" if has_sp_file else "warn"))
    else:
        status_items.append(_status("No IS detections", "none"))

    # Annotations
    has_ann_file = False
    if rec.source_path:
        has_ann_file = annotation_json_path(rec.source_path).is_file()
    tr_anns = state.extra.get("tr_annotations", [])
    if tr_anns:
        ann_counts = {"confirmed": 0, "rejected": 0, "pending": 0, "manual": 0}
        for a in tr_anns:
            lbl = a.get("label", "pending") if isinstance(a, dict) else getattr(a, "label", "pending")
            src = a.get("source", "") if isinstance(a, dict) else getattr(a, "source", "")
            if src == "manual":
                ann_counts["manual"] += 1
            elif lbl in ann_counts:
                ann_counts[lbl] += 1
        total_ann = sum(ann_counts.values())
        reviewed = ann_counts["confirmed"] + ann_counts["rejected"]
        detail = (f"{ann_counts['confirmed']} confirmed, {ann_counts['rejected']} rejected, "
                  f"{ann_counts['pending']} pending, {ann_counts['manual']} manual")
        status_items.append(_status(
            f"Annotations {reviewed}/{total_ann} reviewed",
            "ok" if total_ann and reviewed == total_ann else "info",
            tooltip=detail,
        ))
    elif has_ann_file:
        status_items.append(_status("Annotations saved", "ok"))
    else:
        status_items.append(_status("No annotations", "none"))

    # Background stretches (own file, Training -> Background)
    try:
        bg_counts = training_background.background_counts(state)
    except Exception:
        bg_counts = None
    if bg_counts and bg_counts["total"]:
        bg_reviewed = bg_counts["confirmed"] + bg_counts["rejected"]
        status_items.append(_status(
            f"Background {bg_reviewed}/{bg_counts['total']} reviewed",
            "ok" if bg_reviewed == bg_counts["total"] else "info",
            tooltip=(f"{bg_counts['confirmed']} confirmed, {bg_counts['rejected']} rejected, "
                     f"{bg_counts['pending']} pending"),
        ))

    # Video file
    video_path = state.extra.get("video_path")
    if video_path:
        status_items.append(_status(f"Video: {os.path.basename(video_path)}", "info"))
    else:
        status_items.append(_status("No video", "none"))

    # Hidden placeholder — "Recall Detection Params" moved inline to detection pages
    status_items.append(html.Span(
        dbc.Button(id="sidebar-recall-det-params", style={"display": "none"}),
        style={"display": "none"},
    ))

    # Flash message (set by recall_detection_params, consumed once)
    flash = state.extra.pop("sidebar_flash", None)
    if flash:
        status_items.append(html.Span(flash, className="status-item flash"))

    # Channel selector is now in Viewer/Browse tabs — keep hidden placeholder
    all_indices = list(range(rec.n_channels))
    selected = all_indices if current_selected is None else current_selected

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

    return file_info, status_items, channel_selector, selected


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
            parts.append("detections")
        if has_ann:
            parts.append("annotations")
        label = f"{fname} \u2014 {', '.join(parts)}" if parts else fname
        options.append({"label": label, "value": i})

    bar_children = [
        html.Span(
            f"{n_files} files",
            className="file-selector-count",
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
            className="file-selector-folder",
        ),
    ]

    return bar_children, {"display": "flex"}


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
    for _k in ("trb_annotations", "trb_current_idx", "trb_view", "trb_browse_start",
               "trb_cache", "trb_sz_cache", "trb_for"):
        state.extra.pop(_k, None)
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
            "No file loaded.",
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
            "No saved detections found on disk.",
            style={"fontSize": "0.75rem", "color": "#d29922", "marginTop": "4px"},
        ), no_update

    state.extra["sidebar_flash"] = f"Recalled: {'; '.join(parts)}."
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
