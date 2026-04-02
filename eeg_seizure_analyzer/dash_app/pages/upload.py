"""Upload tab: scan channels, select, load EDF files."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    metric_card,
    alert,
    empty_state,
    section_header,
)
from eeg_seizure_analyzer.io.edf_reader import auto_pair_channels
from eeg_seizure_analyzer.io.channel_ids import (
    load_channel_ids,
    save_channel_ids,
    read_channel_ids_excel,
    generate_channel_ids_template,
)


# ── Native file/folder pickers ─────────────────────────────────────
# On macOS use AppleScript (opens in foreground); fall back to tkinter.


def _browse_folder(title: str = "Select folder") -> str | None:
    """Open a native folder picker. Returns path or None."""
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 f'POSIX path of (choose folder with prompt "{title}")'],
                capture_output=True, text=True, timeout=120,
            )
            folder = r.stdout.strip().rstrip("/")
            return folder if folder else None
        except Exception:
            pass  # fall through to tkinter

    # tkinter fallback (Linux / Windows / macOS fallback)
    try:
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk()",
                "root.withdraw()",
                "root.attributes('-topmost', True)",
                "root.update()",
                f'folder = filedialog.askdirectory(title="{title}")',
                "root.destroy()",
                "print(folder or '')",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        folder = r.stdout.strip()
        return folder if folder else None
    except Exception:
        return None


def _browse_file(title: str = "Select file",
                 filetypes: str = "EDF files|*.edf") -> str | None:
    """Open a native file picker. Returns path or None."""
    if platform.system() == "Darwin":
        try:
            # Build AppleScript file type filter
            # filetypes format: "EDF files|*.edf,ADICHT|*.adicht"
            exts = []
            for part in filetypes.split(","):
                if "|" in part:
                    ext = part.split("|")[1].replace("*.", "").strip()
                    exts.append(f'"{ext}"')
            type_clause = ""
            if exts:
                type_clause = f" of type {{{', '.join(exts)}}}"
            script = (
                f'POSIX path of (choose file with prompt "{title}"'
                f"{type_clause})"
            )
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            path = r.stdout.strip()
            return path if path else None
        except Exception:
            pass  # fall through to tkinter

    # tkinter fallback
    try:
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk()",
                "root.withdraw()",
                "root.attributes('-topmost', True)",
                "root.update()",
                "path = filedialog.askopenfilename(",
                f'    title="{title}",',
                "    filetypes=[",
                '        ("EDF files", "*.edf"),',
                '        ("ADICHT files", "*.adicht"),',
                '        ("All supported", "*.edf *.adicht"),',
                "    ],",
                ")",
                "root.destroy()",
                "print(path)",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        path = r.stdout.strip()
        return path if path else None
    except Exception:
        return None


def layout(sid: str | None) -> html.Div:
    """Return the upload tab layout."""
    state = server_state.get_session(sid)

    # Batch project loaded — show project view
    if state.extra.get("project_files") and state.recording is not None:
        return _batch_loaded_layout(state)

    # If recording already loaded, show info
    if state.recording is not None:
        return _loaded_layout(state)

    # If channels scanned but not yet loaded, show selection
    if state.all_channels_info:
        return _channel_selection_layout(state)

    # Batch browse form
    if state.extra.get("show_batch_form"):
        return _batch_browse_layout(state)

    # If user clicked "Load File" on landing, show upload form
    if state.extra.get("show_upload_form"):
        return _upload_layout()

    # Default: landing page
    return _landing_layout()


def _landing_layout() -> html.Div:
    """Full-width landing page with logo, description, and action buttons."""
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "minHeight": "calc(100vh - 80px)",
            "padding": "40px 24px",
            "textAlign": "center",
        },
        children=[
            html.Img(
                src="/assets/nednet_logo.png",
                style={
                    "width": "260px",
                    "marginBottom": "24px",
                },
            ),
            html.P(
                "Automated detection and annotation of seizures and "
                "interictal spikes in long-term EEG recordings.",
                style={
                    "fontSize": "1.05rem",
                    "color": "#8b949e",
                    "maxWidth": "520px",
                    "marginBottom": "36px",
                    "lineHeight": "1.6",
                },
            ),
            html.Div(
                style={"display": "flex", "gap": "16px"},
                children=[
                    dbc.Button(
                        "Load File",
                        id="landing-load-file-btn",
                        className="btn-ned-primary",
                        size="lg",
                    ),
                    dbc.Button(
                        "Load Multiple...",
                        id="landing-load-project-btn",
                        className="btn-ned-secondary",
                        size="lg",
                    ),
                ],
            ),
            html.Div(
                "v0.1",
                style={
                    "marginTop": "48px",
                    "fontSize": "0.75rem",
                    "color": "#484f58",
                },
            ),
        ],
    )


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("landing-load-file-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def landing_load_file(n_clicks, sid, refresh):
    """Switch from landing to upload form when Load File is clicked."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    state.extra["show_upload_form"] = True
    return (refresh or 0) + 1


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("landing-load-project-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def landing_load_multiple(n_clicks, sid, refresh):
    """Switch from landing to batch browse form."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    state.extra["show_batch_form"] = True
    return (refresh or 0) + 1


def _upload_layout() -> html.Div:
    return html.Div(
        style={"padding": "24px", "maxWidth": "800px", "margin": "0 auto"},
        children=[
            html.H4("Load Recording", style={"marginBottom": "8px"}),
            html.P(
                "Supported files: EDF and ADICHT (Windows only)",
                style={"fontSize": "0.85rem", "color": "#8b949e",
                       "marginBottom": "8px"},
            ),
            html.P(
                "Note: ADICHT files can be opened for viewing only. "
                "Detection, training, ML, and saving results require EDF. "
                "Convert first via Tools \u2192 ADICHT \u2192 EDF.",
                style={"fontSize": "0.82rem", "color": "#d29922",
                       "marginBottom": "24px"},
            ),

            # Path input with Browse button
            dbc.InputGroup([
                dbc.Input(
                    id="upload-path-input",
                    placeholder="/path/to/recording.edf",
                    type="text",
                ),
                dbc.Button(
                    "Browse",
                    id="upload-browse-btn",
                    className="btn-ned-secondary",
                ),
                dbc.Button(
                    "Load",
                    id="upload-path-btn",
                    className="btn-ned-primary",
                ),
            ]),

            # Hidden store for browse result
            dcc.Store(id="upload-browse-result"),

            # Status area
            html.Div(id="upload-status", style={"marginTop": "16px"}),
        ],
    )


def _channel_selection_layout(state: server_state.SessionState) -> html.Div:
    """Channel selection after scanning."""
    channel_info = state.all_channels_info

    df = pd.DataFrame(channel_info)
    col_defs = [
        {"field": "index", "headerName": "#", "width": 60},
        {"field": "label", "headerName": "Label", "flex": 1},
        {"field": "unit", "headerName": "Unit", "width": 80},
        {"field": "fs", "headerName": "Hz", "width": 80},
        {"field": "n_samples", "headerName": "Samples", "width": 120},
    ]

    # Detect mixed sample rates
    rates = sorted(set(ch["fs"] for ch in channel_info))
    has_mixed = len(rates) > 1

    # Auto-pair EEG and activity channels
    eeg_indices, act_indices, pairings = auto_pair_channels(channel_info)
    has_pairs = any(p.activity_index is not None for p in pairings)

    # Build the pairing summary card (shown between table and dropdown)
    pairing_card_children: list = []
    if has_pairs:
        pairing_rows = []
        for p in pairings:
            if p.activity_index is not None:
                pairing_rows.append(
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "padding": "4px 0",
                        },
                        children=[
                            html.Span(
                                f"Ch{p.eeg_index} {p.eeg_label}",
                                style={
                                    "fontWeight": "500",
                                    "color": "#58a6ff",
                                },
                            ),
                            html.Span(
                                "\u2194",
                                style={
                                    "fontSize": "1.1rem",
                                    "color": "#8b949e",
                                },
                            ),
                            html.Span(
                                f"Ch{p.activity_index} {p.activity_label}",
                                style={
                                    "fontWeight": "500",
                                    "color": "#3fb950",
                                },
                            ),
                        ],
                    )
                )
            else:
                pairing_rows.append(
                    html.Div(
                        style={"padding": "4px 0", "color": "#8b949e"},
                        children=f"Ch{p.eeg_index} {p.eeg_label} \u2014 no activity pair",
                    )
                )

        pairing_card_children = [
            dbc.Card(
                dbc.CardBody([
                    html.H6(
                        "Detected Channel Pairings",
                        style={
                            "marginBottom": "12px",
                            "fontWeight": "600",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "6px",
                            "marginBottom": "10px",
                        },
                        children=[
                            html.Span(
                                "EEG (Biopot)",
                                style={
                                    "fontSize": "0.75rem",
                                    "color": "#58a6ff",
                                    "border": "1px solid #58a6ff",
                                    "borderRadius": "4px",
                                    "padding": "2px 8px",
                                },
                            ),
                            html.Span(
                                "Activity",
                                style={
                                    "fontSize": "0.75rem",
                                    "color": "#3fb950",
                                    "border": "1px solid #3fb950",
                                    "borderRadius": "4px",
                                    "padding": "2px 8px",
                                },
                            ),
                        ],
                    ),
                    html.Div(pairing_rows),
                ]),
                style={
                    "backgroundColor": "#161b22",
                    "border": "1px solid #30363d",
                    "marginTop": "16px",
                },
            ),
        ]

    # Default dropdown values: only EEG channels when pairings detected
    if has_pairs:
        default_selected = list(eeg_indices)
    else:
        default_selected = [
            ch["index"] for ch in channel_info
            if "biopot" in ch.get("label", "").lower()
            or ch["fs"] == max(rates)
        ]

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.H4("Channel Selection", style={"marginBottom": "16px"}),

            # Channel table
            dag.AgGrid(
                id="upload-channel-grid",
                rowData=channel_info,
                columnDefs=col_defs,
                defaultColDef={"sortable": True, "resizable": True},
                className="ag-theme-alpine-dark",
                style={"height": "300px"},
                dashGridOptions={"animateRows": False},
            ),

            # Mixed rate info
            html.Div(
                style={"marginTop": "16px"},
                children=[
                    alert(
                        f"Mixed sampling rates detected: "
                        + ", ".join(f"{r:.0f} Hz" for r in rates)
                        + ". EEG and activity channels will be loaded separately.",
                        "warning",
                    )
                ] if has_mixed else [],
            ),

            # Pairing summary card
            html.Div(children=pairing_card_children),

            # Channel selector
            html.Div(
                style={"marginTop": "16px"},
                children=[
                    html.Label(
                        "Select EEG channels to load",
                        style={"fontSize": "0.85rem", "fontWeight": "500",
                               "marginBottom": "8px", "display": "block"},
                    ),
                    dcc.Dropdown(
                        id="upload-channel-dropdown",
                        options=[
                            {"label": f"{ch['index']}: {ch['label']} ({ch['fs']:.0f} Hz)",
                             "value": ch["index"]}
                            for ch in channel_info
                        ],
                        value=default_selected,
                        multi=True,
                        placeholder="Select channels...",
                    ),
                ],
            ),

            # Load button
            html.Div(
                style={"marginTop": "20px", "display": "flex", "gap": "12px"},
                children=[
                    dbc.Button(
                        "Load Selected Channels",
                        id="upload-load-btn",
                        className="btn-ned-primary",
                    ),
                    dbc.Button(
                        "Back",
                        id="upload-back-btn",
                        className="btn-ned-secondary",
                    ),
                ],
            ),

            # Loading spinner wrapping the load status
            dcc.Loading(
                id="upload-load-spinner",
                type="default",
                children=html.Div(
                    id="upload-load-status",
                    style={"marginTop": "12px"},
                ),
            ),
        ],
    )


def _loaded_layout(state: server_state.SessionState) -> html.Div:
    """Show loaded recording info."""
    rec = state.recording
    dur_h = rec.duration_sec / 3600

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.H4("Recording Loaded", style={"marginBottom": "20px"}),

            # Success banner
            alert(
                f"{rec.source_path} \u2014 {rec.n_channels} channels @ {rec.fs:.0f} Hz, "
                f"{rec.duration_sec:.1f}s ({dur_h:.2f}h)",
                "success",
            ),

            # Metric cards
            dbc.Row(
                [
                    dbc.Col(metric_card("Channels", str(rec.n_channels)), width=3),
                    dbc.Col(metric_card("Duration", f"{rec.duration_sec:.1f}s"), width=3),
                    dbc.Col(metric_card("Sample Rate", f"{rec.fs:.0f} Hz"), width=3),
                    dbc.Col(metric_card("Annotations", str(len(rec.annotations))), width=3),
                ],
                className="g-3 mt-3",
            ),

            # Channel list with editable Animal ID
            html.Div(
                style={"marginTop": "24px"},
                children=[
                    html.H6("Loaded Channels", style={"marginBottom": "4px"}),
                    html.P(
                        "Assign an Animal ID to each channel (editable, saved automatically).",
                        style={"color": "#8b949e", "fontSize": "0.82rem",
                               "marginBottom": "12px"},
                    ),
                    dag.AgGrid(
                        id="upload-channel-ids-grid",
                        rowData=_build_channel_id_rows(rec, state),
                        columnDefs=[
                            {"field": "index", "headerName": "#", "width": 60},
                            {"field": "name", "headerName": "Channel", "flex": 1},
                            {"field": "unit", "headerName": "Unit", "width": 80},
                            {"field": "animal_id", "headerName": "Animal ID",
                             "editable": True, "width": 180,
                             "cellStyle": {"color": "#58a6ff",
                                           "fontWeight": "500"}},
                        ],
                        className="ag-theme-alpine-dark",
                        style={"height": f"{min(60 + rec.n_channels * 42, 400)}px"},
                        dashGridOptions={"animateRows": False},
                    ),
                    html.Div(id="upload-channel-ids-status",
                             style={"marginTop": "4px"}),
                ],
            ),

            # Video status + manual path input
            html.Div(
                style={"marginTop": "16px"},
                children=[
                    html.Div([
                        html.Div(
                            [
                                html.Span("\u25B6 ", style={"color": "#3fb950"}),
                                f"Video: {os.path.basename(state.extra['video_path'])}",
                            ],
                            style={"color": "#3fb950", "fontSize": "0.85rem"},
                        ),
                        # Hidden placeholders for callback
                        dbc.Input(id="upload-video-path", type="hidden"),
                        html.Div(dbc.Button(id="upload-video-link-btn",
                                            style={"display": "none"})),
                    ])
                ] if state.extra.get("video_path") else [
                    html.Div(
                        "No video file auto-detected.",
                        style={"color": "#8b949e", "fontSize": "0.85rem",
                               "marginBottom": "8px"},
                    ),
                    dbc.InputGroup([
                        dbc.Input(
                            id="upload-video-path",
                            placeholder="/path/to/video.mp4",
                            type="text",
                        ),
                        dbc.Button("Link Video", id="upload-video-link-btn",
                                   className="btn-ned-secondary", size="sm"),
                    ], size="sm"),
                ],
            ),

            # Action buttons
            html.Div(
                style={"marginTop": "20px", "display": "flex", "gap": "12px"},
                children=[
                    dbc.Button(
                        "Change Channel Selection",
                        id="upload-change-btn",
                        className="btn-ned-secondary",
                    ),
                    dbc.Button(
                        "Load Another File",
                        id="upload-new-file-btn",
                        className="btn-ned-secondary",
                    ),
                ],
            ),
        ],
    )


# ── Batch (Load Multiple) layouts ────────────────────────────────────


def _batch_browse_layout(state: server_state.SessionState) -> html.Div:
    """Form for browsing and scanning a folder of EDF files."""
    prev_folder = state.extra.get("project_folder", "")
    return html.Div(
        style={"padding": "24px", "maxWidth": "900px", "margin": "0 auto"},
        children=[
            html.H4("Load Multiple Recordings", style={"marginBottom": "8px"}),
            html.P(
                "Select a folder containing EDF recordings. Each file will be "
                "scanned for existing detections, annotations, and channel IDs.",
                style={"fontSize": "0.85rem", "color": "#8b949e",
                       "marginBottom": "24px"},
            ),

            # Folder input + Browse + Scan
            html.Label("Recordings folder",
                       style={"fontSize": "0.82rem", "color": "#8b949e"}),
            dbc.InputGroup([
                dbc.Input(
                    id="batch-folder-input",
                    placeholder="/path/to/recordings",
                    value=prev_folder,
                    type="text",
                ),
                dbc.Button("Browse", id="batch-browse-btn",
                           className="btn-ned-secondary"),
                dbc.Button("Scan", id="batch-scan-btn",
                           className="btn-ned-primary"),
            ], className="mb-3"),

            # Scan results area
            dcc.Loading(
                html.Div(id="batch-scan-results"),
                type="circle", color="#58a6ff",
            ),

            # Store for scan data
            dcc.Store(id="batch-scan-data"),

            # Back button
            html.Div(
                style={"marginTop": "20px"},
                children=[
                    dbc.Button("Back", id="batch-back-btn",
                               className="btn-ned-secondary"),
                ],
            ),

            # Status area for template generation
            html.Div(id="batch-template-status", style={"marginTop": "8px"}),
        ],
    )


def _batch_loaded_layout(state: server_state.SessionState) -> html.Div:
    """Show loaded recording info in project context."""
    rec = state.recording
    dur_h = rec.duration_sec / 3600
    project_files = state.extra.get("project_files", [])
    project_folder = state.extra.get("project_folder", "")
    active_idx = state.extra.get("project_active_idx", 0)
    n_files = len(project_files)

    # Build file selector options
    file_options = [
        {"label": f["filename"], "value": i}
        for i, f in enumerate(project_files)
    ]

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.H4("Recording Loaded", style={"marginBottom": "20px"}),

            # Project info banner
            alert(
                f"Project: {project_folder} \u2014 {n_files} file{'s' if n_files != 1 else ''}",
                "info",
            ),

            # File selector
            html.Div(
                style={"marginBottom": "16px", "maxWidth": "500px"},
                children=[
                    html.Label("Active file",
                               style={"fontSize": "0.82rem", "color": "#8b949e"}),
                    dcc.Dropdown(
                        id="batch-file-selector",
                        options=file_options,
                        value=active_idx,
                        clearable=False,
                    ),
                ],
            ),

            # Recording info
            alert(
                f"{rec.source_path} \u2014 {rec.n_channels} channels @ {rec.fs:.0f} Hz, "
                f"{rec.duration_sec:.1f}s ({dur_h:.2f}h)",
                "success",
            ),

            # Metric cards
            dbc.Row(
                [
                    dbc.Col(metric_card("Channels", str(rec.n_channels)), width=3),
                    dbc.Col(metric_card("Duration", f"{rec.duration_sec:.1f}s"), width=3),
                    dbc.Col(metric_card("Sample Rate", f"{rec.fs:.0f} Hz"), width=3),
                    dbc.Col(metric_card("Annotations", str(len(rec.annotations))), width=3),
                ],
                className="g-3 mt-3",
            ),

            # Channel list with editable Animal ID
            html.Div(
                style={"marginTop": "24px"},
                children=[
                    html.H6("Loaded Channels", style={"marginBottom": "4px"}),
                    html.P(
                        "Assign an Animal ID to each channel (editable, saved automatically).",
                        style={"color": "#8b949e", "fontSize": "0.82rem",
                               "marginBottom": "12px"},
                    ),
                    dag.AgGrid(
                        id="upload-channel-ids-grid",
                        rowData=_build_channel_id_rows(rec, state),
                        columnDefs=[
                            {"field": "index", "headerName": "#", "width": 60},
                            {"field": "name", "headerName": "Channel", "flex": 1},
                            {"field": "unit", "headerName": "Unit", "width": 80},
                            {"field": "animal_id", "headerName": "Animal ID",
                             "editable": True, "width": 180,
                             "cellStyle": {"color": "#58a6ff",
                                           "fontWeight": "500"}},
                        ],
                        className="ag-theme-alpine-dark",
                        style={"height": f"{min(60 + rec.n_channels * 42, 400)}px"},
                        dashGridOptions={"animateRows": False},
                    ),
                    html.Div(id="upload-channel-ids-status",
                             style={"marginTop": "4px"}),
                ],
            ),

            # Video status + manual path input
            html.Div(
                style={"marginTop": "16px"},
                children=[
                    html.Div([
                        html.Div(
                            [
                                html.Span("\u25B6 ", style={"color": "#3fb950"}),
                                f"Video: {os.path.basename(state.extra['video_path'])}",
                            ],
                            style={"color": "#3fb950", "fontSize": "0.85rem"},
                        ),
                        dbc.Input(id="upload-video-path", type="hidden"),
                        html.Div(dbc.Button(id="upload-video-link-btn",
                                            style={"display": "none"})),
                    ])
                ] if state.extra.get("video_path") else [
                    html.Div(
                        "No video file auto-detected.",
                        style={"color": "#8b949e", "fontSize": "0.85rem",
                               "marginBottom": "8px"},
                    ),
                    dbc.InputGroup([
                        dbc.Input(
                            id="upload-video-path",
                            placeholder="/path/to/video.mp4",
                            type="text",
                        ),
                        dbc.Button("Link Video", id="upload-video-link-btn",
                                   className="btn-ned-secondary", size="sm"),
                    ], size="sm"),
                ],
            ),

            # Action buttons — no "Load Another File" in project mode
            html.Div(
                style={"marginTop": "20px", "display": "flex", "gap": "12px"},
                children=[
                    dbc.Button(
                        "Change Channel Selection",
                        id="upload-change-btn",
                        className="btn-ned-secondary",
                    ),
                    dbc.Button(
                        "Load Another File",
                        id="upload-new-file-btn",
                        className="btn-ned-secondary",
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _build_channel_id_rows(rec, state):
    """Build row data for the channel IDs grid, loading saved IDs."""
    saved = load_channel_ids(rec.source_path) if rec.source_path else None
    # Also store in session for use by detection/annotation pages
    if saved:
        state.extra["channel_animal_ids"] = saved

    rows = []
    for i, (name, unit) in enumerate(zip(rec.channel_names, rec.units)):
        aid = ""
        if saved and i in saved:
            aid = saved[i]
        rows.append({
            "index": i,
            "name": name,
            "unit": unit,
            "animal_id": aid,
        })
    return rows


def _try_load_saved_detections(state: server_state.SessionState):
    """Attempt to load previously saved detections from disk.

    Populates ``state.seizure_events``, ``state.st_detection_info``,
    and ``state.detected_events`` if a JSON file is found alongside
    the EDF.  Also restores detection parameters, filter settings,
    and channel selection into server state so both the Seizure and
    Training tabs start with the saved configuration.

    Returns a Dash component indicating success or ``None``.
    """
    try:
        rec = state.recording
        src = getattr(rec, "source_path", None) or ""
        if not src or not src.lower().endswith(".edf"):
            return None

        from eeg_seizure_analyzer.io.persistence import load_detections

        result = load_detections(src)
        if result is None:
            return None

        events = result["events"]
        # Assign event IDs if missing (legacy detection files)
        if events and all(ev.event_id == 0 for ev in events):
            events.sort(key=lambda e: (e.channel, e.onset_sec))
            for i, ev in enumerate(events, start=1):
                ev.event_id = i
        state.seizure_events = events
        state.st_detection_info = result.get("detection_info", {})
        state.detected_events = list(events) + state.spike_events

        # Restore detection parameters so Seizure tab shows what was used
        saved_params = result.get("params", {})
        if saved_params:
            state.extra["sz_params"] = saved_params
            # Also restore dropdown values if stored under their keys
            if "sz-bl-method" in saved_params:
                state.extra["sz_bl_method"] = saved_params["sz-bl-method"]
            if "sz-bnd-method" in saved_params:
                state.extra["sz_bnd_method"] = saved_params["sz-bnd-method"]

        # Restore channel selection
        saved_channels = result.get("channels", [])
        if saved_channels:
            state.extra["sz_selected_channels"] = saved_channels

        # Restore filter settings from detection file into BOTH tabs
        fs = result.get("filter_settings", {})
        filter_on = fs.get("filter_enabled", True)  # default ON
        filter_vals = fs.get("filter_values", {})

        # Seizure tab
        state.extra["sz_filter_enabled"] = filter_on
        if filter_vals:
            state.extra["sz_filter_values"] = filter_vals

        # Training tab — map the common keys (min + max)
        state.extra["tr_filter_on"] = filter_on
        if filter_vals:
            state.extra["tr_min_conf"] = filter_vals.get("min_conf", 0)
            state.extra["tr_min_dur"] = filter_vals.get("min_dur", 0)
            state.extra["tr_min_lbl"] = filter_vals.get("min_lbl", 0)
            state.extra["tr_max_conf"] = filter_vals.get("max_conf", None)
            state.extra["tr_max_dur"] = filter_vals.get("max_dur", None)
            state.extra["tr_max_lbl"] = filter_vals.get("max_lbl", None)

        n = len(events)
        return alert(
            f"Loaded {n} saved seizure detection(s) from disk"
            f" (filter {'ON' if filter_on else 'OFF'}).",
            "info",
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _try_load_saved_spikes(state: server_state.SessionState):
    """Attempt to load previously saved interictal spike detections from disk.

    Populates ``state.spike_events``, ``state.sp_detection_info``,
    and merges into ``state.detected_events``.  Restores IS parameters,
    filters, and channel selection.

    Returns a Dash component indicating success or ``None``.
    """
    try:
        rec = state.recording
        src = getattr(rec, "source_path", None) or ""
        if not src or not src.lower().endswith(".edf"):
            return None

        from eeg_seizure_analyzer.io.persistence import load_spike_detections

        result = load_spike_detections(src)
        if result is None:
            return None

        events = result["events"]
        state.spike_events = events
        state.sp_detection_info = result.get("detection_info", {})
        state.detected_events = state.seizure_events + events

        # Restore IS detection parameters so Spikes tab shows what was used
        saved_params = result.get("params", {})
        if saved_params:
            state.extra["sp_params"] = saved_params
            if "sp-bl-method" in saved_params:
                state.extra["sp_bl_method"] = saved_params["sp-bl-method"]

        # Restore channel selection
        saved_channels = result.get("channels", [])
        if saved_channels:
            state.extra["sp_selected_channels"] = saved_channels

        # Restore filter settings
        fs = result.get("filter_settings", {})
        if fs:
            filter_on = fs.get("filter_enabled", True)
            filter_vals = fs.get("filter_values", {})
            filter_vals.pop("channel", None)  # channel filter not persisted
            state.extra["sp_filter_enabled"] = filter_on
            if filter_vals:
                state.extra["sp_filter_values"] = filter_vals

        n = len(events)
        return alert(
            f"Loaded {n} saved interictal spike detection(s) from disk.",
            "info",
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return None



def _try_load_ml_detections(state: server_state.SessionState):
    """Load ML detections from disk sidecar if present."""
    try:
        rec = state.recording
        src = getattr(rec, "source_path", None) or ""
        if not src or not src.lower().endswith(".edf"):
            return None

        from pathlib import Path
        import json

        stem = Path(src).stem
        ml_path = Path(src).parent / f"{stem}_ned_ml_detections.json"
        if not ml_path.exists():
            return None

        with open(ml_path) as f:
            data = json.load(f)

        events = data.get("events", [])
        if events:
            state.extra["ml_detected_events"] = events
            state.extra["ml_det_model"] = data.get("model_name", "")
            return alert(
                f"Loaded {len(events)} saved ML detection(s) from disk.",
                "info",
            )
    except Exception:
        import traceback
        traceback.print_exc()
    return None


def _discover_video(state: server_state.SessionState):
    """Look for an MP4 video file alongside the recording.

    Checks for ``<basename>.mp4`` next to the source file.
    Also checks for an MP4 matching the original upload filename
    in common locations.
    If found, stores the path in ``state.extra["video_path"]``.
    """
    rec = state.recording
    src = getattr(rec, "source_path", None) or ""

    # Try the source path directly (works for path-based loading)
    if src and os.path.isfile(src):
        base = os.path.splitext(src)[0]
        mp4 = base + ".mp4"
        if os.path.isfile(mp4):
            state.extra["video_path"] = mp4
            return

    # Try the upload_source_path (set during path-based load)
    upload_src = state.extra.get("upload_source_path", "")
    if upload_src and os.path.isfile(upload_src):
        base = os.path.splitext(upload_src)[0]
        mp4 = base + ".mp4"
        if os.path.isfile(mp4):
            state.extra["video_path"] = mp4
            return

    state.extra["video_path"] = None


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("upload-browse-result", "data"),
    Input("upload-browse-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_browse(n_clicks):
    """Open a native file dialog and return the selected path."""
    if not n_clicks:
        return no_update
    return _browse_file("Select EEG recording", "EDF files|*.edf,ADICHT|*.adicht")


@callback(
    Output("upload-path-input", "value"),
    Input("upload-browse-result", "data"),
    prevent_initial_call=True,
)
def on_browse_result(path):
    """Fill the path input with the selected file."""
    if not path:
        return no_update
    return path


@callback(
    Output("upload-status", "children"),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-path-btn", "n_clicks"),
    State("upload-path-input", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_path_load(n_clicks, path, sid, refresh):
    """Load from a file path."""
    if not n_clicks or not path:
        return no_update, no_update

    path = path.strip()
    if not os.path.isfile(path):
        return alert(f"File not found: {path}", "danger"), no_update

    state = server_state.get_session(sid)

    try:
        if path.lower().endswith(".edf"):
            from eeg_seizure_analyzer.io.edf_reader import scan_edf_channels
            channel_info = scan_edf_channels(path)
            state.all_channels_info = channel_info
            state.extra["upload_source_path"] = path
            state.extra["upload_filename"] = os.path.basename(path)
            return None, (refresh or 0) + 1

        elif path.lower().endswith(".adicht"):
            from eeg_seizure_analyzer.io.adicht_reader import read_adicht
            recording = read_adicht(path)
            recording.source_path = path
            state.recording = recording
            _discover_video(state)
            return alert(
                "ADICHT file loaded for viewing. Detection, training, ML, "
                "and saving results require EDF format. Convert via "
                "Tools \u2192 ADICHT \u2192 EDF (Windows only).",
                "warning",
            ), (refresh or 0) + 1

        else:
            return alert("Unsupported file type", "danger"), no_update

    except Exception as e:
        return alert(f"Error: {e}", "danger"), no_update


@callback(
    Output("upload-load-status", "children"),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-load-btn", "n_clicks"),
    State("upload-channel-dropdown", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_load_channels(n_clicks, selected_channels, sid, refresh):
    """Load selected channels into the recording."""
    if not n_clicks or not selected_channels:
        return alert("Select at least one channel.", "warning"), no_update

    state = server_state.get_session(sid)

    # Clear previous detections/annotations when loading a new file
    state.seizure_events = []
    state.spike_events = []
    state.detected_events = []
    state.st_detection_info = {}
    state.sp_detection_info = {}
    # Clear training annotations from previous file
    state.extra.pop("tr_annotations", None)
    state.extra.pop("tr_current_idx", None)
    state.extra.pop("sz_selected_event_key", None)

    try:
        source_path = state.extra.get("upload_source_path")
        if not source_path or not os.path.isfile(source_path):
            return alert("No file data available.", "danger"), no_update

        from eeg_seizure_analyzer.io.edf_reader import read_edf, read_edf_paired
        from eeg_seizure_analyzer.io.edf_reader import auto_pair_channels

        channel_info = state.all_channels_info

        # Try auto-pairing
        eeg_indices, act_indices, pairings = auto_pair_channels(channel_info)
        has_pairs = any(p.activity_index is not None for p in pairings)

        if has_pairs:
            sel_eeg = [i for i in selected_channels if i in eeg_indices]
            sel_act = act_indices
            eeg_rec, act_rec = read_edf_paired(source_path, sel_eeg, sel_act)
            eeg_rec.source_path = source_path
            state.recording = eeg_rec
            state.activity_recordings = {"paired": act_rec}
            state.channel_pairings = pairings
        else:
            recording = read_edf(source_path, channels=selected_channels)
            recording.source_path = source_path
            state.recording = recording

        # Discover associated video file
        _discover_video(state)

        # Auto-load saved detections if available
        det_status = _try_load_saved_detections(state)
        sp_status = _try_load_saved_spikes(state)

        # Combine status messages
        combined = []
        if det_status:
            combined.append(det_status)
        if sp_status:
            combined.append(sp_status)
        load_status = html.Div(combined) if combined else None

        return load_status, (refresh or 0) + 1

    except Exception as e:
        return alert(f"Error loading: {e}", "danger"), no_update


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-back-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_back(n_clicks, sid, refresh):
    """Go back to upload screen."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    state.all_channels_info = []

    state.extra.pop("upload_source_path", None)
    return (refresh or 0) + 1


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-change-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_change_channels(n_clicks, sid, refresh):
    """Reset recording to allow re-selection."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    source = state.recording.source_path if state.recording else None
    state.recording = None
    if source and os.path.isfile(source):
        from eeg_seizure_analyzer.io.edf_reader import scan_edf_channels
        state.all_channels_info = scan_edf_channels(source)
        state.extra["upload_source_path"] = source
        state.extra["upload_filename"] = os.path.basename(source)
    return (refresh or 0) + 1


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-new-file-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_load_new_file(n_clicks, sid, refresh):
    """Reset everything to start fresh with a new file."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    state.recording = None
    state.all_channels_info = []
    state.activity_recordings = {}
    state.channel_pairings = []
    state.seizure_events = []
    state.spike_events = []
    state.detected_events = []
    state.st_detection_info = {}
    state.sp_detection_info = {}

    state.extra.pop("upload_source_path", None)
    state.extra.pop("upload_filename", None)
    state.extra.pop("tr_annotations", None)
    state.extra.pop("tr_current_idx", None)
    state.extra.pop("sz_selected_event_key", None)
    state.extra.pop("video_path", None)
    return (refresh or 0) + 1


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-video-link-btn", "n_clicks"),
    State("upload-video-path", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_link_video(n_clicks, video_path, sid, refresh):
    """Manually link a video file to the current recording."""
    if not n_clicks or not video_path:
        return no_update
    state = server_state.get_session(sid)
    if not os.path.isfile(video_path):
        return no_update
    state.extra["video_path"] = video_path
    return (refresh or 0) + 1


# ── Save channel Animal IDs on edit ─────────────────────────────────


@callback(
    Output("upload-channel-ids-status", "children"),
    Input("upload-channel-ids-grid", "cellValueChanged"),
    State("upload-channel-ids-grid", "rowData"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_channel_id_edit(cell_changed, row_data, sid):
    """Save channel-to-animal-ID mapping when user edits a cell."""
    if not cell_changed or not row_data:
        return no_update

    state = server_state.get_session(sid)
    rec = state.recording
    if rec is None or not rec.source_path:
        return no_update

    # Build mapping from grid data
    mapping = {}
    for row in row_data:
        ch_idx = row.get("index")
        aid = row.get("animal_id", "").strip()
        if ch_idx is not None and aid:
            mapping[int(ch_idx)] = aid

    # Save to disk
    save_channel_ids(rec.source_path, mapping)

    # Update session state
    state.extra["channel_animal_ids"] = mapping

    n_assigned = len(mapping)
    return html.Div(
        f"Saved {n_assigned} animal ID{'s' if n_assigned != 1 else ''}.",
        style={"color": "#3fb950", "fontSize": "0.78rem"},
    )


# ── Batch (Load Multiple) callbacks ─────────────────────────────────


@callback(
    Output("batch-folder-input", "value"),
    Input("batch-browse-btn", "n_clicks"),
    prevent_initial_call=True,
)
def batch_browse_folder(n_clicks):
    """Open a native folder picker."""
    if not n_clicks:
        return no_update
    folder = _browse_folder("Select recordings folder")
    return folder if folder else no_update


@callback(
    Output("batch-scan-results", "children"),
    Output("batch-scan-data", "data"),
    Input("batch-scan-btn", "n_clicks"),
    State("batch-folder-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def batch_scan_folder(n_clicks, folder, sid):
    """Scan folder for EDF files and show status table."""
    if not n_clicks or not folder:
        return no_update, no_update

    folder = folder.strip()
    if not os.path.isdir(folder):
        return alert(f"Folder not found: {folder}", "danger"), no_update

    state = server_state.get_session(sid)
    state.extra["project_folder"] = folder

    # Find EDF files (non-recursive)
    edf_files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".edf")
    )

    if not edf_files:
        return (
            alert(f"No EDF files found in {folder}", "warning"),
            no_update,
        )

    # Build status rows
    rows = []
    for fname in edf_files:
        edf_path = os.path.join(folder, fname)
        stem = os.path.splitext(fname)[0]

        # Check channel IDs
        ch_json = os.path.join(folder, stem + "_ned_channels.json")
        has_ids = os.path.isfile(ch_json)

        # Check detections
        det_json = os.path.join(folder, stem + "_ned_detections.json")
        det_count = "\u2014"
        has_detections = False
        if os.path.isfile(det_json):
            has_detections = True
            try:
                with open(det_json) as f:
                    data = json.load(f)
                det_count = str(len(data.get("events", [])))
            except Exception:
                det_count = "\u2714"

        # Check annotations
        ann_json = os.path.join(folder, stem + "_ned_annotations.json")
        ann_count = "\u2014"
        has_annotations = False
        if os.path.isfile(ann_json):
            has_annotations = True
            try:
                with open(ann_json) as f:
                    data = json.load(f)
                ann_count = str(len(data.get("annotations", [])))
            except Exception:
                ann_count = "\u2714"

        rows.append({
            "filename": fname,
            "edf_path": edf_path,
            "animal_ids": "\u2714" if has_ids else "\u2718",
            "detections": det_count,
            "annotations": ann_count,
            "has_ids": has_ids,
            "has_detections": has_detections,
            "has_annotations": has_annotations,
        })

    # Build AgGrid
    col_defs = [
        {"field": "filename", "headerName": "File", "flex": 2, "minWidth": 250},
        {"field": "animal_ids", "headerName": "Animal IDs", "width": 110,
         "cellStyle": {"textAlign": "center"}},
        {"field": "detections", "headerName": "Detections", "width": 110,
         "cellStyle": {"textAlign": "center"}},
        {"field": "annotations", "headerName": "Annotations", "width": 120,
         "cellStyle": {"textAlign": "center"}},
    ]

    grid = dag.AgGrid(
        id="batch-file-grid",
        rowData=rows,
        columnDefs=col_defs,
        defaultColDef={"sortable": True, "resizable": True},
        className="ag-theme-alpine-dark",
        style={"height": f"{min(60 + len(rows) * 42, 500)}px"},
        dashGridOptions={"animateRows": False},
    )

    # Check for channel_ids.xlsx
    xlsx_path = os.path.join(folder, "channel_ids.xlsx")
    template_section: list = []
    if os.path.isfile(xlsx_path):
        template_section = [
            alert(
                "Found channel_ids.xlsx \u2014 animal IDs will be loaded from template.",
                "success",
            ),
        ]
    else:
        template_section = [
            dbc.Button(
                "Generate Template",
                id="batch-gen-template-btn",
                className="btn-ned-secondary",
                size="sm",
                style={"marginTop": "8px"},
            ),
        ]

    content = html.Div([
        html.H6(
            f"Found {len(edf_files)} EDF file{'s' if len(edf_files) != 1 else ''}",
            style={"marginBottom": "12px"},
        ),
        grid,
        html.Div(template_section),
        html.Div(
            style={"marginTop": "16px"},
            children=[
                dbc.Button(
                    "Load All",
                    id="batch-load-all-btn",
                    className="btn-ned-primary",
                ),
            ],
        ),
    ])

    # Serialisable scan data for the store
    scan_data = [
        {
            "filename": r["filename"],
            "edf_path": r["edf_path"],
            "has_ids": r["has_ids"],
            "has_detections": r["has_detections"],
            "has_annotations": r["has_annotations"],
        }
        for r in rows
    ]

    return content, scan_data


@callback(
    Output("batch-template-status", "children"),
    Input("batch-gen-template-btn", "n_clicks"),
    State("batch-scan-data", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def batch_generate_template(n_clicks, scan_data, sid):
    """Generate channel_ids.xlsx template."""
    if not n_clicks or not scan_data:
        return no_update
    state = server_state.get_session(sid)
    folder = state.extra.get("project_folder", "")
    if not folder:
        return alert("No folder set.", "danger")

    edf_files = [r["filename"] for r in scan_data]
    out_path = generate_channel_ids_template(folder, edf_files)
    return alert(f"Template generated: {out_path}", "success")


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("batch-back-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def batch_back(n_clicks, sid, refresh):
    """Return to landing page from batch form."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    state.extra.pop("show_batch_form", None)
    return (refresh or 0) + 1


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("batch-load-all-btn", "n_clicks"),
    State("batch-scan-data", "data"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def batch_load_all(n_clicks, scan_data, sid, refresh):
    """Load all scanned EDF files as a project and open the first one."""
    if not n_clicks or not scan_data:
        return no_update

    state = server_state.get_session(sid)
    folder = state.extra.get("project_folder", "")
    if not folder:
        return no_update

    # 1. Apply channel IDs from Excel if available
    xlsx_ids = read_channel_ids_excel(folder)
    if xlsx_ids:
        for entry in scan_data:
            fname = entry["filename"]
            if fname in xlsx_ids:
                save_channel_ids(entry["edf_path"], xlsx_ids[fname])
                entry["has_ids"] = True

    # 2. Store project file list
    project_files = [
        {
            "edf_path": e["edf_path"],
            "filename": e["filename"],
            "has_detections": e.get("has_detections", False),
            "has_annotations": e.get("has_annotations", False),
            "has_ids": e.get("has_ids", False),
        }
        for e in scan_data
    ]
    state.extra["project_files"] = project_files
    state.extra["project_active_idx"] = 0

    # 3. Load the first file
    first = project_files[0]
    try:
        _load_edf_into_state(state, first["edf_path"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return no_update

    # 4. Clear batch form flag
    state.extra.pop("show_batch_form", None)

    return (refresh or 0) + 1


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("batch-file-selector", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def batch_switch_file(file_idx, sid, refresh):
    """Switch to a different file in the project."""
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
    try:
        _load_edf_into_state(state, entry["edf_path"])
    except Exception:
        import traceback
        traceback.print_exc()
        return no_update

    state.extra["project_active_idx"] = file_idx

    return (refresh or 0) + 1


def _load_edf_into_state(state: server_state.SessionState, edf_path: str):
    """Scan, auto-pair, and load an EDF file into session state.

    Also loads saved detections, spikes, and discovers video.
    """
    from eeg_seizure_analyzer.io.edf_reader import (
        scan_edf_channels, read_edf, read_edf_paired, auto_pair_channels,
    )

    channel_info = scan_edf_channels(edf_path)
    eeg_indices, act_indices, pairings = auto_pair_channels(channel_info)
    has_pairs = any(p.activity_index is not None for p in pairings)

    if has_pairs:
        eeg_rec, act_rec = read_edf_paired(edf_path, list(eeg_indices), act_indices)
        eeg_rec.source_path = edf_path
        state.recording = eeg_rec
        state.activity_recordings = {"paired": act_rec}
        state.channel_pairings = pairings
    else:
        # Load all channels (use biopot/max-rate heuristic)
        rates = sorted(set(ch["fs"] for ch in channel_info))
        default_channels = [
            ch["index"] for ch in channel_info
            if "biopot" in ch.get("label", "").lower()
            or ch["fs"] == max(rates)
        ]
        if not default_channels:
            default_channels = [ch["index"] for ch in channel_info]
        recording = read_edf(edf_path, channels=default_channels)
        recording.source_path = edf_path
        state.recording = recording

    state.extra["upload_source_path"] = edf_path
    state.extra["upload_filename"] = os.path.basename(edf_path)

    # Discover video
    _discover_video(state)

    # Load saved detections and spikes
    _try_load_saved_detections(state)
    _try_load_saved_spikes(state)

    # Load saved ML detections
    _try_load_ml_detections(state)
