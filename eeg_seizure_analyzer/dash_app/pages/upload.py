"""Upload tab: scan channels, select, load EDF files."""

from __future__ import annotations

import base64
import os
import sys
import tempfile

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


def layout(sid: str | None) -> html.Div:
    """Return the upload tab layout."""
    state = server_state.get_session(sid)

    # If recording already loaded, show info
    if state.recording is not None:
        return _loaded_layout(state)

    # If channels scanned but not yet loaded, show selection
    if state.all_channels_info:
        return _channel_selection_layout(state)

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
                        "Load Project",
                        id="landing-load-project-btn",
                        className="btn-ned-secondary",
                        size="lg",
                        disabled=True,
                        title="Coming soon",
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


def _upload_layout() -> html.Div:
    return html.Div(
        style={"padding": "24px", "maxWidth": "800px", "margin": "0 auto"},
        children=[
            html.H4("Load Recording", style={"marginBottom": "8px"}),
            html.P(
                "Supported files: EDF and ADICHT (Windows only)",
                style={"fontSize": "0.85rem", "color": "#8b949e",
                       "marginBottom": "24px"},
            ),

            # Upload area
            dcc.Upload(
                id="upload-edf",
                children=html.Div(
                    className="upload-area",
                    children=[
                        html.Div("\u21E7", className="upload-icon"),
                        html.Div([
                            html.Strong("Drop a file here"),
                            html.Br(),
                            "or click to browse",
                        ], className="upload-text"),
                    ],
                ),
                multiple=False,
            ),

            # Or load from path
            html.Div(
                style={"marginTop": "24px"},
                children=[
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                title="Load from file path",
                                children=[
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="upload-path-input",
                                            placeholder="/path/to/recording.edf",
                                            type="text",
                                        ),
                                        dbc.Button(
                                            "Load",
                                            id="upload-path-btn",
                                            className="btn-ned-primary",
                                        ),
                                    ]),
                                ],
                            ),
                        ],
                        start_collapsed=True,
                        flush=True,
                    ),
                ],
            ),

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
                        style={"backgroundColor": "#1c2128", "color": "#e6edf3"},
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

            # Channel list
            html.Div(
                style={"marginTop": "24px"},
                children=[
                    html.H6("Loaded Channels", style={"marginBottom": "12px"}),
                    dag.AgGrid(
                        rowData=[
                            {"index": i, "name": name, "unit": unit}
                            for i, (name, unit) in enumerate(
                                zip(rec.channel_names, rec.units)
                            )
                        ],
                        columnDefs=[
                            {"field": "index", "headerName": "#", "width": 60},
                            {"field": "name", "headerName": "Name", "flex": 1},
                            {"field": "unit", "headerName": "Unit", "width": 100},
                        ],
                        className="ag-theme-alpine-dark",
                        style={"height": "200px"},
                        dashGridOptions={"animateRows": False},
                    ),
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


# ── Helpers ───────────────────────────────────────────────────────────


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
    Output("upload-status", "children"),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("upload-edf", "contents"),
    State("upload-edf", "filename"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_file_upload(contents, filename, sid, refresh):
    """Handle file upload: decode, scan channels, store in state."""
    if contents is None:
        return no_update, no_update

    try:
        # Decode base64
        content_type, content_string = contents.split(",")
        file_bytes = base64.b64decode(content_string)

        state = server_state.get_session(sid)

        if filename.lower().endswith(".edf"):
            # Scan channels
            from eeg_seizure_analyzer.io.edf_reader import scan_edf_channels

            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                channel_info = scan_edf_channels(tmp_path)
            finally:
                os.unlink(tmp_path)

            state.all_channels_info = channel_info
            state.extra["upload_file_bytes"] = file_bytes
            state.extra["upload_filename"] = filename

            # Re-render to show channel selection
            return None, (refresh or 0) + 1

        elif filename.lower().endswith(".adicht"):
            from eeg_seizure_analyzer.io.adicht_reader import read_adicht

            with tempfile.NamedTemporaryFile(suffix=".adicht", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                recording = read_adicht(tmp_path)
                recording.source_path = filename
            finally:
                os.unlink(tmp_path)

            state.recording = recording
            _discover_video(state)
            return None, (refresh or 0) + 1

        else:
            return alert(f"Unsupported file type: {filename}", "danger"), no_update

    except Exception as e:
        return alert(f"Error: {e}", "danger"), no_update


@callback(
    Output("upload-status", "children", allow_duplicate=True),
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
            return None, (refresh or 0) + 1

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
        file_bytes = state.extra.get("upload_file_bytes")
        source_path = state.extra.get("upload_source_path")
        filename = state.extra.get("upload_filename", "unknown.edf")

        from eeg_seizure_analyzer.io.edf_reader import read_edf, read_edf_paired
        from eeg_seizure_analyzer.io.edf_reader import auto_pair_channels

        channel_info = state.all_channels_info

        # Try auto-pairing
        eeg_indices, act_indices, pairings = auto_pair_channels(channel_info)
        has_pairs = any(p.activity_index is not None for p in pairings)

        # Get or create temp file
        if file_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(file_bytes)
                load_path = tmp.name
            cleanup = True
        elif source_path:
            load_path = source_path
            cleanup = False
        else:
            return alert("No file data available.", "danger"), no_update

        try:
            if has_pairs:
                sel_eeg = [i for i in selected_channels if i in eeg_indices]
                sel_act = act_indices
                eeg_rec, act_rec = read_edf_paired(load_path, sel_eeg, sel_act)
                eeg_rec.source_path = source_path or filename
                state.recording = eeg_rec
                state.activity_recordings = {"paired": act_rec}
                state.channel_pairings = pairings
            else:
                recording = read_edf(load_path, channels=selected_channels)
                recording.source_path = source_path or filename
                state.recording = recording
        finally:
            if cleanup:
                os.unlink(load_path)

        state.extra.pop("upload_file_bytes", None)

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
    state.extra.pop("upload_file_bytes", None)
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
    state.extra.pop("upload_file_bytes", None)
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
