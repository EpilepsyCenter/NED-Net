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

    # Default: upload area
    return _upload_layout()


def _upload_layout() -> html.Div:
    return html.Div(
        style={"padding": "24px", "maxWidth": "800px", "margin": "0 auto"},
        children=[
            html.H4("Upload Recording", style={"marginBottom": "24px"}),

            # Upload area
            dcc.Upload(
                id="upload-edf",
                children=html.Div(
                    className="upload-area",
                    children=[
                        html.Div("\u21E7", className="upload-icon"),
                        html.Div([
                            html.Strong("Drop an EDF file here"),
                            html.Br(),
                            "or click to browse",
                        ], className="upload-text"),
                        html.Div(
                            ".edf" + (" / .adicht" if sys.platform == "win32" else ""),
                            style={
                                "marginTop": "8px",
                                "fontSize": "0.75rem",
                                "opacity": "0.5",
                            },
                        ),
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

            # Change selection button
            html.Div(
                style={"marginTop": "20px"},
                children=[
                    dbc.Button(
                        "Change Channel Selection",
                        id="upload-change-btn",
                        className="btn-ned-secondary",
                    ),
                ],
            ),
        ],
    )


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
        return None, (refresh or 0) + 1

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
