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

# ── Import tab modules ────────────────────────────────────────────────

from eeg_seizure_analyzer.dash_app.pages import upload, viewer, seizures, spikes

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

# ── Clientside callback for slider/input sync ─────────────────────────
# Single callback with both as Input; uses dash_clientside.callback_context
# to determine which triggered and return the value to the other.

from dash import clientside_callback, MATCH, ALL

clientside_callback(
    """
    function(sliderVal, inputVal) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) return [sliderVal, sliderVal];
        const trigger = ctx.triggered[0].prop_id;
        if (trigger.includes('param-slider')) {
            return [sliderVal, sliderVal];
        }
        return [inputVal, inputVal];
    }
    """,
    Output({"type": "param-slider", "key": MATCH}, "value"),
    Output({"type": "param-input", "key": MATCH}, "value"),
    Input({"type": "param-slider", "key": MATCH}, "value"),
    Input({"type": "param-input", "key": MATCH}, "value"),
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
                    # ── Blinding ──────────────────────────────────
                    section_header("BLINDING"),
                    html.Div(id="blinding-badge-container",
                             children=[blinding_badge(True)]),
                    dbc.Switch(
                        id="blinding-toggle",
                        label="Blinding ON",
                        value=True,
                        className="mt-2",
                        style={"fontSize": "0.82rem"},
                    ),

                    sidebar_divider(),

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

                    # ── Channel selector ──────────────────────────
                    section_header("CHANNELS"),
                    html.Div(id="sidebar-channel-selector"),
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

TAB_DEFS = [
    ("upload", "Upload"),
    ("viewer", "Viewer"),
    ("seizures", "Seizure"),
    ("spikes", "Spikes"),
    ("training", "Training"),
    ("results", "Results"),
    ("settings", "Settings"),
]


def _tab_bar():
    return html.Div(
        id="tab-bar",
        children=[
            dbc.Nav(
                [
                    dbc.NavLink(
                        label,
                        id=f"tab-{tid}",
                        active=tid == "upload",
                        n_clicks=0,
                    )
                    for tid, label in TAB_DEFS
                ],
                pills=False,
                className="nav-tabs",
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

        # Sidebar
        _sidebar(),

        # Main content area
        html.Div(
            id="main-content",
            children=[
                _tab_bar(),
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


@callback(
    Output("active-tab", "data"),
    *[Output(f"tab-{tid}", "active") for tid, _ in TAB_DEFS],
    *[Input(f"tab-{tid}", "n_clicks") for tid, _ in TAB_DEFS],
    State("active-tab", "data"),
    prevent_initial_call=True,
)
def switch_tab(*args):
    """Handle tab click — update active tab store and nav link states."""
    n_tabs = len(TAB_DEFS)
    current = args[-1]  # last arg is the State

    triggered = ctx.triggered_id
    if triggered is None:
        return (current,) + tuple(tid == current for tid, _ in TAB_DEFS)

    # Extract tab id from the triggered component
    new_tab = triggered.replace("tab-", "")
    return (new_tab,) + tuple(tid == new_tab for tid, _ in TAB_DEFS)


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
        return _placeholder_tab(
            "Training",
            "ML model training will be available here.",
        )
    elif active_tab == "results":
        return _placeholder_tab(
            "Results",
            "Analysis results and group comparisons will be displayed here.",
        )
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
    Output("sidebar-channel-selector", "children"),
    Output("selected-channels", "data"),
    Input("tab-refresh", "data"),
    State("session-id", "data"),
    State("selected-channels", "data"),
)
def update_sidebar_info(_refresh, sid, current_selected):
    """Update sidebar recording info and channel toggles when a file is loaded."""
    state = server_state.get_session(sid)
    rec = state.recording

    if rec is None:
        return (
            html.Div("No file loaded", className="file-info",
                     style={"opacity": "0.5"}),
            html.Div(),
            None,
        )

    # File info
    import os
    fname = os.path.basename(rec.source_path) if rec.source_path else "Unknown"
    dur_h = rec.duration_sec / 3600
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
    ])

    # Channel toggles
    all_indices = list(range(rec.n_channels))
    if current_selected is None:
        selected = all_indices
    else:
        selected = current_selected

    channel_checklist = dbc.Checklist(
        id="sidebar-channel-checks",
        options=[
            {"label": rec.channel_names[i], "value": i}
            for i in range(rec.n_channels)
        ],
        value=selected,
        inline=False,
        style={"fontSize": "0.82rem"},
        className="channel-checklist",
    )

    select_btns = html.Div(
        style={"display": "flex", "gap": "8px", "marginTop": "8px"},
        children=[
            html.A("All", id="ch-select-all", href="#",
                   style={"fontSize": "0.75rem", "color": "#58a6ff",
                           "cursor": "pointer"}),
            html.A("None", id="ch-select-none", href="#",
                   style={"fontSize": "0.75rem", "color": "#58a6ff",
                           "cursor": "pointer"}),
        ],
    )

    channel_selector = html.Div([channel_checklist, select_btns])

    return file_info, channel_selector, selected


@callback(
    Output("selected-channels", "data", allow_duplicate=True),
    Input("sidebar-channel-checks", "value"),
    prevent_initial_call=True,
)
def on_channel_toggle(checked):
    """Store selected channels when user toggles checkboxes."""
    return checked


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
