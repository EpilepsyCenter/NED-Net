"""Interictal Spike Detection tab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    metric_card,
    no_recording_placeholder,
    param_control,
    collapsible_section,
    alert,
    empty_state,
)
from eeg_seizure_analyzer.config import SpikeDetectionParams

# ── Default parameter values ────────────────────────────────────────

_SP_DEFAULTS = {
    "sp-bp-low": 10.0,
    "sp-bp-high": 70.0,
    "sp-amp-thr": 4.0,
    "sp-min-amp": 0.0,
    "sp-prom": 1.5,
    "sp-maxw": 70.0,
    "sp-minw": 2.0,
    "sp-refr": 200.0,
    "sp-bl-pct": 15,
    "sp-bl-rms": 10.0,
}


def layout(sid: str | None) -> html.Div:
    """Return the spike detection tab layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording

    # Restore persisted slider values
    persisted = state.extra.get("sp_params", {})

    def _val(key):
        if key in persisted:
            return persisted[key]
        return _SP_DEFAULTS[key]

    # ── Explicit param save (belt-and-suspenders with ALL callback) ──
    resolved = {k: _val(k) for k in _SP_DEFAULTS}
    state.extra["sp_params"] = resolved

    # Persisted dropdown value
    persisted_bl_method = state.extra.get("sp_bl_method", "percentile")

    # Rebuild results from existing detections if present
    has_results = bool(state.spike_events)
    existing_results = html.Div()
    if has_results:
        existing_results = _build_results(rec, state.spike_events)

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "20px"},
                children=[
                    html.H4("Interictal Spike Detection", style={"margin": "0"}),
                ],
            ),

            # Parameters
            dbc.Row([
                dbc.Col([
                    _spike_params(_val),
                ], width=4),
                dbc.Col([
                    _morphology_params(_val),
                ], width=4),
                dbc.Col([
                    _baseline_params(_val, persisted_bl_method),
                ], width=4),
            ], className="g-3 mb-3"),

            # Action buttons
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "20px"},
                children=[
                    dbc.Button(
                        "Detect Spikes",
                        id="sp-detect-btn",
                        className="btn-ned-primary",
                    ),
                    dbc.Button(
                        "Clear Results",
                        id="sp-clear-btn",
                        className="btn-ned-danger",
                        style={"display": "inline-block" if has_results else "none"},
                    ),
                ],
            ),

            dcc.Loading(
                html.Div(id="sp-status"),
                type="circle", color="#58a6ff",
            ),

            # Results area — pre-populated if results exist
            html.Div(id="sp-results", children=existing_results),
        ],
    )


def _spike_params(_val) -> html.Div:
    return collapsible_section(
        "Detection", "sp-det",
        default_open=True,
        children=[
            param_control("Bandpass low (Hz)", "sp-bp-low", 1.0, 100.0, 1.0, _val("sp-bp-low")),
            param_control("Bandpass high (Hz)", "sp-bp-high", 10.0, 500.0, 1.0, _val("sp-bp-high")),
            param_control("Threshold (z-score)", "sp-amp-thr", 1.0, 30.0, 0.5, _val("sp-amp-thr"),
                          "Spike must exceed mean + z x std of baseline."),
            param_control("Min amplitude (uV)", "sp-min-amp", 0.0, 500.0, 5.0, _val("sp-min-amp"),
                          "Absolute floor. 0 = disabled."),
        ],
    )


def _morphology_params(_val) -> html.Div:
    return collapsible_section(
        "Morphology", "sp-morph",
        default_open=True,
        children=[
            param_control("Prominence (x baseline)", "sp-prom", 0.5, 10.0, 0.1, _val("sp-prom"),
                          "Must stand out from local context."),
            param_control("Max width (ms)", "sp-maxw", 10.0, 500.0, 5.0, _val("sp-maxw")),
            param_control("Min width (ms)", "sp-minw", 0.5, 50.0, 0.5, _val("sp-minw")),
            param_control("Refractory (ms)", "sp-refr", 10.0, 2000.0, 10.0, _val("sp-refr")),
        ],
    )


def _baseline_params(_val, bl_method="percentile") -> html.Div:
    return collapsible_section(
        "Baseline", "sp-baseline",
        default_open=True,
        children=[
            html.Div([
                html.Label("Method", style={"fontSize": "0.78rem", "color": "#8b949e",
                                            "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="sp-bl-method",
                    options=[
                        {"label": "Percentile", "value": "percentile"},
                        {"label": "Rolling", "value": "rolling"},
                        {"label": "First N min", "value": "first_n"},
                    ],
                    value=bl_method, clearable=False,
                    style={"fontSize": "0.82rem"},
                ),
            ], style={"marginBottom": "12px"}),
            param_control("Percentile", "sp-bl-pct", 1, 50, 1, _val("sp-bl-pct")),
            param_control("RMS window (s)", "sp-bl-rms", 1.0, 60.0, 1.0, _val("sp-bl-rms")),
        ],
    )


# ── Collapse toggles ─────────────────────────────────────────────────

for section_id in ["sp-det", "sp-morph", "sp-baseline"]:
    @callback(
        Output(f"{section_id}-collapse", "is_open"),
        Output(f"{section_id}-chevron", "children"),
        Input(f"{section_id}-header", "n_clicks"),
        State(f"{section_id}-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_section(n, is_open, _id=section_id):
        return not is_open, "\u25BC" if not is_open else "\u25B6"


# ── Auto-save non-MATCH components to server state ──────────────────


@callback(
    Output("store-sp-extras", "data"),
    Input("sp-bl-method", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def auto_save_sp_extras(bl_method, sid):
    """Save non-MATCH spike component values to server state on any change."""
    if not sid:
        return no_update
    state = server_state.get_session(sid)
    state.extra["sp_bl_method"] = bl_method
    return {"saved": True}


# ── Detection callback ───────────────────────────────────────────────


@callback(
    Output("sp-status", "children"),
    Output("sp-results", "children"),
    Output("sp-clear-btn", "style"),
    Input("sp-detect-btn", "n_clicks"),
    Input("sp-clear-btn", "n_clicks"),
    State({"type": "param-slider", "key": "sp-bp-low"}, "value"),
    State({"type": "param-slider", "key": "sp-bp-high"}, "value"),
    State({"type": "param-slider", "key": "sp-amp-thr"}, "value"),
    State({"type": "param-slider", "key": "sp-min-amp"}, "value"),
    State({"type": "param-slider", "key": "sp-prom"}, "value"),
    State({"type": "param-slider", "key": "sp-maxw"}, "value"),
    State({"type": "param-slider", "key": "sp-minw"}, "value"),
    State({"type": "param-slider", "key": "sp-refr"}, "value"),
    State("sp-bl-method", "value"),
    State({"type": "param-slider", "key": "sp-bl-pct"}, "value"),
    State({"type": "param-slider", "key": "sp-bl-rms"}, "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_spike_detection(
    detect_clicks, clear_clicks,
    bp_low, bp_high, amp_thr, min_amp,
    prom, maxw, minw, refr,
    bl_method, bl_pct, bl_rms,
    sid,
):
    """Run interictal spike detection or clear results."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    rec = state.recording

    if rec is None:
        return no_update, no_update, no_update

    if trigger == "sp-clear-btn":
        server_state.clear_detections(sid, "spikes")
        return (
            alert("Results cleared.", "info"),
            html.Div(),
            {"display": "none"},
        )

    if trigger != "sp-detect-btn":
        return no_update, no_update, no_update

    try:
        from eeg_seizure_analyzer.detection.spike import SpikeDetector

        params = SpikeDetectionParams(
            bandpass_low=float(bp_low),
            bandpass_high=float(bp_high),
            amplitude_threshold_zscore=float(amp_thr),
            spike_min_amplitude_uv=float(min_amp),
            spike_prominence_x_baseline=float(prom),
            max_duration_ms=float(maxw),
            min_duration_ms=float(minw),
            refractory_ms=float(refr),
            baseline_method=bl_method,
            baseline_percentile=int(bl_pct),
            baseline_rms_window_sec=float(bl_rms),
        )

        detector = SpikeDetector()
        all_spikes = []
        detection_info = {}

        for ch in range(rec.n_channels):
            ch_spikes = detector.detect(rec, ch, params=params)
            all_spikes.extend(ch_spikes)
            if hasattr(detector, "_last_detection_info"):
                detection_info[ch] = dict(detector._last_detection_info)

        state.spike_events = all_spikes
        state.sp_detection_info = detection_info
        state.detected_events = state.seizure_events + all_spikes

        results = _build_results(rec, all_spikes)

        return (
            alert(f"Found {len(all_spikes)} spike(s).", "success"),
            results,
            {"display": "inline-block"},
        )

    except Exception as e:
        return alert(f"Detection failed: {e}", "danger"), html.Div(), {"display": "none"}


def _build_results(rec, spikes):
    """Build spike results display."""
    if not spikes:
        return empty_state("\u2714", "No Spikes Found",
                           "No interictal spikes detected with current parameters.")

    rate = len(spikes) / (rec.duration_sec / 60) if rec.duration_sec > 0 else 0
    mean_amp = sum(e.features.get("amplitude", 0) for e in spikes) / len(spikes)

    metrics = dbc.Row([
        dbc.Col(metric_card("Total Spikes", str(len(spikes)), accent=True), width=3),
        dbc.Col(metric_card("Rate", f"{rate:.2f}/min"), width=3),
        dbc.Col(metric_card("Mean Amplitude", f"{mean_amp:.1f}"), width=3),
    ], className="g-3 mb-3")

    display = spikes[:200]
    table_data = []
    for i, e in enumerate(display):
        table_data.append({
            "#": i + 1,
            "Channel": rec.channel_names[e.channel],
            "Time (s)": round(e.features.get("peak_time_sec", e.onset_sec), 2),
            "Amplitude": round(e.features.get("amplitude", 0), 1),
            "x Baseline": round(e.features.get("amplitude_x_baseline", 0), 1),
            "Duration (ms)": round(e.features.get("duration_ms", 0), 1)
                if e.features.get("duration_ms") else "\u2014",
            "Confidence": round(e.confidence, 2),
        })

    col_defs = [
        {"field": "#", "width": 50},
        {"field": "Channel", "width": 100},
        {"field": "Time (s)", "width": 90},
        {"field": "Amplitude", "width": 90},
        {"field": "x Baseline", "width": 90},
        {"field": "Duration (ms)", "width": 100},
        {"field": "Confidence", "width": 90},
    ]

    caption = ""
    if len(spikes) > 200:
        caption = f"Showing first 200 of {len(spikes)} spikes."

    table = dag.AgGrid(
        id="sp-results-grid",
        rowData=table_data,
        columnDefs=col_defs,
        defaultColDef={"sortable": True, "resizable": True},
        className="ag-theme-alpine-dark",
        style={"height": "300px"},
        dashGridOptions={
            "animateRows": False,
            "rowSelection": {"mode": "singleRow"},
        },
    )

    # Spike rate over time
    from eeg_seizure_analyzer.detection.burden import compute_spike_rate
    time_bins, rate_arr = compute_spike_rate(spikes, rec.duration_sec, bin_sec=60.0)
    rate_fig = go.Figure()
    rate_fig.add_trace(go.Scatter(
        x=time_bins / 60, y=rate_arr,
        mode="lines", name="Spike rate",
        fill="tozeroy",
        line=dict(color="#58a6ff"),
        fillcolor="rgba(88, 166, 255, 0.15)",
    ))
    rate_fig.update_layout(
        title="Spike Rate Over Time",
        xaxis_title="Time (min)", yaxis_title="Spikes/min",
        height=250,
    )
    apply_fig_theme(rate_fig)

    return html.Div([
        metrics,
        html.Div(caption, style={"fontSize": "0.78rem", "color": "#8b949e",
                                  "marginBottom": "8px"}) if caption else None,
        table,
        html.Div(id="sp-event-inspector", style={"marginTop": "20px"}),
        dcc.Graph(figure=rate_fig, style={"marginTop": "20px", "borderRadius": "8px"}),
    ])
