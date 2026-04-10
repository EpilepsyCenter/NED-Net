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
    "sp-bp-low": 3.0,
    "sp-bp-high": 50.0,
    "sp-amp-thr": 7.0,
    "sp-min-amp": 0.0,
    "sp-prom": 6.0,
    "sp-maxw": 300.0,
    "sp-minw": 10.0,
    "sp-refr": 750.0,
    "sp-bl-pct": 25,
    "sp-bl-rms": 30.0,
    # Isolation
    "sp-iso-win": 2.0,
    "sp-iso-max": 1,
}

_SP_SLIDER_KEYS = list(_SP_DEFAULTS.keys())

# ── Filter defaults ─────────────────────────────────────────────────

# Default inspector options
_SP_INSP_DEFAULTS = {
    "show_baseline": True, "show_threshold": True,
    "bandpass": True, "xrange": 5.0, "yrange": 1.0,
}

_SP_FILTER_DEFAULTS = {
    "min_amp": 0, "max_amp": None,
    "min_xbl": 15, "max_xbl": None,
    "min_dur_ms": 0, "max_dur_ms": None,
    "min_conf": 0.7, "max_conf": None,
    "min_snr": 10, "max_snr": None,
    "min_sharp": 0, "max_sharp": None,
}

_SP_FILTER_MIN_IDS = [
    "sp-filter-min-amp", "sp-filter-min-xbl",
    "sp-filter-min-dur-ms", "sp-filter-min-conf",
    "sp-filter-min-snr", "sp-filter-min-sharp",
]
_SP_FILTER_MAX_IDS = [
    "sp-filter-max-amp", "sp-filter-max-xbl",
    "sp-filter-max-dur-ms", "sp-filter-max-conf",
    "sp-filter-max-snr", "sp-filter-max-sharp",
]
_SP_ALL_FILTER_IDS = _SP_FILTER_MIN_IDS + _SP_FILTER_MAX_IDS


# ── Layout ───────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the spike detection tab layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording

    # Restore persisted slider values (support overrides from recall)
    overrides = state.extra.get("sp_param_overrides", {})
    persisted = state.extra.get("sp_params", {})

    def _val(key):
        if overrides and key in overrides:
            return overrides[key]
        if key in persisted:
            return persisted[key]
        return _SP_DEFAULTS[key]

    # Explicit param save
    resolved = {k: _val(k) for k in _SP_DEFAULTS}
    state.extra["sp_params"] = resolved
    # Clear overrides after applying
    state.extra.pop("sp_param_overrides", None)

    # Persisted dropdown value
    persisted_bl_method = state.extra.get("sp_bl_method", "percentile")

    # Channel selection
    selected_channels = state.extra.get("sp_selected_channels",
                                        list(range(rec.n_channels)))

    # Filter values
    fv = {**_SP_FILTER_DEFAULTS, **state.extra.get("sp_filter_values", {})}
    state.extra["sp_filter_values"] = fv
    filter_enabled = state.extra.get("sp_filter_enabled", True)

    # Rebuild results from existing detections if present
    has_results = bool(state.spike_events)
    existing_results = html.Div()
    existing_rate_graph = html.Div()
    if has_results:
        if filter_enabled:
            filtered = _apply_spike_filters(state.spike_events, **fv)
        else:
            filtered = list(state.spike_events)
        existing_results, existing_rate_graph = _build_results(
            rec, state.spike_events, filtered, len(state.spike_events))

    ch_options = [{"label": rec.channel_names[i], "value": i}
                  for i in range(rec.n_channels)]

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

            # Channel selector
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    html.Label(
                        "Channels to analyze",
                        style={"fontSize": "0.82rem", "fontWeight": "500",
                               "marginBottom": "6px", "display": "block",
                               "color": "var(--ned-text-muted)"},
                    ),
                    dcc.Dropdown(
                        id="sp-channel-selector",
                        options=ch_options,
                        value=selected_channels,
                        multi=True,
                        placeholder="Select channels...",
                        style={"fontSize": "0.82rem"},
                    ),
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

            # Action buttons + settings buttons
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "20px",
                       "alignItems": "center"},
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
                    html.Div(style={"flex": "1"}),
                    dbc.Button("Restore Defaults", id="sp-recall-defaults-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Save User Params", id="sp-save-settings-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Recall User Params", id="sp-recall-settings-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Recall Detection Params", id="sp-recall-det-btn",
                               size="sm",
                               style={"backgroundColor": "#d29922",
                                      "borderColor": "#d29922",
                                      "color": "var(--ned-bg)",
                                      "fontWeight": "600",
                                      "display": "inline-block" if has_results else "none"}),
                ],
            ),

            html.Div(id="sp-settings-status", style={"marginBottom": "8px"}),

            dcc.Loading(
                html.Div(id="sp-status"),
                type="circle", color="#58a6ff",
            ),

            # Filters
            _spike_filter_controls(has_results, rec, fv, filter_enabled),

            # Results area — pre-populated if results exist
            html.Div(id="sp-results", children=existing_results),

            # Inspector controls — outside sp-results so they survive re-renders
            _sp_inspector_controls(has_results),

            # Inspector output
            html.Div(id="sp-event-inspector", style={"marginTop": "12px"}),

            # Rate graph — below inspector
            html.Div(id="sp-rate-graph", children=existing_rate_graph),

            # Export controls
            _sp_export_controls(has_results, state.spike_events, rec),
            dcc.Download(id="sp-export-download"),
        ],
    )


def _sp_export_controls(visible: bool, spike_events, rec) -> html.Div:
    """Build the export section for interictal spikes."""
    import os

    ch_options = []
    if rec is not None:
        ch_options = [{"label": rec.channel_names[i], "value": i}
                      for i in range(rec.n_channels)]

    fname = ""
    if rec and rec.source_path:
        base = os.path.splitext(os.path.basename(rec.source_path))[0]
        fname = f"{base}_spikes.csv"

    return html.Div(
        id="sp-export-section",
        style={"display": "block" if visible else "none",
               "marginTop": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px",
                       "marginBottom": "8px"},
                children=[
                    html.Span("Export Results",
                              style={"fontSize": "0.88rem", "fontWeight": "600",
                                     "color": "var(--ned-accent)"}),
                ],
            ),
            html.Div(
                style={"border": "1px solid var(--ned-border)",
                       "borderRadius": "8px", "padding": "16px",
                       "background": "var(--ned-surface)"},
                children=[
                    # Row 1: Channel
                    dbc.Row([
                        dbc.Col([
                            html.Label("Channel",
                                       style={"fontSize": "0.78rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="sp-export-channel",
                                options=ch_options,
                                value=None,
                                placeholder="All channels",
                                clearable=True,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], width=3),
                    ], className="g-2 mb-3"),
                    # Row 2: Field groups
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fields to export",
                                       style={"fontSize": "0.78rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Checklist(
                                id="sp-export-fields",
                                options=[
                                    {"label": "Core (ID, time, channel, amplitude, confidence)",
                                     "value": "core"},
                                    {"label": "Morphology (duration, sharpness, phase ratio, SNR)",
                                     "value": "morphology"},
                                    {"label": "Context (baseline, threshold, neighbours, slow-wave)",
                                     "value": "context"},
                                    {"label": "Spectral (band powers, dominant freq, entropy)",
                                     "value": "spectral"},
                                ],
                                value=["core", "morphology"],
                                style={"fontSize": "0.82rem"},
                            ),
                        ]),
                    ], className="mb-3"),
                    # Row 3: Filename + export button
                    dbc.Row([
                        dbc.Col([
                            html.Label("Filename",
                                       style={"fontSize": "0.78rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Input(
                                id="sp-export-filename",
                                type="text",
                                value=fname,
                                debounce=True,
                                className="form-control",
                                style={"width": "100%", "fontSize": "0.82rem"},
                            ),
                        ], width=5),
                        dbc.Col([
                            html.Div(style={"height": "20px"}),
                            dbc.Button(
                                "Export CSV",
                                id="sp-export-btn",
                                className="btn-ned-primary",
                                size="sm",
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Div(
                                id="sp-export-status",
                                style={"marginTop": "24px",
                                       "fontSize": "0.82rem",
                                       "color": "var(--ned-text-muted)"},
                            ),
                        ], width=5),
                    ], className="g-2"),
                ],
            ),
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
        "Baseline & Isolation", "sp-baseline",
        default_open=True,
        children=[
            html.Div([
                html.Label("Method", style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
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
            param_control("Isolation window (s)", "sp-iso-win", 0.5, 10.0, 0.5, _val("sp-iso-win"),
                          "Window around spike to count neighbours."),
            param_control("Max neighbours", "sp-iso-max", 1, 30, 1, _val("sp-iso-max"),
                          "Spikes with more neighbours are rejected (burst/seizure)."),
        ],
    )


# ── Filter controls ─────────────────────────────────────────────────


def _filter_range(label, fid_min, fid_max, min_val, max_val, step,
                  value_min, value_max=None):
    """A compact min–max input pair for filter controls."""
    _inp_style = {"width": "100%", "height": "28px", "fontSize": "0.78rem"}
    return dbc.Col([
        html.Label(label, style={"fontSize": "0.75rem", "color": "var(--ned-text-muted)"}),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "4px"},
            children=[
                dcc.Input(
                    id=fid_min, type="number",
                    min=min_val, max=max_val, step=step, value=value_min,
                    placeholder="min",
                    debounce=True, className="form-control",
                    style=_inp_style,
                ),
                html.Span("–", style={"color": "var(--ned-text-muted)", "fontSize": "0.8rem"}),
                dcc.Input(
                    id=fid_max, type="number",
                    min=min_val, max=max_val, step=step, value=value_max,
                    placeholder="max",
                    debounce=True, className="form-control",
                    style=_inp_style,
                ),
            ],
        ),
    ], width=2)


def _spike_filter_controls(visible, rec=None, fv=None,
                            filter_enabled=True):
    fv = fv or dict(_SP_FILTER_DEFAULTS)
    ch_options = []
    if rec is not None:
        ch_options = [{"label": rec.channel_names[i], "value": i}
                      for i in range(rec.n_channels)]

    return html.Div(
        id="sp-filter-section",
        style={"display": "block" if visible else "none",
               "marginBottom": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "8px"},
                children=[
                    html.Span("Result Filters",
                              style={"fontSize": "0.82rem", "fontWeight": "600",
                                     "color": "var(--ned-text-muted)"}),
                    dbc.Switch(
                        id="sp-filter-enabled",
                        value=filter_enabled,
                        style={"fontSize": "0.78rem"},
                    ),
                ],
            ),
            dbc.Row([
                _filter_range("Amplitude", "sp-filter-min-amp", "sp-filter-max-amp",
                              0, 1000, 1, fv.get("min_amp", 0), fv.get("max_amp")),
                _filter_range("x Baseline", "sp-filter-min-xbl", "sp-filter-max-xbl",
                              0, 100, 0.1, fv.get("min_xbl", 0), fv.get("max_xbl")),
                _filter_range("Duration (ms)", "sp-filter-min-dur-ms", "sp-filter-max-dur-ms",
                              0, 500, 1, fv.get("min_dur_ms", 0), fv.get("max_dur_ms")),
                _filter_range("Confidence", "sp-filter-min-conf", "sp-filter-max-conf",
                              0, 1, 0.05, fv.get("min_conf", 0), fv.get("max_conf")),
                _filter_range("Local SNR", "sp-filter-min-snr", "sp-filter-max-snr",
                              0, 100, 0.5, fv.get("min_snr", 0), fv.get("max_snr")),
                _filter_range("Sharpness", "sp-filter-min-sharp", "sp-filter-max-sharp",
                              0, 20, 0.1, fv.get("min_sharp", 0), fv.get("max_sharp")),
                dbc.Col([
                    html.Label("Channel", style={"fontSize": "0.75rem", "color": "var(--ned-text-muted)"}),
                    dcc.Dropdown(
                        id="sp-filter-channel",
                        options=ch_options,
                        value=None,
                        placeholder="All", clearable=True,
                        style={"fontSize": "0.8rem"},
                    ),
                ], width=2),
            ], className="g-2"),
        ],
    )


def _sp_inspector_controls(visible: bool, opts=None) -> html.Div:
    opts = opts or dict(_SP_INSP_DEFAULTS)
    _inp_style = {"width": "80px", "height": "30px", "fontSize": "0.8rem"}
    return html.Div(
        id="sp-inspector-controls",
        style={"display": "block" if visible else "none",
               "marginTop": "16px", "marginBottom": "8px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "flexWrap": "wrap"},
                children=[
                    html.Span("Inspector Options",
                              style={"fontSize": "0.82rem", "fontWeight": "600",
                                     "color": "var(--ned-text-muted)"}),
                    dbc.Checkbox(id="sp-insp-show-baseline", label="Baseline",
                                 value=opts.get("show_baseline", True),
                                 style={"fontSize": "0.8rem"}),
                    dbc.Checkbox(id="sp-insp-show-threshold", label="Threshold",
                                 value=opts.get("show_threshold", True),
                                 style={"fontSize": "0.8rem"}),
                    dbc.Checkbox(id="sp-insp-bandpass", label="Bandpass Filter",
                                 value=opts.get("bandpass", True),
                                 style={"fontSize": "0.8rem"}),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("X range (s):",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
                                              "margin": "0"}),
                            dcc.Input(
                                id="sp-insp-xrange", type="number",
                                min=0.1, step=0.5, value=opts.get("xrange", 5.0),
                                debounce=True, className="form-control",
                                style=_inp_style,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Y range (mV):",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
                                              "margin": "0"}),
                            dcc.Input(
                                id="sp-insp-yrange", type="number",
                                min=0, step=0.1, value=opts.get("yrange", 1.0),
                                debounce=True, className="form-control",
                                style=_inp_style,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ── Filter logic ────────────────────────────────────────────────────


def _apply_spike_filters(spikes, *, min_amp=0, max_amp=None,
                         min_xbl=0, max_xbl=None,
                         min_dur_ms=0, max_dur_ms=None,
                         min_conf=0, max_conf=None,
                         min_snr=0, max_snr=None,
                         min_sharp=0, max_sharp=None,
                         channel=None, **_kw):
    """Apply min/max filters to spike list."""
    filtered = list(spikes)
    min_amp = float(min_amp or 0)
    min_xbl = float(min_xbl or 0)
    min_dur_ms = float(min_dur_ms or 0)
    min_conf = float(min_conf or 0)
    min_snr = float(min_snr or 0)
    min_sharp = float(min_sharp or 0)

    def _fmax(v):
        if v is None or v == "":
            return None
        return float(v)

    max_amp = _fmax(max_amp)
    max_xbl = _fmax(max_xbl)
    max_dur_ms = _fmax(max_dur_ms)
    max_conf = _fmax(max_conf)
    max_snr = _fmax(max_snr)
    max_sharp = _fmax(max_sharp)

    if min_amp > 0:
        filtered = [e for e in filtered
                    if (e.features.get("amplitude") or 0) >= min_amp]
    if max_amp is not None:
        filtered = [e for e in filtered
                    if (e.features.get("amplitude") or 0) <= max_amp]
    if min_xbl > 0:
        filtered = [e for e in filtered
                    if (e.features.get("amplitude_x_baseline") or 0) >= min_xbl]
    if max_xbl is not None:
        filtered = [e for e in filtered
                    if (e.features.get("amplitude_x_baseline") or 0) <= max_xbl]
    if min_dur_ms > 0:
        filtered = [e for e in filtered
                    if (e.features.get("duration_ms") or 0) >= min_dur_ms]
    if max_dur_ms is not None:
        filtered = [e for e in filtered
                    if (e.features.get("duration_ms") or 0) <= max_dur_ms]
    if min_conf > 0:
        filtered = [e for e in filtered if e.confidence >= min_conf]
    if max_conf is not None:
        filtered = [e for e in filtered if e.confidence <= max_conf]
    if min_snr > 0:
        filtered = [e for e in filtered
                    if (e.features.get("local_snr") or 0) >= min_snr]
    if max_snr is not None:
        filtered = [e for e in filtered
                    if (e.features.get("local_snr") or 0) <= max_snr]
    if min_sharp > 0:
        filtered = [e for e in filtered
                    if (e.features.get("sharpness") or 0) >= min_sharp]
    if max_sharp is not None:
        filtered = [e for e in filtered
                    if (e.features.get("sharpness") or 0) <= max_sharp]
    if channel is not None and channel != "":
        filtered = [e for e in filtered if e.channel == int(channel)]
    return filtered


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
    Input("sp-channel-selector", "value"),
    *[Input(fid, "value") for fid in _SP_ALL_FILTER_IDS],
    Input("sp-filter-channel", "value"),
    Input("sp-filter-enabled", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def auto_save_sp_extras(*args):
    """Save non-MATCH spike component values to server state on any change."""
    n_filt = len(_SP_ALL_FILTER_IDS)
    bl_method = args[0]
    channels = args[1]
    filt_vals = args[2:2 + n_filt]
    filt_channel = args[2 + n_filt]
    filt_enabled = args[2 + n_filt + 1]
    sid = args[-1]

    if not sid:
        return no_update
    state = server_state.get_session(sid)
    if bl_method is not None:
        state.extra["sp_bl_method"] = bl_method
    if channels is not None:
        state.extra["sp_selected_channels"] = list(channels)

    # Merge filter values
    _min_keys = ["min_amp", "min_xbl", "min_dur_ms", "min_conf", "min_snr", "min_sharp"]
    _max_keys = ["max_amp", "max_xbl", "max_dur_ms", "max_conf", "max_snr", "max_sharp"]
    n_min = len(_SP_FILTER_MIN_IDS)
    existing_fv = state.extra.get("sp_filter_values", dict(_SP_FILTER_DEFAULTS))
    for k, v in zip(_min_keys, filt_vals[:n_min]):
        if v is not None:
            existing_fv[k] = v
    for k, v in zip(_max_keys, filt_vals[n_min:]):
        existing_fv[k] = v  # None is valid for max
    # Don't persist channel filter — it causes stale values on tab switch
    state.extra["sp_filter_values"] = existing_fv
    if filt_enabled is not None:
        state.extra["sp_filter_enabled"] = bool(filt_enabled)

    return {"saved": True}


# ── Detection callback ───────────────────────────────────────────────


@callback(
    Output("sp-status", "children"),
    Output("sp-results", "children"),
    Output("sp-clear-btn", "style"),
    Output("sp-filter-section", "style"),
    Output("sp-inspector-controls", "style"),
    Output("sp-event-inspector", "children"),
    Output("sp-rate-graph", "children"),
    Input("sp-detect-btn", "n_clicks"),
    Input("sp-clear-btn", "n_clicks"),
    # Filters trigger re-filter
    *[Input(fid, "value") for fid in _SP_ALL_FILTER_IDS],
    Input("sp-filter-channel", "value"),
    Input("sp-filter-enabled", "value"),
    # Detection params
    State("sp-channel-selector", "value"),
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
    State({"type": "param-slider", "key": "sp-iso-win"}, "value"),
    State({"type": "param-slider", "key": "sp-iso-max"}, "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_spike_detection(
    detect_clicks, clear_clicks,
    # Filter inputs (min + max)
    filt_min_amp, filt_min_xbl, filt_min_dur_ms, filt_min_conf, filt_min_snr, filt_min_sharp,
    filt_max_amp, filt_max_xbl, filt_max_dur_ms, filt_max_conf, filt_max_snr, filt_max_sharp,
    filt_channel, filt_enabled,
    # Detection params
    selected_channels,
    bp_low, bp_high, amp_thr, min_amp,
    prom, maxw, minw, refr,
    bl_method, bl_pct, bl_rms,
    iso_win, iso_max,
    sid,
):
    """Run interictal spike detection, clear results, or apply filters."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    rec = state.recording

    if rec is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    show_style = {"display": "inline-block"}
    hide_style = {"display": "none"}
    block_style = {"display": "block", "marginBottom": "16px"}
    insp_show = {"display": "block", "marginTop": "16px", "marginBottom": "8px"}

    filter_kwargs = dict(
        min_amp=filt_min_amp, min_xbl=filt_min_xbl,
        min_dur_ms=filt_min_dur_ms, min_conf=filt_min_conf,
        min_snr=filt_min_snr, min_sharp=filt_min_sharp,
        max_amp=filt_max_amp, max_xbl=filt_max_xbl,
        max_dur_ms=filt_max_dur_ms, max_conf=filt_max_conf,
        max_snr=filt_max_snr, max_sharp=filt_max_sharp,
        channel=filt_channel,
    )

    if trigger == "sp-clear-btn":
        server_state.clear_detections(sid, "spikes")
        state.extra.pop("sp_selected_idx", None)
        return (
            alert("Results cleared.", "info"),
            html.Div(), hide_style, hide_style, hide_style, html.Div(), html.Div(),
        )

    # Filter change — re-filter existing results
    is_filter_trigger = isinstance(trigger, str) and trigger.startswith("sp-filter-")
    if is_filter_trigger and state.spike_events:
        if filt_enabled:
            filtered = _apply_spike_filters(state.spike_events, **filter_kwargs)
        else:
            filtered = list(state.spike_events)
        n_total = len(state.spike_events)
        n_shown = len(filtered)
        results, rate_graph = _build_results(rec, state.spike_events, filtered, n_total)
        status = alert(f"Showing {n_shown} of {n_total} spike(s) after filtering.", "info")
        return status, results, show_style, block_style, no_update, no_update, rate_graph

    if trigger != "sp-detect-btn":
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # Warn if Animal IDs not assigned
    ch_ids = state.extra.get("channel_animal_ids", {})
    sel_channels = selected_channels or list(range(rec.n_channels))
    missing_ids = [ch for ch in sel_channels if ch not in ch_ids or not ch_ids[ch]]
    if missing_ids:
        return (
            alert(
                f"Animal IDs not assigned for channel(s): {missing_ids}. "
                "Go to the Load tab and fill in the Animal ID column before detecting.",
                "warning",
            ),
            no_update, no_update, no_update, no_update, no_update, no_update,
        )

    try:
        from eeg_seizure_analyzer.detection.spike import SpikeDetector
        from eeg_seizure_analyzer.io.persistence import save_spike_detections

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
            isolation_window_sec=float(iso_win),
            isolation_max_neighbours=int(iso_max),
        )

        detector = SpikeDetector()
        all_spikes = []
        detection_info = {}

        channels = selected_channels or list(range(rec.n_channels))

        # Use chunked detection for large files (>30 min per channel)
        _src = getattr(rec, "source_path", "") or ""
        _use_chunked = (
            _src.lower().endswith(".edf")
            and rec.duration_sec > 1800
        )

        if _use_chunked:
            from eeg_seizure_analyzer.detection.base import detect_chunked
            all_spikes, detection_info = detect_chunked(
                detector,
                path=_src,
                channels=channels,
                chunk_duration_sec=1800.0,
                overlap_sec=10.0,  # shorter overlap for spikes (short events)
                params=params,
            )
        else:
            for ch in channels:
                ch_spikes = detector.detect(rec, ch, params=params)
                all_spikes.extend(ch_spikes)
                if hasattr(detector, "_last_detection_info"):
                    detection_info[ch] = dict(detector._last_detection_info)

        # Assign animal IDs from channel mapping
        ch_ids = state.extra.get("channel_animal_ids", {})
        for ev in all_spikes:
            aid = ch_ids.get(ev.channel, "")
            if aid:
                ev.animal_id = aid

        state.spike_events = all_spikes
        state.sp_detection_info = detection_info
        state.detected_events = state.seizure_events + all_spikes

        # ── Refresh training-tab annotations ──────────────────────
        # Clear stale in-memory annotations so the training tab
        # re-initialises from fresh detections on next visit.
        try:
            from eeg_seizure_analyzer.io.annotation_store import (
                detections_to_annotations, merge_annotations,
                load_spike_annotations, save_spike_annotations,
            )
            new_annotations = detections_to_annotations(
                all_spikes, rec.source_path or "",
                animal_id=state.extra.get("trs_animal_id", ""),
            )
            existing = None
            if rec.source_path:
                existing = load_spike_annotations(rec.source_path)
            if existing:
                merged = merge_annotations(existing, new_annotations,
                                           tolerance_sec=0.05)
            else:
                merged = new_annotations
            state.extra["trs_annotations"] = [a.to_dict() for a in merged]
            state.extra["trs_current_idx"] = 0
            if rec.source_path:
                save_spike_annotations(rec.source_path, merged)
        except Exception:
            # Fallback: just clear so layout() rebuilds from spike_events
            state.extra.pop("trs_annotations", None)
            state.extra["trs_current_idx"] = 0

        # Save to disk
        if rec.source_path:
            _params = {k: getattr(params, k, None)
                       for k in params.__dataclass_fields__}
            # Also save slider keys for recall
            _params.update(state.extra.get("sp_params", {}))
            _params["sp-bl-method"] = bl_method
            try:
                save_spike_detections(
                    rec.source_path,
                    events=all_spikes,
                    detection_info=detection_info,
                    params_dict=_params,
                    channels=channels,
                    filter_settings={
                        "filter_enabled": state.extra.get("sp_filter_enabled", True),
                        "filter_values": state.extra.get("sp_filter_values", {}),
                    },
                )
            except Exception:
                import traceback
                traceback.print_exc()

        # Apply filters
        if filt_enabled:
            filtered = _apply_spike_filters(all_spikes, **filter_kwargs)
        else:
            filtered = all_spikes
        results, rate_graph = _build_results(rec, all_spikes, filtered, len(all_spikes))
        state.extra.pop("sp_selected_idx", None)

        return (
            alert(f"Found {len(all_spikes)} spike(s).", "success"),
            results, show_style, block_style, insp_show, html.Div(), rate_graph,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (alert(f"Detection failed: {e}", "danger"),
                html.Div(), hide_style, hide_style, hide_style, html.Div(), html.Div())


def _build_results(rec, all_spikes, spikes, total_count=None):
    """Build spike results display.

    ``all_spikes`` is the full unfiltered list (for index mapping).
    ``spikes`` is the filtered list to display.
    """
    if not spikes:
        return (empty_state("\u2714", "No Spikes Found",
                            "No interictal spikes detected with current parameters."),
                html.Div())

    n_shown = len(spikes)
    total = total_count or n_shown
    # Rate and mean amplitude reflect the filtered (shown) spikes
    shown_rate = n_shown / (rec.duration_sec / 60) if rec.duration_sec > 0 else 0
    mean_amp = sum(e.features.get("amplitude", 0) for e in spikes) / len(spikes)

    metrics = dbc.Row([
        dbc.Col(metric_card("Total Spikes", str(total), accent=True), width=3),
        dbc.Col(metric_card("Shown", str(n_shown)), width=3),
        dbc.Col(metric_card("Rate", f"{shown_rate:.2f}/min"), width=3),
        dbc.Col(metric_card("Mean Amplitude", f"{mean_amp:.1f}"), width=3),
    ], className="g-3 mb-3")

    # Build id(event) → original index map for correct inspector lookup
    id_to_orig = {id(e): i for i, e in enumerate(all_spikes)}

    display = spikes[:500]
    table_data = []
    for i, e in enumerate(display):
        f = e.features or {}
        orig_idx = id_to_orig.get(id(e), i)
        table_data.append({
            "#": i + 1,
            "Channel": rec.channel_names[e.channel] if e.channel < len(rec.channel_names) else f"Ch{e.channel}",
            "Time (s)": round(f.get("peak_time_sec", e.onset_sec), 2),
            "Amplitude": round(f.get("amplitude", 0), 1),
            "x Baseline": round(f.get("amplitude_x_baseline", 0), 1),
            "Duration (ms)": round(float(f["duration_ms"]), 1)
                if f.get("duration_ms") is not None else "",
            "SNR": round(f.get("local_snr", 0), 1),
            "Sharp": round(float(f["sharpness"]), 2)
                if f.get("sharpness") is not None else "",
            "ASW": "\u2713" if f.get("after_slow_wave") else "",
            "Confidence": round(e.confidence, 2),
            "_idx": orig_idx,
        })

    col_defs = [
        {"field": "#", "flex": 0.4},
        {"field": "Channel", "flex": 0.8},
        {"field": "Time (s)", "flex": 0.8},
        {"field": "Amplitude", "flex": 0.8},
        {"field": "x Baseline", "flex": 0.8},
        {"field": "Duration (ms)", "flex": 0.8},
        {"field": "SNR", "flex": 0.6, "headerTooltip": "Local signal-to-noise ratio"},
        {"field": "Sharp", "flex": 0.6, "headerTooltip": "Sharpness (rise/fall slope ratio)"},
        {"field": "ASW", "flex": 0.4, "headerTooltip": "After-slow-wave detected"},
        {"field": "Confidence", "flex": 0.7},
        {"field": "_idx", "hide": True},
    ]

    caption = ""
    if len(spikes) > 500:
        caption = f"Showing first 500 of {len(spikes)} spikes."

    table = dag.AgGrid(
        id="sp-results-grid",
        rowData=table_data,
        columnDefs=col_defs,
        defaultColDef={"sortable": True, "resizable": True},
        className="ag-theme-alpine-dark",
        style={"height": "300px", "width": "100%"},
        dashGridOptions={
            "animateRows": False,
            "rowSelection": {"mode": "singleRow"},
        },
    )

    # Spike rate over time (uses filtered spikes)
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
        title="Spike Rate Over Time (filtered)" if n_shown < total else "Spike Rate Over Time",
        xaxis_title="Time (min)", yaxis_title="Spikes/min",
        height=250,
    )
    apply_fig_theme(rate_fig)

    return html.Div([
        metrics,
        html.Div(caption, style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
                                  "marginBottom": "8px"}) if caption else None,
        table,
    ]), dcc.Graph(figure=rate_fig, style={"marginTop": "20px", "borderRadius": "8px"})


# ── Inspector callback ──────────────────────────────────────────────


@callback(
    Output("sp-event-inspector", "children", allow_duplicate=True),
    Input("sp-results-grid", "selectedRows"),
    Input("sp-insp-show-baseline", "value"),
    Input("sp-insp-show-threshold", "value"),
    Input("sp-insp-bandpass", "value"),
    Input("sp-insp-xrange", "value"),
    Input("sp-insp-yrange", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def show_spike_inspector(selected_rows, show_baseline, show_threshold,
                         bandpass_on, x_range, y_range, sid):
    """Show EEG trace around selected spike with baseline/threshold lines."""
    state = server_state.get_session(sid)
    rec = state.recording
    if rec is None or not state.spike_events:
        return no_update

    # Determine which spike to show — from grid click or persisted selection
    idx = None
    if selected_rows:
        row = selected_rows[0] if isinstance(selected_rows, list) else selected_rows
        idx = row.get("_idx", 0)
        state.extra["sp_selected_idx"] = idx
    else:
        idx = state.extra.get("sp_selected_idx")

    if idx is None:
        return html.Div()

    if idx >= len(state.spike_events):
        return html.Div("Spike not found.", style={"color": "var(--ned-danger)"})

    event = state.spike_events[idx]
    ch = event.channel
    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch{ch}"
    peak_time = event.features.get("peak_time_sec", event.onset_sec)
    features = event.features or {}

    # Window: x_range seconds centred on the spike
    if not x_range or x_range <= 0:
        x_range = 5.0
    half_x = x_range / 2.0
    start_sec = max(0, peak_time - half_x)
    end_sec = min(rec.duration_sec, peak_time + half_x)
    start_idx = int(start_sec * rec.fs)
    end_idx = min(int(end_sec * rec.fs), rec.n_samples)

    data = rec.data[ch, start_idx:end_idx].astype(np.float64)
    time_arr = np.arange(start_idx, end_idx) / rec.fs

    # Optional bandpass
    if bandpass_on:
        from eeg_seizure_analyzer.processing.preprocess import bandpass_filter
        bp_low = state.extra.get("sp_params", {}).get("sp-bp-low", 10.0)
        bp_high = state.extra.get("sp_params", {}).get("sp-bp-high", 70.0)
        try:
            data = bandpass_filter(data, rec.fs, float(bp_low), float(bp_high))
        except Exception:
            pass

    unit_label = rec.units[ch] if ch < len(rec.units) else ""

    # Y range
    if not y_range or y_range <= 0:
        y_range = 1.0
    half_yr = y_range / 2.0

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time_arr, y=data, mode="lines",
        name=ch_name, line=dict(color="#1b2a4a", width=1),
    ))
    # Mark the peak
    fig.add_vline(x=peak_time, line=dict(color="#f85149", width=1.5, dash="dash"))

    # Baseline line
    baseline_val = features.get("baseline_mean")
    if show_baseline and baseline_val is not None:
        fig.add_hline(
            y=baseline_val,
            line=dict(color="#3fb950", width=1, dash="dot"),
            annotation_text="Baseline",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-baseline_val,
            line=dict(color="#3fb950", width=1, dash="dot"),
        )

    # Threshold line
    threshold_val = features.get("threshold")
    if show_threshold and threshold_val is not None:
        fig.add_hline(
            y=threshold_val,
            line=dict(color="#d29922", width=1, dash="dash"),
            annotation_text="Threshold",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-threshold_val,
            line=dict(color="#d29922", width=1, dash="dash"),
        )

    fig.update_yaxes(
        title_text=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
        range=[-half_yr, half_yr],
        fixedrange=False,
        uirevision=f"sp_insp_y_{y_range}",
    )
    fig.update_xaxes(
        title_text="Time (s)",
        uirevision=f"sp_insp_x_{peak_time}_{ch}",
    )
    fig.update_layout(
        height=300,
        showlegend=False, dragmode="zoom",
        uirevision=f"sp_insp_{peak_time}_{ch}",
    )
    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    detail_metrics = dbc.Row([
        dbc.Col(metric_card("Channel", ch_name), width=2),
        dbc.Col(metric_card("Time", f"{peak_time:.3f}s"), width=2),
        dbc.Col(metric_card("Amplitude", f"{features.get('amplitude', 0):.1f}"), width=2),
        dbc.Col(metric_card("x Baseline", f"{features.get('amplitude_x_baseline', 0):.1f}"), width=2),
        dbc.Col(metric_card("Duration", f"{features.get('duration_ms', 0):.1f}ms"
                            if features.get('duration_ms') else "—"), width=2),
        dbc.Col(metric_card("Confidence", f"{event.confidence:.2f}"), width=2),
    ], className="g-3 mb-2")

    detail_metrics_2 = dbc.Row([
        dbc.Col(metric_card("Local SNR", f"{features.get('local_snr', 0):.1f}"), width=2),
        dbc.Col(metric_card("Sharpness", f"{features.get('sharpness', 1.0):.2f}"
                            if features.get('sharpness') else "—"), width=2),
        dbc.Col(metric_card("Phase Ratio", f"{features.get('phase_ratio', 1.0):.2f}"
                            if features.get('phase_ratio') else "—"), width=2),
        dbc.Col(metric_card("After Slow-Wave", "Yes" if features.get('after_slow_wave') else "No"), width=2),
        dbc.Col(metric_card("Neighbours", str(features.get('neighbours', 0))), width=2),
    ], className="g-3 mb-3")

    return html.Div([
        html.Hr(style={"borderColor": "var(--ned-border)", "margin": "24px 0"}),
        html.H5("Spike Inspector",
                 style={"marginBottom": "16px", "color": "var(--ned-accent)"}),
        detail_metrics,
        detail_metrics_2,
        dcc.Graph(figure=fig, config={"scrollZoom": True, "displayModeBar": True}),
    ])


# ── Save / Recall settings callbacks ────────────────────────────────


@callback(
    Output("sp-settings-status", "children"),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("sp-recall-defaults-btn", "n_clicks"),
    Input("sp-save-settings-btn", "n_clicks"),
    Input("sp-recall-settings-btn", "n_clicks"),
    Input("sp-recall-det-btn", "n_clicks"),
    *[State({"type": "param-slider", "key": k}, "value") for k in _SP_SLIDER_KEYS],
    State("sp-bl-method", "value"),
    State("sp-channel-selector", "value"),
    # Filter values
    *[State(fid, "value") for fid in _SP_ALL_FILTER_IDS],
    State("sp-filter-channel", "value"),
    State("sp-filter-enabled", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def handle_sp_settings(*args):
    """Handle recall defaults, save, and recall user settings for spikes."""
    trigger = ctx.triggered_id
    if trigger not in ("sp-recall-defaults-btn", "sp-save-settings-btn",
                       "sp-recall-settings-btn", "sp-recall-det-btn"):
        return no_update, no_update
    # Check actual click
    btn_clicks = {"sp-recall-defaults-btn": args[0],
                  "sp-save-settings-btn": args[1],
                  "sp-recall-settings-btn": args[2],
                  "sp-recall-det-btn": args[3]}
    if not btn_clicks.get(trigger):
        return no_update, no_update

    n_keys = len(_SP_SLIDER_KEYS)
    n_filt = len(_SP_ALL_FILTER_IDS)
    current_values = args[4:4 + n_keys]
    bl_method = args[4 + n_keys]
    channels = args[4 + n_keys + 1]
    # Filter states
    filter_offset = 4 + n_keys + 2
    filter_vals = args[filter_offset:filter_offset + n_filt]
    filt_channel = args[filter_offset + n_filt]
    filt_enabled = args[filter_offset + n_filt + 1]
    sid = args[-2]
    refresh = args[-1]
    state = server_state.get_session(sid)

    if trigger == "sp-recall-det-btn":
        # Load saved spike detection params from disk
        rec = state.recording
        if rec is None or not rec.source_path:
            return alert("No file loaded.", "warning"), no_update
        from eeg_seizure_analyzer.io.persistence import load_spike_detections
        result = load_spike_detections(rec.source_path)
        if result is None:
            return alert("No saved spike detections found on disk.", "warning"), no_update
        saved_params = result.get("params", {})
        if saved_params:
            state.extra["sp_param_overrides"] = dict(saved_params)
            state.extra["sp_params"] = dict(saved_params)
            if "sp-bl-method" in saved_params:
                state.extra["sp_bl_method"] = saved_params["sp-bl-method"]
        sp_channels = result.get("channels", [])
        if sp_channels:
            state.extra["sp_selected_channels"] = sp_channels
        sp_fs = result.get("filter_settings", {})
        if sp_fs:
            sp_filter_on = sp_fs.get("filter_enabled", True)
            sp_filter_vals_d = sp_fs.get("filter_values", {})
            sp_filter_vals_d.pop("channel", None)
            state.extra["sp_filter_enabled"] = sp_filter_on
            if sp_filter_vals_d:
                state.extra["sp_filter_values"] = sp_filter_vals_d
        n_p = len(saved_params)
        return alert(f"Detection params recalled ({n_p} params).", "success"), (refresh or 0) + 1

    if trigger == "sp-recall-defaults-btn":
        state.extra["sp_param_overrides"] = dict(_SP_DEFAULTS)
        return alert("Default parameters restored.", "info"), (refresh or 0) + 1

    if trigger == "sp-save-settings-btn":
        from eeg_seizure_analyzer.dash_app.components import save_spike_user_defaults
        params = {k: v for k, v in zip(_SP_SLIDER_KEYS, current_values)}
        params["sp-bl-method"] = bl_method
        # Include filter settings
        n_min = len(_SP_FILTER_MIN_IDS)
        _min_keys = ["min_amp", "min_xbl", "min_dur_ms", "min_conf", "min_snr"]
        _max_keys = ["max_amp", "max_xbl", "max_dur_ms", "max_conf", "max_snr"]
        filter_params = {}
        for k, v in zip(_min_keys, filter_vals[:n_min]):
            filter_params[f"filter-{k}"] = v
        for k, v in zip(_max_keys, filter_vals[n_min:]):
            filter_params[f"filter-{k}"] = v
        filter_params["filter-enabled"] = bool(filt_enabled) if filt_enabled is not None else True
        params.update(filter_params)
        path = save_spike_user_defaults(params)
        return alert(f"User params saved to {path}", "success"), no_update

    if trigger == "sp-recall-settings-btn":
        from eeg_seizure_analyzer.dash_app.components import load_spike_user_defaults
        saved = load_spike_user_defaults()
        if saved is None:
            return alert("No saved user params found.", "warning"), no_update
        # Separate filter settings from param overrides
        filter_vals_dict = {}
        filter_keys_map = {
            "filter-min_amp": "min_amp", "filter-min_xbl": "min_xbl",
            "filter-min_dur_ms": "min_dur_ms", "filter-min_conf": "min_conf",
            "filter-min_snr": "min_snr", "filter-min_sharp": "min_sharp",
            "filter-max_amp": "max_amp", "filter-max_xbl": "max_xbl",
            "filter-max_dur_ms": "max_dur_ms", "filter-max_conf": "max_conf",
            "filter-max_snr": "max_snr", "filter-max_sharp": "max_sharp",
        }
        for fk, mk in filter_keys_map.items():
            if fk in saved:
                filter_vals_dict[mk] = saved.pop(fk)
        filter_en = saved.pop("filter-enabled", True)
        if filter_vals_dict:
            state.extra["sp_filter_values"] = {**_SP_FILTER_DEFAULTS, **filter_vals_dict}
        state.extra["sp_filter_enabled"] = filter_en
        state.extra["sp_param_overrides"] = saved
        return alert("User params loaded.", "success"), (refresh or 0) + 1

    return no_update, no_update


# ── Export callback ────────────────────────────────────────────────────


def _round_or_none(val, decimals=4):
    """Round a value if not None."""
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return val


@callback(
    Output("sp-export-download", "data"),
    Output("sp-export-status", "children"),
    Input("sp-export-btn", "n_clicks"),
    State("sp-export-channel", "value"),
    State("sp-export-fields", "value"),
    State("sp-export-filename", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def export_spike_csv(n_clicks, channel, fields, filename, sid):
    """Export detected interictal spikes to CSV."""
    import pandas as pd
    from scipy.signal import welch as _welch

    if not n_clicks:
        return no_update, no_update

    state = server_state.get_session(sid)
    rec = state.recording
    if not state.spike_events:
        return no_update, "No spikes to export."

    events = list(state.spike_events)

    # Filter by channel
    if channel is not None and channel != "":
        events = [e for e in events if e.channel == int(channel)]

    if not events:
        return no_update, "No spikes match the selected filters."

    fields = fields or ["core"]

    rows = []
    for ev in events:
        row = {}
        feat = ev.features or {}

        if "core" in fields:
            ch_name = ""
            if rec and ev.channel < len(rec.channel_names):
                ch_name = rec.channel_names[ev.channel]
            peak_time = feat.get("peak_time_sec", ev.onset_sec)
            row.update({
                "event_id": ev.event_id,
                "peak_time_sec": round(peak_time, 4),
                "channel": ev.channel,
                "channel_name": ch_name,
                "animal_id": ev.animal_id,
                "amplitude": _round_or_none(feat.get("amplitude")),
                "amplitude_x_baseline": _round_or_none(feat.get("amplitude_x_baseline")),
                "confidence": round(ev.confidence, 4),
            })

        if "morphology" in fields:
            row.update({
                "duration_ms": _round_or_none(feat.get("duration_ms")),
                "sharpness": _round_or_none(feat.get("sharpness")),
                "phase_ratio": _round_or_none(feat.get("phase_ratio")),
                "local_snr": _round_or_none(feat.get("local_snr")),
                "rise_time_ms": _round_or_none(feat.get("rise_time_ms")),
                "fall_time_ms": _round_or_none(feat.get("fall_time_ms")),
            })

        if "context" in fields:
            row.update({
                "baseline_mean": _round_or_none(feat.get("baseline_mean")),
                "threshold": _round_or_none(feat.get("threshold")),
                "neighbours": feat.get("neighbours"),
                "after_slow_wave": bool(feat.get("after_slow_wave")),
                "isolation_score": _round_or_none(feat.get("isolation_score")),
            })

        # Spectral — compute from raw EEG around spike
        if "spectral" in fields and rec is not None:
            try:
                ch = ev.channel
                peak_time = feat.get("peak_time_sec", ev.onset_sec)
                # Use a 1-second window centred on the spike
                half_win = 0.5
                onset_s = int(max(0, (peak_time - half_win)) * rec.fs)
                offset_s = int(min(rec.duration_sec, (peak_time + half_win)) * rec.fs)
                offset_s = min(rec.n_samples, offset_s)
                segment = rec.data[ch, onset_s:offset_s].astype(np.float64)

                if len(segment) > int(rec.fs * 0.25):
                    nperseg = min(int(rec.fs), len(segment))
                    freqs, psd = _welch(segment, fs=rec.fs, nperseg=nperseg)

                    def _band_power(f_low, f_high):
                        mask = (freqs >= f_low) & (freqs <= f_high)
                        return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

                    delta = _band_power(0.5, 4)
                    theta = _band_power(4, 8)
                    alpha = _band_power(8, 13)
                    beta = _band_power(13, 30)
                    gamma = _band_power(30, 100)
                    total = delta + theta + alpha + beta + gamma
                    dom_freq = float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0

                    psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
                    psd_norm = psd_norm[psd_norm > 0]
                    entropy = float(-np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0.0

                    row.update({
                        "delta_power": round(delta, 4),
                        "theta_power": round(theta, 4),
                        "alpha_power": round(alpha, 4),
                        "beta_power": round(beta, 4),
                        "gamma_power": round(gamma, 4),
                        "total_power": round(total, 4),
                        "delta_rel": round(delta / total, 4) if total > 0 else 0,
                        "theta_rel": round(theta / total, 4) if total > 0 else 0,
                        "alpha_rel": round(alpha / total, 4) if total > 0 else 0,
                        "beta_rel": round(beta / total, 4) if total > 0 else 0,
                        "gamma_rel": round(gamma / total, 4) if total > 0 else 0,
                        "dominant_freq_hz": round(dom_freq, 2),
                        "spectral_entropy": round(entropy, 4),
                    })
                else:
                    row.update({k: None for k in [
                        "delta_power", "theta_power", "alpha_power",
                        "beta_power", "gamma_power", "total_power",
                        "delta_rel", "theta_rel", "alpha_rel",
                        "beta_rel", "gamma_rel",
                        "dominant_freq_hz", "spectral_entropy",
                    ]})
            except Exception:
                row.update({k: None for k in [
                    "delta_power", "theta_power", "alpha_power",
                    "beta_power", "gamma_power", "total_power",
                    "delta_rel", "theta_rel", "alpha_rel",
                    "beta_rel", "gamma_rel",
                    "dominant_freq_hz", "spectral_entropy",
                ]})

        rows.append(row)

    df = pd.DataFrame(rows)
    fname = filename or "spikes.csv"
    if not fname.endswith(".csv"):
        fname += ".csv"

    n = len(df)
    return (
        dcc.send_data_frame(df.to_csv, fname, index=False),
        f"Exported {n} spike(s).",
    )
