"""Seizure Detection tab: spike-train method with parameter controls."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from eeg_seizure_analyzer.config import SpikeTrainSeizureParams

# ── Default parameter values ────────────────────────────────────────

_SZ_DEFAULTS = {
    "sz-bp-low": 1.0,
    "sz-bp-high": 100.0,
    "sz-spike-amp": 3.0,
    "sz-spike-min-uv": 0.0,
    "sz-spike-prom": 1.5,
    "sz-spike-maxw": 70.0,
    "sz-spike-minw": 2.0,
    "sz-spike-refr": 50.0,
    "sz-max-isi": 500.0,
    "sz-min-spikes": 5,
    "sz-min-dur": 5.0,
    "sz-min-iei": 3.0,
    "sz-bl-pct": 15,
    "sz-bl-rms": 10.0,
    # Boundary — signal method
    "sz-bnd-rms-win": 100.0,
    "sz-bnd-rms-thr": 2.0,
    "sz-bnd-max-trim": 5.0,
    # Boundary — spike_density method
    "sz-bnd-window": 2.0,
    "sz-bnd-rate": 2.0,
    "sz-bnd-amp-x": 2.0,
    # HVSW
    "sz-hvsw-amp": 3.0,
    "sz-hvsw-freq": 2.0,
    "sz-hvsw-dur": 5.0,
    "sz-hvsw-max-ev": 0.4,
    # HPD
    "sz-hpd-amp": 2.0,
    "sz-hpd-freq": 5.0,
    "sz-hpd-dur": 10.0,
    # Convulsive
    "sz-conv-dur": 20.0,
    "sz-conv-amp": 5.0,
    "sz-conv-postictal": 5.0,
    # Local baseline (pre-ictal comparison window)
    "sz-lbl-start": 20.0,   # seconds before onset (stored positive, used negative)
    "sz-lbl-end": 5.0,      # seconds before onset (stored positive, used negative)
    "sz-lbl-trim-pct": 30.0,  # % of top-amplitude samples to trim from baseline
}

_SZ_SLIDER_KEYS = list(_SZ_DEFAULTS.keys())

# Default filter values
_FILTER_DEFAULTS = {
    "min_conf": 0, "max_conf": None,
    "min_dur": 0, "max_dur": None,
    "min_spikes": 0, "max_spikes": None,
    "min_amp": 0, "max_amp": None,
    "min_lbl": 0, "max_lbl": None,
    "min_top_amp": 0, "max_top_amp": None,
    "min_ll": 0, "max_ll": None,
    "min_energy": 0, "max_energy": None,
    "min_sigbl": 0, "max_sigbl": None,
    "min_freq": 0, "max_freq": None,
    "min_td": 0, "max_td": None,
    "channel": None, "severity": "",
}

# Default inspector options
_INSP_DEFAULTS = {
    "show_spikes": True, "show_baseline": True,
    "show_threshold": True, "bandpass": True,
}


def layout(sid: str | None) -> html.Div:
    """Return the seizure detection tab layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording

    # Check for param overrides (from recall defaults / recall settings)
    overrides = state.extra.pop("sz_param_overrides", {})
    persisted = state.extra.get("sz_params", {})

    def _val(key):
        if key in overrides:
            return overrides[key]
        if key in persisted:
            return persisted[key]
        return _SZ_DEFAULTS[key]

    # ── Explicit param save (belt-and-suspenders with ALL callback) ──
    # Ensures params survive tab switches even if the ALL-pattern callback
    # in main.py doesn't fire reliably during component mount/unmount.
    resolved = {k: _val(k) for k in _SZ_DEFAULTS}
    # Include dropdown values so they're saved in the detection file too
    resolved["sz-bl-method"] = state.extra.get("sz_bl_method", "percentile")
    resolved["sz-bnd-method"] = state.extra.get("sz_bnd_method", "signal")
    state.extra["sz_params"] = resolved

    # Channel selector — restore persisted selection
    ch_options = [
        {"label": rec.channel_names[i], "value": i}
        for i in range(rec.n_channels)
    ]
    persisted_channels = state.extra.get("sz_selected_channels", list(range(rec.n_channels)))
    persisted_channels = [c for c in persisted_channels if 0 <= c < rec.n_channels]
    if not persisted_channels:
        persisted_channels = list(range(rec.n_channels))

    # Persisted dropdown values
    persisted_bl_method = state.extra.get("sz_bl_method", "percentile")
    persisted_bnd_method = state.extra.get("sz_bnd_method", "signal")
    persisted_classify = state.extra.get("sz_classify_subtypes", False)
    if overrides:
        persisted_bl_method = overrides.get("sz-bl-method", persisted_bl_method)
        persisted_bnd_method = overrides.get("sz-bnd-method", persisted_bnd_method)

    # Persisted filter values
    # Start from defaults, then overlay any saved values (handles schema changes)
    fv = {**_FILTER_DEFAULTS, **state.extra.get("sz_filter_values", {})}
    # Force hidden filters to 0/None — they have no visible UI to reset them
    for _hk in ("min_ll", "min_energy", "min_sigbl", "min_td"):
        fv[_hk] = 0
    for _hk in ("max_ll", "max_energy", "max_sigbl", "max_td"):
        fv[_hk] = None
    state.extra["sz_filter_values"] = fv  # persist cleaned values
    filter_enabled = state.extra.get("sz_filter_enabled", True)

    # Inspector options
    insp_opts = state.extra.get("sz_inspector_opts", dict(_INSP_DEFAULTS))
    viewer_yr = state.extra.get("viewer_settings", {}).get("yrange", 1.0)
    insp_yr = insp_opts.get("yrange", state.extra.get("sz_inspector_yrange", viewer_yr))

    # Rebuild results from existing detections if present
    has_results = bool(state.seizure_events)
    existing_results = html.Div()
    selected_event_key = state.extra.get("sz_selected_event_key")
    if has_results:
        # Apply persisted filters to get the same filtered list
        if filter_enabled:
            filtered = _apply_filters(rec, state.seizure_events, persisted_classify,
                                      **{k: fv.get(k, v) for k, v in _FILTER_DEFAULTS.items()})
        else:
            filtered = list(state.seizure_events)
        # Keep detected_events in sync with the filtered set for viewer overlays
        state.detected_events = filtered + state.spike_events
        existing_results = _build_results(
            rec, filtered, persisted_classify,
            selected_event_key=selected_event_key,
            all_channels=persisted_channels,
        )

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "20px"},
                children=[
                    html.H4("Seizure Detection", style={"margin": "0"}),
                    html.Span(
                        "Spike-train method",
                        style={"fontSize": "0.78rem", "color": "#8b949e",
                               "border": "1px solid #2d333b", "borderRadius": "12px",
                               "padding": "2px 10px"},
                    ),
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
                               "color": "#8b949e"},
                    ),
                    dcc.Dropdown(
                        id="sz-channel-selector",
                        options=ch_options,
                        value=persisted_channels,
                        multi=True,
                        placeholder="Select channels...",
                        style={"fontSize": "0.82rem"},
                    ),
                ],
            ),

            # Parameter sections
            dbc.Row([
                dbc.Col([_spike_frontend_params(_val)], width=4),
                dbc.Col([_train_grouping_params(_val)], width=4),
                dbc.Col([_baseline_params(_val, persisted_bl_method)], width=4),
            ], className="g-3 mb-3"),

            dbc.Row([
                dbc.Col([_boundary_params(_val, persisted_bnd_method)], width=6),
                dbc.Col([_subtype_params(_val, persisted_classify)], width=6),
            ], className="g-3 mb-3"),

            # Action buttons
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "20px",
                       "flexWrap": "wrap"},
                children=[
                    dbc.Button("Detect Seizures", id="sz-detect-btn",
                               className="btn-ned-primary"),
                    dbc.Button("Clear Results", id="sz-clear-btn",
                               className="btn-ned-danger",
                               style={"display": "inline-block" if has_results else "none"}),
                    html.Div(style={"flex": "1"}),
                    dbc.Button("Recall Defaults", id="sz-recall-defaults-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Save User Params", id="sz-save-settings-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Recall User Params", id="sz-recall-settings-btn",
                               className="btn-ned-secondary", size="sm"),
                ],
            ),

            html.Div(id="sz-settings-status", style={"marginBottom": "8px"}),

            dcc.Loading(html.Div(id="sz-status"), type="circle", color="#58a6ff"),

            # Confidence filtering controls
            _confidence_filter_controls(has_results, rec, fv,
                                        filter_enabled=filter_enabled),

            # Results area — pre-populated if results exist
            html.Div(id="sz-results", children=existing_results),

            # Inspector controls
            _inspector_controls(has_results, insp_yr, insp_opts),

            # Event inspector — pre-render if event was selected before tab switch
            html.Div(
                id="sz-event-inspector",
                style={"marginTop": "4px"},
                children=_prerender_inspector(state, rec, selected_event_key, insp_opts, insp_yr)
                if has_results and selected_event_key else [],
            ),
        ],
    )


# ── Confidence filtering controls ──────────────────────────────────


def _filter_range(label, fid_min, fid_max, min_val, max_val, step,
                   value_min, value_max=None):
    """A compact min–max input pair for filter controls."""
    _inp_style = {"width": "100%", "height": "28px", "fontSize": "0.78rem"}
    return dbc.Col([
        html.Label(label, style={"fontSize": "0.75rem", "color": "#8b949e"}),
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
                html.Span("–", style={"color": "#8b949e", "fontSize": "0.8rem"}),
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


def _confidence_filter_controls(visible: bool, rec=None, fv=None,
                                filter_enabled: bool = True) -> html.Div:
    fv = fv or dict(_FILTER_DEFAULTS)
    ch_options = []
    if rec is not None:
        ch_options = [{"label": rec.channel_names[i], "value": i}
                      for i in range(rec.n_channels)]

    return html.Div(
        id="sz-confidence-section",
        style={"display": "block" if visible else "none",
               "marginBottom": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "8px"},
                children=[
                    html.Span("Result Filters",
                              style={"fontSize": "0.82rem", "fontWeight": "600",
                                     "color": "#8b949e"}),
                    dbc.Switch(
                        id="sz-filter-enabled",
                        value=filter_enabled,
                        style={"fontSize": "0.78rem"},
                    ),
                ],
            ),
            # Row 1: basic filters (min–max)
            dbc.Row([
                _filter_range("Confidence", "sz-filter-min-conf", "sz-filter-max-conf",
                              0, 1, 0.05, fv.get("min_conf", 0), fv.get("max_conf")),
                _filter_range("Duration (s)", "sz-filter-min-dur", "sz-filter-max-dur",
                              0, 120, 0.5, fv.get("min_dur", 0), fv.get("max_dur")),
                _filter_range("Spikes", "sz-filter-min-spikes", "sz-filter-max-spikes",
                              0, 100, 1, fv.get("min_spikes", 0), fv.get("max_spikes")),
                _filter_range("Amp (xBL)", "sz-filter-min-amp", "sz-filter-max-amp",
                              0, 50, 0.5, fv.get("min_amp", 0), fv.get("max_amp")),
                _filter_range("Local BL", "sz-filter-min-lbl", "sz-filter-max-lbl",
                              0, 10, 0.1, fv.get("min_lbl", 0), fv.get("max_lbl")),
                _filter_range("Top Amp", "sz-filter-min-top-amp", "sz-filter-max-top-amp",
                              0, 20, 0.5, fv.get("min_top_amp", 0), fv.get("max_top_amp")),
            ], className="g-2 mb-2"),
            # Row 2: quality metric filters + channel/severity
            dbc.Row([
                _filter_range("Freq (Hz)", "sz-filter-min-freq", "sz-filter-max-freq",
                              0, 50, 0.5, fv.get("min_freq", 0), fv.get("max_freq")),
                dbc.Col([
                    html.Label("Channel", style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    dcc.Dropdown(
                        id="sz-filter-channel",
                        options=ch_options,
                        value=fv.get("channel", None),
                        placeholder="All", clearable=True,
                        style={"fontSize": "0.8rem"},
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Severity", style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    dcc.Dropdown(
                        id="sz-filter-severity",
                        options=[
                            {"label": "All", "value": ""},
                            {"label": "Mild", "value": "mild"},
                            {"label": "Moderate", "value": "moderate"},
                            {"label": "Severe", "value": "severe"},
                        ],
                        value=fv.get("severity", ""), clearable=False,
                        style={"fontSize": "0.8rem"},
                    ),
                ], width=2),
            ], className="g-2"),
            # Hidden filters — kept for callback compatibility / future use
            html.Div(style={"display": "none"}, children=[
                _filter_range("LL (z)", "sz-filter-min-ll", "sz-filter-max-ll",
                              0, 50, 0.5, fv.get("min_ll", 0), fv.get("max_ll")),
                _filter_range("Energy (z)", "sz-filter-min-energy", "sz-filter-max-energy",
                              0, 50, 0.5, fv.get("min_energy", 0), fv.get("max_energy")),
                _filter_range("Sig/BL", "sz-filter-min-sigbl", "sz-filter-max-sigbl",
                              0, 50, 0.5, fv.get("min_sigbl", 0), fv.get("max_sigbl")),
                _filter_range("θ/δ", "sz-filter-min-td", "sz-filter-max-td",
                              0, 10, 0.1, fv.get("min_td", 0), fv.get("max_td")),
            ]),
        ],
    )


# ── Filter input IDs ──────────────────────────────────────────────

_ALL_FILTER_MIN_IDS = [
    "sz-filter-min-conf", "sz-filter-min-dur",
    "sz-filter-min-spikes", "sz-filter-min-amp",
    "sz-filter-min-lbl", "sz-filter-min-top-amp",
    "sz-filter-min-ll", "sz-filter-min-energy",
    "sz-filter-min-sigbl", "sz-filter-min-freq",
    "sz-filter-min-td",
]

_ALL_FILTER_MAX_IDS = [
    "sz-filter-max-conf", "sz-filter-max-dur",
    "sz-filter-max-spikes", "sz-filter-max-amp",
    "sz-filter-max-lbl", "sz-filter-max-top-amp",
    "sz-filter-max-ll", "sz-filter-max-energy",
    "sz-filter-max-sigbl", "sz-filter-max-freq",
    "sz-filter-max-td",
]

_ALL_FILTER_IDS = _ALL_FILTER_MIN_IDS + _ALL_FILTER_MAX_IDS


# ── Inspector controls ─────────────────────────────────────────────


def _inspector_controls(visible: bool, y_range: float, opts=None) -> html.Div:
    opts = opts or dict(_INSP_DEFAULTS)
    return html.Div(
        id="sz-inspector-controls",
        style={"display": "block" if visible else "none",
               "marginTop": "16px", "marginBottom": "8px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "flexWrap": "wrap"},
                children=[
                    html.Span("Inspector Options",
                              style={"fontSize": "0.82rem", "fontWeight": "600",
                                     "color": "#8b949e"}),
                    dbc.Checkbox(id="sz-insp-show-spikes", label="Spikes",
                                 value=opts.get("show_spikes", True),
                                 style={"fontSize": "0.8rem"}),
                    dbc.Checkbox(id="sz-insp-show-baseline", label="Baseline",
                                 value=opts.get("show_baseline", True),
                                 style={"fontSize": "0.8rem"}),
                    dbc.Checkbox(id="sz-insp-show-threshold", label="Threshold",
                                 value=opts.get("show_threshold", True),
                                 style={"fontSize": "0.8rem"}),
                    dbc.Checkbox(id="sz-insp-bandpass", label="Bandpass Filter",
                                 value=opts.get("bandpass", True),
                                 style={"fontSize": "0.8rem"}),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Y range:",
                                       style={"fontSize": "0.78rem", "color": "#8b949e",
                                              "margin": "0"}),
                            dcc.Input(
                                id="sz-insp-yrange", type="number",
                                min=0, step=0.01, value=y_range,
                                debounce=True, className="form-control",
                                style={"width": "80px", "height": "30px",
                                       "fontSize": "0.8rem"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ── Parameter section builders ────────────────────────────────────────


def _spike_frontend_params(_val) -> html.Div:
    return collapsible_section(
        "Spike Front-end", "sz-spike",
        default_open=True,
        children=[
            param_control("Bandpass low (Hz)", "sz-bp-low", 0.5, 100.0, 0.5, _val("sz-bp-low")),
            param_control("Bandpass high (Hz)", "sz-bp-high", 10.0, 500.0, 1.0, _val("sz-bp-high")),
            param_control("Threshold (z-score)", "sz-spike-amp", 1.0, 10.0, 0.5, _val("sz-spike-amp"),
                          "Z-score multiplier: threshold = mean + z x std"),
            param_control("Min amplitude (uV)", "sz-spike-min-uv", 0.0, 500.0, 5.0, _val("sz-spike-min-uv"),
                          "Absolute floor. 0 = disabled."),
            param_control("Prominence (x baseline)", "sz-spike-prom", 0.5, 10.0, 0.5, _val("sz-spike-prom"),
                          "Spike must stand out from local context by this multiple."),
            param_control("Max width (ms)", "sz-spike-maxw", 10.0, 200.0, 5.0, _val("sz-spike-maxw")),
            param_control("Min width (ms)", "sz-spike-minw", 0.5, 20.0, 0.5, _val("sz-spike-minw")),
            param_control("Refractory (ms)", "sz-spike-refr", 5.0, 200.0, 5.0, _val("sz-spike-refr")),
        ],
    )


def _train_grouping_params(_val) -> html.Div:
    return collapsible_section(
        "Train Grouping", "sz-train",
        default_open=True,
        children=[
            param_control("Max ISI (ms)", "sz-max-isi", 50.0, 2000.0, 50.0, _val("sz-max-isi"),
                          "Max inter-spike interval within a train."),
            param_control("Min spikes", "sz-min-spikes", 2, 50, 1, _val("sz-min-spikes")),
            param_control("Min duration (s)", "sz-min-dur", 1.0, 60.0, 0.5, _val("sz-min-dur")),
            param_control("Min inter-event (s)", "sz-min-iei", 0.5, 30.0, 0.5, _val("sz-min-iei")),
        ],
    )


def _baseline_params(_val, bl_method="percentile") -> html.Div:
    return collapsible_section(
        "Baseline", "sz-baseline",
        default_open=True,
        children=[
            html.Div([
                html.Label("Method", style={"fontSize": "0.78rem", "color": "#8b949e",
                                            "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="sz-bl-method",
                    options=[
                        {"label": "Percentile", "value": "percentile"},
                        {"label": "Rolling", "value": "rolling"},
                        {"label": "First N min", "value": "first_n"},
                    ],
                    value=bl_method, clearable=False,
                    style={"fontSize": "0.82rem"},
                ),
            ], style={"marginBottom": "12px"}),
            param_control("Percentile", "sz-bl-pct", 1, 50, 1, _val("sz-bl-pct")),
            param_control("RMS window (s)", "sz-bl-rms", 1.0, 60.0, 1.0, _val("sz-bl-rms")),
            html.Hr(style={"margin": "10px 0", "borderColor": "#30363d"}),
            html.Span("Pre-ictal local baseline",
                       style={"fontSize": "0.75rem", "color": "#8b949e",
                              "display": "block", "marginBottom": "6px"}),
            param_control("Window start (s before onset)", "sz-lbl-start",
                          1.0, 120.0, 1.0, _val("sz-lbl-start")),
            param_control("Window end (s before onset)", "sz-lbl-end",
                          1.0, 60.0, 1.0, _val("sz-lbl-end")),
            param_control("Trim top % (spike removal)", "sz-lbl-trim-pct",
                          0, 80, 5, _val("sz-lbl-trim-pct")),
        ],
    )


def _boundary_params(_val, bnd_method="signal") -> html.Div:
    # Show controls for the selected method
    signal_controls = html.Div(
        id="sz-bnd-signal-controls",
        style={"display": "block" if bnd_method == "signal" else "none"},
        children=[
            param_control("RMS window (ms)", "sz-bnd-rms-win", 10.0, 500.0, 10.0, _val("sz-bnd-rms-win")),
            param_control("RMS threshold (x BL)", "sz-bnd-rms-thr", 0.5, 10.0, 0.5, _val("sz-bnd-rms-thr")),
            param_control("Max trim (s)", "sz-bnd-max-trim", 0.5, 20.0, 0.5, _val("sz-bnd-max-trim")),
        ],
    )
    density_controls = html.Div(
        id="sz-bnd-density-controls",
        style={"display": "block" if bnd_method == "spike_density" else "none"},
        children=[
            param_control("Boundary window (s)", "sz-bnd-window", 0.5, 10.0, 0.5, _val("sz-bnd-window")),
            param_control("Min rate (Hz)", "sz-bnd-rate", 0.5, 20.0, 0.5, _val("sz-bnd-rate")),
            param_control("Min amplitude (x BL)", "sz-bnd-amp-x", 0.5, 10.0, 0.5, _val("sz-bnd-amp-x")),
        ],
    )
    return collapsible_section(
        "Boundary Refinement", "sz-boundary",
        default_open=True,
        children=[
            html.Div([
                html.Label("Method", style={"fontSize": "0.78rem", "color": "#8b949e",
                                            "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="sz-bnd-method",
                    options=[
                        {"label": "Signal (RMS)", "value": "signal"},
                        {"label": "Spike density", "value": "spike_density"},
                        {"label": "None", "value": "none"},
                    ],
                    value=bnd_method, clearable=False,
                    style={"fontSize": "0.82rem"},
                ),
            ], style={"marginBottom": "12px"}),
            signal_controls,
            density_controls,
        ],
    )


def _subtype_params(_val, classify_on=False) -> html.Div:
    _sub_label = {"fontSize": "0.72rem", "color": "#8b949e",
                  "fontWeight": "600", "letterSpacing": "1px",
                  "marginBottom": "6px", "marginTop": "10px"}
    return collapsible_section(
        "Subtype Classification", "sz-subtype",
        default_open=False,
        children=[
            dbc.Checkbox(
                id="sz-classify-subtypes", label="Enable subtype classification",
                value=classify_on, style={"fontSize": "0.82rem", "marginBottom": "12px"},
            ),
            html.Div([
                html.Div("HVSW", style=_sub_label),
                param_control("Amplitude (x BL)", "sz-hvsw-amp", 1.0, 10.0, 0.5, _val("sz-hvsw-amp")),
                param_control("Min freq (Hz)", "sz-hvsw-freq", 0.5, 10.0, 0.5, _val("sz-hvsw-freq")),
                param_control("Min duration (s)", "sz-hvsw-dur", 1.0, 30.0, 0.5, _val("sz-hvsw-dur")),
                param_control("Max evolution (CV)", "sz-hvsw-max-ev", 0.1, 1.0, 0.05, _val("sz-hvsw-max-ev"),
                              "Max CV of ISI — monomorphic if below this."),

                html.Div("HPD", style=_sub_label),
                param_control("Amplitude (x BL)", "sz-hpd-amp", 1.0, 10.0, 0.5, _val("sz-hpd-amp")),
                param_control("Min freq (Hz)", "sz-hpd-freq", 1.0, 20.0, 0.5, _val("sz-hpd-freq")),
                param_control("Min duration (s)", "sz-hpd-dur", 1.0, 60.0, 1.0, _val("sz-hpd-dur")),

                html.Div("Electroclinical / Convulsive", style=_sub_label),
                param_control("Min duration (s)", "sz-conv-dur", 5.0, 120.0, 1.0, _val("sz-conv-dur")),
                param_control("Amplitude (x BL)", "sz-conv-amp", 2.0, 20.0, 0.5, _val("sz-conv-amp")),
                param_control("Post-ictal suppression (s)", "sz-conv-postictal", 1.0, 30.0, 1.0, _val("sz-conv-postictal")),
            ]),
        ],
    )


# ── Collapse toggle callbacks ─────────────────────────────────────────

for section_id in ["sz-spike", "sz-train", "sz-baseline", "sz-boundary", "sz-subtype"]:
    @callback(
        Output(f"{section_id}-collapse", "is_open"),
        Output(f"{section_id}-chevron", "children"),
        Input(f"{section_id}-header", "n_clicks"),
        State(f"{section_id}-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_section(n, is_open, _id=section_id):
        return not is_open, "\u25BC" if not is_open else "\u25B6"


# ── Boundary method toggle ──────────────────────────────────────────


@callback(
    Output("sz-bnd-signal-controls", "style"),
    Output("sz-bnd-density-controls", "style"),
    Input("sz-bnd-method", "value"),
    prevent_initial_call=True,
)
def toggle_boundary_controls(method):
    """Show/hide boundary sub-controls based on selected method."""
    show = {"display": "block"}
    hide = {"display": "none"}
    if method == "signal":
        return show, hide
    elif method == "spike_density":
        return hide, show
    return hide, hide


# ── Auto-save non-MATCH components to server state ──────────────────


@callback(
    Output("store-sz-extras", "data"),
    Input("sz-channel-selector", "value"),
    Input("sz-bl-method", "value"),
    Input("sz-bnd-method", "value"),
    Input("sz-classify-subtypes", "value"),
    # Filter values (min + max)
    *[Input(fid, "value") for fid in _ALL_FILTER_IDS],
    Input("sz-filter-channel", "value"),
    Input("sz-filter-severity", "value"),
    # Filter enabled toggle
    Input("sz-filter-enabled", "value"),
    # Inspector options
    Input("sz-insp-show-spikes", "value"),
    Input("sz-insp-show-baseline", "value"),
    Input("sz-insp-show-threshold", "value"),
    Input("sz-insp-bandpass", "value"),
    Input("sz-insp-yrange", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def auto_save_sz_extras(*args):
    """Save all non-MATCH seizure component values to server state on any change."""
    # Unpack *args: 4 fixed + n_min + n_max filter IDs + 2 dropdowns + 1 toggle + 5 inspector + sid
    n_min = len(_ALL_FILTER_MIN_IDS)
    n_max = len(_ALL_FILTER_MAX_IDS)
    channels, bl_method, bnd_method, classify = args[0:4]
    filt_min_vals = args[4:4 + n_min]
    filt_max_vals = args[4 + n_min:4 + n_min + n_max]
    filt_channel = args[4 + n_min + n_max]
    filt_severity = args[4 + n_min + n_max + 1]
    filt_enabled = args[4 + n_min + n_max + 2]
    insp_spikes, insp_baseline, insp_threshold, insp_bp, insp_yr = args[-6:-1]
    sid = args[-1]

    if not sid:
        return no_update
    state = server_state.get_session(sid)
    # Only update values that are not None (None means component was unmounted)
    if channels is not None:
        state.extra["sz_selected_channels"] = list(channels)
    if bl_method is not None:
        state.extra["sz_bl_method"] = bl_method
    if bnd_method is not None:
        state.extra["sz_bnd_method"] = bnd_method
    if classify is not None:
        state.extra["sz_classify_subtypes"] = classify
    # Merge filter values — only update keys with non-None values
    _min_keys = ["min_conf", "min_dur", "min_spikes", "min_amp", "min_lbl",
                 "min_top_amp", "min_ll", "min_energy", "min_sigbl",
                 "min_freq", "min_td"]
    _max_keys = ["max_conf", "max_dur", "max_spikes", "max_amp", "max_lbl",
                 "max_top_amp", "max_ll", "max_energy", "max_sigbl",
                 "max_freq", "max_td"]
    new_fv = {}
    for k, v in zip(_min_keys, filt_min_vals):
        new_fv[k] = v
    for k, v in zip(_max_keys, filt_max_vals):
        new_fv[k] = v
    new_fv["channel"] = filt_channel
    new_fv["severity"] = filt_severity
    existing_fv = state.extra.get("sz_filter_values", dict(_FILTER_DEFAULTS))
    for k, v in new_fv.items():
        if v is not None:
            existing_fv[k] = v
    state.extra["sz_filter_values"] = existing_fv
    if filt_enabled is not None:
        state.extra["sz_filter_enabled"] = bool(filt_enabled)
    # Merge inspector opts — only update non-None values
    existing_insp = state.extra.get("sz_inspector_opts", dict(_INSP_DEFAULTS))
    new_insp = {
        "show_spikes": insp_spikes, "show_baseline": insp_baseline,
        "show_threshold": insp_threshold, "bandpass": insp_bp,
    }
    for k, v in new_insp.items():
        if v is not None:
            existing_insp[k] = v
    if insp_yr is not None and insp_yr > 0:
        existing_insp["yrange"] = float(insp_yr)
    state.extra["sz_inspector_opts"] = existing_insp

    # Auto-save filter settings to the detection JSON file on every change
    # so they persist across sessions and load into both Seizure & Training tabs
    try:
        rec = state.recording
        _src = getattr(rec, "source_path", None) or "" if rec else ""
        if _src and _src.lower().endswith(".edf") and state.seizure_events:
            from dataclasses import asdict
            from eeg_seizure_analyzer.io.persistence import save_detections

            _params = state.extra.get("sz_params", {})
            save_detections(
                edf_path=_src,
                events=state.seizure_events,
                detection_info=state.st_detection_info,
                params_dict=_params,
                detector_name="SpikeTrainSeizureDetector",
                channels=state.extra.get("sz_selected_channels", []),
                animal_id=getattr(state, "animal_id", ""),
                filter_settings={
                    "filter_enabled": state.extra.get("sz_filter_enabled", False),
                    "filter_values": state.extra.get("sz_filter_values", {}),
                },
            )
    except Exception:
        import traceback
        traceback.print_exc()

    return {"saved": True}


# ── Detection callback ───────────────────────────────────────────────


@callback(
    Output("sz-status", "children"),
    Output("sz-results", "children"),
    Output("sz-clear-btn", "style"),
    Output("sz-confidence-section", "style"),
    Output("sz-inspector-controls", "style"),
    Input("sz-detect-btn", "n_clicks"),
    Input("sz-clear-btn", "n_clicks"),
    # Confidence filters — min + max (inputs trigger re-filter)
    *[Input(fid, "value") for fid in _ALL_FILTER_MIN_IDS],
    *[Input(fid, "value") for fid in _ALL_FILTER_MAX_IDS],
    Input("sz-filter-channel", "value"),
    Input("sz-filter-severity", "value"),
    Input("sz-filter-enabled", "value"),
    # Channel selector
    State("sz-channel-selector", "value"),
    # Spike params
    State({"type": "param-slider", "key": "sz-bp-low"}, "value"),
    State({"type": "param-slider", "key": "sz-bp-high"}, "value"),
    State({"type": "param-slider", "key": "sz-spike-amp"}, "value"),
    State({"type": "param-slider", "key": "sz-spike-min-uv"}, "value"),
    State({"type": "param-slider", "key": "sz-spike-prom"}, "value"),
    State({"type": "param-slider", "key": "sz-spike-maxw"}, "value"),
    State({"type": "param-slider", "key": "sz-spike-minw"}, "value"),
    State({"type": "param-slider", "key": "sz-spike-refr"}, "value"),
    # Train params
    State({"type": "param-slider", "key": "sz-max-isi"}, "value"),
    State({"type": "param-slider", "key": "sz-min-spikes"}, "value"),
    State({"type": "param-slider", "key": "sz-min-dur"}, "value"),
    State({"type": "param-slider", "key": "sz-min-iei"}, "value"),
    # Baseline
    State("sz-bl-method", "value"),
    State({"type": "param-slider", "key": "sz-bl-pct"}, "value"),
    State({"type": "param-slider", "key": "sz-bl-rms"}, "value"),
    # Boundary — signal
    State("sz-bnd-method", "value"),
    State({"type": "param-slider", "key": "sz-bnd-rms-win"}, "value"),
    State({"type": "param-slider", "key": "sz-bnd-rms-thr"}, "value"),
    State({"type": "param-slider", "key": "sz-bnd-max-trim"}, "value"),
    # Boundary — spike_density
    State({"type": "param-slider", "key": "sz-bnd-window"}, "value"),
    State({"type": "param-slider", "key": "sz-bnd-rate"}, "value"),
    State({"type": "param-slider", "key": "sz-bnd-amp-x"}, "value"),
    # Subtype
    State("sz-classify-subtypes", "value"),
    # HVSW params
    State({"type": "param-slider", "key": "sz-hvsw-amp"}, "value"),
    State({"type": "param-slider", "key": "sz-hvsw-freq"}, "value"),
    State({"type": "param-slider", "key": "sz-hvsw-dur"}, "value"),
    State({"type": "param-slider", "key": "sz-hvsw-max-ev"}, "value"),
    # HPD params
    State({"type": "param-slider", "key": "sz-hpd-amp"}, "value"),
    State({"type": "param-slider", "key": "sz-hpd-freq"}, "value"),
    State({"type": "param-slider", "key": "sz-hpd-dur"}, "value"),
    # Convulsive params
    State({"type": "param-slider", "key": "sz-conv-dur"}, "value"),
    State({"type": "param-slider", "key": "sz-conv-amp"}, "value"),
    State({"type": "param-slider", "key": "sz-conv-postictal"}, "value"),
    # Local baseline params
    State({"type": "param-slider", "key": "sz-lbl-start"}, "value"),
    State({"type": "param-slider", "key": "sz-lbl-end"}, "value"),
    State({"type": "param-slider", "key": "sz-lbl-trim-pct"}, "value"),
    # Session
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_detection(
    detect_clicks, clear_clicks,
    filt_min_conf, filt_min_dur, filt_min_spikes, filt_min_amp,
    filt_min_lbl, filt_min_top_amp,
    filt_min_ll, filt_min_energy, filt_min_sigbl, filt_min_freq, filt_min_td,
    filt_max_conf, filt_max_dur, filt_max_spikes, filt_max_amp,
    filt_max_lbl, filt_max_top_amp,
    filt_max_ll, filt_max_energy, filt_max_sigbl, filt_max_freq, filt_max_td,
    filt_channel, filt_severity, filt_enabled,
    selected_channels,
    bp_low, bp_high, spike_amp, spike_min_uv, spike_prom,
    spike_maxw, spike_minw, spike_refr,
    max_isi, min_spikes, min_dur, min_iei,
    bl_method, bl_pct, bl_rms,
    bnd_method, bnd_rms_win, bnd_rms_thr, bnd_max_trim,
    bnd_window, bnd_rate, bnd_amp_x,
    classify_subtypes,
    hvsw_amp, hvsw_freq, hvsw_dur, hvsw_max_ev,
    hpd_amp, hpd_freq, hpd_dur,
    conv_dur, conv_amp, conv_postictal,
    lbl_start, lbl_end, lbl_trim_pct,
    sid,
):
    """Run spike-train seizure detection, clear results, or apply filters."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    rec = state.recording

    if rec is None:
        return no_update, no_update, no_update, no_update, no_update

    show_style = {"display": "inline-block"}
    hide_style = {"display": "none"}
    block_style = {"display": "block", "marginBottom": "16px"}
    insp_block = {"display": "block", "marginTop": "16px", "marginBottom": "8px"}

    filter_kwargs = dict(
        min_conf=filt_min_conf, min_dur=filt_min_dur,
        min_spikes=filt_min_spikes, min_amp=filt_min_amp,
        min_lbl=filt_min_lbl, min_top_amp=filt_min_top_amp,
        min_ll=filt_min_ll, min_energy=filt_min_energy,
        min_sigbl=filt_min_sigbl, min_freq=filt_min_freq,
        min_td=filt_min_td,
        max_conf=filt_max_conf, max_dur=filt_max_dur,
        max_spikes=filt_max_spikes, max_amp=filt_max_amp,
        max_lbl=filt_max_lbl, max_top_amp=filt_max_top_amp,
        max_ll=filt_max_ll, max_energy=filt_max_energy,
        max_sigbl=filt_max_sigbl, max_freq=filt_max_freq,
        max_td=filt_max_td,
        channel=filt_channel, severity=filt_severity,
    )

    # Clear
    if trigger == "sz-clear-btn":
        server_state.clear_detections(sid, "seizures")
        state.extra.pop("sz_selected_event_key", None)
        return (
            alert("Results cleared.", "info"),
            html.Div(), hide_style, hide_style, hide_style,
        )

    # Filter change — re-filter existing results
    is_filter_trigger = isinstance(trigger, str) and trigger.startswith("sz-filter-")
    if is_filter_trigger and state.seizure_events:
        if filt_enabled:
            filtered = _apply_filters(rec, state.seizure_events, classify_subtypes,
                                      **filter_kwargs)
        else:
            filtered = list(state.seizure_events)
        # Update detected_events so viewer shadows reflect the filtered set
        state.detected_events = filtered + state.spike_events
        selected_ek = state.extra.get("sz_selected_event_key")
        results = _build_results(rec, filtered, classify_subtypes,
                                 selected_event_key=selected_ek,
                                 all_channels=selected_channels)
        n_total = len(state.seizure_events)
        n_shown = len(filtered)
        status = alert(f"Showing {n_shown} of {n_total} seizure(s) after filtering.", "info")
        return status, results, show_style, block_style, insp_block

    # Detect
    if trigger != "sz-detect-btn":
        return no_update, no_update, no_update, no_update, no_update

    if not selected_channels:
        return (
            alert("Select at least one channel.", "warning"),
            no_update, no_update, no_update, no_update,
        )

    try:
        from eeg_seizure_analyzer.detection.spike_train_seizure import (
            SpikeTrainSeizureDetector,
        )
        from eeg_seizure_analyzer.detection.confidence import (
            compute_event_quality,
            compute_confidence_score,
            compute_local_baseline_ratio,
            compute_top_spike_amplitude,
        )

        params = SpikeTrainSeizureParams(
            classify_subtypes=bool(classify_subtypes),
            bandpass_low=float(bp_low),
            bandpass_high=float(bp_high),
            spike_amplitude_x_baseline=float(spike_amp),
            spike_min_amplitude_uv=float(spike_min_uv),
            spike_prominence_x_baseline=float(spike_prom),
            spike_max_width_ms=float(spike_maxw),
            spike_min_width_ms=float(spike_minw),
            spike_refractory_ms=float(spike_refr),
            max_interspike_interval_ms=float(max_isi),
            min_train_spikes=int(min_spikes),
            min_train_duration_sec=float(min_dur),
            min_interevent_interval_sec=float(min_iei),
            baseline_method=bl_method,
            baseline_percentile=int(bl_pct),
            baseline_rms_window_sec=float(bl_rms),
            boundary_method=bnd_method,
            boundary_rms_window_ms=float(bnd_rms_win),
            boundary_rms_threshold_x=float(bnd_rms_thr),
            boundary_max_trim_sec=float(bnd_max_trim),
            boundary_window_sec=float(bnd_window),
            boundary_min_rate_hz=float(bnd_rate),
            boundary_min_amplitude_x=float(bnd_amp_x),
            hvsw_min_amplitude_x=float(hvsw_amp),
            hvsw_min_frequency_hz=float(hvsw_freq),
            hvsw_min_duration_sec=float(hvsw_dur),
            hvsw_max_evolution=float(hvsw_max_ev),
            hpd_min_amplitude_x=float(hpd_amp),
            hpd_min_frequency_hz=float(hpd_freq),
            hpd_min_duration_sec=float(hpd_dur),
            convulsive_min_duration_sec=float(conv_dur),
            convulsive_min_amplitude_x=float(conv_amp),
            convulsive_postictal_suppression_sec=float(conv_postictal),
        )

        detector = SpikeTrainSeizureDetector()
        seizures = []
        detection_info = {}

        for ch in selected_channels:
            if ch < 0 or ch >= rec.n_channels:
                continue
            ch_events = detector.detect(rec, ch, params=params)
            seizures.extend(ch_events)
            if hasattr(detector, "_last_detection_info"):
                detection_info[ch] = detector._last_detection_info.copy()

        seizures.sort(key=lambda e: e.onset_sec)

        # Compute proper confidence scores using the confidence module
        _lbl_start = float(lbl_start) if lbl_start is not None else -20.0
        _lbl_end = float(lbl_end) if lbl_end is not None else -5.0
        _lbl_trim = float(lbl_trim_pct) if lbl_trim_pct is not None else 30.0
        # Ensure they are negative offsets
        if _lbl_start > 0:
            _lbl_start = -_lbl_start
        if _lbl_end > 0:
            _lbl_end = -_lbl_end

        # ── Pass 1: compute quality metrics with basic local baseline ───
        for event in seizures:
            bl_rms_val = detection_info.get(event.channel, {}).get("baseline_mean")
            try:
                qm = compute_event_quality(
                    rec, event,
                    baseline_rms=bl_rms_val,
                    bandpass_low=float(bp_low),
                    bandpass_high=float(bp_high),
                )
                # Basic local baseline ratio (no event-awareness yet)
                lbr = compute_local_baseline_ratio(
                    rec, event,
                    local_start_sec=_lbl_start,
                    local_end_sec=_lbl_end,
                    bandpass_low=float(bp_low),
                    bandpass_high=float(bp_high),
                )
                qm["local_baseline_ratio"] = round(lbr, 2)
                # Top spike amplitude (top 10% of spikes)
                qm["top_spike_amplitude_x"] = round(
                    compute_top_spike_amplitude(event), 2
                )
                event.quality_metrics = qm
                event.confidence = compute_confidence_score(qm)
            except Exception:
                event.quality_metrics = {}
                event.confidence = 0.0

        # ── Pass 2: refine local baseline with event-aware shifting
        #    and trimmed RMS.  Now that all events are known we can
        #    skip windows that overlap other detections. ──────────────
        for event in seizures:
            try:
                lbr_refined = compute_local_baseline_ratio(
                    rec, event,
                    local_start_sec=_lbl_start,
                    local_end_sec=_lbl_end,
                    bandpass_low=float(bp_low),
                    bandpass_high=float(bp_high),
                    all_events=seizures,
                    trim_pct=_lbl_trim,
                )
                qm = event.quality_metrics
                qm["local_baseline_ratio_raw"] = qm.get("local_baseline_ratio", 0.0)
                qm["local_baseline_ratio"] = round(lbr_refined, 2)
                event.confidence = compute_confidence_score(qm)
            except Exception:
                pass  # keep pass-1 values

        # Assign stable event IDs (1-based, sorted by channel then onset)
        seizures.sort(key=lambda e: (e.channel, e.onset_sec))
        for i, ev in enumerate(seizures, start=1):
            ev.event_id = i

        state.seizure_events = seizures
        state.st_detection_info = detection_info
        state.extra.pop("sz_selected_event_key", None)  # new detection, clear selection

        # Auto-save detections to disk — use slider-key format for params
        # so they can be directly restored into the Seizure tab UI.
        try:
            from eeg_seizure_analyzer.io.persistence import save_detections

            _src = getattr(rec, "source_path", None) or ""
            if _src and _src.lower().endswith(".edf"):
                save_detections(
                    edf_path=_src,
                    events=seizures,
                    detection_info=detection_info,
                    params_dict=state.extra.get("sz_params", {}),
                    detector_name="SpikeTrainSeizureDetector",
                    channels=selected_channels,
                    animal_id=getattr(state, "animal_id", ""),
                    filter_settings={
                        "filter_enabled": state.extra.get("sz_filter_enabled", False),
                        "filter_values": state.extra.get("sz_filter_values", {}),
                    },
                )
        except Exception:
            pass  # save failure must not break detection

        # Apply any active filters
        if filt_enabled:
            filtered = _apply_filters(rec, seizures, classify_subtypes,
                                      **filter_kwargs)
        else:
            filtered = list(seizures)
        # Update detected_events with filtered set so viewer shows only these
        state.detected_events = filtered + state.spike_events
        results = _build_results(rec, filtered, classify_subtypes,
                                 all_channels=selected_channels)
        n_ch = len(selected_channels)

        return (
            alert(f"Found {len(seizures)} seizure(s) across {n_ch} channel(s).", "success"),
            results, show_style, block_style, insp_block,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            alert(f"Detection failed: {e}", "danger"),
            html.Div(), hide_style, hide_style, hide_style,
        )


# ── Filter helpers ──────────────────────────────────────────────────


def _apply_filters(rec, seizures, classify_on, *,
                   min_conf=0, min_dur=0, min_spikes=0, min_amp=0,
                   min_lbl=0, min_top_amp=0,
                   min_ll=0, min_energy=0, min_sigbl=0,
                   min_freq=0, min_td=0,
                   max_conf=None, max_dur=None, max_spikes=None, max_amp=None,
                   max_lbl=None, max_top_amp=None,
                   max_ll=None, max_energy=None, max_sigbl=None,
                   max_freq=None, max_td=None,
                   channel=None, severity=""):
    """Apply min/max filters to seizure list."""
    filtered = list(seizures)
    min_conf = float(min_conf or 0)
    min_dur = float(min_dur or 0)
    min_spikes = int(min_spikes or 0)
    min_amp = float(min_amp or 0)
    min_lbl = float(min_lbl or 0)
    min_top_amp = float(min_top_amp or 0)
    min_ll = float(min_ll or 0)
    min_energy = float(min_energy or 0)
    min_sigbl = float(min_sigbl or 0)
    min_freq = float(min_freq or 0)
    min_td = float(min_td or 0)

    def _fmax(v):
        """Convert max filter value: None/empty → None, else float."""
        if v is None or v == "":
            return None
        return float(v)

    max_conf = _fmax(max_conf)
    max_dur = _fmax(max_dur)
    max_spikes = _fmax(max_spikes)
    max_amp = _fmax(max_amp)
    max_lbl = _fmax(max_lbl)
    max_top_amp = _fmax(max_top_amp)
    max_ll = _fmax(max_ll)
    max_energy = _fmax(max_energy)
    max_sigbl = _fmax(max_sigbl)
    max_freq = _fmax(max_freq)
    max_td = _fmax(max_td)

    # --- Confidence ---
    if min_conf > 0:
        filtered = [e for e in filtered if e.confidence >= min_conf]
    if max_conf is not None:
        filtered = [e for e in filtered if e.confidence <= max_conf]
    # --- Duration ---
    if min_dur > 0:
        filtered = [e for e in filtered if e.duration_sec >= min_dur]
    if max_dur is not None:
        filtered = [e for e in filtered if e.duration_sec <= max_dur]
    # --- Spikes ---
    if min_spikes > 0:
        filtered = [e for e in filtered
                    if (e.features.get("n_spikes") or 0) >= min_spikes]
    if max_spikes is not None:
        filtered = [e for e in filtered
                    if (e.features.get("n_spikes") or 0) <= max_spikes]
    # --- Amp (xBL) ---
    if min_amp > 0:
        filtered = [e for e in filtered
                    if (e.features.get("max_amplitude_x_baseline") or 0) >= min_amp]
    if max_amp is not None:
        filtered = [e for e in filtered
                    if (e.features.get("max_amplitude_x_baseline") or 0) <= max_amp]
    # --- Local BL ---
    if min_lbl > 0:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("local_baseline_ratio", 0) >= min_lbl]
    if max_lbl is not None:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("local_baseline_ratio", 0) <= max_lbl]
    # --- Top Amp ---
    if min_top_amp > 0:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("top_spike_amplitude_x", 0) >= min_top_amp]
    if max_top_amp is not None:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("top_spike_amplitude_x", 0) <= max_top_amp]
    # --- LL z-score ---
    if min_ll > 0:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("peak_ll_zscore", 0) >= min_ll]
    if max_ll is not None:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("peak_ll_zscore", 0) <= max_ll]
    # --- Energy z-score ---
    if min_energy > 0:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("peak_energy_zscore", 0) >= min_energy]
    if max_energy is not None:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("peak_energy_zscore", 0) <= max_energy]
    # --- Sig/BL ---
    if min_sigbl > 0:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("signal_to_baseline_ratio", 0) >= min_sigbl]
    if max_sigbl is not None:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("signal_to_baseline_ratio", 0) <= max_sigbl]
    # --- Freq ---
    if min_freq > 0:
        filtered = [e for e in filtered
                    if (e.features.get("mean_spike_frequency_hz") or 0) >= min_freq]
    if max_freq is not None:
        filtered = [e for e in filtered
                    if (e.features.get("mean_spike_frequency_hz") or 0) <= max_freq]
    # --- θ/δ ---
    if min_td > 0:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("theta_delta_ratio", 0) >= min_td]
    if max_td is not None:
        filtered = [e for e in filtered
                    if (e.quality_metrics or {}).get("theta_delta_ratio", 0) <= max_td]
    # --- Channel / Severity ---
    if channel is not None and channel != "":
        filtered = [e for e in filtered if e.channel == int(channel)]
    if severity:
        filtered = [e for e in filtered if e.severity == severity]
    return filtered


def _event_key(event):
    """Stable unique key for an event: (onset_sec rounded, channel)."""
    return f"{event.onset_sec:.4f}_{event.channel}"


def _prerender_inspector(state, rec, selected_event_key, insp_opts, insp_yr):
    """Pre-render the inspector for a previously selected event (tab restoration)."""
    if not selected_event_key or not state.seizure_events:
        return []
    event = None
    for e in state.seizure_events:
        if _event_key(e) == selected_event_key:
            event = e
            break
    if event is None:
        return []
    try:
        det_info = state.st_detection_info.get(event.channel, {})
        return _render_inspector(
            rec, event, det_info, state,
            show_spikes=insp_opts.get("show_spikes", True),
            show_baseline=insp_opts.get("show_baseline", True),
            show_threshold=insp_opts.get("show_threshold", True),
            bandpass_on=insp_opts.get("bandpass", False),
            y_range=float(insp_yr) if insp_yr and insp_yr > 0 else None,
        )
    except Exception:
        return []


def _build_results(rec, seizures, classify_on, *, selected_event_key=None,
                    all_channels=None):
    """Build the results display (metrics + table)."""
    if not seizures:
        return empty_state("\u2714", "No Seizures Found",
                           "No seizure events match the current parameters / filters.")

    from eeg_seizure_analyzer.detection.burden import compute_burden

    # Compute per-channel metrics — show ALL selected channels
    channels_with_results = set(e.channel for e in seizures)
    if all_channels is None:
        all_channels = sorted(channels_with_results)
    else:
        all_channels = sorted(all_channels)

    metric_rows = []
    for ch in all_channels:
        ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch{ch}"
        if ch in channels_with_results:
            burden = compute_burden(seizures, rec.duration_sec, channel=ch)
            metric_rows.append(
                dbc.Row([
                    dbc.Col(metric_card(ch_name, str(burden.n_seizures), accent=True), width=2),
                    dbc.Col(metric_card("Total Time", f"{burden.total_seizure_time_sec:.1f}s"), width=2),
                    dbc.Col(metric_card("Per Hour", f"{burden.seizure_frequency_per_hour:.2f}"), width=2),
                    dbc.Col(metric_card("% Time", f"{burden.percent_time_in_seizure:.2f}%"), width=2),
                    dbc.Col(metric_card("Mean Dur", f"{burden.mean_duration_sec:.1f}s"), width=2),
                ], className="g-3 mb-2")
            )
        else:
            metric_rows.append(
                dbc.Row([
                    dbc.Col(metric_card(ch_name, "0", accent=False), width=2),
                    dbc.Col(metric_card("Total Time", "-"), width=2),
                    dbc.Col(metric_card("Per Hour", "-"), width=2),
                    dbc.Col(metric_card("% Time", "-"), width=2),
                    dbc.Col(metric_card("Mean Dur", "-"), width=2),
                ], className="g-3 mb-2")
            )
    metrics = html.Div(metric_rows)

    # Table data with all quality metric columns
    # Include a hidden _event_key for stable event identification across filters
    table_data = []
    selected_rows = []
    for i, e in enumerate(seizures):
        feat = e.features or {}
        qm = e.quality_metrics or {}
        ek = _event_key(e)
        is_manual = getattr(e, "source", "detector") == "manual"

        # Spike features may be None for manually added seizures
        n_spikes = feat.get("n_spikes")
        spike_freq = feat.get("mean_spike_frequency_hz")
        max_amp = feat.get("max_amplitude_x_baseline")

        row = {
            "#": i + 1,
            "ID": e.event_id if e.event_id > 0 else i + 1,
            "_event_key": ek,
            "_source": "manual" if is_manual else "detector",
            "Channel": rec.channel_names[e.channel],
            "Onset (s)": round(e.onset_sec, 2),
            "Offset (s)": round(e.offset_sec, 2),
            "Duration (s)": round(e.duration_sec, 2),
            "Spikes": "\u2014" if n_spikes is None else n_spikes,
            "Freq (Hz)": "\u2014" if spike_freq is None else round(spike_freq, 1),
            "Max Amp (xBL)": "\u2014" if max_amp is None else round(max_amp, 1),
            "Confidence": round(e.confidence, 2),
            "Local BL": round(qm.get("local_baseline_ratio", 0), 1),
            "Top Amp": round(qm.get("top_spike_amplitude_x", 0), 1),
            "LL (z)": round(qm.get("peak_ll_zscore", 0), 1),
            "Energy (z)": round(qm.get("peak_energy_zscore", 0), 1),
            "Sig/BL": round(qm.get("signal_to_baseline_ratio", 0), 1),
            "Spec Ent": round(qm.get("spectral_entropy", 0), 1),
            "Peak Freq": round(qm.get("dominant_freq_hz", 0), 1),
            "\u03b8/\u03b4": round(qm.get("theta_delta_ratio", 0), 2),
            "Severity": e.severity,
        }
        if classify_on:
            row["Type"] = feat.get("seizure_subtype", "\u2014")
        table_data.append(row)
        if selected_event_key and ek == selected_event_key:
            selected_rows = [row]

    col_defs = [
        {"field": "#", "maxWidth": 55, "minWidth": 40},
        {"field": "ID", "maxWidth": 55, "minWidth": 40, "headerTooltip": "Stable event ID"},
        {"field": "_event_key", "hide": True},
        {"field": "_source", "hide": True},
        {"field": "Channel", "flex": 1, "minWidth": 75},
        {"field": "Onset (s)", "flex": 1, "minWidth": 70},
        {"field": "Offset (s)", "flex": 1, "minWidth": 70},
        {"field": "Duration (s)", "flex": 1, "minWidth": 70},
        {"field": "Spikes", "flex": 1, "minWidth": 55},
        {"field": "Freq (Hz)", "flex": 1, "minWidth": 65},
        {"field": "Max Amp (xBL)", "flex": 1, "minWidth": 75},
        {"field": "Confidence", "flex": 1, "minWidth": 70},
        {"field": "Local BL", "flex": 1, "minWidth": 65,
         "headerTooltip": "Signal-to-local-baseline ratio (pre-ictal comparison)"},
        {"field": "Top Amp", "flex": 1, "minWidth": 60,
         "headerTooltip": "Top 10% spike amplitude (\u00d7baseline)"},
        {"field": "Severity", "flex": 1, "minWidth": 65},
        # Hidden columns — kept for ML export / future use
        {"field": "LL (z)", "hide": True},
        {"field": "Energy (z)", "hide": True},
        {"field": "Sig/BL", "hide": True},
        {"field": "Spec Ent", "hide": True},
        {"field": "Peak Freq", "hide": True},
        {"field": "\u03b8/\u03b4", "hide": True},
    ]
    if classify_on:
        col_defs.append({"field": "Type", "flex": 1, "minWidth": 65})

    grid_props = {
        "animateRows": False,
        "rowSelection": {"mode": "singleRow"},
        "headerHeight": 32,
        "enableCellTextSelection": True,
    }

    # Row style callback: highlight manual seizures in purple
    manual_row_style = {
        "styleConditions": [
            {
                "condition": "params.data._source === 'manual'",
                "style": {"backgroundColor": "rgba(188, 140, 255, 0.12)"},
            },
        ],
    }

    table = dag.AgGrid(
        id="sz-results-grid",
        rowData=table_data,
        columnDefs=col_defs,
        selectedRows=selected_rows if selected_rows else [],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        className="ag-theme-alpine-dark",
        style={"height": "300px", "width": "100%"},
        dashGridOptions=grid_props,
        getRowStyle=manual_row_style,
    )

    return html.Div([
        metrics,
        html.Div(
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                   "marginBottom": "12px"},
            children=[
                html.H6("Detected Seizures", style={"margin": "0"}),
                html.Span("Click a row to inspect (use \u2191\u2193 arrows to navigate)",
                          style={"fontSize": "0.75rem", "color": "#8b949e"}),
            ],
        ),
        table,
    ])


# ── Event Inspector callback ────────────────────────────────────────


@callback(
    Output("sz-event-inspector", "children"),
    Input("sz-results-grid", "selectedRows"),
    Input("sz-insp-show-spikes", "value"),
    Input("sz-insp-show-baseline", "value"),
    Input("sz-insp-show-threshold", "value"),
    Input("sz-insp-bandpass", "value"),
    Input("sz-insp-yrange", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_row_select(selected_rows, show_spikes, show_baseline, show_threshold,
                  bandpass_on, insp_yrange, sid):
    """Render the event inspector when a seizure row is clicked or controls change."""
    if not selected_rows:
        return html.Div()

    row = selected_rows[0]
    ek = row.get("_event_key", "")

    state = server_state.get_session(sid)
    rec = state.recording
    if rec is None:
        return html.Div()

    # Find event by stable key (onset_sec + channel), not by table row index
    event = None
    for e in state.seizure_events:
        if _event_key(e) == ek:
            event = e
            break
    if event is None:
        # Fallback: try by "#" index (legacy)
        event_idx = row.get("#", 1) - 1
        if 0 <= event_idx < len(state.seizure_events):
            event = state.seizure_events[event_idx]
        else:
            return html.Div()

    # Persist selected event for tab-switch restoration
    state.extra["sz_selected_event_key"] = _event_key(event)

    det_info = state.st_detection_info.get(event.channel, {})

    try:
        return _render_inspector(
            rec, event, det_info, state,
            show_spikes=show_spikes,
            show_baseline=show_baseline,
            show_threshold=show_threshold,
            bandpass_on=bandpass_on,
            y_range=float(insp_yrange) if insp_yrange and insp_yrange > 0 else None,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return alert(f"Inspector error: {e}", "danger")


def _render_inspector(rec, event, det_info, state, *,
                      show_spikes=True, show_baseline=True,
                      show_threshold=True, bandpass_on=False,
                      y_range=None):
    """Build full event inspector: EEG trace, activity, spectrogram, power over time."""
    context_sec = 10.0
    ch = event.channel
    onset, offset = event.onset_sec, event.offset_sec

    win_start = max(0, onset - context_sec)
    win_end = min(rec.duration_sec, offset + context_sec)

    start_idx = int(win_start * rec.fs)
    end_idx = min(int(win_end * rec.fs), rec.n_samples)
    data = rec.data[ch, start_idx:end_idx].astype(np.float64)
    time_axis = np.linspace(win_start, win_end, len(data))

    if bandpass_on:
        bp_low = state.extra.get("sz_params", {}).get("sz-bp-low", _SZ_DEFAULTS["sz-bp-low"])
        bp_high = state.extra.get("sz_params", {}).get("sz-bp-high", _SZ_DEFAULTS["sz-bp-high"])
        from eeg_seizure_analyzer.processing.preprocess import bandpass_filter
        data = bandpass_filter(data, rec.fs, float(bp_low), float(bp_high))

    unit_label = rec.units[ch] if ch < len(rec.units) else ""
    ch_name = rec.channel_names[ch]

    # Activity channel
    act_rec = state.activity_recordings.get("paired")
    pairings = state.channel_pairings
    act_data = None
    act_time = None
    if act_rec and pairings:
        for p in pairings:
            if p.eeg_index == ch and p.activity_index is not None:
                act_start = int(win_start * act_rec.fs)
                act_end = min(int(win_end * act_rec.fs), act_rec.n_samples)
                act_data = act_rec.data[p.activity_index, act_start:act_end]
                act_time = np.linspace(win_start, win_end, len(act_data))
                break

    has_act = act_data is not None

    # Y range
    if y_range is None or y_range <= 0:
        y_range = state.extra.get("viewer_settings", {}).get("yrange", 1.0)
    half_yr = y_range / 2.0

    # ── EEG trace figure ────────────────────────────────────────
    n_rows = 2 if has_act else 1
    row_heights = [0.75, 0.25] if has_act else [1.0]
    fig_eeg = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.05,
    )

    fig_eeg.add_trace(go.Scattergl(
        x=time_axis, y=data,
        mode="lines", name=ch_name,
        line=dict(width=0.8, color="#58a6ff"),
    ), row=1, col=1)

    # Event shadow
    fig_eeg.add_vrect(
        x0=onset, x1=offset,
        fillcolor="rgba(248, 81, 73, 0.15)",
        line=dict(color="#f85149", width=1),
        layer="below", row=1, col=1,
    )
    for t in [onset, offset]:
        fig_eeg.add_vline(x=t, line=dict(color="#f85149", width=1, dash="dash"),
                          row=1, col=1)

    # Spike dots
    if show_spikes:
        spike_times = det_info.get("all_spike_times", [])
        spike_samples = det_info.get("all_spike_samples", [])
        if spike_times:
            in_t, in_y, out_t, out_y = [], [], [], []
            for i, t in enumerate(spike_times):
                if win_start <= t <= win_end:
                    local = spike_samples[i] - start_idx
                    yv = float(data[local]) if 0 <= local < len(data) else 0.0
                    if onset <= t <= offset:
                        in_t.append(t); in_y.append(yv)
                    else:
                        out_t.append(t); out_y.append(yv)
            if in_t:
                fig_eeg.add_trace(go.Scatter(
                    x=in_t, y=in_y, mode="markers",
                    marker=dict(color="#f85149", size=5),
                    showlegend=False, name="In-event spikes",
                    hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
                ), row=1, col=1)
            if out_t:
                fig_eeg.add_trace(go.Scatter(
                    x=out_t, y=out_y, mode="markers",
                    marker=dict(color="#ffb347", size=5, opacity=1.0),
                    showlegend=False, name="Out-of-event spikes",
                    hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
                ), row=1, col=1)

    # Baseline / threshold lines
    baseline_val = det_info.get("baseline_mean")
    if show_baseline and baseline_val is not None:
        # Positive side — with annotation
        fig_eeg.add_hline(
            y=baseline_val, row=1, col=1,
            line=dict(color="#3fb950", width=1, dash="dot"),
            annotation_text="Baseline",
            annotation_position="top right",
        )
        # Negative side — no annotation
        fig_eeg.add_hline(
            y=-baseline_val, row=1, col=1,
            line=dict(color="#3fb950", width=1, dash="dot"),
        )
    threshold_val = det_info.get("threshold")
    if show_threshold and threshold_val is not None:
        # Positive side — with annotation
        fig_eeg.add_hline(
            y=threshold_val, row=1, col=1,
            line=dict(color="#d29922", width=1, dash="dash"),
            annotation_text="Threshold",
            annotation_position="top right",
        )
        # Negative side — no annotation
        fig_eeg.add_hline(
            y=-threshold_val, row=1, col=1,
            line=dict(color="#d29922", width=1, dash="dash"),
        )

    # Activity
    if has_act:
        fig_eeg.add_trace(go.Scattergl(
            x=act_time, y=act_data, mode="lines", name="Activity",
            line=dict(width=1, color="#3fb950"),
        ), row=2, col=1)
        fig_eeg.add_vrect(x0=onset, x1=offset,
                          fillcolor="rgba(248, 81, 73, 0.1)",
                          line=dict(width=0), layer="below", row=2, col=1)
        act_unit = act_rec.units[0] if act_rec.units else ""
        fig_eeg.update_yaxes(
            title_text=f"Activity ({act_unit})" if act_unit else "Activity",
            row=2, col=1)
        fig_eeg.update_xaxes(title_text="Time (s)", row=2, col=1)
    else:
        fig_eeg.update_xaxes(title_text="Time (s)", row=1, col=1)

    # Per-axis uirevision:
    #   Y resets only when y_range input changes
    #   X stable across checkbox toggles (only changes on new event)
    fig_eeg.update_yaxes(
        title_text=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
        range=[-half_yr, half_yr],
        fixedrange=False,
        uirevision=f"insp_y_{y_range}",
        row=1, col=1,
    )
    fig_eeg.update_xaxes(
        uirevision=f"insp_x_{onset}_{ch}",
        row=1, col=1,
    )

    fig_eeg.update_layout(
        height=400 if has_act else 300,
        showlegend=False, dragmode="zoom",
        uirevision=f"insp_{onset}_{ch}",
    )
    apply_fig_theme(fig_eeg)
    fig_eeg.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    # ── Spectrogram ─────────────────────────────────────────────
    from scipy.signal import spectrogram as scipy_spectrogram

    nperseg = min(int(1.0 * rec.fs), len(data) // 4)
    nperseg = max(nperseg, 64)
    noverlap = int(nperseg * 0.9)

    f_spec, t_spec, Sxx = scipy_spectrogram(
        data, fs=rec.fs, nperseg=nperseg, noverlap=noverlap)
    t_spec = t_spec + win_start

    freq_mask = f_spec <= 100
    f_spec = f_spec[freq_mask]
    Sxx = Sxx[freq_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig_spec = go.Figure(go.Heatmap(
        x=t_spec, y=f_spec, z=Sxx_db,
        colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.1f}Hz<br>Power: %{z:.1f}dB<extra></extra>",
    ))
    fig_spec.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.update_layout(
        height=250, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        showlegend=False, uirevision=f"spec_{onset}_{ch}",
    )
    apply_fig_theme(fig_spec)
    fig_spec.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    # ── Power Over Time (absolute, stacked area) ────────────────
    bands = {
        "Delta (0.5-4)": (0.5, 4, "#1f77b4"),
        "Theta (4-8)": (4, 8, "#ff7f0e"),
        "Alpha (8-13)": (8, 13, "#2ca02c"),
        "Beta (13-30)": (13, 30, "#d62728"),
        "Gamma-low (30-50)": (30, 50, "#9467bd"),
        "Gamma-high (50-100)": (50, 100, "#8c564b"),
    }

    from scipy.signal import welch

    win_samples = int(2.0 * rec.fs)
    step_samples = int(1.0 * rec.fs)
    band_power_data = {name: [] for name in bands}
    bp_times = []

    for start_s in range(0, max(1, len(data) - win_samples), step_samples):
        end_s = start_s + win_samples
        segment = data[start_s:end_s]
        bp_times.append(win_start + (start_s + win_samples / 2) / rec.fs)

        f_welch, psd = welch(segment, fs=rec.fs, nperseg=min(win_samples, len(segment)))
        for name, (flo, fhi, _) in bands.items():
            mask = (f_welch >= flo) & (f_welch <= fhi)
            bp = np.trapezoid(psd[mask], f_welch[mask]) if mask.sum() > 1 else 0.0
            band_power_data[name].append(bp)

    fig_bp = go.Figure()
    for name, (_, _, color) in bands.items():
        fig_bp.add_trace(go.Scatter(
            x=bp_times, y=band_power_data[name],
            name=name, mode="lines",
            line=dict(color=color),
            stackgroup="bands",
        ))

    fig_bp.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_bp.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))

    power_unit = f"{unit_label}\u00b2/Hz" if unit_label else "Power"
    fig_bp.update_layout(
        height=250,
        xaxis_title="Time (s)",
        yaxis_title=f"Power ({power_unit})",
        yaxis_rangemode="tozero",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)),
        uirevision=f"bp_{onset}_{ch}",
    )
    apply_fig_theme(fig_bp)
    fig_bp.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    # ── Event detail metrics ────────────────────────────────────
    features = event.features or {}
    qm = event.quality_metrics or {}
    detail_metrics = dbc.Row([
        dbc.Col(metric_card("Channel", ch_name), width=2),
        dbc.Col(metric_card("Onset", f"{onset:.2f}s"), width=2),
        dbc.Col(metric_card("Duration", f"{event.duration_sec:.2f}s"), width=2),
        dbc.Col(metric_card("Spikes", str(features.get("n_spikes", 0))), width=2),
        dbc.Col(metric_card("Confidence", f"{event.confidence:.2f}"), width=2),
        dbc.Col(metric_card("Severity", event.severity), width=2),
    ], className="g-3 mb-3")

    return html.Div([
        html.Hr(style={"borderColor": "#2d333b", "margin": "24px 0"}),
        html.H5("Event Inspector",
                 style={"marginBottom": "16px", "color": "#58a6ff"}),
        detail_metrics,
        html.Div("EEG Trace", style={"fontSize": "0.82rem", "fontWeight": "600",
                                      "color": "#8b949e", "marginBottom": "4px"}),
        dcc.Graph(figure=fig_eeg, config={"scrollZoom": True, "displayModeBar": True}),
        dbc.Row([
            dbc.Col([
                html.Div("Spectrogram", style={"fontSize": "0.82rem", "fontWeight": "600",
                                                "color": "#8b949e", "marginBottom": "4px",
                                                "marginTop": "16px"}),
                dcc.Graph(figure=fig_spec, config={"scrollZoom": True}),
            ], width=6),
            dbc.Col([
                html.Div("Power Over Time", style={"fontSize": "0.82rem", "fontWeight": "600",
                                                    "color": "#8b949e", "marginBottom": "4px",
                                                    "marginTop": "16px"}),
                dcc.Graph(figure=fig_bp, config={"scrollZoom": True}),
            ], width=6),
        ]),
    ])


# ── Save / Recall settings callbacks ────────────────────────────────


@callback(
    Output("sz-settings-status", "children"),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("sz-recall-defaults-btn", "n_clicks"),
    Input("sz-save-settings-btn", "n_clicks"),
    Input("sz-recall-settings-btn", "n_clicks"),
    *[State({"type": "param-slider", "key": k}, "value") for k in _SZ_SLIDER_KEYS],
    State("sz-bl-method", "value"),
    State("sz-bnd-method", "value"),
    State("sz-classify-subtypes", "value"),
    State("sz-channel-selector", "value"),
    # Filter values
    *[State(fid, "value") for fid in _ALL_FILTER_IDS],
    State("sz-filter-channel", "value"),
    State("sz-filter-severity", "value"),
    State("sz-filter-enabled", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def handle_settings(*args):
    """Handle recall defaults, save, and recall user settings."""
    trigger = ctx.triggered_id
    # Guard: only proceed if an actual button was clicked
    if trigger not in ("sz-recall-defaults-btn", "sz-save-settings-btn",
                       "sz-recall-settings-btn"):
        return no_update, no_update
    # Guard: check the corresponding button actually has clicks
    btn_clicks = {"sz-recall-defaults-btn": args[0],
                  "sz-save-settings-btn": args[1],
                  "sz-recall-settings-btn": args[2]}
    if not btn_clicks.get(trigger):
        return no_update, no_update
    n_keys = len(_SZ_SLIDER_KEYS)
    n_filter_ids = len(_ALL_FILTER_IDS)
    current_values = args[3:3 + n_keys]
    bl_method = args[3 + n_keys]
    bnd_method = args[3 + n_keys + 1]
    classify = args[3 + n_keys + 2]
    channels = args[3 + n_keys + 3]
    # Filter states
    filter_offset = 3 + n_keys + 4
    filter_slider_vals = args[filter_offset:filter_offset + n_filter_ids]
    filt_channel = args[filter_offset + n_filter_ids]
    filt_severity = args[filter_offset + n_filter_ids + 1]
    filt_enabled = args[filter_offset + n_filter_ids + 2]
    sid = args[-2]
    refresh = args[-1]
    state = server_state.get_session(sid)

    if trigger == "sz-recall-defaults-btn":
        state.extra["sz_param_overrides"] = dict(_SZ_DEFAULTS)
        return alert("Default parameters restored.", "info"), (refresh or 0) + 1

    if trigger == "sz-save-settings-btn":
        from eeg_seizure_analyzer.dash_app.components import save_user_defaults
        params = {k: v for k, v in zip(_SZ_SLIDER_KEYS, current_values)}
        params["sz-bl-method"] = bl_method
        params["sz-bnd-method"] = bnd_method
        # Include filter settings (min + max)
        n_min = len(_ALL_FILTER_MIN_IDS)
        filter_min_vals = filter_slider_vals[:n_min]
        filter_max_vals = filter_slider_vals[n_min:]
        _min_keys = ["min_conf", "min_dur", "min_spikes", "min_amp",
                     "min_lbl", "min_top_amp",
                     "min_ll", "min_energy", "min_sigbl", "min_freq", "min_td"]
        _max_keys = ["max_conf", "max_dur", "max_spikes", "max_amp",
                     "max_lbl", "max_top_amp",
                     "max_ll", "max_energy", "max_sigbl", "max_freq", "max_td"]
        filter_params = {}
        for k, v in zip(_min_keys, filter_min_vals):
            filter_params[f"filter-{k}"] = v
        for k, v in zip(_max_keys, filter_max_vals):
            filter_params[f"filter-{k}"] = v
        filter_params["filter-channel"] = filt_channel
        filter_params["filter-severity"] = filt_severity
        filter_params["filter-enabled"] = bool(filt_enabled)
        params.update(filter_params)
        path = save_user_defaults(params)
        return alert(f"User params saved to {path}", "success"), no_update

    if trigger == "sz-recall-settings-btn":
        from eeg_seizure_analyzer.dash_app.components import load_user_defaults
        saved = load_user_defaults()
        if saved is None:
            return alert("No saved user params found.", "warning"), no_update
        # Separate filter settings from param overrides
        filter_vals = {}
        filter_keys_map = {
            "filter-min_conf": "min_conf", "filter-min_dur": "min_dur",
            "filter-min_spikes": "min_spikes", "filter-min_amp": "min_amp",
            "filter-min_lbl": "min_lbl", "filter-min_top_amp": "min_top_amp",
            "filter-min_ll": "min_ll", "filter-min_energy": "min_energy",
            "filter-min_sigbl": "min_sigbl", "filter-min_freq": "min_freq",
            "filter-min_td": "min_td",
            "filter-max_conf": "max_conf", "filter-max_dur": "max_dur",
            "filter-max_spikes": "max_spikes", "filter-max_amp": "max_amp",
            "filter-max_lbl": "max_lbl", "filter-max_top_amp": "max_top_amp",
            "filter-max_ll": "max_ll", "filter-max_energy": "max_energy",
            "filter-max_sigbl": "max_sigbl", "filter-max_freq": "max_freq",
            "filter-max_td": "max_td",
            "filter-channel": "channel", "filter-severity": "severity",
        }
        for fk, mk in filter_keys_map.items():
            if fk in saved:
                filter_vals[mk] = saved.pop(fk)
        filter_enabled = saved.pop("filter-enabled", True)
        if filter_vals:
            state.extra["sz_filter_values"] = {**_FILTER_DEFAULTS, **filter_vals}
        state.extra["sz_filter_enabled"] = filter_enabled
        state.extra["sz_param_overrides"] = saved
        return alert("User params loaded.", "success"), (refresh or 0) + 1

    return no_update, no_update
