"""Seizure Detection tab: multi-method detection with parameter controls."""

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
from eeg_seizure_analyzer.config import (
    SpikeTrainSeizureParams,
    SpectralBandParams,
    AutocorrelationParams,
    EnsembleParams,
)

# Detection method labels
_METHOD_OPTIONS = [
    {"label": "Spike-Train", "value": "spike_train"},
    {"label": "Spectral Band (17–25 Hz)", "value": "spectral_band"},
    {"label": "Autocorrelation", "value": "autocorrelation"},
    {"label": "Ensemble", "value": "ensemble"},
]

_DETECTOR_NAMES = {
    "spike_train": "SpikeTrainSeizureDetector",
    "spectral_band": "SpectralBandDetector",
    "autocorrelation": "AutocorrelationDetector",
    "ensemble": "EnsembleDetector",
}

# ── Default parameter values ────────────────────────────────────────

_SZ_DEFAULTS = {
    "sz-bp-low": 1.0,
    "sz-bp-high": 50.0,
    "sz-spike-amp": 3.0,
    "sz-spike-min-uv": 0.0,
    "sz-spike-prom": 2.5,
    "sz-spike-maxw": 200.0,
    "sz-spike-minw": 10.0,
    "sz-spike-refr": 75.0,
    "sz-max-isi": 500.0,
    "sz-min-spikes": 10,
    "sz-min-dur": 5.0,
    "sz-min-iei": 3.0,
    "sz-bl-pct": 25,
    "sz-bl-rms": 30.0,
    # Boundary — signal method
    "sz-bnd-rms-win": 100.0,
    "sz-bnd-rms-thr": 3.0,
    "sz-bnd-max-trim": 2.0,
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
    "sz-lbl-start": 15.0,
    "sz-lbl-end": 5.0,
    "sz-lbl-trim-pct": 30.0,
}

_SZ_SLIDER_KEYS = list(_SZ_DEFAULTS.keys())

# ── Spectral band defaults ─────────────────────────────────────────
_SB_DEFAULTS = {
    "sz-sb-band-low": 17.0,
    "sz-sb-band-high": 25.0,
    "sz-sb-ref-low": 1.0,
    "sz-sb-ref-high": 50.0,
    "sz-sb-window": 2.0,
    "sz-sb-step": 1.0,
    "sz-sb-thr-z": 3.0,
    "sz-sb-bl-pct": 15,
    "sz-sb-min-dur": 5.0,
    "sz-sb-merge-gap": 3.0,
    # Boundary refinement
    "sz-sb-bnd-rms-win": 100.0,
    "sz-sb-bnd-rms-thr": 2.0,
    "sz-sb-bnd-max-trim": 5.0,
    # Pre-ictal local baseline
    "sz-sb-lbl-start": 15.0,
    "sz-sb-lbl-end": 5.0,
    "sz-sb-lbl-trim-pct": 30.0,
}
_SB_SLIDER_KEYS = list(_SB_DEFAULTS.keys())

# ── Autocorrelation defaults ───────────────────────────────────────
_AC_DEFAULTS = {
    "sz-ac-bp-low": 1.0,
    "sz-ac-bp-high": 100.0,
    "sz-ac-spike-amp": 3.0,
    "sz-ac-spike-refr": 50.0,
    "sz-ac-subwin": 30,
    "sz-ac-lookahead": 60,
    "sz-ac-window": 12.0,
    "sz-ac-step": 4.0,
    "sz-ac-min-freq": 2.0,
    "sz-ac-thr-z": 3.0,
    "sz-ac-min-dur": 5.0,
    "sz-ac-merge-gap": 3.0,
    "sz-ac-bl-pct": 15,
    "sz-ac-bl-rms": 10.0,
    # Boundary refinement
    "sz-ac-bnd-rms-win": 100.0,
    "sz-ac-bnd-rms-thr": 2.0,
    "sz-ac-bnd-max-trim": 5.0,
    "sz-ac-bnd-window": 2.0,
    "sz-ac-bnd-rate": 2.0,
    "sz-ac-bnd-amp-x": 2.0,
    # Pre-ictal local baseline
    "sz-ac-lbl-start": 15.0,
    "sz-ac-lbl-end": 5.0,
    "sz-ac-lbl-trim-pct": 30.0,
}
_AC_SLIDER_KEYS = list(_AC_DEFAULTS.keys())

# ── Ensemble defaults ──────────────────────────────────────────────
_ENS_DEFAULTS = {
    "sz-ens-vote-thr": 2,
}
_ENS_SLIDER_KEYS = list(_ENS_DEFAULTS.keys())

# Default filter values
_FILTER_DEFAULTS = {
    "min_conf": 0, "max_conf": None,
    "min_dur": 0, "max_dur": 100,
    "min_spikes": 0, "max_spikes": None,
    "min_amp": 0, "max_amp": None,
    "min_lbl": 3, "max_lbl": None,
    "min_top_amp": 0, "max_top_amp": None,
    "min_ll": 0, "max_ll": None,
    "min_energy": 0, "max_energy": None,
    "min_sigbl": 0, "max_sigbl": None,
    "min_freq": 0, "max_freq": None,
    "min_td": 0, "max_td": None,
    "channel": None, "severity": "",
}

# ── Help text for detection methods ────────────────────────────────
_METHOD_HELP_TEXT = """\
## Spike-Train
**Reference:** Twele et al., 2017

Detects individual spikes exceeding a z-score amplitude threshold, \
then groups them into trains by temporal proximity (inter-spike interval). \
Trains are classified as HVSW, HPD, or electroclinical/convulsive based on \
spike frequency, amplitude evolution, and morphology.

**Parameter sections:**

| Section | Description |
|---------|-------------|
| **Baseline** | How the quiet-period baseline amplitude is estimated. *Percentile*: takes the quietest P% of short RMS windows. *Rolling*: recomputes every N minutes using a lookback window. *First N min*: uses the first 5 minutes. |
| **Spike Front-end** | Bandpass filter range, z-score amplitude threshold (threshold = mean + z x std), minimum prominence (spike must stand out from local context), width constraints (rejects slow waves and single-sample noise), refractory period (minimum time between successive spikes). |
| **Boundary Refinement** | *Signal (RMS)*: computes a short-window RMS envelope and walks from the first/last spike to find where signal energy drops below threshold - produces precise electrographic onset/offset. *Spike density*: trims event edges to where local spike rate and amplitude exceed thresholds. *None*: uses raw spike-based boundaries. |
| **Train Grouping** | Max inter-spike interval (spikes further apart start a new train), minimum spikes per train, minimum duration, and minimum inter-event interval (gap to separate distinct events). |

Best for models with clear spike-and-wave patterns (kainate, pilocarpine).

---

## Spectral Band (17-25 Hz)
**Reference:** Casillas-Espinosa et al., 2019 (ASSYST)

Computes a Spectral Band Index (SBI) = power in target band (default 17-25 Hz) \
divided by total power in a reference band (default 1-50 Hz) per sliding window. \
Thresholds the SBI timeseries against the quiet baseline distribution. \
100% sensitivity across 179 rats (10,600 seizures) in the original study.

**Parameter sections:**

| Section | Description |
|---------|-------------|
| **Baseline** | Percentile of SBI distribution used to estimate quiet-state statistics. |
| **Detection (SBI)** | Target frequency band (low/high Hz), reference band (for ratio normalisation), sliding window size and step, z-score threshold above baseline SBI. |
| **Boundary Refinement** | *Signal (RMS)*: refines coarse window-based boundaries using the raw signal's RMS envelope for sample-level precision. *None*: uses SBI window edges directly. |
| **Event Grouping** | Minimum event duration and merge gap (events closer than this are merged). |

Best as a first-pass detector for models with consistent spectral signature. \
Works well when spike morphology varies but the seizure frequency band is stable.

---

## Autocorrelation
**Reference:** White et al., 2006

Combines two complementary metrics computed per sliding window:

1. **Range autocorrelation**: compresses signal into min/max sub-windows and \
measures overlap between consecutive ranges. High overlap = rhythmic, \
repetitive activity.
2. **Spike frequency**: counts detected spikes per window using the same \
spike front-end as the Spike-Train method.

A window is flagged as ictal when **both** metrics exceed their thresholds \
simultaneously. 96% PPV, 100% sensitivity in the original study.

**Parameter sections:**

| Section | Description |
|---------|-------------|
| **Baseline** | Same options as Spike-Train (percentile, rolling, first N). Controls both the spike amplitude threshold and the autocorrelation baseline. |
| **Spike Front-end + Detection** | Spike detection params (same as Spike-Train) plus autocorrelation-specific: sub-window size (data points for range computation), lookahead (points to compare), analysis window size/step, minimum spike frequency, autocorrelation z-score threshold. |
| **Boundary Refinement** | All three options: *Signal (RMS)*, *Spike density*, *None*. Uses first/last spike as anchor for RMS refinement. |
| **Event Grouping** | Minimum duration and merge gap. |

Best for models with rhythmic seizure patterns. High specificity due to \
dual-metric requirement.

---

## Ensemble

Runs 2 or 3 of the above detectors independently, then combines results \
via temporal overlap voting. An event survives only if at least N methods \
detected overlapping activity.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| **Methods to combine** | Select which detectors to include. |
| **Voting threshold** | Minimum number of methods that must agree (e.g., 2 of 3). |
| **Merge strategy** | *Union*: merged event spans the widest boundaries. *Intersection*: spans only the overlap region. |
| **Confidence merge** | How confidence scores are combined (mean or max). |

Individual method parameters are inherited from the other method tabs. \
Switch to each method to tune its parameters, then select Ensemble to combine.

Best for maximum specificity - reduces false positives by requiring agreement.

---

## Common Concepts

### Baseline Methods
- **Percentile**: Computes RMS in short windows, takes the quietest P% as \
baseline. Default P=15. Most robust for recordings with variable activity.
- **Rolling**: Recomputes baseline every N minutes using a lookback window. \
Adapts to slow drift in signal amplitude.
- **First N min**: Uses the first 5 minutes of the recording as baseline. \
Only suitable if the recording starts with quiet EEG.

### Boundary Refinement
- **Signal (RMS)**: Computes a short-window RMS envelope and walks from the \
candidate boundary to find where signal energy crosses above/below a \
threshold. Produces precise electrographic onset/offset.
- **Spike density**: Trims event edges to the first/last point where local \
spike rate and amplitude exceed thresholds. Only available for methods \
that detect spikes (Spike-Train, Autocorrelation).
- **None**: Uses raw candidate boundaries without refinement.

### Pre-ictal Local Baseline
A separate comparison window (default: 20-5 seconds before event onset) \
is used to compute a local signal-to-baseline ratio for confidence scoring. \
This runs for all methods post-detection and is not a detection parameter.
"""

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
    # All method params go into one dict — keys have distinct prefixes so no collision.
    resolved = {k: _val(k) for k in _SZ_DEFAULTS}
    # Include dropdown values so they're saved in the detection file too
    resolved["sz-bl-method"] = state.extra.get("sz_bl_method", "percentile")
    resolved["sz-bnd-method"] = state.extra.get("sz_bnd_method", "signal")
    # Merge spectral band / autocorrelation / ensemble params into same dict
    for k in _SB_DEFAULTS:
        if k in overrides:
            resolved[k] = overrides[k]
        elif k in persisted:
            resolved[k] = persisted[k]
        else:
            resolved[k] = _SB_DEFAULTS[k]
    for k in _AC_DEFAULTS:
        if k in overrides:
            resolved[k] = overrides[k]
        elif k in persisted:
            resolved[k] = persisted[k]
        else:
            resolved[k] = _AC_DEFAULTS[k]
    for k in _ENS_DEFAULTS:
        if k in overrides:
            resolved[k] = overrides[k]
        elif k in persisted:
            resolved[k] = persisted[k]
        else:
            resolved[k] = _ENS_DEFAULTS[k]
    # Also persist the selected method and ensemble sub-options
    resolved["sz-method"] = state.extra.get("sz_method", "spike_train")
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
    persisted_classify = False  # subtype classification removed
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
            n_total=len(state.seizure_events),
        )

    # Persisted detection method
    persisted_method = state.extra.get("sz_method", "spike_train")

    # Value helpers for new method params — all in same persisted dict now
    def _sb_val(key):
        if key in overrides:
            return overrides[key]
        if key in persisted:
            return persisted[key]
        return _SB_DEFAULTS[key]

    def _ac_val(key):
        if key in overrides:
            return overrides[key]
        if key in persisted:
            return persisted[key]
        return _AC_DEFAULTS[key]

    def _ens_val(key):
        if key in overrides:
            return overrides[key]
        if key in persisted:
            return persisted[key]
        return _ENS_DEFAULTS[key]

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "20px"},
                children=[
                    html.H4("Seizure Detection", style={"margin": "0"}),
                    html.Span(
                        id="sz-method-badge",
                        children={o["value"]: o["label"] for o in _METHOD_OPTIONS}.get(
                            persisted_method, "Spike-Train"),
                        style={"fontSize": "0.78rem", "color": "#8b949e",
                               "border": "1px solid #2d333b", "borderRadius": "12px",
                               "padding": "2px 10px"},
                    ),
                ],
            ),

            # ── Detection method selector ───────────────────────────
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px",
                               "marginBottom": "6px"},
                        children=[
                            html.Label(
                                "Detection method",
                                style={"fontSize": "0.82rem", "fontWeight": "500",
                                       "color": "#8b949e", "margin": "0"},
                            ),
                            dbc.Button(
                                "?", id="sz-method-help-btn", size="sm",
                                color="secondary", outline=True,
                                style={"padding": "0px 7px", "fontSize": "0.75rem",
                                       "lineHeight": "1.4", "borderRadius": "50%"},
                            ),
                        ],
                    ),
                    dbc.RadioItems(
                        id="sz-method-selector",
                        options=_METHOD_OPTIONS,
                        value=persisted_method,
                        inline=True,
                        className="mb-2",
                        style={"fontSize": "0.82rem"},
                    ),
                ],
            ),

            # ── Help modal ─────────────────────────────────────────────
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Detection Methods Reference")),
                    dbc.ModalBody(dcc.Markdown(
                        _METHOD_HELP_TEXT,
                        style={"fontSize": "0.85rem", "lineHeight": "1.6"},
                    )),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="sz-method-help-close",
                                   className="ms-auto btn-ned-secondary"),
                    ),
                ],
                id="sz-method-help-modal",
                size="xl",
                scrollable=True,
                is_open=False,
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

            # ── Spike-Train params (shown/hidden) ──────────────────
            # Flow: Baseline → Spike Frontend → Boundary → Grouping
            html.Div(
                id="sz-params-spike-train",
                style={"display": "block" if persisted_method == "spike_train" else "none"},
                children=[
                    dbc.Row([
                        dbc.Col([_baseline_params(_val, persisted_bl_method)], width=3),
                        dbc.Col([_spike_frontend_params(_val)], width=3),
                        dbc.Col([_boundary_params(_val, persisted_bnd_method)], width=3),
                        dbc.Col([_train_grouping_params(_val)], width=3),
                    ], className="g-3 mb-3"),
                ],
            ),

            # ── Spectral Band params (shown/hidden) ────────────────
            html.Div(
                id="sz-params-spectral-band",
                style={"display": "block" if persisted_method == "spectral_band" else "none"},
                children=[_spectral_band_params(
                    _sb_val,
                    sb_bnd_method=state.extra.get("sz_sb_bnd_method", "none"),
                )],
            ),

            # ── Autocorrelation params (shown/hidden) ──────────────
            html.Div(
                id="sz-params-autocorrelation",
                style={"display": "block" if persisted_method == "autocorrelation" else "none"},
                children=[_autocorrelation_params(
                    _ac_val,
                    ac_bl_method=state.extra.get("sz_ac_bl_method", "percentile"),
                    ac_bnd_method=state.extra.get("sz_ac_bnd_method", "signal"),
                )],
            ),

            # ── Ensemble params (shown/hidden) ─────────────────────
            html.Div(
                id="sz-params-ensemble",
                style={"display": "block" if persisted_method == "ensemble" else "none"},
                children=[_ensemble_params(_ens_val, state.extra.get("sz_ens_methods",
                          ["spike_train", "spectral_band"]))],
            ),

            # Hidden: subtype params (kept for callback compatibility —
            # the detect callback still reads these param-slider States)
            html.Div(
                children=[
                    dbc.Checkbox(id="sz-classify-subtypes", value=False),
                    _subtype_params(_val, classify_on=False),
                ],
                style={"display": "none"},
            ),

            # Action buttons
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "20px",
                       "flexWrap": "wrap"},
                children=[
                    dbc.Button("Detect Seizures", id="sz-detect-btn",
                               className="btn-ned-primary"),
                    dbc.Button(
                        "\U0001F4C1 Detect All Files",
                        id="sz-detect-all-btn",
                        style={"display": "inline-block"
                               if state.extra.get("project_files")
                               else "none",
                               "backgroundColor": "#238636",
                               "borderColor": "#238636",
                               "color": "#ffffff",
                               "fontWeight": "600"},
                    ),
                    dbc.Button("Clear Results", id="sz-clear-btn",
                               className="btn-ned-danger",
                               style={"display": "inline-block" if has_results else "none"}),
                    html.Div(style={"flex": "1"}),
                    dbc.Button("Restore Defaults", id="sz-recall-defaults-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Save User Params", id="sz-save-settings-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Recall User Params", id="sz-recall-settings-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Recall Detection Params", id="sz-recall-det-btn",
                               size="sm",
                               style={"backgroundColor": "#d29922",
                                      "borderColor": "#d29922",
                                      "color": "#0d1117",
                                      "fontWeight": "600",
                                      "display": "inline-block" if has_results else "none"}),
                ],
            ),

            # Detect All progress area
            html.Div(id="sz-detect-all-status", style={"marginBottom": "8px"}),
            # Hidden interval + store for polling Detect All progress
            dcc.Interval(id="sz-detect-all-poll", interval=800, disabled=True),
            dcc.Store(id="sz-detect-all-running", data=False),

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
                children=_prerender_inspector(state, rec, selected_event_key, insp_opts, insp_yr, sid=sid)
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
                # Hidden: severity filter (kept for callback compatibility)
                html.Div(
                    dcc.Dropdown(id="sz-filter-severity", value="", clearable=False),
                    style={"display": "none"},
                ),
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


# ── Spectral band parameter builder ───────────────────────────────────


def _spectral_band_params(_val, sb_bnd_method="none") -> html.Div:
    # Flow: Baseline → Detection Frontend → Boundary → Grouping
    sb_signal_controls = html.Div(
        id="sz-sb-bnd-signal-controls",
        style={"display": "block" if sb_bnd_method == "signal" else "none"},
        children=[
            param_control("RMS window (ms)", "sz-sb-bnd-rms-win",
                          10.0, 500.0, 10.0, _val("sz-sb-bnd-rms-win")),
            param_control("RMS threshold (x BL)", "sz-sb-bnd-rms-thr",
                          0.5, 10.0, 0.5, _val("sz-sb-bnd-rms-thr")),
            param_control("Max trim (s)", "sz-sb-bnd-max-trim",
                          0.5, 20.0, 0.5, _val("sz-sb-bnd-max-trim")),
        ],
    )
    return dbc.Row([
        dbc.Col([
            collapsible_section(
                "Baseline", "sz-sb-baseline",
                default_open=True,
                children=[
                    param_control("Baseline percentile", "sz-sb-bl-pct",
                                  1, 50, 1, _val("sz-sb-bl-pct")),
                    html.Hr(style={"margin": "10px 0", "borderColor": "#30363d"}),
                    html.Span("Pre-ictal local baseline",
                              style={"fontSize": "0.75rem", "color": "#8b949e",
                                     "display": "block", "marginBottom": "6px"}),
                    param_control("Window start (s before onset)", "sz-sb-lbl-start",
                                  1.0, 120.0, 1.0, _val("sz-sb-lbl-start")),
                    param_control("Window end (s before onset)", "sz-sb-lbl-end",
                                  1.0, 60.0, 1.0, _val("sz-sb-lbl-end")),
                    param_control("Trim top % (spike removal)", "sz-sb-lbl-trim-pct",
                                  0, 80, 5, _val("sz-sb-lbl-trim-pct")),
                ],
            ),
        ], width=3),
        dbc.Col([
            collapsible_section(
                "Detection (SBI)", "sz-sb-band",
                default_open=True,
                children=[
                    param_control("Band low (Hz)", "sz-sb-band-low",
                                  1.0, 50.0, 1.0, _val("sz-sb-band-low")),
                    param_control("Band high (Hz)", "sz-sb-band-high",
                                  5.0, 100.0, 1.0, _val("sz-sb-band-high")),
                    param_control("Ref band low (Hz)", "sz-sb-ref-low",
                                  0.5, 20.0, 0.5, _val("sz-sb-ref-low"),
                                  "Lower bound of reference band for SBI ratio."),
                    param_control("Ref band high (Hz)", "sz-sb-ref-high",
                                  10.0, 200.0, 1.0, _val("sz-sb-ref-high"),
                                  "Upper bound of reference band for SBI ratio."),
                    param_control("Window (s)", "sz-sb-window",
                                  0.5, 10.0, 0.5, _val("sz-sb-window")),
                    param_control("Step (s)", "sz-sb-step",
                                  0.25, 5.0, 0.25, _val("sz-sb-step")),
                    param_control("Threshold (z-score)", "sz-sb-thr-z",
                                  1.0, 10.0, 0.5, _val("sz-sb-thr-z"),
                                  "SBI must exceed baseline_mean + z × baseline_std"),
                ],
            ),
        ], width=3),
        dbc.Col([
            collapsible_section(
                "Boundary Refinement", "sz-sb-boundary",
                default_open=True,
                children=[
                    html.Div([
                        html.Label("Method", style={"fontSize": "0.78rem", "color": "#8b949e",
                                                    "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="sz-sb-bnd-method",
                            options=[
                                {"label": "Signal (RMS)", "value": "signal"},
                                {"label": "None", "value": "none"},
                            ],
                            value=sb_bnd_method, clearable=False,
                            style={"fontSize": "0.82rem"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    sb_signal_controls,
                ],
            ),
        ], width=3),
        dbc.Col([
            collapsible_section(
                "Event Grouping", "sz-sb-grp",
                default_open=True,
                children=[
                    param_control("Min duration (s)", "sz-sb-min-dur",
                                  1.0, 60.0, 0.5, _val("sz-sb-min-dur")),
                    param_control("Merge gap (s)", "sz-sb-merge-gap",
                                  0.5, 30.0, 0.5, _val("sz-sb-merge-gap")),
                ],
            ),
        ], width=3),
    ], className="g-3 mb-3")


# ── Autocorrelation parameter builder ─────────────────────────────────


def _autocorrelation_params(_val, ac_bl_method="percentile",
                            ac_bnd_method="signal") -> html.Div:
    # Flow: Baseline → Spike Frontend + Detection → Boundary → Grouping
    ac_bnd_signal = html.Div(
        id="sz-ac-bnd-signal-controls",
        style={"display": "block" if ac_bnd_method == "signal" else "none"},
        children=[
            param_control("RMS window (ms)", "sz-ac-bnd-rms-win",
                          10.0, 500.0, 10.0, _val("sz-ac-bnd-rms-win")),
            param_control("RMS threshold (x BL)", "sz-ac-bnd-rms-thr",
                          0.5, 10.0, 0.5, _val("sz-ac-bnd-rms-thr")),
            param_control("Max trim (s)", "sz-ac-bnd-max-trim",
                          0.5, 20.0, 0.5, _val("sz-ac-bnd-max-trim")),
        ],
    )
    ac_bnd_density = html.Div(
        id="sz-ac-bnd-density-controls",
        style={"display": "block" if ac_bnd_method == "spike_density" else "none"},
        children=[
            param_control("Boundary window (s)", "sz-ac-bnd-window",
                          0.5, 10.0, 0.5, _val("sz-ac-bnd-window")),
            param_control("Min rate (Hz)", "sz-ac-bnd-rate",
                          0.5, 20.0, 0.5, _val("sz-ac-bnd-rate")),
            param_control("Min amplitude (x BL)", "sz-ac-bnd-amp-x",
                          0.5, 10.0, 0.5, _val("sz-ac-bnd-amp-x")),
        ],
    )
    return dbc.Row([
        dbc.Col([
            collapsible_section(
                "Baseline", "sz-ac-baseline",
                default_open=True,
                children=[
                    html.Div([
                        html.Label("Method", style={"fontSize": "0.78rem", "color": "#8b949e",
                                                    "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="sz-ac-bl-method",
                            options=[
                                {"label": "Percentile", "value": "percentile"},
                                {"label": "Rolling", "value": "rolling"},
                                {"label": "First N min", "value": "first_n"},
                            ],
                            value=ac_bl_method, clearable=False,
                            style={"fontSize": "0.82rem"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    param_control("Percentile", "sz-ac-bl-pct",
                                  1, 50, 1, _val("sz-ac-bl-pct")),
                    param_control("RMS window (s)", "sz-ac-bl-rms",
                                  1.0, 60.0, 1.0, _val("sz-ac-bl-rms")),
                    html.Hr(style={"margin": "10px 0", "borderColor": "#30363d"}),
                    html.Span("Pre-ictal local baseline",
                              style={"fontSize": "0.75rem", "color": "#8b949e",
                                     "display": "block", "marginBottom": "6px"}),
                    param_control("Window start (s before onset)", "sz-ac-lbl-start",
                                  1.0, 120.0, 1.0, _val("sz-ac-lbl-start")),
                    param_control("Window end (s before onset)", "sz-ac-lbl-end",
                                  1.0, 60.0, 1.0, _val("sz-ac-lbl-end")),
                    param_control("Trim top % (spike removal)", "sz-ac-lbl-trim-pct",
                                  0, 80, 5, _val("sz-ac-lbl-trim-pct")),
                ],
            ),
        ], width=3),
        dbc.Col([
            collapsible_section(
                "Spike Front-end", "sz-ac-spike",
                default_open=True,
                children=[
                    param_control("Bandpass low (Hz)", "sz-ac-bp-low",
                                  0.5, 100.0, 0.5, _val("sz-ac-bp-low")),
                    param_control("Bandpass high (Hz)", "sz-ac-bp-high",
                                  10.0, 500.0, 1.0, _val("sz-ac-bp-high")),
                    param_control("Threshold (z-score)", "sz-ac-spike-amp",
                                  1.0, 10.0, 0.5, _val("sz-ac-spike-amp")),
                    param_control("Refractory (ms)", "sz-ac-spike-refr",
                                  5.0, 200.0, 5.0, _val("sz-ac-spike-refr")),
                    html.Hr(style={"margin": "8px 0", "borderColor": "#30363d"}),
                    html.Span("Autocorrelation",
                              style={"fontSize": "0.75rem", "color": "#8b949e",
                                     "display": "block", "marginBottom": "6px"}),
                    param_control("Sub-window (pts)", "sz-ac-subwin",
                                  10, 100, 5, _val("sz-ac-subwin"),
                                  "Data points per sub-window for range computation."),
                    param_control("Lookahead (pts)", "sz-ac-lookahead",
                                  20, 200, 10, _val("sz-ac-lookahead"),
                                  "Data points to look ahead for overlap."),
                    param_control("Window (s)", "sz-ac-window",
                                  2.0, 30.0, 1.0, _val("sz-ac-window")),
                    param_control("Step (s)", "sz-ac-step",
                                  0.5, 10.0, 0.5, _val("sz-ac-step")),
                    param_control("Min spike freq (Hz)", "sz-ac-min-freq",
                                  0.5, 10.0, 0.5, _val("sz-ac-min-freq")),
                    param_control("Threshold (z-score)", "sz-ac-thr-z",
                                  1.0, 10.0, 0.5, _val("sz-ac-thr-z")),
                ],
            ),
        ], width=3),
        dbc.Col([
            collapsible_section(
                "Boundary Refinement", "sz-ac-boundary",
                default_open=True,
                children=[
                    html.Div([
                        html.Label("Method", style={"fontSize": "0.78rem", "color": "#8b949e",
                                                    "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="sz-ac-bnd-method",
                            options=[
                                {"label": "Signal (RMS)", "value": "signal"},
                                {"label": "Spike density", "value": "spike_density"},
                                {"label": "None", "value": "none"},
                            ],
                            value=ac_bnd_method, clearable=False,
                            style={"fontSize": "0.82rem"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    ac_bnd_signal,
                    ac_bnd_density,
                ],
            ),
        ], width=3),
        dbc.Col([
            collapsible_section(
                "Event Grouping", "sz-ac-grp",
                default_open=True,
                children=[
                    param_control("Min duration (s)", "sz-ac-min-dur",
                                  1.0, 60.0, 0.5, _val("sz-ac-min-dur")),
                    param_control("Merge gap (s)", "sz-ac-merge-gap",
                                  0.5, 30.0, 0.5, _val("sz-ac-merge-gap")),
                ],
            ),
        ], width=3),
    ], className="g-3 mb-3")


# ── Ensemble parameter builder ────────────────────────────────────────


def _ensemble_params(_val, selected_methods=None) -> html.Div:
    if selected_methods is None:
        selected_methods = ["spike_train", "spectral_band"]
    return html.Div([
        dbc.Row([
            dbc.Col([
                collapsible_section(
                    "Ensemble Settings", "sz-ens-cfg",
                    default_open=True,
                    children=[
                        html.Div([
                            html.Label("Methods to combine",
                                       style={"fontSize": "0.78rem", "color": "#8b949e",
                                              "marginBottom": "4px"}),
                            dbc.Checklist(
                                id="sz-ens-methods",
                                options=[
                                    {"label": "Spike-Train", "value": "spike_train"},
                                    {"label": "Spectral Band", "value": "spectral_band"},
                                    {"label": "Autocorrelation", "value": "autocorrelation"},
                                ],
                                value=selected_methods,
                                inline=True,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], style={"marginBottom": "12px"}),
                        param_control("Voting threshold", "sz-ens-vote-thr",
                                      1, 3, 1, _val("sz-ens-vote-thr"),
                                      "Min methods that must agree for an event to survive."),
                        html.Div([
                            html.Label("Merge strategy",
                                       style={"fontSize": "0.78rem", "color": "#8b949e",
                                              "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="sz-ens-merge",
                                options=[
                                    {"label": "Union (widest)", "value": "union"},
                                    {"label": "Intersection (tightest)", "value": "intersection"},
                                ],
                                value="union", clearable=False,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], style={"marginBottom": "12px"}),
                        html.Div([
                            html.Label("Confidence merge",
                                       style={"fontSize": "0.78rem", "color": "#8b949e",
                                              "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="sz-ens-conf-merge",
                                options=[
                                    {"label": "Mean", "value": "mean"},
                                    {"label": "Max", "value": "max"},
                                ],
                                value="mean", clearable=False,
                                style={"fontSize": "0.82rem"},
                            ),
                        ]),
                    ],
                ),
            ], width=6),
            dbc.Col([
                html.Div(
                    "Individual method parameters are inherited from the other "
                    "tabs above. Switch to each method to tune its parameters, "
                    "then select Ensemble to combine results.",
                    style={"fontSize": "0.78rem", "color": "#8b949e",
                           "padding": "12px", "border": "1px solid #30363d",
                           "borderRadius": "6px", "marginTop": "8px"},
                ),
            ], width=6),
        ], className="g-3 mb-3"),
    ])


# ── Collapse toggle callbacks ─────────────────────────────────────────

_ALL_COLLAPSE_SECTIONS = [
    "sz-spike", "sz-train", "sz-baseline", "sz-boundary", "sz-subtype",
    # Spectral band
    "sz-sb-baseline", "sz-sb-band", "sz-sb-boundary", "sz-sb-grp",
    # Autocorrelation
    "sz-ac-baseline", "sz-ac-spike", "sz-ac-boundary", "sz-ac-grp",
    # Ensemble
    "sz-ens-cfg",
]

for section_id in _ALL_COLLAPSE_SECTIONS:
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


@callback(
    Output("sz-sb-bnd-signal-controls", "style"),
    Input("sz-sb-bnd-method", "value"),
    prevent_initial_call=True,
)
def toggle_sb_boundary_controls(method):
    """Show/hide spectral band boundary sub-controls."""
    return {"display": "block"} if method == "signal" else {"display": "none"}


@callback(
    Output("sz-ac-bnd-signal-controls", "style"),
    Output("sz-ac-bnd-density-controls", "style"),
    Input("sz-ac-bnd-method", "value"),
    prevent_initial_call=True,
)
def toggle_ac_boundary_controls(method):
    """Show/hide autocorrelation boundary sub-controls."""
    show = {"display": "block"}
    hide = {"display": "none"}
    if method == "signal":
        return show, hide
    elif method == "spike_density":
        return hide, show
    return hide, hide


# ── Method selector toggle ───────────────────────────────────────────


@callback(
    Output("sz-params-spike-train", "style"),
    Output("sz-params-spectral-band", "style"),
    Output("sz-params-autocorrelation", "style"),
    Output("sz-params-ensemble", "style"),
    Output("sz-method-badge", "children"),
    Input("sz-method-selector", "value"),
    prevent_initial_call=True,
)
def toggle_method_params(method):
    """Show/hide parameter sections based on selected detection method."""
    show = {"display": "block"}
    hide = {"display": "none"}
    labels = {o["value"]: o["label"] for o in _METHOD_OPTIONS}
    styles = {
        "spike_train": hide, "spectral_band": hide,
        "autocorrelation": hide, "ensemble": hide,
    }
    styles[method] = show
    return (
        styles["spike_train"],
        styles["spectral_band"],
        styles["autocorrelation"],
        styles["ensemble"],
        labels.get(method, "Spike-Train"),
    )


# ── Help modal toggle ─────────────────────────────────────────────


@callback(
    Output("sz-method-help-modal", "is_open"),
    Input("sz-method-help-btn", "n_clicks"),
    Input("sz-method-help-close", "n_clicks"),
    State("sz-method-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_help_modal(open_clicks, close_clicks, is_open):
    """Open/close the detection methods help modal."""
    return not is_open


# ── Auto-save non-MATCH components to server state ──────────────────


@callback(
    Output("store-sz-extras", "data"),
    Input("sz-channel-selector", "value"),
    Input("sz-bl-method", "value"),
    Input("sz-bnd-method", "value"),
    Input("sz-classify-subtypes", "value"),
    Input("sz-method-selector", "value"),
    # New method dropdowns
    Input("sz-sb-bnd-method", "value"),
    Input("sz-ac-bl-method", "value"),
    Input("sz-ac-bnd-method", "value"),
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
    # Unpack *args: 8 fixed + n_min + n_max filter IDs + 2 dropdowns + 1 toggle + 5 inspector + sid
    n_min = len(_ALL_FILTER_MIN_IDS)
    n_max = len(_ALL_FILTER_MAX_IDS)
    (channels, bl_method, bnd_method, classify, method_sel,
     sb_bnd_method, ac_bl_method, ac_bnd_method) = args[0:8]
    filt_min_vals = args[8:8 + n_min]
    filt_max_vals = args[8 + n_min:8 + n_min + n_max]
    filt_channel = args[8 + n_min + n_max]
    filt_severity = args[8 + n_min + n_max + 1]
    filt_enabled = args[8 + n_min + n_max + 2]
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
    if method_sel is not None:
        state.extra["sz_method"] = method_sel
    if sb_bnd_method is not None:
        state.extra["sz_sb_bnd_method"] = sb_bnd_method
    if ac_bl_method is not None:
        state.extra["sz_ac_bl_method"] = ac_bl_method
    if ac_bnd_method is not None:
        state.extra["sz_ac_bnd_method"] = ac_bnd_method
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
    # ── Method selector ──
    State("sz-method-selector", "value"),
    # ── Spectral band params ──
    *[State({"type": "param-slider", "key": k}, "value") for k in _SB_SLIDER_KEYS],
    State("sz-sb-bnd-method", "value"),
    # ── Autocorrelation params ──
    *[State({"type": "param-slider", "key": k}, "value") for k in _AC_SLIDER_KEYS],
    State("sz-ac-bl-method", "value"),
    State("sz-ac-bnd-method", "value"),
    # ── Ensemble params ──
    *[State({"type": "param-slider", "key": k}, "value") for k in _ENS_SLIDER_KEYS],
    State("sz-ens-methods", "value"),
    State("sz-ens-merge", "value"),
    State("sz-ens-conf-merge", "value"),
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
    # New method params
    detection_method,
    # Spectral band (13 params + 3 pre-ictal + 1 dropdown)
    sb_band_low, sb_band_high, sb_ref_low, sb_ref_high,
    sb_window, sb_step, sb_thr_z, sb_bl_pct, sb_min_dur, sb_merge_gap,
    sb_bnd_rms_win, sb_bnd_rms_thr, sb_bnd_max_trim,
    sb_lbl_start, sb_lbl_end, sb_lbl_trim_pct,
    sb_bnd_method,
    # Autocorrelation (19 params + 3 pre-ictal + 2 dropdowns)
    ac_bp_low, ac_bp_high, ac_spike_amp, ac_spike_refr,
    ac_subwin, ac_lookahead, ac_window, ac_step,
    ac_min_freq, ac_thr_z, ac_min_dur, ac_merge_gap, ac_bl_pct, ac_bl_rms,
    ac_bnd_rms_win, ac_bnd_rms_thr, ac_bnd_max_trim,
    ac_bnd_window, ac_bnd_rate, ac_bnd_amp_x,
    ac_lbl_start, ac_lbl_end, ac_lbl_trim_pct,
    ac_bl_method, ac_bnd_method,
    # Ensemble
    ens_vote_thr,
    ens_methods, ens_merge, ens_conf_merge,
    sid,
):
    """Run seizure detection (multi-method), clear results, or apply filters."""
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
        n_total = len(state.seizure_events)
        results = _build_results(rec, filtered, classify_subtypes,
                                 selected_event_key=selected_ek,
                                 all_channels=selected_channels,
                                 n_total=n_total)
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

    # Warn if Animal IDs not assigned
    ch_ids = state.extra.get("channel_animal_ids", {})
    missing_ids = [ch for ch in selected_channels if ch not in ch_ids or not ch_ids[ch]]
    if missing_ids:
        return (
            alert(
                f"Animal IDs not assigned for channel(s): {missing_ids}. "
                "Go to the Load tab and fill in the Animal ID column before detecting.",
                "warning",
            ),
            no_update, no_update, no_update, no_update,
        )

    try:
        from eeg_seizure_analyzer.detection.spike_train_seizure import (
            SpikeTrainSeizureDetector,
        )
        from eeg_seizure_analyzer.detection.spectral_band_seizure import (
            SpectralBandDetector,
        )
        from eeg_seizure_analyzer.detection.autocorrelation_seizure import (
            AutocorrelationDetector,
        )
        from eeg_seizure_analyzer.detection.ensemble_seizure import (
            EnsembleDetector,
        )
        from eeg_seizure_analyzer.detection.confidence import (
            compute_event_quality,
            compute_confidence_score,
            compute_local_baseline_ratio,
            compute_top_spike_amplitude,
        )

        method = detection_method or "spike_train"

        # ── Build detector + params based on selected method ────────
        if method == "spike_train":
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

        elif method == "spectral_band":
            params = SpectralBandParams(
                band_low=float(sb_band_low),
                band_high=float(sb_band_high),
                ref_band_low=float(sb_ref_low),
                ref_band_high=float(sb_ref_high),
                window_sec=float(sb_window),
                step_sec=float(sb_step),
                threshold_z=float(sb_thr_z),
                baseline_percentile=int(sb_bl_pct),
                min_duration_sec=float(sb_min_dur),
                merge_gap_sec=float(sb_merge_gap),
                boundary_method=sb_bnd_method or "none",
                boundary_rms_window_ms=float(sb_bnd_rms_win or 100),
                boundary_rms_threshold_x=float(sb_bnd_rms_thr or 2.0),
                boundary_max_trim_sec=float(sb_bnd_max_trim or 5.0),
            )
            detector = SpectralBandDetector()

        elif method == "autocorrelation":
            params = AutocorrelationParams(
                bandpass_low=float(ac_bp_low),
                bandpass_high=float(ac_bp_high),
                spike_amplitude_x_baseline=float(ac_spike_amp),
                spike_refractory_ms=float(ac_spike_refr),
                subwindow_points=int(ac_subwin),
                lookahead_points=int(ac_lookahead),
                acorr_window_sec=float(ac_window),
                acorr_step_sec=float(ac_step),
                min_spike_freq_hz=float(ac_min_freq),
                acorr_threshold_z=float(ac_thr_z),
                min_duration_sec=float(ac_min_dur),
                merge_gap_sec=float(ac_merge_gap),
                baseline_method=ac_bl_method or "percentile",
                baseline_percentile=int(ac_bl_pct),
                baseline_rms_window_sec=float(ac_bl_rms or 10.0),
                boundary_method=ac_bnd_method or "signal",
                boundary_rms_window_ms=float(ac_bnd_rms_win or 100),
                boundary_rms_threshold_x=float(ac_bnd_rms_thr or 2.0),
                boundary_max_trim_sec=float(ac_bnd_max_trim or 5.0),
                boundary_window_sec=float(ac_bnd_window or 2.0),
                boundary_min_rate_hz=float(ac_bnd_rate or 2.0),
                boundary_min_amplitude_x=float(ac_bnd_amp_x or 2.0),
            )
            detector = AutocorrelationDetector()

        elif method == "ensemble":
            detector = None  # handled separately below
            params = None
        else:
            return (
                alert(f"Unknown method: {method}", "danger"),
                no_update, no_update, no_update, no_update,
            )

        seizures = []
        detection_info = {}

        _src = getattr(rec, "source_path", "") or ""

        if method == "ensemble":
            # Run each selected sub-detector and pass results to ensemble
            ens_method_list = ens_methods or ["spike_train", "spectral_band"]
            method_events = {}

            for sub_method in ens_method_list:
                if sub_method == "spike_train":
                    sub_params = SpikeTrainSeizureParams(
                        classify_subtypes=bool(classify_subtypes),
                        bandpass_low=float(bp_low), bandpass_high=float(bp_high),
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
                    )
                    sub_det = SpikeTrainSeizureDetector()
                elif sub_method == "spectral_band":
                    sub_params = SpectralBandParams(
                        band_low=float(sb_band_low), band_high=float(sb_band_high),
                        ref_band_low=float(sb_ref_low), ref_band_high=float(sb_ref_high),
                        window_sec=float(sb_window), step_sec=float(sb_step),
                        threshold_z=float(sb_thr_z), baseline_percentile=int(sb_bl_pct),
                        min_duration_sec=float(sb_min_dur), merge_gap_sec=float(sb_merge_gap),
                        boundary_method=sb_bnd_method or "none",
                        boundary_rms_window_ms=float(sb_bnd_rms_win or 100),
                        boundary_rms_threshold_x=float(sb_bnd_rms_thr or 2.0),
                        boundary_max_trim_sec=float(sb_bnd_max_trim or 5.0),
                    )
                    sub_det = SpectralBandDetector()
                elif sub_method == "autocorrelation":
                    sub_params = AutocorrelationParams(
                        bandpass_low=float(ac_bp_low), bandpass_high=float(ac_bp_high),
                        spike_amplitude_x_baseline=float(ac_spike_amp),
                        spike_refractory_ms=float(ac_spike_refr),
                        subwindow_points=int(ac_subwin), lookahead_points=int(ac_lookahead),
                        acorr_window_sec=float(ac_window), acorr_step_sec=float(ac_step),
                        min_spike_freq_hz=float(ac_min_freq),
                        acorr_threshold_z=float(ac_thr_z),
                        min_duration_sec=float(ac_min_dur), merge_gap_sec=float(ac_merge_gap),
                        baseline_method=ac_bl_method or "percentile",
                        baseline_percentile=int(ac_bl_pct),
                        baseline_rms_window_sec=float(ac_bl_rms or 10.0),
                        boundary_method=ac_bnd_method or "signal",
                        boundary_rms_window_ms=float(ac_bnd_rms_win or 100),
                        boundary_rms_threshold_x=float(ac_bnd_rms_thr or 2.0),
                        boundary_max_trim_sec=float(ac_bnd_max_trim or 5.0),
                        boundary_window_sec=float(ac_bnd_window or 2.0),
                        boundary_min_rate_hz=float(ac_bnd_rate or 2.0),
                        boundary_min_amplitude_x=float(ac_bnd_amp_x or 2.0),
                    )
                    sub_det = AutocorrelationDetector()
                else:
                    continue

                sub_events = []
                for ch in selected_channels:
                    if ch < 0 or ch >= rec.n_channels:
                        continue
                    ch_events = sub_det.detect(rec, ch, params=sub_params)
                    sub_events.extend(ch_events)
                    if hasattr(sub_det, "_last_detection_info"):
                        detection_info[ch] = sub_det._last_detection_info.copy()
                method_events[sub_method] = sub_events

            ens_det = EnsembleDetector()
            ens_params = EnsembleParams(
                methods=ens_method_list,
                voting_threshold=int(ens_vote_thr or 2),
                merge_strategy=ens_merge or "union",
                confidence_merge=ens_conf_merge or "mean",
            )
            seizures = ens_det.detect_ensemble(method_events, params=ens_params)
            if hasattr(ens_det, "_last_detection_info"):
                for ch in selected_channels:
                    if ch not in detection_info:
                        detection_info[ch] = ens_det._last_detection_info.copy()

        else:
            # Single-method detection (spike_train, spectral_band, autocorrelation)
            # Use chunked detection for large files (>30 min per channel)
            _use_chunked = (
                _src.lower().endswith(".edf")
                and rec.duration_sec > 1800
                and method == "spike_train"  # chunked only for spike-train for now
            )

            if _use_chunked:
                from eeg_seizure_analyzer.detection.base import detect_chunked
                valid_channels = [ch for ch in selected_channels
                                  if 0 <= ch < rec.n_channels]
                seizures, detection_info = detect_chunked(
                    detector,
                    path=_src,
                    channels=valid_channels,
                    chunk_duration_sec=1800.0,
                    overlap_sec=30.0,
                    params=params,
                )
            else:
                for ch in selected_channels:
                    if ch < 0 or ch >= rec.n_channels:
                        continue
                    ch_events = detector.detect(rec, ch, params=params)
                    seizures.extend(ch_events)
                    if hasattr(detector, "_last_detection_info"):
                        detection_info[ch] = detector._last_detection_info.copy()

        seizures.sort(key=lambda e: e.onset_sec)

        # Compute proper confidence scores using the confidence module
        # Use method-specific pre-ictal local baseline params
        if method == "spectral_band":
            _lbl_start = float(sb_lbl_start) if sb_lbl_start is not None else -15.0
            _lbl_end = float(sb_lbl_end) if sb_lbl_end is not None else -5.0
            _lbl_trim = float(sb_lbl_trim_pct) if sb_lbl_trim_pct is not None else 30.0
        elif method == "autocorrelation":
            _lbl_start = float(ac_lbl_start) if ac_lbl_start is not None else -15.0
            _lbl_end = float(ac_lbl_end) if ac_lbl_end is not None else -5.0
            _lbl_trim = float(ac_lbl_trim_pct) if ac_lbl_trim_pct is not None else 30.0
        else:
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

        # Assign animal IDs from channel mapping
        ch_ids = state.extra.get("channel_animal_ids", {})
        for ev in seizures:
            aid = ch_ids.get(ev.channel, "")
            if aid:
                ev.animal_id = aid

        # ── Activity channel z-score ──────────────────────────────
        # If paired activity channels exist, compute activity z-score
        # for each event (stored as a feature for the ML model).
        act_rec = state.activity_recordings.get("paired")
        pairings = state.channel_pairings or []
        if act_rec is not None and pairings and seizures:
            try:
                from eeg_seizure_analyzer.processing.activity import (
                    flag_events_activity,
                )
                eeg_to_act = {}
                for p in pairings:
                    if p.activity_index is not None:
                        eeg_to_act[p.eeg_index] = p.activity_index

                for act_idx in set(eeg_to_act.values()):
                    act_data = act_rec.data[act_idx]
                    act_fs = act_rec.fs
                    ch_events = [
                        ev for ev in seizures
                        if eeg_to_act.get(ev.channel) == act_idx
                    ]
                    if ch_events:
                        flag_events_activity(ch_events, act_data, act_fs)
            except Exception:
                import traceback
                traceback.print_exc()

        state.seizure_events = seizures
        state.st_detection_info = detection_info
        state.extra.pop("sz_selected_event_key", None)  # new detection, clear selection

        # ── Persist all method params to sz_params ──────────────────
        # Capture actual slider values used during this detection so they
        # survive tab switches and are saved in the detection JSON.
        all_params = dict(state.extra.get("sz_params", {}))
        # Spike-train params
        for k, v in zip(_SZ_SLIDER_KEYS, [
            bp_low, bp_high, spike_amp, spike_min_uv, spike_prom,
            spike_maxw, spike_minw, spike_refr, max_isi, min_spikes,
            min_dur, min_iei, bl_pct, bl_rms,
            bnd_rms_win, bnd_rms_thr, bnd_max_trim,
            bnd_window, bnd_rate, bnd_amp_x,
            hvsw_amp, hvsw_freq, hvsw_dur, hvsw_max_ev,
            hpd_amp, hpd_freq, hpd_dur,
            conv_dur, conv_amp, conv_postictal,
            lbl_start, lbl_end, lbl_trim_pct,
        ]):
            if v is not None:
                all_params[k] = v
        all_params["sz-bl-method"] = bl_method
        all_params["sz-bnd-method"] = bnd_method
        # Spectral band params
        for k, v in zip(_SB_SLIDER_KEYS, [
            sb_band_low, sb_band_high, sb_ref_low, sb_ref_high,
            sb_window, sb_step, sb_thr_z, sb_bl_pct, sb_min_dur, sb_merge_gap,
            sb_bnd_rms_win, sb_bnd_rms_thr, sb_bnd_max_trim,
            sb_lbl_start, sb_lbl_end, sb_lbl_trim_pct,
        ]):
            if v is not None:
                all_params[k] = v
        all_params["sz-sb-bnd-method"] = sb_bnd_method or "none"
        # Autocorrelation params
        for k, v in zip(_AC_SLIDER_KEYS, [
            ac_bp_low, ac_bp_high, ac_spike_amp, ac_spike_refr,
            ac_subwin, ac_lookahead, ac_window, ac_step,
            ac_min_freq, ac_thr_z, ac_min_dur, ac_merge_gap, ac_bl_pct, ac_bl_rms,
            ac_bnd_rms_win, ac_bnd_rms_thr, ac_bnd_max_trim,
            ac_bnd_window, ac_bnd_rate, ac_bnd_amp_x,
            ac_lbl_start, ac_lbl_end, ac_lbl_trim_pct,
        ]):
            if v is not None:
                all_params[k] = v
        all_params["sz-ac-bl-method"] = ac_bl_method or "percentile"
        all_params["sz-ac-bnd-method"] = ac_bnd_method or "signal"
        # Ensemble params
        all_params["sz-ens-vote-thr"] = ens_vote_thr
        all_params["sz-method"] = method
        state.extra["sz_params"] = all_params

        # ── Refresh training-tab annotations ──────────────────────
        # Clear stale in-memory annotations so the training tab
        # re-initialises from fresh detections on next visit.
        try:
            from eeg_seizure_analyzer.io.annotation_store import (
                detections_to_annotations, merge_annotations,
                load_annotations, save_annotations,
            )
            _src = getattr(rec, "source_path", None) or ""
            new_annotations = detections_to_annotations(
                seizures, _src,
                animal_id=state.extra.get("tr_animal_id", ""),
            )
            existing = load_annotations(_src) if _src else None
            if existing:
                merged = merge_annotations(existing, new_annotations,
                                           tolerance_sec=1.0)
            else:
                merged = new_annotations
            state.extra["tr_annotations"] = [a.to_dict() for a in merged]
            state.extra["tr_current_idx"] = 0
            if _src:
                save_annotations(_src, merged)
        except Exception:
            # Fallback: just clear so layout() rebuilds from seizure_events
            state.extra.pop("tr_annotations", None)
            state.extra["tr_current_idx"] = 0

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
                    detector_name=_DETECTOR_NAMES.get(method, "SpikeTrainSeizureDetector"),
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
                                 all_channels=selected_channels,
                                 n_total=len(seizures))
        n_ch = len(selected_channels)

        return (
            alert(f"[{({o['value']: o['label'] for o in _METHOD_OPTIONS}).get(method, method)}] "
                  f"Found {len(seizures)} seizure(s) across {n_ch} channel(s).", "success"),
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


def _prerender_inspector(state, rec, selected_event_key, insp_opts, insp_yr, sid=None):
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
            sid=sid,
        )
    except Exception:
        return []


def _build_results(rec, seizures, classify_on, *, selected_event_key=None,
                    all_channels=None, n_total=None):
    """Build the results display (summary line + table)."""
    if not seizures:
        return empty_state("\u2714", "No Seizures Found",
                           "No seizure events match the current parameters / filters.")

    n_shown = len(seizures)
    n_all = n_total if n_total is not None else n_shown
    summary = html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "16px",
               "marginBottom": "12px", "padding": "8px 16px",
               "background": "#161b22", "borderRadius": "8px",
               "border": "1px solid #2d333b"},
        children=[
            html.Span(
                f"Total detected: {n_all}",
                style={"fontSize": "0.88rem", "fontWeight": "600",
                       "color": "#c9d1d9"},
            ),
            html.Span("\u2022", style={"color": "#484f58"}),
            html.Span(
                f"Shown after filtering: {n_shown}",
                style={"fontSize": "0.88rem", "fontWeight": "500",
                       "color": "#58a6ff" if n_shown < n_all else "#c9d1d9"},
            ),
        ],
    )

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

        # Detection method label
        det_method = feat.get("detection_method", "spike_train")
        method_labels = {"spike_train": "Spike-Train", "spectral_band": "Spectral",
                         "autocorrelation": "Autocorr", "ensemble": "Ensemble"}
        method_label = method_labels.get(det_method, det_method)

        row = {
            "#": i + 1,
            "ID": e.event_id if e.event_id > 0 else i + 1,
            "_event_key": ek,
            "_source": "manual" if is_manual else "detector",
            "Method": method_label,
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
        }
        table_data.append(row)
        if selected_event_key and ek == selected_event_key:
            selected_rows = [row]

    col_defs = [
        {"field": "#", "maxWidth": 55, "minWidth": 40},
        {"field": "ID", "maxWidth": 55, "minWidth": 40, "headerTooltip": "Stable event ID"},
        {"field": "_event_key", "hide": True},
        {"field": "_source", "hide": True},
        {"field": "Method", "flex": 1, "minWidth": 75,
         "headerTooltip": "Detection method used"},
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
        # Hidden columns — kept for ML export / future use
        {"field": "LL (z)", "hide": True},
        {"field": "Energy (z)", "hide": True},
        {"field": "Sig/BL", "hide": True},
        {"field": "Spec Ent", "hide": True},
        {"field": "Peak Freq", "hide": True},
        {"field": "\u03b8/\u03b4", "hide": True},
    ]
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
        summary,
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


# ── Detect All Files — threaded with progress polling ─────────────

import json as _json
import threading
from pathlib import Path as _Path

_DETECT_ALL_PROGRESS_DIR = _Path.home() / ".eeg_seizure_analyzer" / "cache"


def _progress_path(sid: str) -> _Path:
    """Return the progress file path for a session."""
    _DETECT_ALL_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return _DETECT_ALL_PROGRESS_DIR / f"detect_all_{sid}.json"


def _write_progress(sid: str, current: int, total: int,
                    current_file: str, events_so_far: int,
                    done: bool = False, error_msg: str = ""):
    """Write progress info to a JSON file for polling."""
    _json.dump(
        {"current": current, "total": total, "file": current_file,
         "events": events_so_far, "done": done, "error": error_msg},
        open(_progress_path(sid), "w"),
    )


def _detect_all_worker(sid: str, project_files: list, sz_params: dict):
    """Run detection on all project files in a background thread."""
    from eeg_seizure_analyzer.detection.spike_train_seizure import (
        SpikeTrainSeizureDetector,
    )
    from eeg_seizure_analyzer.detection.confidence import (
        compute_event_quality,
        compute_confidence_score,
        compute_local_baseline_ratio,
        compute_top_spike_amplitude,
    )
    from eeg_seizure_analyzer.io.edf_reader import (
        scan_edf_channels, read_edf, read_edf_paired, auto_pair_channels,
    )
    from eeg_seizure_analyzer.io.persistence import save_detections
    from eeg_seizure_analyzer.io.channel_ids import load_channel_ids
    from eeg_seizure_analyzer.io.annotation_store import (
        detections_to_annotations, merge_annotations,
        load_annotations, save_annotations,
    )

    def _p(key):
        return sz_params.get(key, _SZ_DEFAULTS.get(key))

    params = SpikeTrainSeizureParams(
        bandpass_low=float(_p("sz-bp-low")),
        bandpass_high=float(_p("sz-bp-high")),
        spike_amplitude_x_baseline=float(_p("sz-spike-amp")),
        spike_min_amplitude_uv=float(_p("sz-spike-min-uv")),
        spike_prominence_x_baseline=float(_p("sz-spike-prom")),
        spike_max_width_ms=float(_p("sz-spike-maxw")),
        spike_min_width_ms=float(_p("sz-spike-minw")),
        spike_refractory_ms=float(_p("sz-spike-refr")),
        max_interspike_interval_ms=float(_p("sz-max-isi")),
        min_train_spikes=int(_p("sz-min-spikes")),
        min_train_duration_sec=float(_p("sz-min-dur")),
        min_interevent_interval_sec=float(_p("sz-min-iei")),
        baseline_method=sz_params.get("sz-bl-method", "percentile"),
        baseline_percentile=int(_p("sz-bl-pct")),
        baseline_rms_window_sec=float(_p("sz-bl-rms")),
        boundary_method=sz_params.get("sz-bnd-method", "rms"),
        boundary_rms_window_ms=float(_p("sz-bnd-rms-win")),
        boundary_rms_threshold_x=float(_p("sz-bnd-rms-thr")),
        boundary_max_trim_sec=float(_p("sz-bnd-max-trim")),
        boundary_window_sec=float(_p("sz-bnd-window")),
        boundary_min_rate_hz=float(_p("sz-bnd-rate")),
        boundary_min_amplitude_x=float(_p("sz-bnd-amp-x")),
        hvsw_min_amplitude_x=float(_p("sz-hvsw-amp")),
        hvsw_min_frequency_hz=float(_p("sz-hvsw-freq")),
        hvsw_min_duration_sec=float(_p("sz-hvsw-dur")),
        hvsw_max_evolution=float(_p("sz-hvsw-max-ev")),
        hpd_min_amplitude_x=float(_p("sz-hpd-amp")),
        hpd_min_frequency_hz=float(_p("sz-hpd-freq")),
        hpd_min_duration_sec=float(_p("sz-hpd-dur")),
        convulsive_min_duration_sec=float(_p("sz-conv-dur")),
        convulsive_min_amplitude_x=float(_p("sz-conv-amp")),
        convulsive_postictal_suppression_sec=float(_p("sz-conv-postictal")),
    )

    bp_low = float(_p("sz-bp-low"))
    bp_high = float(_p("sz-bp-high"))

    detector = SpikeTrainSeizureDetector()
    total_events = 0
    errors = []
    n_total = len(project_files)

    for i, pf in enumerate(project_files):
        edf_path = pf["edf_path"]
        fname = pf["filename"]
        _write_progress(sid, i, n_total, fname, total_events)
        try:
            channel_info = scan_edf_channels(edf_path)
            eeg_indices, act_indices, pairings = auto_pair_channels(channel_info)
            has_pairs = any(p.activity_index is not None for p in pairings)

            if has_pairs:
                rec, act_rec = read_edf_paired(edf_path, list(eeg_indices), act_indices)
            else:
                rates = sorted(set(ch["fs"] for ch in channel_info))
                default_ch = [
                    ch["index"] for ch in channel_info
                    if "biopot" in ch.get("label", "").lower()
                    or ch["fs"] == max(rates)
                ]
                if not default_ch:
                    default_ch = [ch["index"] for ch in channel_info]
                rec = read_edf(edf_path, channels=default_ch)
                act_rec = None
            rec.source_path = edf_path

            ch_ids = load_channel_ids(edf_path) or {}
            selected_channels = list(range(rec.n_channels))

            seizures = []
            detection_info = {}

            use_chunked = rec.duration_sec > 1800 and edf_path.lower().endswith(".edf")
            if use_chunked:
                from eeg_seizure_analyzer.detection.base import detect_chunked
                seizures, detection_info = detect_chunked(
                    detector, path=edf_path,
                    channels=selected_channels,
                    chunk_duration_sec=1800.0, overlap_sec=30.0,
                    params=params,
                )
            else:
                for ch in selected_channels:
                    ch_events = detector.detect(rec, ch, params=params)
                    seizures.extend(ch_events)
                    if hasattr(detector, "_last_detection_info"):
                        detection_info[ch] = detector._last_detection_info.copy()

            for event in seizures:
                bl_rms_val = detection_info.get(event.channel, {}).get("baseline_mean")
                try:
                    qm = compute_event_quality(rec, event, baseline_rms=bl_rms_val,
                                               bandpass_low=bp_low, bandpass_high=bp_high)
                    lbr = compute_local_baseline_ratio(rec, event,
                                                       bandpass_low=bp_low, bandpass_high=bp_high)
                    qm["local_baseline_ratio"] = round(lbr, 2)
                    qm["top_spike_amplitude_x"] = round(compute_top_spike_amplitude(event), 2)
                    event.quality_metrics = qm
                    event.confidence = compute_confidence_score(qm)
                except Exception:
                    event.quality_metrics = {}
                    event.confidence = 0.0

            seizures.sort(key=lambda e: (e.channel, e.onset_sec))
            for idx, ev in enumerate(seizures, start=1):
                ev.event_id = idx
                ev.animal_id = ch_ids.get(ev.channel, "")

            if act_rec is not None and pairings and seizures:
                try:
                    from eeg_seizure_analyzer.processing.activity import flag_events_activity
                    eeg_to_act = {}
                    for p in pairings:
                        if p.activity_index is not None:
                            eeg_to_act[p.eeg_index] = p.activity_index
                    for act_idx in set(eeg_to_act.values()):
                        act_data = act_rec.data[act_idx]
                        act_fs = act_rec.fs
                        ch_events = [ev for ev in seizures if eeg_to_act.get(ev.channel) == act_idx]
                        if ch_events:
                            flag_events_activity(ch_events, act_data, act_fs)
                except Exception:
                    pass

            if edf_path.lower().endswith(".edf"):
                save_detections(
                    edf_path=edf_path,
                    events=seizures,
                    detection_info=detection_info,
                    params_dict=sz_params,
                    detector_name="SpikeTrainSeizureDetector",
                    channels=selected_channels,
                )

            try:
                new_anns = detections_to_annotations(seizures, edf_path)
                existing = load_annotations(edf_path)
                if existing:
                    merged = merge_annotations(existing, new_anns, tolerance_sec=1.0)
                else:
                    merged = new_anns
                save_annotations(edf_path, merged)
            except Exception:
                pass

            total_events += len(seizures)
            pf["has_detections"] = True

        except Exception as e:
            errors.append(f"{fname}: {e}")
            import traceback
            traceback.print_exc()

    # Write final progress
    err_msg = ""
    if errors:
        err_msg = "; ".join(errors[:3])
    _write_progress(sid, n_total, n_total, "", total_events,
                    done=True, error_msg=err_msg)

    # Reload the active file into session state
    try:
        state = server_state.get_session(sid)
        original_idx = state.extra.get("project_active_idx", 0)
        state.recording = None
        state.seizure_events = []
        state.spike_events = []
        state.detected_events = []
        state.st_detection_info = {}
        state.sp_detection_info = {}
        from eeg_seizure_analyzer.dash_app.pages.upload import _load_edf_into_state
        _load_edf_into_state(state, project_files[original_idx]["edf_path"])
    except Exception:
        import traceback
        traceback.print_exc()


@callback(
    Output("sz-detect-all-running", "data"),
    Output("sz-detect-all-poll", "disabled"),
    Output("sz-detect-all-status", "children", allow_duplicate=True),
    Output("sz-detect-all-btn", "disabled"),
    Input("sz-detect-all-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_detect_all(n_clicks, sid):
    """Launch the background detection thread and start polling."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    state = server_state.get_session(sid)
    project_files = state.extra.get("project_files", [])
    if not project_files:
        return False, True, alert("No project loaded.", "warning"), False

    sz_params = dict(state.extra.get("sz_params", {}))

    # Write initial progress
    _write_progress(sid, 0, len(project_files), project_files[0]["filename"], 0)

    # Launch background thread
    t = threading.Thread(
        target=_detect_all_worker,
        args=(sid, project_files, sz_params),
        daemon=True,
    )
    t.start()

    n = len(project_files)
    progress_bar = html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "12px"},
        children=[
            dbc.Progress(
                value=0, max=n,
                style={"flex": "1", "height": "22px"},
                color="success",
                striped=True, animated=True,
            ),
            html.Span(
                f"0/{n} files — starting...",
                style={"color": "#8b949e", "fontSize": "0.82rem",
                       "whiteSpace": "nowrap"},
            ),
        ],
    )
    return True, False, progress_bar, True  # running=True, poll enabled, disable btn


@callback(
    Output("sz-detect-all-status", "children", allow_duplicate=True),
    Output("sz-detect-all-poll", "disabled", allow_duplicate=True),
    Output("sz-detect-all-running", "data", allow_duplicate=True),
    Output("sz-detect-all-btn", "disabled", allow_duplicate=True),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("sz-detect-all-poll", "n_intervals"),
    State("session-id", "data"),
    State("sz-detect-all-running", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def poll_detect_all(n_intervals, sid, is_running, refresh):
    """Poll progress file and update the progress bar."""
    if not is_running:
        return no_update, no_update, no_update, no_update, no_update

    p = _progress_path(sid)
    if not p.exists():
        return no_update, no_update, no_update, no_update, no_update

    try:
        data = _json.loads(p.read_text())
    except Exception:
        return no_update, no_update, no_update, no_update, no_update

    current = data.get("current", 0)
    total = data.get("total", 1)
    fname = data.get("file", "")
    events = data.get("events", 0)
    done = data.get("done", False)
    error_msg = data.get("error", "")

    if done:
        # Clean up progress file
        try:
            p.unlink()
        except OSError:
            pass

        msg = f"Detected {events} seizure(s) across {total} files."
        if error_msg:
            msg += f" Errors: {error_msg}"
            result = alert(msg, "warning")
        else:
            result = alert(msg, "success")

        # Stop polling, re-enable button, refresh tab to show updated results
        return result, True, False, False, (refresh or 0) + 1

    # Still running — update progress bar
    pct = int(100 * current / total) if total else 0
    short_name = fname if len(fname) <= 40 else fname[:37] + "..."
    progress_bar = html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "12px"},
        children=[
            dbc.Progress(
                value=current, max=total,
                label=f"{current}/{total}",
                style={"flex": "1", "height": "22px"},
                color="success",
                striped=True, animated=True,
            ),
            html.Span(
                f"{current}/{total} files — {short_name}  ({events} events)",
                style={"color": "#8b949e", "fontSize": "0.82rem",
                       "whiteSpace": "nowrap"},
            ),
        ],
    )
    return progress_bar, no_update, no_update, no_update, no_update


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
            sid=sid,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return alert(f"Inspector error: {e}", "danger")


def _render_inspector(rec, event, det_info, state, *,
                      show_spikes=True, show_baseline=True,
                      show_threshold=True, bandpass_on=False,
                      y_range=None, sid=None):
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

    # Determine detection method for this event
    features = event.features or {}
    det_method = features.get("detection_method", "spike_train")
    has_spikes = det_method in ("spike_train", "autocorrelation", "ensemble")

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

    # Spike dots (spike-train, autocorrelation, ensemble — not spectral_band)
    if show_spikes and has_spikes:
        spike_times = det_info.get("all_spike_times", [])
        spike_samples = det_info.get("all_spike_samples", [])
        # For ensemble events, also try event-level spike data
        if not spike_times and det_method == "ensemble":
            spike_times = features.get("spike_times", [])
            spike_samples = features.get("spike_samples", [])
        if spike_times:
            in_t, in_y, out_t, out_y = [], [], [], []
            for i, t in enumerate(spike_times):
                if win_start <= t <= win_end:
                    local = spike_samples[i] - start_idx if i < len(spike_samples) else -1
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

    # Baseline / threshold lines (spike-train and autocorrelation only)
    if has_spikes:
        baseline_val = det_info.get("baseline_mean")
        if show_baseline and baseline_val is not None:
            fig_eeg.add_hline(
                y=baseline_val, row=1, col=1,
                line=dict(color="#3fb950", width=1, dash="dot"),
                annotation_text="Baseline",
                annotation_position="top right",
            )
            fig_eeg.add_hline(
                y=-baseline_val, row=1, col=1,
                line=dict(color="#3fb950", width=1, dash="dot"),
            )
        threshold_val = det_info.get("threshold")
        if show_threshold and threshold_val is not None:
            fig_eeg.add_hline(
                y=threshold_val, row=1, col=1,
                line=dict(color="#d29922", width=1, dash="dash"),
                annotation_text="Threshold",
                annotation_position="top right",
            )
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

    # ── Method-specific extra plots ───────────────────────────────
    method_plots = []

    if det_method == "spectral_band":
        # SBI timeseries plot
        sbi_times = det_info.get("sbi_times", [])
        sbi_values = det_info.get("sbi_values", [])
        sbi_thr = det_info.get("threshold", 0)
        if sbi_times and sbi_values:
            fig_sbi = go.Figure()
            fig_sbi.add_trace(go.Scatter(
                x=sbi_times, y=sbi_values,
                mode="lines", name="SBI",
                line=dict(width=1.2, color="#58a6ff"),
            ))
            fig_sbi.add_hline(
                y=sbi_thr,
                line=dict(color="#d29922", width=1, dash="dash"),
                annotation_text="Threshold",
                annotation_position="top right",
            )
            fig_sbi.add_vrect(
                x0=onset, x1=offset,
                fillcolor="rgba(248, 81, 73, 0.15)",
                line=dict(width=0), layer="below",
            )
            fig_sbi.update_layout(
                height=200,
                xaxis_title="Time (s)", yaxis_title="SBI (target/ref power)",
                xaxis_range=[win_start, win_end],
                showlegend=False,
                uirevision=f"sbi_{onset}_{ch}",
            )
            apply_fig_theme(fig_sbi)
            fig_sbi.update_layout(margin=dict(l=60, r=20, t=30, b=40))
            method_plots.append(html.Div([
                html.Div("Spectral Band Index (SBI)",
                         style={"fontSize": "0.82rem", "fontWeight": "600",
                                "color": "#8b949e", "marginBottom": "4px",
                                "marginTop": "12px"}),
                dcc.Graph(figure=fig_sbi, config={"scrollZoom": True}),
            ]))

    elif det_method == "autocorrelation":
        # Autocorrelation metric + spike frequency timeseries
        acorr_times = det_info.get("acorr_times", [])
        acorr_values = det_info.get("acorr_values", [])
        spike_freqs = det_info.get("spike_freqs", [])
        acorr_thr = det_info.get("acorr_threshold", 0)
        if acorr_times and acorr_values:
            fig_acorr = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.5], vertical_spacing=0.08,
            )
            fig_acorr.add_trace(go.Scatter(
                x=acorr_times, y=acorr_values,
                mode="lines", name="Range Autocorrelation",
                line=dict(width=1.2, color="#58a6ff"),
            ), row=1, col=1)
            fig_acorr.add_hline(
                y=acorr_thr, row=1, col=1,
                line=dict(color="#d29922", width=1, dash="dash"),
                annotation_text="Threshold",
                annotation_position="top right",
            )
            if spike_freqs:
                fig_acorr.add_trace(go.Scatter(
                    x=acorr_times, y=spike_freqs,
                    mode="lines", name="Spike Frequency",
                    line=dict(width=1.2, color="#3fb950"),
                ), row=2, col=1)
            fig_acorr.add_vrect(
                x0=onset, x1=offset,
                fillcolor="rgba(248, 81, 73, 0.15)",
                line=dict(width=0), layer="below", row=1, col=1,
            )
            fig_acorr.add_vrect(
                x0=onset, x1=offset,
                fillcolor="rgba(248, 81, 73, 0.15)",
                line=dict(width=0), layer="below", row=2, col=1,
            )
            fig_acorr.update_xaxes(range=[win_start, win_end], row=1, col=1)
            fig_acorr.update_xaxes(range=[win_start, win_end], title_text="Time (s)",
                                    row=2, col=1)
            fig_acorr.update_yaxes(title_text="Autocorrelation", row=1, col=1)
            fig_acorr.update_yaxes(title_text="Spike Freq (Hz)", row=2, col=1)
            fig_acorr.update_layout(
                height=300, showlegend=False,
                uirevision=f"acorr_{onset}_{ch}",
            )
            apply_fig_theme(fig_acorr)
            fig_acorr.update_layout(margin=dict(l=60, r=20, t=30, b=40))
            method_plots.append(html.Div([
                html.Div("Autocorrelation Metrics",
                         style={"fontSize": "0.82rem", "fontWeight": "600",
                                "color": "#8b949e", "marginBottom": "4px",
                                "marginTop": "12px"}),
                dcc.Graph(figure=fig_acorr, config={"scrollZoom": True}),
            ]))

    # ── Event detail metrics ────────────────────────────────────
    qm = event.quality_metrics or {}

    # Method-specific detail cards
    method_label_map = {"spike_train": "Spike-Train", "spectral_band": "Spectral Band",
                        "autocorrelation": "Autocorrelation", "ensemble": "Ensemble"}
    method_card = metric_card("Method", method_label_map.get(det_method, det_method))

    extra_cards = []
    if det_method == "spectral_band":
        sbi_peak = features.get("sbi_peak")
        if sbi_peak is not None:
            extra_cards.append(dbc.Col(metric_card("SBI Peak", f"{sbi_peak:.3f}"), width=2))
    elif det_method == "autocorrelation":
        peak_acorr = features.get("peak_acorr")
        mean_freq = features.get("mean_spike_frequency_hz")
        if peak_acorr is not None:
            extra_cards.append(dbc.Col(metric_card("Peak Autocorr", f"{peak_acorr:.3f}"), width=2))
        if mean_freq is not None:
            extra_cards.append(dbc.Col(metric_card("Spike Freq", f"{mean_freq:.1f} Hz"), width=2))
    elif det_method == "ensemble":
        contrib = features.get("contributing_methods", [])
        n_methods = features.get("n_methods", len(contrib))
        extra_cards.append(dbc.Col(metric_card("Methods", f"{n_methods} agreed"), width=2))
        if contrib:
            short = [m.replace("_", " ").title()[:8] for m in contrib]
            extra_cards.append(dbc.Col(metric_card("Sources", ", ".join(short)), width=2))

    detail_metrics = dbc.Row([
        dbc.Col(method_card, width=2),
        dbc.Col(metric_card("Channel", ch_name), width=2),
        dbc.Col(metric_card("Onset", f"{onset:.2f}s"), width=2),
        dbc.Col(metric_card("Duration", f"{event.duration_sec:.2f}s"), width=2),
        dbc.Col(metric_card("Spikes", str(features.get("n_spikes", "\u2014"))), width=2),
        dbc.Col(metric_card("Confidence", f"{event.confidence:.2f}"), width=2),
        *extra_cards,
    ], className="g-3 mb-3")

    # ── Video player (if available) ────────────────────────────────
    video_section = []
    video_path = state.extra.get("video_path")
    if video_path and sid:
        import os
        vname = os.path.basename(video_path)
        graph_id = "sz-insp-eeg-graph"
        video_id = "sz-insp-video"
        video_section = [
            html.Div(
                style={"marginTop": "16px"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center",
                               "gap": "12px", "marginBottom": "8px"},
                        children=[
                            html.Label("Video", style={"fontSize": "0.82rem",
                                                        "fontWeight": "600",
                                                        "color": "#8b949e"}),
                            html.Span(vname, style={"fontSize": "0.78rem",
                                                     "color": "#484f58"}),
                        ],
                    ),
                    html.Video(
                        id=video_id,
                        src=f"/video/{sid}#t={win_start:.1f}",
                        controls=True,
                        style={
                            "width": "100%",
                            "maxHeight": "360px",
                            "borderRadius": "8px",
                            "backgroundColor": "#000",
                        },
                    ),
                ],
            ),
        ]
    else:
        graph_id = "sz-insp-eeg-graph"

    return html.Div([
        html.Hr(style={"borderColor": "#2d333b", "margin": "24px 0"}),
        html.H5("Event Inspector",
                 style={"marginBottom": "16px", "color": "#58a6ff"}),
        detail_metrics,
        html.Div("EEG Trace", style={"fontSize": "0.82rem", "fontWeight": "600",
                                      "color": "#8b949e", "marginBottom": "4px"}),
        dcc.Graph(id=graph_id, figure=fig_eeg,
                  config={"scrollZoom": True, "displayModeBar": True}),
        *method_plots,
        *video_section,
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
    Input("sz-recall-det-btn", "n_clicks"),
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
                       "sz-recall-settings-btn", "sz-recall-det-btn"):
        return no_update, no_update
    # Guard: check the corresponding button actually has clicks
    btn_clicks = {"sz-recall-defaults-btn": args[0],
                  "sz-save-settings-btn": args[1],
                  "sz-recall-settings-btn": args[2],
                  "sz-recall-det-btn": args[3]}
    if not btn_clicks.get(trigger):
        return no_update, no_update
    n_keys = len(_SZ_SLIDER_KEYS)
    n_filter_ids = len(_ALL_FILTER_IDS)
    current_values = args[4:4 + n_keys]
    bl_method = args[4 + n_keys]
    bnd_method = args[4 + n_keys + 1]
    classify = args[4 + n_keys + 2]
    channels = args[4 + n_keys + 3]
    # Filter states
    filter_offset = 4 + n_keys + 4
    filter_slider_vals = args[filter_offset:filter_offset + n_filter_ids]
    filt_channel = args[filter_offset + n_filter_ids]
    filt_severity = args[filter_offset + n_filter_ids + 1]
    filt_enabled = args[filter_offset + n_filter_ids + 2]
    sid = args[-2]
    refresh = args[-1]
    state = server_state.get_session(sid)

    if trigger == "sz-recall-det-btn":
        # Load saved detection params from disk (same logic as sidebar callback)
        rec = state.recording
        if rec is None or not rec.source_path:
            return alert("No file loaded.", "warning"), no_update
        from eeg_seizure_analyzer.io.persistence import load_detections
        result = load_detections(rec.source_path)
        if result is None:
            return alert("No saved seizure detections found on disk.", "warning"), no_update
        saved_params = result.get("params", {})
        if saved_params:
            state.extra["sz_param_overrides"] = dict(saved_params)
            state.extra["sz_params"] = dict(saved_params)
            if "sz-bl-method" in saved_params:
                state.extra["sz_bl_method"] = saved_params["sz-bl-method"]
            if "sz-bnd-method" in saved_params:
                state.extra["sz_bnd_method"] = saved_params["sz-bnd-method"]
            if "sz-method" in saved_params:
                state.extra["sz_method"] = saved_params["sz-method"]
        saved_channels = result.get("channels", [])
        if saved_channels:
            state.extra["sz_selected_channels"] = saved_channels
        fs = result.get("filter_settings", {})
        if fs:
            filter_on = fs.get("filter_enabled", True)
            filter_vals_d = fs.get("filter_values", {})
            state.extra["sz_filter_enabled"] = filter_on
            if filter_vals_d:
                state.extra["sz_filter_values"] = filter_vals_d
            state.extra["tr_filter_on"] = filter_on
            if filter_vals_d:
                state.extra["tr_min_conf"] = filter_vals_d.get("min_conf", 0)
                state.extra["tr_min_dur"] = filter_vals_d.get("min_dur", 0)
                state.extra["tr_min_lbl"] = filter_vals_d.get("min_lbl", 0)
                state.extra["tr_max_conf"] = filter_vals_d.get("max_conf", None)
                state.extra["tr_max_dur"] = filter_vals_d.get("max_dur", None)
                state.extra["tr_max_lbl"] = filter_vals_d.get("max_lbl", None)
        n_p = len(saved_params)
        return alert(f"Detection params recalled ({n_p} params).", "success"), (refresh or 0) + 1

    if trigger == "sz-recall-defaults-btn":
        all_defaults = {**_SZ_DEFAULTS, **_SB_DEFAULTS, **_AC_DEFAULTS, **_ENS_DEFAULTS}
        state.extra["sz_param_overrides"] = all_defaults
        return alert("Default parameters restored.", "info"), (refresh or 0) + 1

    if trigger == "sz-save-settings-btn":
        from eeg_seizure_analyzer.dash_app.components import save_user_defaults
        params = {k: v for k, v in zip(_SZ_SLIDER_KEYS, current_values)}
        params["sz-bl-method"] = bl_method
        params["sz-bnd-method"] = bnd_method
        # Include all method params from server state
        all_method_params = state.extra.get("sz_params", {})
        for k in list(_SB_DEFAULTS) + list(_AC_DEFAULTS) + list(_ENS_DEFAULTS):
            if k in all_method_params:
                params[k] = all_method_params[k]
        # Save selected method
        params["sz-method"] = state.extra.get("sz_method", "spike_train")
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
        # Remove method selection from overrides — recall should restore
        # parameter values without changing the currently selected method
        saved.pop("sz-method", None)
        if filter_vals:
            state.extra["sz_filter_values"] = {**_FILTER_DEFAULTS, **filter_vals}
        state.extra["sz_filter_enabled"] = filter_enabled
        state.extra["sz_param_overrides"] = saved
        return alert("User params loaded.", "success"), (refresh or 0) + 1

    return no_update, no_update
