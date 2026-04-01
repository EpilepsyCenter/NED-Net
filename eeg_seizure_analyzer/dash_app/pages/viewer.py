"""Viewer tab: multi-channel EEG trace viewer with overlays."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    no_recording_placeholder,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter, notch_filter


# ── Helpers ───────────────────────────────────────────────────────────


def _minmax_downsample(
    time_arr: np.ndarray, data: np.ndarray, target_points: int = 2400,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max downsampling preserving spike morphology."""
    n = len(data)
    if n <= target_points:
        return time_arr, data
    n_buckets = max(1, target_points // 2)
    bucket_size = n // n_buckets
    times_out, data_out = [], []
    for i in range(n_buckets):
        s = i * bucket_size
        e = min(s + bucket_size, n)
        chunk, t_chunk = data[s:e], time_arr[s:e]
        mi, ma = np.argmin(chunk), np.argmax(chunk)
        if mi <= ma:
            times_out.extend([t_chunk[mi], t_chunk[ma]])
            data_out.extend([chunk[mi], chunk[ma]])
        else:
            times_out.extend([t_chunk[ma], t_chunk[mi]])
            data_out.extend([chunk[ma], chunk[mi]])
    return np.array(times_out), np.array(data_out)


def _nice_round(value: float) -> float:
    """Round to nearest 'nice' value (1, 2, 5 series)."""
    nice = [float(m * 10 ** e) for e in range(-3, 6) for m in [1, 2, 5]]
    return float(min(nice, key=lambda v: abs(v - value)))


def _activity_controls(state, act_ymin=0.0, act_ymax=1.0) -> html.Div:
    """Activity Y-range controls, shown only when activity channels are paired."""
    act_rec = state.activity_recordings.get("paired")
    has_act = act_rec is not None and bool(state.channel_pairings)

    return html.Div(
        id="viewer-act-controls",
        style={"display": "block" if has_act else "none"},
        children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Activity Y min",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Input(
                        id="viewer-act-ymin", type="number",
                        value=act_ymin, step=0.1, debounce=True,
                        className="form-control", style={"width": "100%"},
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Activity Y max",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Input(
                        id="viewer-act-ymax", type="number",
                        value=act_ymax, step=0.1, debounce=True,
                        className="form-control", style={"width": "100%"},
                    ),
                ], width=2),
            ], className="g-2 mb-3"),
        ],
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _filter_spikes_for_viewer(spikes, fv):
    """Lightweight IS filter for viewer overlay (avoids cross-module import)."""
    filtered = list(spikes)

    def _fmin(v):
        return float(v) if v is not None and v != "" else 0.0

    def _fmax(v):
        return float(v) if v is not None and v != "" else None

    checks = [
        ("min_amp", "amplitude", True),
        ("max_amp", "amplitude", False),
        ("min_xbl", "amplitude_x_baseline", True),
        ("max_xbl", "amplitude_x_baseline", False),
        ("min_dur_ms", "duration_ms", True),
        ("max_dur_ms", "duration_ms", False),
        ("min_snr", "local_snr", True),
        ("max_snr", "local_snr", False),
        ("min_sharp", "sharpness", True),
        ("max_sharp", "sharpness", False),
    ]
    for fk, feat_key, is_min in checks:
        v = fv.get(fk)
        if is_min:
            v = _fmin(v)
            if v > 0:
                filtered = [e for e in filtered
                            if (e.features.get(feat_key) or 0) >= v]
        else:
            v = _fmax(v)
            if v is not None:
                filtered = [e for e in filtered
                            if (e.features.get(feat_key) or 0) <= v]

    # Confidence
    min_conf = _fmin(fv.get("min_conf"))
    if min_conf > 0:
        filtered = [e for e in filtered if e.confidence >= min_conf]
    max_conf = _fmax(fv.get("max_conf"))
    if max_conf is not None:
        filtered = [e for e in filtered if e.confidence <= max_conf]

    return filtered


# ── Layout ────────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the viewer tab layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording

    # Compute default Y range
    n_samp = min(int(10 * rec.fs), rec.n_samples)
    ptps = [float(np.ptp(rec.data[i, :n_samp])) for i in range(rec.n_channels)]
    default_yrange = _nice_round(float(np.median(ptps)) * 1.5) if ptps else 1.0

    unit_label = rec.units[0] if rec.units else ""

    # Restore saved settings (or use defaults)
    saved = state.extra.get("viewer_settings", {})
    v_window = saved.get("window", 150)
    v_start = saved.get("start", 0)
    v_yrange = saved.get("yrange", default_yrange)
    v_height = saved.get("height", 600)
    v_bp_on = saved.get("bp_on", True)
    v_notch_on = saved.get("notch_on", False)
    v_bp_low = saved.get("bp_low", 1.0)
    v_bp_high = saved.get("bp_high", 50.0)
    v_notch_freq = saved.get("notch_freq", 50)
    v_show_events = saved.get("show_events", True)
    v_show_is = saved.get("show_is", False)
    v_show_spikes = saved.get("show_spikes", False)
    v_show_baseline = saved.get("show_baseline", True)
    v_show_threshold = saved.get("show_threshold", True)
    v_act_ymin = saved.get("act_ymin", 0.0)
    v_act_ymax = saved.get("act_ymax", 4.0)
    v_channels = saved.get("channels", list(range(rec.n_channels)))

    # Store default so callback can use it as fallback (not 100)
    state.extra["_viewer_default_yrange"] = default_yrange

    return html.Div(
        style={"padding": "24px"},
        children=[
            # Channel selection (first — most important)
            html.Div(
                style={"marginBottom": "8px", "display": "flex",
                       "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
                children=[
                    html.Label("Channels:",
                               style={"fontSize": "0.78rem", "color": "#8b949e",
                                      "margin": "0", "fontWeight": "500"}),
                    dbc.Checklist(
                        id="viewer-channel-checks",
                        options=[
                            {"label": rec.channel_names[i], "value": i}
                            for i in range(rec.n_channels)
                        ],
                        value=v_channels,
                        inline=True,
                        style={"fontSize": "0.8rem"},
                    ),
                    html.A("All", id="viewer-ch-all", href="#",
                           style={"fontSize": "0.75rem", "color": "#58a6ff",
                                  "cursor": "pointer", "marginLeft": "4px"}),
                    html.A("None", id="viewer-ch-none", href="#",
                           style={"fontSize": "0.75rem", "color": "#58a6ff",
                                  "cursor": "pointer"}),
                ],
            ),

            # Controls row
            dbc.Row(
                [
                    dbc.Col([
                        html.Label("Window (s)", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Input(
                            id="viewer-window-input", type="number",
                            min=1, max=600, step=1, value=v_window,
                            debounce=True, className="form-control",
                            style={"width": "100%"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Start (s)", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Input(
                            id="viewer-start-input", type="number",
                            min=0, max=rec.duration_sec, step=1, value=v_start,
                            debounce=True, className="form-control",
                            style={"width": "100%"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label(f"Y range ({unit_label})" if unit_label else "Y range",
                                   style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Input(
                            id="viewer-yrange-input", type="number",
                            min=0, step=0.01,
                            value=v_yrange,
                            debounce=True, className="form-control",
                            style={"width": "100%"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Height (px)", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Input(
                            id="viewer-height-input", type="number",
                            min=300, max=3000, step=50, value=v_height,
                            debounce=True, className="form-control",
                            style={"width": "100%"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Filters", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        html.Div([
                            dbc.Checkbox(id="viewer-bp-check", label="Bandpass", value=v_bp_on,
                                         style={"fontSize": "0.8rem"}),
                            dbc.Checkbox(id="viewer-notch-check", label="Notch", value=v_notch_on,
                                         style={"fontSize": "0.8rem"}),
                        ]),
                    ], width=2),
                    dbc.Col([
                        html.Label("Overlays", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        html.Div([
                            dbc.Checkbox(id="viewer-show-events", label="Seizures", value=v_show_events,
                                         style={"fontSize": "0.8rem"}),
                            dbc.Checkbox(id="viewer-show-is", label="IS events", value=v_show_is,
                                         style={"fontSize": "0.8rem"}),
                            dbc.Checkbox(id="viewer-show-spikes", label="Spike dots", value=v_show_spikes,
                                         style={"fontSize": "0.8rem"}),
                            dbc.Checkbox(id="viewer-show-baseline", label="Baseline", value=v_show_baseline,
                                         style={"fontSize": "0.8rem"}),
                            dbc.Checkbox(id="viewer-show-threshold", label="Threshold", value=v_show_threshold,
                                         style={"fontSize": "0.8rem"}),
                        ]),
                    ], width=2),
                ],
                className="g-2 mb-3",
            ),

            # Activity Y range (shown when activity channels exist)
            _activity_controls(state, v_act_ymin, v_act_ymax),

            # Filter params (shown when bandpass is on)
            dbc.Collapse(
                dbc.Row([
                    dbc.Col([
                        html.Label("BP Low (Hz)", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Input(id="viewer-bp-low", type="number", min=0.5, max=500,
                                  step=0.5, value=v_bp_low, debounce=True, className="form-control"),
                    ], width=2),
                    dbc.Col([
                        html.Label("BP High (Hz)", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Input(id="viewer-bp-high", type="number", min=1, max=1000,
                                  step=1, value=v_bp_high, debounce=True, className="form-control"),
                    ], width=2),
                    dbc.Col([
                        html.Label("Notch (Hz)", style={"fontSize": "0.78rem", "color": "#8b949e"}),
                        dcc.Dropdown(
                            id="viewer-notch-freq",
                            options=[{"label": "50 Hz", "value": 50},
                                     {"label": "60 Hz", "value": 60}],
                            value=v_notch_freq, clearable=False,
                            style={"fontSize": "0.82rem"},
                        ),
                    ], width=2),
                ], className="g-2 mb-3"),
                id="viewer-filter-collapse",
                is_open=v_bp_on or v_notch_on,
            ),

            # Navigation bar — just above the graph
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px",
                       "marginBottom": "4px"},
                children=[
                    dbc.Button("\u23EE", id="nav-start", size="sm",
                               className="btn-ned-secondary",
                               title="Go to start",
                               style={"fontSize": "1rem"}),
                    dbc.Button("\u23EA", id="nav-back-big", size="sm",
                               className="btn-ned-secondary", style={"fontSize": "1rem"}),
                    dbc.Button("\u25C0", id="nav-back", size="sm",
                               className="btn-ned-secondary", style={"fontSize": "1rem"}),
                    html.Div(
                        id="nav-time-display",
                        style={"flex": "1", "textAlign": "center", "fontWeight": "600",
                               "fontSize": "0.9rem"},
                    ),
                    dbc.Button("\u25B6", id="nav-fwd", size="sm",
                               className="btn-ned-secondary", style={"fontSize": "1rem"}),
                    dbc.Button("\u23E9", id="nav-fwd-big", size="sm",
                               className="btn-ned-secondary", style={"fontSize": "1rem"}),
                ],
            ),

            # Main graph
            dcc.Loading(
                dcc.Graph(
                    id="viewer-graph",
                    config={"scrollZoom": True, "displayModeBar": True},
                    style={"borderRadius": "8px"},
                ),
                type="circle",
                color="#58a6ff",
            ),

            # Video player (shown if MP4 exists)
            _video_player(state, sid, v_start),
        ],
    )


def _video_player(state, sid, start_sec):
    """Return a video player div if an MP4 is available, else a hidden div."""
    video_path = state.extra.get("video_path")
    if not video_path:
        return html.Div(id="viewer-video-container", style={"display": "none"})

    import os
    fname = os.path.basename(video_path)

    return html.Div(
        id="viewer-video-container",
        style={"marginTop": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px",
                       "marginBottom": "8px"},
                children=[
                    html.Label("Video", style={"fontSize": "0.85rem",
                                                "fontWeight": "600",
                                                "color": "#c9d1d9"}),
                    html.Span(fname, style={"fontSize": "0.78rem",
                                             "color": "#8b949e"}),
                    dbc.Button("Sync to EEG", id="viewer-video-sync",
                               size="sm", className="btn-ned-secondary"),
                ],
            ),
            html.Video(
                id="viewer-video-player",
                src=f"/video/{sid}",
                controls=True,
                style={
                    "width": "100%",
                    "maxHeight": "400px",
                    "borderRadius": "8px",
                    "backgroundColor": "#000",
                },
            ),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("viewer-filter-collapse", "is_open"),
    Input("viewer-bp-check", "value"),
    Input("viewer-notch-check", "value"),
)
def toggle_filter_collapse(bp_on, notch_on):
    return bp_on or notch_on


@callback(
    Output("viewer-channel-checks", "value"),
    Input("viewer-ch-all", "n_clicks"),
    Input("viewer-ch-none", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def viewer_channel_all_none(all_clicks, none_clicks, sid):
    """Handle All/None links for viewer channel selection."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update
    if trigger == "viewer-ch-all":
        return list(range(state.recording.n_channels))
    if trigger == "viewer-ch-none":
        return []
    return no_update


@callback(
    Output("viewer-start-input", "value"),
    Output("nav-time-display", "children"),
    Input("nav-start", "n_clicks"),
    Input("nav-back-big", "n_clicks"),
    Input("nav-back", "n_clicks"),
    Input("nav-fwd", "n_clicks"),
    Input("nav-fwd-big", "n_clicks"),
    Input("viewer-start-input", "value"),
    State("viewer-window-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_navigation(ns, bb, b, f, fb, start_val, window, sid):
    """Handle navigation button clicks and start input changes."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update, no_update

    rec = state.recording
    window = float(window or 10)
    max_start = max(0.0, rec.duration_sec - window)
    current = float(start_val or 0)

    trigger = ctx.triggered_id
    if trigger == "nav-start":
        current = 0.0
    elif trigger == "nav-back-big":
        current -= window * 5
    elif trigger == "nav-back":
        current -= window
    elif trigger == "nav-fwd":
        current += window
    elif trigger == "nav-fwd-big":
        current += window * 5

    current = max(0.0, min(max_start, current))
    time_display = f"{current:.1f}s \u2013 {current + window:.1f}s"
    return round(current, 2), time_display


@callback(
    Output("viewer-graph", "figure"),
    Input("viewer-start-input", "value"),
    Input("viewer-window-input", "value"),
    Input("viewer-yrange-input", "value"),
    Input("viewer-height-input", "value"),
    Input("viewer-bp-check", "value"),
    Input("viewer-notch-check", "value"),
    Input("viewer-bp-low", "value"),
    Input("viewer-bp-high", "value"),
    Input("viewer-notch-freq", "value"),
    Input("viewer-show-events", "value"),
    Input("viewer-show-is", "value"),
    Input("viewer-show-spikes", "value"),
    Input("viewer-show-baseline", "value"),
    Input("viewer-show-threshold", "value"),
    Input("viewer-channel-checks", "value"),
    Input("viewer-act-ymin", "value"),
    Input("viewer-act-ymax", "value"),
    State("session-id", "data"),
)
def update_viewer(
    start_sec, window_sec, y_range, plot_height,
    bp_on, notch_on, bp_low, bp_high, notch_freq,
    show_events, show_is, show_spikes, show_baseline, show_threshold,
    selected_channels, act_ymin, act_ymax,
    sid,
):
    """Build and return the EEG viewer figure."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return go.Figure()

    rec = state.recording
    start_sec = float(start_sec or 0)
    window_sec = float(window_sec or 10)
    # Use saved default as fallback, NOT 100
    default_yr = state.extra.get("_viewer_default_yrange", 1.0)
    if y_range is not None and y_range > 0:
        y_range = float(y_range)
    else:
        y_range = float(default_yr)
    plot_height = int(plot_height or 600)
    bp_low = float(bp_low or 1)
    bp_high = float(bp_high or 50)
    notch_freq = float(notch_freq or 50)
    act_ymin = float(act_ymin) if act_ymin is not None else 0.0
    act_ymax = float(act_ymax) if act_ymax is not None else 1.0

    end_sec = min(start_sec + window_sec, rec.duration_sec)
    start_idx = int(start_sec * rec.fs)
    end_idx = min(int(end_sec * rec.fs), rec.n_samples)

    # Use selected channels from viewer checklist, default to all
    if selected_channels is not None and len(selected_channels) > 0:
        channels = [ch for ch in selected_channels if 0 <= ch < rec.n_channels]
    else:
        channels = list(range(rec.n_channels))
    if not channels:
        channels = list(range(rec.n_channels))
    n_ch = len(channels)

    # Check for activity channels
    act_rec = state.activity_recordings.get("paired")
    pairings = state.channel_pairings
    has_paired_act = False
    if act_rec is not None and pairings:
        for ch_idx in channels:
            for p in pairings:
                if p.eeg_index == ch_idx and p.activity_index is not None:
                    has_paired_act = True
                    break
            if has_paired_act:
                break

    # Create figure
    if has_paired_act:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
    else:
        fig = go.Figure()

    spacing = float(y_range)
    channel_offsets = {}
    channel_displayed_data = {}

    # EEG traces
    for i, ch_idx in enumerate(channels):
        data = rec.data[ch_idx, start_idx:end_idx].copy()

        if bp_on:
            data = bandpass_filter(data, rec.fs, bp_low, bp_high)
        if notch_on:
            data = notch_filter(data, rec.fs, notch_freq)

        offset = -i * spacing
        channel_offsets[ch_idx] = offset
        channel_displayed_data[ch_idx] = data
        time_axis = np.linspace(start_sec, end_sec, len(data))

        ds_time, ds_data = _minmax_downsample(time_axis, data + offset)

        trace = go.Scattergl(
            x=ds_time, y=ds_data,
            mode="lines", name=rec.channel_names[ch_idx],
            line=dict(width=0.8),
        )
        if has_paired_act:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    # Event overlays
    _event_colors = [
        "rgba(88, 166, 255, 0.18)", "rgba(63, 185, 80, 0.18)",
        "rgba(210, 153, 34, 0.18)", "rgba(248, 81, 73, 0.18)",
        "rgba(188, 140, 255, 0.18)", "rgba(247, 120, 186, 0.18)",
    ]
    _event_borders = [
        "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#f778ba",
    ]

    # Seizure event overlays (rectangles) — use filtered set from detected_events
    _sz_events = [e for e in (state.detected_events or [])
                  if e.event_type == "seizure"]
    if show_events and _sz_events:
        for event in _sz_events:
            if event.offset_sec < start_sec or event.onset_sec > end_sec:
                continue
            if event.channel not in channel_offsets:
                continue

            ch_offset = channel_offsets[event.channel]
            half = spacing / 2.0
            ch_pos = list(channel_offsets.keys()).index(event.channel)

            fig.add_shape(
                type="rect",
                x0=max(event.onset_sec, start_sec),
                x1=min(event.offset_sec, end_sec),
                y0=ch_offset - half, y1=ch_offset + half,
                fillcolor=_event_colors[ch_pos % len(_event_colors)],
                line=dict(color=_event_borders[ch_pos % len(_event_borders)], width=1),
                layer="below",
            )

    # Interictal spike event overlays (green dots at peak)
    # Apply current IS filters if enabled
    if show_is and state.spike_events:
        sp_filter_on = state.extra.get("sp_filter_enabled", True)
        if sp_filter_on:
            sp_fv = state.extra.get("sp_filter_values", {})
            visible_spikes = _filter_spikes_for_viewer(state.spike_events, sp_fv)
        else:
            visible_spikes = state.spike_events

        is_t, is_y = [], []
        for event in visible_spikes:
            peak_t = event.features.get("peak_time_sec", event.onset_sec) if event.features else event.onset_sec
            if peak_t < start_sec or peak_t > end_sec:
                continue
            if event.channel not in channel_offsets:
                continue

            ch_offset = channel_offsets[event.channel]
            # Place dot at actual signal value if available
            displayed = channel_displayed_data.get(event.channel)
            sample_local = int((peak_t - start_sec) * rec.fs)
            if displayed is not None and 0 <= sample_local < len(displayed):
                yv = float(displayed[sample_local]) + ch_offset
            else:
                yv = ch_offset

            is_t.append(peak_t)
            is_y.append(yv)

        if is_t:
            is_trace = go.Scatter(
                x=is_t, y=is_y,
                mode="markers",
                marker=dict(color="#3fb950", size=6, symbol="circle", opacity=0.85),
                showlegend=False,
                hovertemplate="IS @ %{x:.3f}s<extra></extra>",
            )
            if has_paired_act:
                fig.add_trace(is_trace, row=1, col=1)
            else:
                fig.add_trace(is_trace)

    # Spike / baseline / threshold overlays from detection info
    det_info_all = state.st_detection_info
    if det_info_all:
        for ch_idx in channels:
            det_info = det_info_all.get(ch_idx)
            if det_info is None:
                continue
            ch_offset = channel_offsets[ch_idx]
            displayed = channel_displayed_data.get(ch_idx)

            # Spike dots
            if show_spikes:
                spike_times = det_info.get("all_spike_times", [])
                spike_samples = det_info.get("all_spike_samples", [])
                visible = [
                    i for i, t in enumerate(spike_times) if start_sec <= t <= end_sec
                ]
                if visible and displayed is not None:
                    sp_t = [spike_times[i] for i in visible]
                    sp_y = []
                    for i in visible:
                        local = spike_samples[i] - start_idx
                        if 0 <= local < len(displayed):
                            sp_y.append(float(displayed[local]) + ch_offset)
                        else:
                            sp_y.append(ch_offset)

                    # Color-code: in-event spikes = red, out-of-event = orange/bright
                    # Determine which spikes are inside detected events
                    events = state.detected_events or []
                    sp_colors = []
                    for st in sp_t:
                        in_event = any(
                            e.onset_sec <= st <= (e.onset_sec + e.duration_sec)
                            for e in events if e.channel == ch_idx
                        )
                        sp_colors.append("#f85149" if in_event else "#ffb347")
                    spike_trace = go.Scatter(
                        x=sp_t, y=sp_y,
                        mode="markers",
                        marker=dict(color=sp_colors, size=5, symbol="circle",
                                    opacity=1.0),
                        showlegend=False,
                        hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
                    )
                    if has_paired_act:
                        fig.add_trace(spike_trace, row=1, col=1)
                    else:
                        fig.add_trace(spike_trace)

            # Baseline lines
            if show_baseline:
                baseline_val = det_info.get("baseline_mean")
                if baseline_val is not None:
                    for sign in [1, -1]:
                        fig.add_shape(
                            type="line",
                            x0=start_sec, x1=end_sec,
                            y0=ch_offset + sign * baseline_val,
                            y1=ch_offset + sign * baseline_val,
                            line=dict(color="#3fb950", width=1, dash="dot"),
                            layer="above",
                        )

            # Threshold lines
            if show_threshold:
                threshold_val = det_info.get("threshold")
                if threshold_val is not None:
                    for sign in [1, -1]:
                        fig.add_shape(
                            type="line",
                            x0=start_sec, x1=end_sec,
                            y0=ch_offset + sign * threshold_val,
                            y1=ch_offset + sign * threshold_val,
                            line=dict(color="#d29922", width=1, dash="dash"),
                            layer="above",
                        )

    # Activity traces
    if has_paired_act:
        act_start = int(start_sec * act_rec.fs)
        act_end = min(int(end_sec * act_rec.fs), act_rec.n_samples)
        for ch_idx in channels:
            for p in pairings:
                if p.eeg_index == ch_idx and p.activity_index is not None:
                    act_data = act_rec.data[p.activity_index, act_start:act_end]
                    act_time = np.linspace(start_sec, end_sec, len(act_data))
                    fig.add_trace(
                        go.Scattergl(
                            x=act_time, y=act_data,
                            mode="lines", name=f"Act: {p.activity_label}",
                            line=dict(width=1),
                        ),
                        row=2, col=1,
                    )

    # Layout
    y_ticks = [channel_offsets[ch] for ch in channels]
    y_labels = [rec.channel_names[ch] for ch in channels]
    y_upper = spacing / 2
    y_lower = -(n_ch - 1) * spacing - spacing * 1.5

    # Per-axis uirevision:
    #   Y-axis: resets only when y_range or n_channels changes
    #   X-axis: always stable (mouse zoom / pan preserved)
    y_ui = f"y_{y_range}_{n_ch}"

    if has_paired_act:
        fig.update_xaxes(title_text="Time (s)", fixedrange=False,
                         uirevision="x_stable", row=2, col=1)
        fig.update_xaxes(fixedrange=False, uirevision="x_stable",
                         row=1, col=1)
        fig.update_yaxes(
            tickvals=y_ticks, ticktext=y_labels,
            zeroline=False, showgrid=False,
            fixedrange=False,
            range=[y_lower, y_upper],
            uirevision=y_ui,
            row=1, col=1,
        )
        act_unit = act_rec.units[0] if act_rec.units else ""
        fig.update_yaxes(
            title_text=f"Activity ({act_unit})" if act_unit else "Activity",
            zeroline=False, fixedrange=False,
            range=[act_ymin, act_ymax],
            uirevision=f"y_act_{act_ymin}_{act_ymax}",
            row=2, col=1,
        )
    else:
        fig.update_layout(
            xaxis=dict(title="Time (s)", fixedrange=False,
                       uirevision="x_stable"),
            yaxis=dict(
                tickvals=y_ticks, ticktext=y_labels,
                zeroline=False, showgrid=False,
                fixedrange=False,
                range=[y_lower, y_upper],
                uirevision=y_ui,
            ),
        )

    fig.update_layout(
        height=plot_height,
        uirevision="viewer_stable",
        showlegend=False,
        dragmode="zoom",
    )

    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=80, r=60, t=10, b=40))

    # Scale bar: vertical bar on the right side showing Y range
    unit_label = rec.units[0] if rec.units else ""
    scale_label = f"{y_range:.4g} {unit_label}".strip()
    bar_y_center = channel_offsets[channels[0]]
    bar_y0 = bar_y_center - spacing / 2
    bar_y1 = bar_y_center + spacing / 2
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=1.02, x1=1.02,
        y0=bar_y0, y1=bar_y1,
        line=dict(color="#8b949e", width=2),
    )
    fig.add_annotation(
        xref="paper", yref="y",
        x=1.04, y=(bar_y0 + bar_y1) / 2,
        text=scale_label,
        showarrow=False,
        textangle=-90,
        font=dict(size=11, color="#8b949e"),
    )

    # Save current settings to server state so they persist across tab switches
    state.extra["viewer_settings"] = {
        "window": window_sec,
        "start": start_sec,
        "yrange": y_range,
        "height": plot_height,
        "bp_on": bp_on,
        "notch_on": notch_on,
        "bp_low": bp_low,
        "bp_high": bp_high,
        "notch_freq": notch_freq,
        "show_events": show_events,
        "show_is": show_is,
        "show_spikes": show_spikes,
        "show_baseline": show_baseline,
        "show_threshold": show_threshold,
        "act_ymin": act_ymin,
        "act_ymax": act_ymax,
        "channels": channels,
    }

    return fig
