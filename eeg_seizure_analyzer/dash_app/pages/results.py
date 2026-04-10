"""Results tab — reads from SQLite, shows summary stats, daily burden,
circadian analysis, event table with filters, and event inspector.

All data queries go through the ``db`` module — no raw SQL here.
Clicking an event row opens an inspector with EEG trace, PSD,
spectrogram (power over time), and all measured/computed parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch, spectrogram as scipy_spectrogram

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    alert,
    get_plotly_theme,
    metric_card,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter
from eeg_seizure_analyzer import db


# ── Layout ─────────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Build Results tab with filter controls and data panels."""
    try:
        animals = db.get_all_animals()
        date_min, date_max = db.get_date_range()
        files = db.get_all_files()
    except Exception:
        animals = []
        date_min = date_max = ""
        files = []

    file_options = [
        {"label": Path(f["path"]).name, "value": str(f["id"])}
        for f in files
    ]

    return html.Div(
        style={"padding": "24px", "maxWidth": "1200px"},
        children=[
            html.H4("Results", style={"marginBottom": "8px"}),
            html.P(
                "Analysis results from all modes (single, batch, live). "
                "Use filters to scope what is shown. Click an event to inspect.",
                style={"color": "var(--ned-text-muted)", "fontSize": "0.9rem",
                       "marginBottom": "16px"},
            ),

            # ── Event category selector ───────────────────────────
            dbc.RadioItems(
                id="res-source-selector",
                options=[
                    {"label": " Seizures", "value": "seizure_cnn"},
                    {"label": " Interictal Spikes", "value": "spike_cnn"},
                ],
                value="seizure_cnn",
                inline=True,
                className="mb-3",
                style={"fontSize": "0.95rem", "fontWeight": "600"},
            ),

            # ── Filter controls ────────────────────────────────────
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Source file",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-file-filter",
                                options=file_options,
                                multi=True,
                                placeholder="All files",
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Date range",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Row([
                                dbc.Col(dbc.Input(
                                    id="res-date-start", type="text",
                                    placeholder="Start",
                                    value=date_min or "", size="sm",
                                    style={"backgroundColor": "var(--ned-bg)",
                                           "color": "var(--ned-text)",
                                           "border": "1px solid var(--ned-border)"},
                                ), width=6),
                                dbc.Col(dbc.Input(
                                    id="res-date-end", type="text",
                                    placeholder="End",
                                    value=date_max or "", size="sm",
                                    style={"backgroundColor": "var(--ned-bg)",
                                           "color": "var(--ned-text)",
                                           "border": "1px solid var(--ned-border)"},
                                ), width=6),
                            ], className="g-1"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Mode",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Checklist(
                                id="res-mode-filter",
                                options=[
                                    {"label": "Single", "value": "single"},
                                    {"label": "Batch", "value": "batch"},
                                    {"label": "Live", "value": "live"},
                                ],
                                value=["single", "batch", "live"],
                                inline=True,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Animals",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-animal-filter",
                                options=[{"label": a, "value": a}
                                         for a in animals],
                                multi=True,
                                placeholder="All",
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Event type",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Checklist(
                                id="res-type-filter",
                                options=[
                                    {"label": "Conv", "value": "convulsive"},
                                    {"label": "Non-conv", "value": "non_convulsive"},
                                ],
                                value=["convulsive", "non_convulsive"],
                                inline=True,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Min conf",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Input(
                                id="res-min-conf", type="number",
                                value=0, min=0, max=1, step=0.05, size="sm",
                                style={"backgroundColor": "var(--ned-bg)",
                                       "color": "var(--ned-text)",
                                       "border": "1px solid var(--ned-border)"},
                            ),
                        ], width=1),
                    ], className="g-2"),
                    dbc.Button(
                        "Apply filters", id="res-apply",
                        size="sm", outline=True, color="info",
                        className="mt-2",
                    ),
                ]),
                style={"backgroundColor": "var(--ned-sidebar)",
                       "border": "1px solid #21262d",
                       "marginBottom": "20px"},
            ),

            # ── Summary cards ──────────────────────────────────────
            html.Div(id="res-summary"),

            # ── Plots ──────────────────────────────────────────────
            dbc.Row([
                dbc.Col(dcc.Graph(id="res-daily-burden"), width=6),
                dbc.Col(dcc.Graph(id="res-circadian"), width=6),
            ], className="mb-3"),

            # ── Events table ───────────────────────────────────────
            html.H6("Events", style={"color": "var(--ned-accent)", "marginTop": "16px"}),
            html.Div(id="res-events-table"),

            # ── Event inspector ────────────────────────────────────
            html.Div(id="res-inspector", style={"marginTop": "16px"}),

            # Hidden store for selected event data
            dcc.Store(id="res-selected-event"),

            # ── Export ─────────────────────────────────────────────
            dbc.Button("Export CSV", id="res-export-csv",
                       outline=True, color="secondary", size="sm",
                       className="mt-3"),
            dcc.Download(id="res-download"),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════


# ── Main filter callback ───────────────────────────────────────────────


@callback(
    Output("res-summary", "children"),
    Output("res-daily-burden", "figure"),
    Output("res-circadian", "figure"),
    Output("res-events-table", "children"),
    Input("res-apply", "n_clicks"),
    Input("res-source-selector", "value"),
    State("res-date-start", "value"),
    State("res-date-end", "value"),
    State("res-mode-filter", "value"),
    State("res-animal-filter", "value"),
    State("res-type-filter", "value"),
    State("res-min-conf", "value"),
    State("res-file-filter", "value"),
)
def update_results(n, source, date_start, date_end, modes, animals, types,
                   min_conf, file_ids):
    """Re-query SQLite and update all panels."""
    animal_id = animals[0] if animals and len(animals) == 1 else None
    event_type = types[0] if types and len(types) == 1 else None
    min_confidence = float(min_conf) if min_conf and float(min_conf) > 0 else None

    filter_kw = {
        "date_start": date_start or None,
        "date_end": date_end or None,
        "animal_id": animal_id,
        "min_confidence": min_confidence,
        "event_type": event_type,
        "source": source or None,
    }
    if modes and len(modes) < 3:
        filter_kw["mode"] = modes[0] if len(modes) == 1 else None

    try:
        summary = db.get_summary(**filter_kw)
        events = db.get_events(**filter_kw)
        daily = db.get_daily_burden(
            animal_id=animal_id, min_confidence=min_confidence,
            source=source or None)
        circadian = db.get_circadian(
            animal_id=animal_id, min_confidence=min_confidence,
            source=source or None)
    except Exception as e:
        empty_fig = go.Figure()
        apply_fig_theme(empty_fig)
        return (alert(f"Database error: {e}", "danger"),
                empty_fig, empty_fig, html.Div())

    # Post-filter by file IDs
    if file_ids:
        chunk_ids = {int(fid) for fid in file_ids}
        events = [e for e in events if e.get("chunk_id") in chunk_ids]

    # Post-filter by multiple animals
    if animals and len(animals) > 1:
        events = [e for e in events if e.get("animal_id") in animals]
    if types and len(types) < 2:
        events = [e for e in events if e.get("type") in types]

    # Summary cards — adapt layout based on source type
    n_total = summary["total_events"]

    if source == "spike_cnn":
        summary_cards = dbc.Row([
            dbc.Col(metric_card("Files", str(summary["n_files"])), width=2),
            dbc.Col(metric_card("Animals", str(summary["n_animals"])), width=2),
            dbc.Col(metric_card("Total spikes", str(n_total), accent=True), width=2),
            dbc.Col(metric_card("Flagged", str(summary["n_flagged"])), width=2),
        ], className="g-2 mb-3")
    else:
        n_conv = summary["n_convulsive"]
        n_nonconv = summary["n_nonconvulsive"]
        pct_c = f"({round(100*n_conv/n_total)}%)" if n_total else ""
        pct_nc = f"({round(100*n_nonconv/n_total)}%)" if n_total else ""
        summary_cards = dbc.Row([
            dbc.Col(metric_card("Files", str(summary["n_files"])), width=2),
            dbc.Col(metric_card("Animals", str(summary["n_animals"])), width=2),
            dbc.Col(metric_card("Total events", str(n_total), accent=True), width=2),
            dbc.Col(metric_card("Convulsive", f"{n_conv} {pct_c}"), width=2),
            dbc.Col(metric_card("Non-conv", f"{n_nonconv} {pct_nc}"), width=2),
            dbc.Col(metric_card("Flagged", str(summary["n_flagged"])), width=2),
        ], className="g-2 mb-3")

    daily_fig = _build_daily_burden(daily)
    circ_fig = _build_circadian(circadian)
    table = _build_events_table(events)

    return summary_cards, daily_fig, circ_fig, table


# ── Event selection from AG Grid ───────────────────────────────────────


@callback(
    Output("res-selected-event", "data"),
    Input("res-grid", "selectedRows"),
    prevent_initial_call=True,
)
def select_event(selected):
    if not selected:
        return no_update
    row = selected[0]
    return {
        "path": row.get("_path", ""),
        "start_sec": row.get("Start (s)", 0),
        "end_sec": row.get("End (s)", 0),
        "duration": row.get("Duration", 0),
        "animal": row.get("Animal", ""),
        "type": row.get("Type", ""),
        "subtype": row.get("Subtype", ""),
        "confidence": row.get("Confidence", 0),
        "conv_pct": row.get("Conv %", 0),
        "flagged": row.get("Flagged", ""),
        "hour": row.get("Hour", ""),
        "mode": row.get("Mode", ""),
        "date": row.get("Date", ""),
        "file": row.get("File", ""),
        "_channel_idx": row.get("_channel_idx", 0),
    }


# ── Event inspector ────────────────────────────────────────────────────


@callback(
    Output("res-inspector", "children"),
    Input("res-selected-event", "data"),
    prevent_initial_call=True,
)
def show_inspector(ev_data):
    if not ev_data:
        return html.Div()

    edf_path = ev_data.get("path", "")
    if not edf_path or not os.path.isfile(edf_path):
        return _inspector_params_only(ev_data)

    try:
        return _build_full_inspector(edf_path, ev_data)
    except Exception as e:
        return html.Div([
            _inspector_params_panel(ev_data),
            alert(f"Could not load EEG data: {e}", "warning"),
        ])


def _inspector_params_only(ev_data: dict):
    """Show just the parameters table when EDF file is not available."""
    return html.Div([
        html.H6("Event Inspector",
                style={"color": "var(--ned-accent)", "marginBottom": "12px"}),
        alert("EDF file not found — showing parameters only.", "info"),
        _inspector_params_panel(ev_data),
    ])


def _build_full_inspector(edf_path: str, ev_data: dict):
    """Build inspector with EEG trace, PSD, spectrogram, and params."""
    from eeg_seizure_analyzer.io.edf_reader import read_edf_window, scan_edf_channels, auto_pair_channels

    onset = float(ev_data["start_sec"])
    offset = float(ev_data["end_sec"])
    context_sec = 10.0
    bp_low, bp_high = 1.0, 50.0

    # Determine channel index
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, _, _ = auto_pair_channels(ch_info)
    channel_idx = int(ev_data.get("_channel_idx", 0))
    if channel_idx not in eeg_idx and eeg_idx:
        channel_idx = eeg_idx[0]

    # Read window around event
    win_start = max(0, onset - context_sec)
    win_end = offset + context_sec
    rec = read_edf_window(edf_path, channels=[channel_idx],
                          start_sec=win_start, duration_sec=win_end - win_start)

    data = rec.data[0].astype(np.float64)
    fs = rec.fs
    data_filt = bandpass_filter(data, fs, bp_low, bp_high)
    time_axis = np.linspace(win_start, win_start + len(data) / fs, len(data))

    # ── EEG trace ──────────────────────────────────────────────────
    ds_time, ds_data = _minmax_downsample(time_axis, data_filt)

    _eeg_color = "#1b2a4a" if get_plotly_theme() == "light" else "#58a6ff"
    fig_eeg = go.Figure()
    fig_eeg.add_trace(go.Scattergl(
        x=ds_time, y=ds_data, mode="lines",
        line=dict(width=0.8, color=_eeg_color),
        name="EEG",
    ))
    fig_eeg.add_shape(
        type="rect", x0=onset, x1=offset, y0=0, y1=1, yref="paper",
        fillcolor="rgba(88,166,255,0.15)",
        line=dict(color="#58a6ff", width=1.5), layer="below",
    )
    fig_eeg.update_layout(
        height=280, xaxis_title="Time (s)", yaxis_title="Amplitude",
        showlegend=False, dragmode="zoom",
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_eeg)

    # ── PSD of event window ────────────────────────────────────────
    event_start_idx = max(0, int((onset - win_start) * fs))
    event_end_idx = min(len(data_filt), int((offset - win_start) * fs))
    event_data = data_filt[event_start_idx:event_end_idx]

    nperseg_psd = min(int(2 * fs), len(event_data))
    nperseg_psd = max(nperseg_psd, 64)
    freqs_psd, psd_vals = welch(event_data, fs=fs, nperseg=nperseg_psd)
    psd_mask = freqs_psd <= 100
    freqs_psd, psd_vals = freqs_psd[psd_mask], psd_vals[psd_mask]

    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(
        x=freqs_psd, y=10 * np.log10(psd_vals + 1e-12),
        mode="lines", line=dict(color="#58a6ff"),
        name="PSD",
    ))
    fig_psd.update_layout(
        height=250, xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)",
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_psd)

    # ── Spectrogram (power over time) ──────────────────────────────
    nperseg_spec = min(int(1.0 * fs), len(data_filt) // 4)
    nperseg_spec = max(nperseg_spec, 64)
    noverlap = int(nperseg_spec * 0.9)
    f_spec, t_spec, Sxx = scipy_spectrogram(
        data_filt, fs=fs, nperseg=nperseg_spec, noverlap=noverlap)
    t_spec = t_spec + win_start
    freq_mask = f_spec <= 100
    f_spec, Sxx = f_spec[freq_mask], Sxx[freq_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig_spec = go.Figure(go.Heatmap(
        x=t_spec, y=f_spec, z=Sxx_db, colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8),
    ))
    fig_spec.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.update_layout(
        height=250, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_spec)

    # ── Band power over time ───────────────────────────────────────
    bands = {
        "Delta (0.5-4)": (0.5, 4, "#1f77b4"),
        "Theta (4-8)": (4, 8, "#ff7f0e"),
        "Alpha (8-13)": (8, 13, "#2ca02c"),
        "Beta (13-30)": (13, 30, "#d62728"),
        "Gamma (30-50)": (30, 50, "#9467bd"),
    }
    win_samples = int(2.0 * fs)
    step_samples = int(1.0 * fs)
    band_power_data = {name: [] for name in bands}
    bp_times = []
    for start_s in range(0, max(1, len(data_filt) - win_samples), step_samples):
        seg = data_filt[start_s:start_s + win_samples]
        bp_times.append(win_start + (start_s + win_samples / 2) / fs)
        f_w, psd_w = welch(seg, fs=fs, nperseg=min(win_samples, len(seg)))
        for name, (flo, fhi, _) in bands.items():
            mask = (f_w >= flo) & (f_w <= fhi)
            bp = np.trapezoid(psd_w[mask], f_w[mask]) if mask.sum() > 1 else 0.0
            band_power_data[name].append(bp)

    fig_bp = go.Figure()
    for name, (_, _, color) in bands.items():
        fig_bp.add_trace(go.Scatter(
            x=bp_times, y=band_power_data[name],
            name=name, mode="lines", line=dict(color=color),
            stackgroup="bands",
        ))
    fig_bp.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_bp.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_bp.update_layout(
        height=250, xaxis_title="Time (s)", yaxis_title="Power",
        yaxis_rangemode="tozero", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_bp)

    # ── Computed features from event ───────────────────────────────
    # Compute spectral features on the event window
    computed = {}
    if len(event_data) > 0 and len(psd_vals) > 0:
        total_power = np.sum(psd_vals)
        if total_power > 0:
            dominant_freq = freqs_psd[np.argmax(psd_vals)]
            computed["Dominant freq (Hz)"] = f"{dominant_freq:.1f}"

            # Band powers
            for bname, (flo, fhi) in [("Delta", (0.5, 4)), ("Theta", (4, 8)),
                                       ("Alpha", (8, 13)), ("Beta", (13, 30)),
                                       ("Gamma", (30, 50))]:
                mask = (freqs_psd >= flo) & (freqs_psd <= fhi)
                rel = np.sum(psd_vals[mask]) / total_power * 100
                computed[f"{bname} power (%)"] = f"{rel:.1f}"

            # Spectral entropy
            psd_norm = psd_vals / total_power
            psd_norm = psd_norm[psd_norm > 0]
            spec_entropy = -np.sum(psd_norm * np.log2(psd_norm))
            computed["Spectral entropy"] = f"{spec_entropy:.2f}"

        # RMS amplitude
        rms = np.sqrt(np.mean(event_data ** 2))
        computed["RMS amplitude"] = f"{rms:.2f}"

        # Peak-to-peak
        ptp = float(np.ptp(event_data))
        computed["Peak-to-peak"] = f"{ptp:.2f}"

    # Build layout
    return html.Div([
        html.Hr(style={"borderColor": "#58a6ff", "margin": "16px 0"}),
        html.H6("Event Inspector",
                style={"color": "var(--ned-accent)", "marginBottom": "12px"}),

        # EEG trace
        dcc.Graph(figure=fig_eeg, config={"displayModeBar": False}),

        # PSD + Spectrogram side by side
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_psd, config={"displayModeBar": False}),
                    width=6),
            dbc.Col(dcc.Graph(figure=fig_spec, config={"displayModeBar": False}),
                    width=6),
        ]),

        # Band power over time
        dcc.Graph(figure=fig_bp, config={"displayModeBar": False}),

        # Parameters panel
        _inspector_params_panel(ev_data, computed),
    ])


def _inspector_params_panel(ev_data: dict, computed: dict | None = None):
    """Build a card showing all event parameters."""
    params = {
        "File": ev_data.get("file", Path(ev_data.get("path", "")).name),
        "Animal": ev_data.get("animal", ""),
        "Date": ev_data.get("date", ""),
        "Onset (s)": ev_data.get("start_sec", ""),
        "Offset (s)": ev_data.get("end_sec", ""),
        "Duration (s)": ev_data.get("duration", ""),
        "Type": ev_data.get("type", ""),
        "Subtype": ev_data.get("subtype", ""),
        "CNN confidence": ev_data.get("confidence", ""),
        "Convulsive %": ev_data.get("conv_pct", ""),
        "Movement flagged": ev_data.get("flagged", "No"),
        "Hour of day": ev_data.get("hour", ""),
        "Analysis mode": ev_data.get("mode", ""),
    }

    # Merge computed spectral features
    if computed:
        params.update(computed)

    rows = []
    for k, v in params.items():
        if v == "" or v is None:
            continue
        rows.append(
            html.Tr([
                html.Td(k, style={"color": "var(--ned-text-muted)", "fontSize": "0.82rem",
                                   "paddingRight": "16px", "whiteSpace": "nowrap"}),
                html.Td(str(v), style={"color": "var(--ned-text)", "fontSize": "0.82rem"}),
            ])
        )

    return dbc.Card(
        dbc.CardBody([
            html.H6("Parameters", style={"color": "var(--ned-text-muted)",
                                          "fontSize": "0.82rem",
                                          "marginBottom": "8px"}),
            html.Table(
                html.Tbody(rows),
                style={"width": "100%"},
            ),
        ]),
        style={"backgroundColor": "var(--ned-sidebar)", "border": "1px solid #21262d",
               "marginTop": "12px"},
    )


# ═══════════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════════


def _build_daily_burden(daily: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not daily:
        apply_fig_theme(fig)
        fig.update_layout(title="Daily Seizure Burden")
        return fig

    conv_dates, conv_counts = [], []
    nonconv_dates, nonconv_counts = [], []
    for row in daily:
        if row["type"] == "convulsive":
            conv_dates.append(row["date"])
            conv_counts.append(row["n_events"])
        else:
            nonconv_dates.append(row["date"])
            nonconv_counts.append(row["n_events"])

    if conv_dates:
        fig.add_trace(go.Bar(x=conv_dates, y=conv_counts,
                             name="Convulsive", marker_color="#f85149"))
    if nonconv_dates:
        fig.add_trace(go.Bar(x=nonconv_dates, y=nonconv_counts,
                             name="Non-convulsive", marker_color="#58a6ff"))

    fig.update_layout(
        barmode="stack", title="Daily Seizure Burden",
        xaxis_title="Date", yaxis_title="Events",
        legend=dict(orientation="h", y=1.1),
    )
    apply_fig_theme(fig)
    return fig


def _build_circadian(circadian: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not circadian:
        apply_fig_theme(fig)
        fig.update_layout(title="Circadian Distribution")
        return fig

    conv_by_hour = [0] * 24
    nonconv_by_hour = [0] * 24
    for row in circadian:
        h = row["hour_of_day"]
        if h is None:
            continue
        if row["type"] == "convulsive":
            conv_by_hour[h] += row["n_events"]
        else:
            nonconv_by_hour[h] += row["n_events"]

    hours = [f"{h:02d}:00" for h in range(24)]
    fig.add_trace(go.Bar(x=hours, y=conv_by_hour,
                         name="Convulsive", marker_color="#f85149"))
    fig.add_trace(go.Bar(x=hours, y=nonconv_by_hour,
                         name="Non-convulsive", marker_color="#58a6ff"))

    fig.update_layout(
        barmode="stack", title="Circadian Distribution",
        xaxis_title="Hour of day", yaxis_title="Events",
        legend=dict(orientation="h", y=1.1),
    )
    apply_fig_theme(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Events table
# ═══════════════════════════════════════════════════════════════════════


def _build_events_table(events: list[dict]):
    if not events:
        return html.P("No events found.",
                      style={"color": "var(--ned-text-muted)", "fontSize": "0.85rem"})

    rows = []
    for ev in events[:500]:
        edf_path = ev.get("path", "")
        rows.append({
            "Animal": ev.get("animal_id", ""),
            "File": Path(edf_path).name if edf_path else "",
            "Date": ev.get("date", ev.get("chunk_date", "")),
            "Start (s)": round(ev.get("start_sec", 0), 1),
            "End (s)": round(ev.get("end_sec", 0), 1),
            "Duration": round(ev.get("duration_sec", 0), 1),
            "Type": ev.get("type", ""),
            "Subtype": ev.get("subtype", "") or "",
            "Confidence": round(ev.get("cnn_confidence", 0), 3),
            "Conv %": round((ev.get("convulsive_confidence") or 0) * 100, 0),
            "Flagged": "Yes" if ev.get("movement_flag") else "",
            "Hour": ev.get("hour_of_day", ""),
            "Mode": ev.get("mode", ""),
            # Hidden fields for inspector
            "_path": edf_path,
            "_channel_idx": ev.get("chunk_id", 0),  # will improve with per-event channel
        })

    columns = [
        {"field": "Animal", "width": 80},
        {"field": "File", "width": 150},
        {"field": "Date", "width": 95},
        {"field": "Start (s)", "width": 80},
        {"field": "End (s)", "width": 80},
        {"field": "Duration", "width": 75},
        {"field": "Type", "width": 100},
        {"field": "Subtype", "width": 75},
        {"field": "Confidence", "width": 90},
        {"field": "Conv %", "width": 65},
        {"field": "Flagged", "width": 65},
        {"field": "Hour", "width": 50},
        {"field": "Mode", "width": 65},
        # Hidden columns
        {"field": "_path", "hide": True},
        {"field": "_channel_idx", "hide": True},
    ]

    return dag.AgGrid(
        id="res-grid",
        rowData=rows,
        columnDefs=columns,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        style={"height": "400px"},
        dashGridOptions={
            "rowSelection": "single",
            "animateRows": False,
        },
        className="ag-theme-alpine-dark",
    )


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _minmax_downsample(time_axis, data, max_points=6000):
    """Downsample for display keeping min/max per bin."""
    n = len(data)
    if n <= max_points:
        return time_axis, data
    bin_size = n // (max_points // 2)
    n_bins = n // bin_size
    out_t, out_d = [], []
    for i in range(n_bins):
        s = i * bin_size
        e = min(s + bin_size, n)
        seg = data[s:e]
        idx_min = s + np.argmin(seg)
        idx_max = s + np.argmax(seg)
        if idx_min < idx_max:
            out_t.extend([time_axis[idx_min], time_axis[idx_max]])
            out_d.extend([data[idx_min], data[idx_max]])
        else:
            out_t.extend([time_axis[idx_max], time_axis[idx_min]])
            out_d.extend([data[idx_max], data[idx_min]])
    return np.array(out_t), np.array(out_d)


# ═══════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════


@callback(
    Output("res-download", "data"),
    Input("res-export-csv", "n_clicks"),
    State("res-date-start", "value"),
    State("res-date-end", "value"),
    State("res-animal-filter", "value"),
    State("res-min-conf", "value"),
    prevent_initial_call=True,
)
def export_csv(n, date_start, date_end, animals, min_conf):
    if not n:
        return no_update

    animal_id = animals[0] if animals and len(animals) == 1 else None
    min_confidence = float(min_conf) if min_conf and float(min_conf) > 0 else None

    events = db.get_events(
        date_start=date_start or None,
        date_end=date_end or None,
        animal_id=animal_id,
        min_confidence=min_confidence,
    )
    if not events:
        return no_update

    import csv
    import io

    output = io.StringIO()
    fields = [
        "animal_id", "date", "start_sec", "end_sec", "duration_sec",
        "type", "subtype", "cnn_confidence", "convulsive_confidence",
        "movement_flag", "hour_of_day", "path", "mode",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for ev in events:
        writer.writerow(ev)

    return dict(content=output.getvalue(), filename="analysis_results.csv")
