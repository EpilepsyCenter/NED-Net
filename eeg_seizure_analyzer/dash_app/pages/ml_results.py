"""ML Results — inspect ML detections, compare with spike-train, per-animal stats, export."""

from __future__ import annotations

import json as _json
import os
from pathlib import Path as _Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    alert,
    metric_card,
    no_recording_placeholder,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


# ── Helpers ──────────────────────────────────────────────────────────


def _load_ml_events(state) -> list[dict]:
    """Load ML detection events from state or disk."""
    events = state.extra.get("ml_detected_events", [])
    if events:
        return events
    rec = state.recording
    if rec and rec.source_path:
        stem = _Path(rec.source_path).stem
        ml_path = _Path(rec.source_path).parent / f"{stem}_ned_ml_detections.json"
        if ml_path.exists():
            try:
                with open(ml_path) as f:
                    data = _json.load(f)
                events = data.get("events", [])
                state.extra["ml_detected_events"] = events
                return events
            except Exception:
                pass
    return []


def _load_st_events(state) -> list[dict]:
    """Get spike-train detected events as dicts for comparison."""
    st_events = state.seizure_events or []
    return [
        {
            "onset_sec": e.onset_sec,
            "offset_sec": e.offset_sec,
            "channel": e.channel,
            "confidence": e.confidence,
            "duration_sec": e.duration_sec,
        }
        for e in st_events
    ]


def _overlap_fraction(a_on, a_off, b_on, b_off):
    """Return overlap fraction relative to event A."""
    overlap = max(0, min(a_off, b_off) - max(a_on, b_on))
    dur_a = a_off - a_on
    return overlap / dur_a if dur_a > 0 else 0.0


def _match_events(ml_events, st_events, overlap_thresh=0.2):
    """Match ML events to spike-train events by overlap."""
    matched_ml = set()
    matched_st = set()
    for i, ml in enumerate(ml_events):
        for j, st in enumerate(st_events):
            if ml["channel"] != st["channel"]:
                continue
            frac = _overlap_fraction(
                ml["onset_sec"], ml["offset_sec"],
                st["onset_sec"], st["offset_sec"],
            )
            if frac >= overlap_thresh:
                matched_ml.add(i)
                matched_st.add(j)
    unique_ml = set(range(len(ml_events))) - matched_ml
    unique_st = set(range(len(st_events))) - matched_st
    return matched_ml, matched_st, unique_ml, unique_st


def _is_matched(ev_idx, ml_events, st_events):
    """Check if ML event at ev_idx overlaps a spike-train event."""
    ml = ml_events[ev_idx]
    for st in st_events:
        if ml["channel"] != st["channel"]:
            continue
        frac = _overlap_fraction(
            ml["onset_sec"], ml["offset_sec"],
            st["onset_sec"], st["offset_sec"],
        )
        if frac >= 0.2:
            return True
    return False


def _build_comparison_summary(ml_events, st_events):
    """Build comparison badges between ML and spike-train detections."""
    if not st_events:
        return html.Div(
            "No spike-train detections to compare against.",
            style={"color": "#484f58", "fontSize": "0.85rem",
                   "marginBottom": "12px"},
        )
    matched_ml, matched_st, unique_ml, unique_st = _match_events(
        ml_events, st_events)
    return html.Div(
        style={"marginBottom": "16px"},
        children=[
            html.Div("Comparison with spike-train detections",
                     style={"fontSize": "0.82rem", "fontWeight": "600",
                            "color": "#8b949e", "marginBottom": "6px"}),
            dbc.Badge(f"Matched: {len(matched_ml)}", color="success",
                      className="me-2", style={"fontSize": "0.82rem"}),
            dbc.Badge(f"ML-only: {len(unique_ml)}", color="info",
                      className="me-2", style={"fontSize": "0.82rem"}),
            dbc.Badge(f"Spike-train-only: {len(unique_st)}", color="warning",
                      className="me-2", style={"fontSize": "0.82rem"}),
        ],
    )


# ── Filter helper ────────────────────────────────────────────────────


def _apply_filters(events, *, ch_filter="all",
                   min_dur=0, max_dur=None,
                   min_conf=0, max_conf=None,
                   min_freq=0, max_freq=None):
    """Apply filters to ML event list."""
    out = []
    for ev in events:
        if ch_filter != "all" and ev["channel"] != int(ch_filter):
            continue
        dur = ev.get("duration_sec", ev["offset_sec"] - ev["onset_sec"])
        if dur < float(min_dur or 0):
            continue
        if max_dur is not None and max_dur != "" and dur > float(max_dur):
            continue
        conf = ev.get("confidence", 0)
        if conf < float(min_conf or 0):
            continue
        if max_conf is not None and max_conf != "" and conf > float(max_conf):
            continue
        qm = ev.get("quality_metrics", {})
        dom_freq = qm.get("dominant_freq_hz", 0)
        if dom_freq < float(min_freq or 0):
            continue
        if max_freq is not None and max_freq != "" and dom_freq > float(max_freq):
            continue
        out.append(ev)
    return out


# ── Statistics ───────────────────────────────────────────────────────


def _build_statistics(ml_events, rec, state):
    """Build per-animal statistics table."""
    if not ml_events:
        return html.Div()

    ch_ids = {}
    for ev in ml_events:
        ch = ev["channel"]
        if ch not in ch_ids:
            animal = ev.get("animal_id", "")
            if not animal and rec:
                ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
                animal = ch_name
            ch_ids[ch] = animal

    stats = {}
    for ev in ml_events:
        animal = ch_ids.get(ev["channel"], f"Ch {ev['channel']}")
        if animal not in stats:
            stats[animal] = {
                "count": 0, "total_duration": 0.0,
                "durations": [], "confidences": [], "channel": ev["channel"],
            }
        dur = ev.get("duration_sec", ev["offset_sec"] - ev["onset_sec"])
        stats[animal]["count"] += 1
        stats[animal]["total_duration"] += dur
        stats[animal]["durations"].append(dur)
        stats[animal]["confidences"].append(ev.get("confidence", 0))

    rec_hours = (rec.duration_sec / 3600) if rec and rec.duration_sec > 0 else 1

    rows = []
    for animal, s in sorted(stats.items()):
        mean_dur = np.mean(s["durations"]) if s["durations"] else 0
        median_dur = np.median(s["durations"]) if s["durations"] else 0
        mean_conf = np.mean(s["confidences"]) if s["confidences"] else 0
        freq = s["count"] / rec_hours
        rows.append(html.Tr([
            html.Td(animal, style={"fontWeight": "600"}),
            html.Td(str(s["count"]), style={"textAlign": "center"}),
            html.Td(f"{s['total_duration']:.1f}", style={"textAlign": "center"}),
            html.Td(f"{mean_dur:.1f}", style={"textAlign": "center"}),
            html.Td(f"{median_dur:.1f}", style={"textAlign": "center"}),
            html.Td(f"{freq:.2f}", style={"textAlign": "center"}),
            html.Td(f"{mean_conf:.3f}", style={"textAlign": "center"}),
        ]))

    return html.Div([
        html.Hr(style={"borderColor": "#30363d", "margin": "24px 0"}),
        html.H5("Statistics per Animal",
                style={"color": "#58a6ff", "marginBottom": "12px"}),
        dbc.Table(
            children=[
                html.Thead(html.Tr([
                    html.Th("Animal"),
                    html.Th("Events", style={"textAlign": "center"}),
                    html.Th("Total (s)", style={"textAlign": "center"}),
                    html.Th("Mean dur (s)", style={"textAlign": "center"}),
                    html.Th("Median dur (s)", style={"textAlign": "center"}),
                    html.Th("Events/hour", style={"textAlign": "center"}),
                    html.Th("Mean conf", style={"textAlign": "center"}),
                ])),
                html.Tbody(rows),
            ],
            bordered=True, hover=True, striped=True,
            color="dark", size="sm",
            style={"fontSize": "0.85rem"},
        ),
    ])


# ── Export ────────────────────────────────────────────────────────────


def _export_csv(ml_events, rec, edf_path):
    """Export ML detections to CSV and return the path."""
    import csv
    stem = _Path(edf_path).stem
    out_path = _Path(edf_path).parent / f"{stem}_ml_detections.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "animal_id", "channel", "channel_name",
            "onset_sec", "offset_sec", "duration_sec", "confidence",
            "dominant_freq_hz", "spectral_entropy", "signal_to_baseline",
            "local_baseline_ratio",
        ])
        for ev in ml_events:
            ch = ev["channel"]
            ch_name = rec.channel_names[ch] if rec and ch < len(rec.channel_names) else f"Ch {ch}"
            dur = ev.get("duration_sec", ev["offset_sec"] - ev["onset_sec"])
            qm = ev.get("quality_metrics", {})
            writer.writerow([
                os.path.basename(edf_path),
                ev.get("animal_id", ""),
                ch, ch_name,
                f"{ev['onset_sec']:.3f}",
                f"{ev['offset_sec']:.3f}",
                f"{dur:.3f}",
                f"{ev.get('confidence', 0):.4f}",
                f"{qm.get('dominant_freq_hz', 0):.1f}",
                f"{qm.get('spectral_entropy', 0):.2f}",
                f"{qm.get('signal_to_baseline_ratio', 0):.2f}",
                f"{qm.get('local_baseline_ratio', 0):.2f}",
            ])
    return str(out_path)


# ── Inspector figure builders ────────────────────────────────────────


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


def _build_eeg_figure(rec, ev, state, bp_low=1.0, bp_high=50.0):
    """Build EEG trace figure for an ML detection event."""
    context_sec = 10.0
    ch = ev["channel"]
    onset, offset = ev["onset_sec"], ev["offset_sec"]

    win_start = max(0, onset - context_sec)
    win_end = min(rec.duration_sec, offset + context_sec)
    start_idx = int(win_start * rec.fs)
    end_idx = min(int(win_end * rec.fs), rec.n_samples)
    data = rec.data[ch, start_idx:end_idx].astype(np.float64)
    data = bandpass_filter(data, rec.fs, bp_low, bp_high)
    time_axis = np.linspace(win_start, win_end, len(data))
    ds_time, ds_data = _minmax_downsample(time_axis, data)

    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
    unit_label = rec.units[ch] if ch < len(rec.units) else ""

    # Activity channel
    act_rec = state.activity_recordings.get("paired")
    pairings = state.channel_pairings or []
    has_act = False
    act_pairing = None
    if act_rec and pairings:
        for p in pairings:
            if p.eeg_index == ch and p.activity_index is not None:
                has_act = True
                act_pairing = p
                break

    if has_act:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)
    else:
        fig = go.Figure()

    trace = go.Scattergl(
        x=ds_time, y=ds_data, mode="lines", name=ch_name,
        line=dict(width=0.8, color="#58a6ff"),
    )
    if has_act:
        fig.add_trace(trace, row=1, col=1)
    else:
        fig.add_trace(trace)

    # Seizure region
    fig.add_shape(
        type="rect", x0=onset, x1=offset, y0=0, y1=1, yref="paper",
        fillcolor="rgba(88,166,255,0.15)",
        line=dict(color="#58a6ff", width=1.5), layer="below",
    )

    # Activity trace
    if has_act and act_rec and act_pairing:
        act_start = int(win_start * act_rec.fs)
        act_end = min(int(win_end * act_rec.fs), act_rec.n_samples)
        act_data = act_rec.data[act_pairing.activity_index, act_start:act_end]
        act_time = np.linspace(win_start, win_end, len(act_data))
        fig.add_trace(
            go.Scattergl(
                x=act_time, y=act_data, mode="lines",
                name=f"Act: {act_pairing.activity_label}",
                line=dict(width=1, color="#d29922"),
            ), row=2, col=1,
        )
        fig.add_shape(
            type="rect", x0=onset, x1=offset, y0=0, y1=1, yref="y2 domain",
            fillcolor="rgba(88,166,255,0.1)",
            line=dict(color="#58a6ff", width=0.5), layer="below",
        )

    height = 400 if has_act else 300
    y_ptp = float(np.ptp(data)) if len(data) > 0 else 1.0
    half_yr = y_ptp * 0.6
    y_center = float(np.mean(data)) if len(data) > 0 else 0.0

    if has_act:
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(
            title_text=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
            range=[y_center - half_yr, y_center + half_yr], row=1, col=1,
        )
        fig.update_yaxes(title_text="Activity", row=2, col=1)
    else:
        fig.update_layout(
            xaxis=dict(title="Time (s)"),
            yaxis=dict(
                title=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
                range=[y_center - half_yr, y_center + half_yr],
            ),
        )

    fig.update_layout(
        height=height, showlegend=False, dragmode="zoom",
        uirevision=f"ml_insp_{onset}_{ch}",
    )
    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=30, b=40))
    return fig


def _build_spectral_plots(rec, ev, bp_low=1.0, bp_high=50.0):
    """Build spectrogram + band-power figures for ML event."""
    from scipy.signal import spectrogram as scipy_spectrogram, welch

    context_sec = 10.0
    ch = ev["channel"]
    onset, offset = ev["onset_sec"], ev["offset_sec"]
    win_start = max(0, onset - context_sec)
    win_end = min(rec.duration_sec, offset + context_sec)
    start_idx = int(win_start * rec.fs)
    end_idx = min(int(win_end * rec.fs), rec.n_samples)
    data = rec.data[ch, start_idx:end_idx].astype(np.float64)
    data = bandpass_filter(data, rec.fs, bp_low, bp_high)
    unit_label = rec.units[ch] if ch < len(rec.units) else ""

    # Spectrogram
    nperseg = min(int(1.0 * rec.fs), len(data) // 4)
    nperseg = max(nperseg, 64)
    noverlap = int(nperseg * 0.9)
    f_spec, t_spec, Sxx = scipy_spectrogram(
        data, fs=rec.fs, nperseg=nperseg, noverlap=noverlap)
    t_spec = t_spec + win_start
    freq_mask = f_spec <= 100
    f_spec, Sxx = f_spec[freq_mask], Sxx[freq_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig_spec = go.Figure(go.Heatmap(
        x=t_spec, y=f_spec, z=Sxx_db, colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.1f}Hz<br>Power: %{z:.1f}dB<extra></extra>",
    ))
    fig_spec.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.update_layout(
        height=250, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        showlegend=False, uirevision=f"ml_spec_{onset}_{ch}",
    )
    apply_fig_theme(fig_spec)
    fig_spec.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    # Power over time
    bands = {
        "Delta (0.5-4)": (0.5, 4, "#1f77b4"),
        "Theta (4-8)": (4, 8, "#ff7f0e"),
        "Alpha (8-13)": (8, 13, "#2ca02c"),
        "Beta (13-30)": (13, 30, "#d62728"),
        "Gamma-low (30-50)": (30, 50, "#9467bd"),
        "Gamma-high (50-100)": (50, 100, "#8c564b"),
    }
    win_samples = int(2.0 * rec.fs)
    step_samples = int(1.0 * rec.fs)
    band_power_data = {name: [] for name in bands}
    bp_times = []
    for start_s in range(0, max(1, len(data) - win_samples), step_samples):
        segment = data[start_s:start_s + win_samples]
        bp_times.append(win_start + (start_s + win_samples / 2) / rec.fs)
        f_welch, psd = welch(segment, fs=rec.fs,
                             nperseg=min(win_samples, len(segment)))
        for name, (flo, fhi, _) in bands.items():
            mask = (f_welch >= flo) & (f_welch <= fhi)
            bp = np.trapezoid(psd[mask], f_welch[mask]) if mask.sum() > 1 else 0.0
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
    power_unit = f"{unit_label}\u00b2/Hz" if unit_label else "Power"
    fig_bp.update_layout(
        height=250, xaxis_title="Time (s)",
        yaxis_title=f"Power ({power_unit})",
        yaxis_rangemode="tozero", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                    x=0, font=dict(size=10)),
        uirevision=f"ml_bp_{onset}_{ch}",
    )
    apply_fig_theme(fig_bp)
    fig_bp.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    return fig_spec, fig_bp


# ── Events table builder ─────────────────────────────────────────────


def _event_key(ev):
    return f"{ev['onset_sec']:.4f}_{ev['channel']}"


def _build_events_table(filtered_events, rec, ml_events, st_events,
                        selected_key=None):
    """Build AgGrid table of ML events."""
    table_data = []
    selected_rows = []
    for i, ev in enumerate(filtered_events):
        ch = ev["channel"]
        ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
        dur = ev.get("duration_sec", ev["offset_sec"] - ev["onset_sec"])
        qm = ev.get("quality_metrics", {})

        # Check match against spike-train
        full_idx = next((j for j, e in enumerate(ml_events)
                         if e["onset_sec"] == ev["onset_sec"]
                         and e["channel"] == ev["channel"]), 0)
        matched = "✓" if _is_matched(full_idx, ml_events, st_events) else ""

        ek = _event_key(ev)
        row = {
            "#": i + 1,
            "_event_key": ek,
            "Channel": ch_name,
            "Animal": ev.get("animal_id", ""),
            "Onset (s)": round(ev["onset_sec"], 2),
            "Offset (s)": round(ev["offset_sec"], 2),
            "Duration (s)": round(dur, 2),
            "Confidence": round(ev.get("confidence", 0), 3),
            "Peak Freq": round(qm.get("dominant_freq_hz", 0), 1),
            "Spec Ent": round(qm.get("spectral_entropy", 0), 1),
            "Sig/BL": round(qm.get("signal_to_baseline_ratio", 0), 1),
            "Local BL": round(qm.get("local_baseline_ratio", 0), 1),
            "θ/δ": round(qm.get("theta_delta_ratio", 0), 2),
            "Conv": "✓" if (ev.get("features") or {}).get("convulsive", False) else "",
            "Conv %": round((ev.get("features") or {}).get("convulsive_probability", 0) * 100, 0),
            "ST Match": matched,
        }
        table_data.append(row)
        if selected_key and ek == selected_key:
            selected_rows = [row]

    col_defs = [
        {"field": "#", "maxWidth": 55, "minWidth": 40},
        {"field": "_event_key", "hide": True},
        {"field": "Channel", "flex": 1, "minWidth": 80},
        {"field": "Animal", "flex": 1, "minWidth": 75},
        {"field": "Onset (s)", "flex": 1, "minWidth": 70},
        {"field": "Offset (s)", "flex": 1, "minWidth": 70},
        {"field": "Duration (s)", "flex": 1, "minWidth": 75},
        {"field": "Confidence", "flex": 1, "minWidth": 80},
        {"field": "Peak Freq", "flex": 1, "minWidth": 70,
         "headerTooltip": "Dominant frequency (Hz)"},
        {"field": "Spec Ent", "flex": 1, "minWidth": 65,
         "headerTooltip": "Spectral entropy"},
        {"field": "Sig/BL", "flex": 1, "minWidth": 60,
         "headerTooltip": "Signal-to-baseline ratio"},
        {"field": "Local BL", "flex": 1, "minWidth": 65,
         "headerTooltip": "Local baseline ratio"},
        {"field": "θ/δ", "flex": 1, "minWidth": 50,
         "headerTooltip": "Theta/delta ratio"},
        {"field": "Conv", "maxWidth": 55, "minWidth": 45,
         "headerTooltip": "Predicted convulsive"},
        {"field": "Conv %", "flex": 1, "minWidth": 60,
         "headerTooltip": "Convulsive probability (%)"},
        {"field": "ST Match", "maxWidth": 75, "minWidth": 60,
         "headerTooltip": "Matched by spike-train detector"},
    ]

    return dag.AgGrid(
        id="ml-res-grid",
        rowData=table_data,
        columnDefs=col_defs,
        selectedRows=selected_rows if selected_rows else [],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        className="ag-theme-alpine-dark",
        style={"height": "300px", "width": "100%"},
        dashGridOptions={
            "animateRows": False,
            "rowSelection": {"mode": "singleRow"},
            "headerHeight": 32,
            "enableCellTextSelection": True,
        },
    )


# ── Video section ────────────────────────────────────────────────────


def _build_video_section(state, sid, ev):
    """Build video player for the current event."""
    video_path = state.extra.get("video_path")
    if not video_path or not sid:
        return html.Div()
    vname = os.path.basename(video_path)
    win_start = max(0, ev["onset_sec"] - 10)
    return html.Div(
        style={"marginTop": "12px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "12px", "marginBottom": "6px"},
                children=[
                    html.Label("Video", style={"fontSize": "0.82rem",
                                                "fontWeight": "600",
                                                "color": "#8b949e"}),
                    html.Span(vname, style={"fontSize": "0.78rem",
                                             "color": "#484f58"}),
                ],
            ),
            html.Video(
                id="ml-res-video-player",
                src=f"/video/{sid}#t={win_start:.1f}",
                controls=True,
                style={
                    "width": "100%", "maxHeight": "360px",
                    "borderRadius": "8px", "backgroundColor": "#000",
                },
            ),
        ],
    )


# ── Layout ───────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the ML Results layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording
    ml_events = _load_ml_events(state)
    st_events = _load_st_events(state)

    if not ml_events:
        return html.Div(
            style={"padding": "24px", "maxWidth": "1200px"},
            children=[
                html.H4("ML Detection Results", style={"marginBottom": "8px"}),
                alert(
                    "No ML detections found. Run ML detection on the Detection "
                    "subtab first.",
                    "info",
                ),
            ],
        )

    # Channel filter options
    ch_set = sorted(set(ev["channel"] for ev in ml_events))
    ch_options = [{"label": "All channels", "value": "all"}]
    for ch in ch_set:
        ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
        animal = ""
        for ev in ml_events:
            if ev["channel"] == ch and ev.get("animal_id"):
                animal = ev["animal_id"]
                break
        label = f"{ch_name}" + (f" ({animal})" if animal else "")
        ch_options.append({"label": label, "value": ch})

    # Pre-render first event
    current_idx = 0
    ev = ml_events[current_idx]
    selected_key = _event_key(ev)
    state.extra["ml_res_selected_key"] = selected_key

    try:
        fig_eeg = _build_eeg_figure(rec, ev, state)
    except Exception:
        fig_eeg = go.Figure()
    try:
        fig_spec, fig_bp = _build_spectral_plots(rec, ev)
    except Exception:
        fig_spec, fig_bp = go.Figure(), go.Figure()

    n_all = len(ml_events)

    return html.Div(
        style={"padding": "24px", "maxWidth": "1200px"},
        children=[
            html.H4("ML Detection Results", style={"marginBottom": "8px"}),

            # Comparison summary
            _build_comparison_summary(ml_events, st_events),

            # ── Filters ─────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("Channel",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Dropdown(
                        id="ml-res-ch-filter", options=ch_options,
                        value="all", clearable=False,
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Min duration (s)",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(id="ml-res-min-dur", type="number",
                              value=0, min=0, step=0.5, size="sm",
                              className="form-control"),
                ], width=2),
                dbc.Col([
                    html.Label("Max duration (s)",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(id="ml-res-max-dur", type="number",
                              value=None, min=0, step=0.5, size="sm",
                              className="form-control"),
                ], width=2),
                dbc.Col([
                    html.Label("Min confidence",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(id="ml-res-min-conf", type="number",
                              value=0, min=0, max=1, step=0.05, size="sm",
                              className="form-control"),
                ], width=2),
                dbc.Col([
                    html.Label("Min peak freq",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(id="ml-res-min-freq", type="number",
                              value=0, min=0, step=1, size="sm",
                              className="form-control"),
                ], width=2),
                dbc.Col([
                    html.Label("Max peak freq",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(id="ml-res-max-freq", type="number",
                              value=None, min=0, step=1, size="sm",
                              className="form-control"),
                ], width=2),
            ], className="g-2 mb-3"),

            # ── Filter summary + table ──────────────────────────────
            html.Div(id="ml-res-filter-summary",
                     style={"marginBottom": "8px"},
                     children=html.Span(
                         f"Total: {n_all} events — Shown: {n_all}",
                         style={"fontSize": "0.88rem", "color": "#c9d1d9"},
                     )),

            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between",
                       "marginBottom": "8px"},
                children=[
                    html.H6("ML Detected Events",
                            style={"margin": "0"}),
                    html.Span(
                        "Click a row to inspect",
                        style={"fontSize": "0.75rem", "color": "#8b949e"},
                    ),
                ],
            ),

            html.Div(
                id="ml-res-table-container",
                children=_build_events_table(ml_events, rec, ml_events,
                                             st_events, selected_key),
            ),

            # ── Inspector ───────────────────────────────────────────
            html.Hr(style={"borderColor": "#2d333b", "margin": "20px 0"}),
            html.H5("Event Inspector",
                     style={"marginBottom": "12px", "color": "#58a6ff"}),

            html.Div("EEG Trace",
                     style={"fontSize": "0.82rem", "fontWeight": "600",
                            "color": "#8b949e", "marginBottom": "4px"}),
            dcc.Graph(id="ml-res-eeg", figure=fig_eeg,
                      config={"scrollZoom": True, "displayModeBar": True}),

            # Video
            html.Div(id="ml-res-video",
                     children=_build_video_section(state, sid, ev)),

            # Spectral plots
            dbc.Row([
                dbc.Col([
                    html.Div("Spectrogram",
                             style={"fontSize": "0.82rem", "fontWeight": "600",
                                    "color": "#8b949e", "marginBottom": "4px",
                                    "marginTop": "16px"}),
                    dcc.Graph(id="ml-res-spec", figure=fig_spec,
                              config={"scrollZoom": True}),
                ], width=6),
                dbc.Col([
                    html.Div("Power Over Time",
                             style={"fontSize": "0.82rem", "fontWeight": "600",
                                    "color": "#8b949e", "marginBottom": "4px",
                                    "marginTop": "16px"}),
                    dcc.Graph(id="ml-res-bp", figure=fig_bp,
                              config={"scrollZoom": True}),
                ], width=6),
            ]),

            # Statistics
            html.Div(id="ml-res-statistics",
                     children=_build_statistics(ml_events, rec, state)),

            # Export
            html.Hr(style={"borderColor": "#30363d", "margin": "24px 0"}),
            html.Div(
                style={"display": "flex", "gap": "12px",
                       "alignItems": "center"},
                children=[
                    dbc.Button("📊 Export CSV", id="ml-res-export-btn",
                               className="btn-ned-primary", size="sm"),
                    html.Div(id="ml-res-export-status"),
                ],
            ),
        ],
    )


# ── Callbacks ────────────────────────────────────────────────────────


@callback(
    Output("ml-res-table-container", "children"),
    Output("ml-res-filter-summary", "children"),
    Input("ml-res-ch-filter", "value"),
    Input("ml-res-min-dur", "value"),
    Input("ml-res-max-dur", "value"),
    Input("ml-res-min-conf", "value"),
    Input("ml-res-min-freq", "value"),
    Input("ml-res-max-freq", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_table(ch_filter, min_dur, max_dur, min_conf, min_freq, max_freq,
                 sid):
    """Update events table when filters change."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update, no_update

    rec = state.recording
    ml_events = _load_ml_events(state)
    st_events = _load_st_events(state)

    if not ml_events:
        return no_update, no_update

    filtered = _apply_filters(
        ml_events, ch_filter=ch_filter or "all",
        min_dur=min_dur, max_dur=max_dur,
        min_conf=min_conf,
        min_freq=min_freq, max_freq=max_freq,
    )

    selected_key = state.extra.get("ml_res_selected_key")
    table = _build_events_table(filtered, rec, ml_events, st_events,
                                selected_key)

    n_all = len(ml_events)
    n_shown = len(filtered)
    summary = html.Div(
        style={"display": "flex", "gap": "12px"},
        children=[
            html.Span(f"Total: {n_all}", style={"fontSize": "0.88rem",
                                                  "fontWeight": "600",
                                                  "color": "#c9d1d9"}),
            html.Span("•", style={"color": "#484f58"}),
            html.Span(
                f"Shown: {n_shown}",
                style={"fontSize": "0.88rem", "fontWeight": "500",
                       "color": "#58a6ff" if n_shown < n_all else "#c9d1d9"},
            ),
        ],
    )

    return table, summary


@callback(
    Output("ml-res-eeg", "figure"),
    Output("ml-res-spec", "figure"),
    Output("ml-res-bp", "figure"),
    Output("ml-res-video", "children"),
    Input("ml-res-grid", "selectedRows"),
    State("ml-res-ch-filter", "value"),
    State("ml-res-min-dur", "value"),
    State("ml-res-max-dur", "value"),
    State("ml-res-min-conf", "value"),
    State("ml-res-min-freq", "value"),
    State("ml-res-max-freq", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def inspect_event(selected_rows, ch_filter, min_dur, max_dur, min_conf,
                  min_freq, max_freq, sid):
    """Update inspector when a row is clicked in the events table."""
    _N = 4
    if not selected_rows:
        return (no_update,) * _N

    state = server_state.get_session(sid)
    if state.recording is None:
        return (no_update,) * _N

    rec = state.recording
    ml_events = _load_ml_events(state)

    # Find the event by key
    ek = selected_rows[0].get("_event_key", "")
    ev = None
    for e in ml_events:
        if _event_key(e) == ek:
            ev = e
            break

    if ev is None:
        return (no_update,) * _N

    state.extra["ml_res_selected_key"] = ek

    try:
        fig_eeg = _build_eeg_figure(rec, ev, state)
    except Exception:
        fig_eeg = go.Figure()
    try:
        fig_spec, fig_bp = _build_spectral_plots(rec, ev)
    except Exception:
        fig_spec, fig_bp = go.Figure(), go.Figure()

    video = _build_video_section(state, sid, ev)

    return fig_eeg, fig_spec, fig_bp, video


@callback(
    Output("ml-res-export-status", "children"),
    Input("ml-res-export-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, sid):
    """Export ML detections to CSV."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    if state.recording is None:
        return alert("No recording loaded.", "warning")
    ml_events = _load_ml_events(state)
    if not ml_events:
        return alert("No ML detections to export.", "warning")
    edf_path = state.recording.source_path
    if not edf_path:
        return alert("No source path.", "warning")
    try:
        out_path = _export_csv(ml_events, state.recording, edf_path)
        return alert(f"Exported to {out_path}", "success")
    except Exception as e:
        return alert(f"Export failed: {e}", "danger")
