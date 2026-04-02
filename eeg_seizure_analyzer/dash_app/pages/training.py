"""Training tab: annotate detected seizures to build ML training data."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from dash import html, dcc, callback, Input, Output, State, no_update, ctx, clientside_callback
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    alert,
    empty_state,
    metric_card,
    no_recording_placeholder,
)
from eeg_seizure_analyzer.io.annotation_store import (
    AnnotatedEvent,
    save_annotations,
    load_annotations,
    detections_to_annotations,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


# ── Helpers ───────────────────────────────────────────────────────────


def _get_annotations(state) -> list[AnnotatedEvent]:
    """Retrieve annotations from state.extra, deserialising if needed."""
    raw = state.extra.get("tr_annotations", [])
    if not raw:
        return []
    out: list[AnnotatedEvent] = []
    seen: set[tuple[float, int, str]] = set()
    for item in raw:
        if isinstance(item, AnnotatedEvent):
            ann = item
        elif isinstance(item, dict):
            ann = AnnotatedEvent.from_dict(item)
        else:
            continue
        # Deduplicate by (onset_sec rounded, channel, source)
        key = (round(ann.onset_sec, 4), ann.channel, ann.source)
        if key not in seen:
            seen.add(key)
            out.append(ann)
    return out


def _set_annotations(state, annotations: list[AnnotatedEvent]):
    """Store annotations into state.extra as dicts."""
    state.extra["tr_annotations"] = [a.to_dict() for a in annotations]


def _auto_save(state, annotations: list[AnnotatedEvent]):
    """Persist annotations to disk and to state.

    Filter settings are saved in the *detection* file (not here).
    """
    _set_annotations(state, annotations)
    rec = state.recording
    if rec and rec.source_path:
        annotator = state.extra.get("tr_annotator", "")
        animal_id = state.extra.get("tr_animal_id", "")
        try:
            save_annotations(rec.source_path, annotations,
                             annotator=annotator, animal_id=animal_id)
        except Exception as e:
            import traceback
            traceback.print_exc()  # log save failures instead of silencing


def _progress_counts(annotations: list[AnnotatedEvent]) -> dict:
    """Count confirmed/rejected/pending annotations."""
    counts = {"confirmed": 0, "rejected": 0, "pending": 0, "total": 0}
    for a in annotations:
        counts["total"] += 1
        if a.label in counts:
            counts[a.label] += 1
    return counts


def _filter_by_channel(annotations: list[AnnotatedEvent],
                       channel_filter) -> list[AnnotatedEvent]:
    """Filter annotations by channel if a filter is set."""
    if channel_filter is None or channel_filter == "":
        return annotations
    try:
        ch = int(channel_filter)
        return [a for a in annotations if a.channel == ch]
    except (ValueError, TypeError):
        return annotations


def _apply_annotation_filters(annotations: list[AnnotatedEvent], *,
                              min_conf=0, min_dur=0, min_lbl=0,
                              max_conf=None, max_dur=None, max_lbl=None,
                              min_spikes=0, max_spikes=None,
                              min_amp=0, max_amp=None,
                              min_top_amp=0, max_top_amp=None,
                              min_freq=0, max_freq=None):
    """Apply confidence/duration/local-BL/spike min-max filters to annotation list."""
    filtered = list(annotations)
    min_conf = float(min_conf or 0)
    min_dur = float(min_dur or 0)
    min_lbl = float(min_lbl or 0)
    min_spikes = int(min_spikes or 0)
    min_amp = float(min_amp or 0)
    min_top_amp = float(min_top_amp or 0)
    min_freq = float(min_freq or 0)

    def _fmax(v):
        if v is None or v == "":
            return None
        return float(v)
    max_conf = _fmax(max_conf)
    max_dur = _fmax(max_dur)
    max_lbl = _fmax(max_lbl)
    max_spikes = _fmax(max_spikes)
    max_amp = _fmax(max_amp)
    max_top_amp = _fmax(max_top_amp)
    max_freq = _fmax(max_freq)

    if min_conf > 0:
        filtered = [a for a in filtered if a.detector_confidence >= min_conf]
    if max_conf is not None:
        filtered = [a for a in filtered if a.detector_confidence <= max_conf]
    if min_dur > 0:
        filtered = [a for a in filtered
                    if (a.offset_sec - a.onset_sec) >= min_dur]
    if max_dur is not None:
        filtered = [a for a in filtered
                    if (a.offset_sec - a.onset_sec) <= max_dur]
    if min_lbl > 0:
        filtered = [a for a in filtered
                    if (a.quality_metrics or {}).get("local_baseline_ratio", 0) >= min_lbl]
    if max_lbl is not None:
        filtered = [a for a in filtered
                    if (a.quality_metrics or {}).get("local_baseline_ratio", 0) <= max_lbl]
    # --- Spikes ---
    if min_spikes > 0:
        filtered = [a for a in filtered
                    if (a.features or {}).get("n_spikes", 0) is not None
                    and (a.features or {}).get("n_spikes", 0) >= min_spikes]
    if max_spikes is not None:
        filtered = [a for a in filtered
                    if (a.features or {}).get("n_spikes") is None
                    or (a.features or {}).get("n_spikes", 0) <= max_spikes]
    # --- Amp (xBL) ---
    if min_amp > 0:
        filtered = [a for a in filtered
                    if (a.features or {}).get("max_amplitude_x_baseline") is not None
                    and (a.features or {}).get("max_amplitude_x_baseline", 0) >= min_amp]
    if max_amp is not None:
        filtered = [a for a in filtered
                    if (a.features or {}).get("max_amplitude_x_baseline") is None
                    or (a.features or {}).get("max_amplitude_x_baseline", 0) <= max_amp]
    # --- Top Amp ---
    if min_top_amp > 0:
        filtered = [a for a in filtered
                    if (a.quality_metrics or {}).get("top_spike_amplitude_x", 0) >= min_top_amp]
    if max_top_amp is not None:
        filtered = [a for a in filtered
                    if (a.quality_metrics or {}).get("top_spike_amplitude_x", 0) <= max_top_amp]
    # --- Freq ---
    if min_freq > 0:
        filtered = [a for a in filtered
                    if (a.features or {}).get("mean_spike_frequency_hz") is not None
                    and (a.features or {}).get("mean_spike_frequency_hz", 0) >= min_freq]
    if max_freq is not None:
        filtered = [a for a in filtered
                    if (a.features or {}).get("mean_spike_frequency_hz") is None
                    or (a.features or {}).get("mean_spike_frequency_hz", 0) <= max_freq]
    return filtered


def _recompute_activity_zscore(state, ann):
    """Recompute activity z-score for an annotation after boundary change."""
    act_rec = state.activity_recordings.get("paired")
    pairings = state.channel_pairings or []
    if act_rec is None or not pairings:
        return

    # Find the activity channel paired to this annotation's EEG channel
    act_idx = None
    for p in pairings:
        if p.eeg_index == ann.channel and p.activity_index is not None:
            act_idx = p.activity_index
            break
    if act_idx is None:
        return

    try:
        import numpy as _np
        act_data = act_rec.data[act_idx]
        act_fs = act_rec.fs
        pad_sec = 2.0

        # Global stats for z-score
        abs_act = _np.abs(act_data)
        global_mean = float(_np.mean(abs_act))
        global_std = float(_np.std(abs_act))
        if global_std < 1e-12:
            return

        # Event window
        start = max(0, int((ann.onset_sec - pad_sec) * act_fs))
        end = min(len(act_data), int((ann.offset_sec + pad_sec) * act_fs))
        if end <= start:
            return

        event_mean = float(_np.mean(_np.abs(act_data[start:end])))
        z = (event_mean - global_mean) / global_std

        if ann.features is None:
            ann.features = {}
        ann.features["activity_zscore"] = round(z, 2)
        ann.features["mean_activity"] = round(event_mean, 4)

        # Also update the matching seizure event in state
        for ev in state.seizure_events:
            if ev.channel == ann.channel and abs(ev.onset_sec - ann.onset_sec) < 0.01:
                if ev.features is None:
                    ev.features = {}
                ev.features["activity_zscore"] = round(z, 2)
                ev.features["mean_activity"] = round(event_mean, 4)
                break
    except Exception:
        import traceback
        traceback.print_exc()


def _sync_boundary_to_seizure_events(state, channel: int,
                                     original_onset: float,
                                     new_onset: float, new_offset: float):
    """Push boundary changes from Training tab back to state.seizure_events.

    This ensures the Seizure tab table reflects modified boundaries.
    Also re-saves the detection JSON with updated events.
    """
    if not state.seizure_events:
        return
    for ev in state.seizure_events:
        if ev.channel == channel and abs(ev.onset_sec - original_onset) < 0.01:
            ev.onset_sec = new_onset
            ev.offset_sec = new_offset
            ev.duration_sec = new_offset - new_onset
            break
    # Update detected_events too
    state.detected_events = list(state.seizure_events) + state.spike_events
    # Re-save the detection file with updated boundaries
    _save_detection_file(state)


def _compute_manual_event_metrics(rec, event, state):
    """Compute quality metrics for a manually added seizure.

    Only computes signal-level quality metrics (LL/energy z-scores,
    spectral features, local baseline ratio, etc.) — does NOT estimate
    spike counts or spike frequency because the simplified threshold-
    crossing approach would be inaccurate compared to the full spike
    detection pipeline and could bias ML training.

    Spike-related features are explicitly set to None so downstream
    code knows they are unavailable.
    """
    from eeg_seizure_analyzer.detection.confidence import (
        compute_event_quality,
        compute_local_baseline_ratio,
        compute_confidence_score,
    )

    bp_low = float(state.extra.get("sz_params", {}).get("sz-bp-low", 1.0))
    bp_high = float(state.extra.get("sz_params", {}).get("sz-bp-high", 100.0))
    lbl_start = float(state.extra.get("sz_params", {}).get("sz-lbl-start", 20.0))
    lbl_end = float(state.extra.get("sz_params", {}).get("sz-lbl-end", 5.0))

    # Mark spike features as unavailable (manual source — no spike detection)
    event.features = {
        "n_spikes": None,
        "mean_spike_frequency_hz": None,
        "max_amplitude_x_baseline": None,
        "source": "manual",
    }

    # Quality metrics (signal-level — accurate without spike detection)
    qm = compute_event_quality(
        rec, event, bandpass_low=bp_low, bandpass_high=bp_high,
    )
    lbr = compute_local_baseline_ratio(
        rec, event,
        local_start_sec=lbl_start,
        local_end_sec=lbl_end,
        bandpass_low=bp_low,
        bandpass_high=bp_high,
    )
    qm["local_baseline_ratio"] = round(lbr, 2)
    qm["top_spike_amplitude_x"] = 0.0  # requires spike data
    event.quality_metrics = qm
    event.confidence = compute_confidence_score(qm)

    return event


def _backfill_event_ids(annotations: list[AnnotatedEvent],
                        seizure_events) -> None:
    """Assign event_ids to annotations that don't have one yet.

    Matches by channel + onset proximity to seizure_events. Any
    remaining unmatched annotations get IDs starting from max+1.
    """
    # First try matching to seizure_events
    for ann in annotations:
        if ann.event_id > 0:
            continue
        for ev in seizure_events:
            if ev.channel == ann.channel and abs(ev.onset_sec - ann.onset_sec) < 0.5:
                if ev.event_id > 0:
                    ann.event_id = ev.event_id
                break

    # Assign sequential IDs to any still without one
    all_ids = [a.event_id for a in annotations if a.event_id > 0]
    for ev in seizure_events:
        if ev.event_id > 0:
            all_ids.append(ev.event_id)
    next_id = max(all_ids) + 1 if all_ids else 1

    for ann in annotations:
        if ann.event_id == 0:
            ann.event_id = next_id
            next_id += 1


def _save_detection_file(state):
    """Re-save the detection JSON with current seizure_events."""
    try:
        rec = state.recording
        _src = getattr(rec, "source_path", None) or "" if rec else ""
        if _src and _src.lower().endswith(".edf") and state.seizure_events:
            from eeg_seizure_analyzer.io.persistence import save_detections
            save_detections(
                edf_path=_src,
                events=state.seizure_events,
                detection_info=state.st_detection_info,
                params_dict=state.extra.get("sz_params", {}),
                detector_name="SpikeTrainSeizureDetector",
                channels=state.extra.get("sz_selected_channels", []),
                animal_id=getattr(state, "animal_id", ""),
                filter_settings={
                    "filter_enabled": state.extra.get("sz_filter_enabled", True),
                    "filter_values": state.extra.get("sz_filter_values", {}),
                },
            )
    except Exception:
        import traceback
        traceback.print_exc()


_LABEL_COLORS = {
    "confirmed": {"fill": "rgba(63, 185, 80, 0.2)", "line": "#3fb950", "text": "#3fb950"},
    "rejected":  {"fill": "rgba(248, 81, 73, 0.15)", "line": "#f85149", "text": "#f85149"},
    "pending":   {"fill": "rgba(210, 153, 34, 0.2)", "line": "#d29922", "text": "#d29922"},
    "manual":    {"fill": "rgba(188, 140, 255, 0.25)", "line": "#bc8cff", "text": "#bc8cff"},
}


def _label_badge(label: str) -> html.Span:
    """Small colored badge for an annotation label."""
    colors = _LABEL_COLORS.get(label, _LABEL_COLORS["pending"])
    return html.Span(
        label.upper(),
        style={
            "fontSize": "0.72rem", "fontWeight": "600",
            "letterSpacing": "0.5px", "textTransform": "uppercase",
            "padding": "2px 10px", "borderRadius": "12px",
            "color": colors["text"],
            "background": colors["fill"],
            "border": f"1px solid {colors['line']}",
        },
    )


def _build_annotation_counts(counts, counts_filtered, progress_pct,
                             ch_counts, filter_on):
    """Build the annotation progress section at the bottom of the page."""
    return html.Div(
        style={"marginBottom": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "marginBottom": "4px"},
                children=[
                    html.Span(
                        f"{counts_filtered['confirmed']} confirmed, "
                        f"{counts_filtered['rejected']} rejected, "
                        f"{counts_filtered['pending']} pending"
                        + (f" (of {counts['total']} total)"
                           if filter_on else ""),
                        style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    html.Span(f"{progress_pct}%",
                              style={"fontSize": "0.78rem", "color": "#8b949e"}),
                ],
            ),
            dbc.Progress(
                value=progress_pct,
                style={"height": "6px", "backgroundColor": "#2d333b"},
                color="success" if progress_pct >= 80 else (
                    "warning" if progress_pct >= 40 else "info"
                ),
            ),
            # Per-channel counts
            html.Div(
                style={"display": "flex", "gap": "16px", "marginTop": "6px",
                       "flexWrap": "wrap"},
                children=[
                    html.Span(
                        f"{ch_name}: {cc['confirmed']}\u2713 {cc['rejected']}\u2717 {cc['pending']}?",
                        style={"fontSize": "0.72rem", "color": "#8b949e",
                               "border": "1px solid #2d333b",
                               "borderRadius": "8px", "padding": "1px 8px"},
                    )
                    for ch_name, cc in ch_counts.items()
                ],
            ),
        ],
    )


def _event_badge(event) -> html.Span:
    """Label badge + optional activity z-score indicator."""
    children = [_label_badge(event.label)]
    act_z = (event.features or {}).get("activity_zscore")
    if act_z is not None:
        # Color by severity: grey <1, yellow 1-3, red >3
        if act_z > 3:
            _c = {"bg": "#f8514922", "border": "#f85149", "text": "#f85149"}
        elif act_z > 1:
            _c = {"bg": "#d2992222", "border": "#d29922", "text": "#d29922"}
        else:
            _c = {"bg": "#8b949e22", "border": "#8b949e", "text": "#8b949e"}
        children.append(html.Span(
            f"Act {act_z:+.1f}\u03c3",
            style={
                "fontSize": "0.68rem", "fontWeight": "600",
                "padding": "2px 8px", "borderRadius": "12px",
                "marginLeft": "6px",
                "color": _c["text"],
                "background": _c["bg"],
                "border": f"1px solid {_c['border']}",
            },
        ))
    return html.Span(children, style={"display": "flex", "gap": "4px",
                                       "alignItems": "center"})


def _build_event_properties(rec, event) -> dbc.Row:
    """Build the seizure property info boxes for the current event."""
    if event is None:
        return dbc.Row(className="g-2 mb-2")
    ch = event.channel
    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch{ch}"
    feat = event.features or {}
    qm = event.quality_metrics or {}
    n_spikes = feat.get("n_spikes")
    return dbc.Row([
        dbc.Col(metric_card("Channel", ch_name), width=2),
        dbc.Col(metric_card("Duration",
                            f"{event.offset_sec - event.onset_sec:.2f}s"), width=2),
        dbc.Col(metric_card("Spikes",
                            str(n_spikes) if n_spikes is not None else "\u2014"), width=2),
        dbc.Col(metric_card("Confidence",
                            f"{event.detector_confidence:.2f}"), width=2),
        dbc.Col(metric_card("Local BL",
                            f"{qm.get('local_baseline_ratio', 0):.1f}"), width=2),
        dbc.Col(metric_card("Top Amp",
                            f"{qm.get('top_spike_amplitude_x', 0):.1f}"), width=2),
    ], className="g-2 mb-2")


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


def _training_video_player(state, sid, onset_sec):
    """Return a video player for training review, or an empty div."""
    import os
    video_path = state.extra.get("video_path")
    if not video_path or not sid:
        return html.Div(id="tr-video-container", style={"display": "none"})

    vname = os.path.basename(video_path)
    graph_id = "tr-review-graph"
    video_id = "tr-review-video"

    return html.Div(
        id="tr-video-container",
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
                id=video_id,
                src=f"/video/{sid}#t={max(0, onset_sec - 10):.1f}",
                controls=True,
                style={
                    "width": "100%",
                    "maxHeight": "360px",
                    "borderRadius": "8px",
                    "backgroundColor": "#000",
                },
            ),
        ],
    )


def _initial_spectral_row(rec, current_event, state):
    """Pre-render spectral plots for layout, or empty placeholders."""
    if current_event and rec is not None:
        try:
            fig_spec, fig_bp = _build_spectral_plots(rec, current_event, state)
        except Exception:
            fig_spec, fig_bp = go.Figure(), go.Figure()
    else:
        fig_spec, fig_bp = go.Figure(), go.Figure()

    return [
        dbc.Col([
            html.Div("Spectrogram",
                     style={"fontSize": "0.82rem", "fontWeight": "600",
                            "color": "#8b949e", "marginBottom": "4px",
                            "marginTop": "16px"}),
            dcc.Graph(id="tr-spectrogram", figure=fig_spec,
                      config={"scrollZoom": True}),
        ], width=6),
        dbc.Col([
            html.Div("Power Over Time",
                     style={"fontSize": "0.82rem", "fontWeight": "600",
                            "color": "#8b949e", "marginBottom": "4px",
                            "marginTop": "16px"}),
            dcc.Graph(id="tr-band-power", figure=fig_bp,
                      config={"scrollZoom": True}),
        ], width=6),
    ]


# ── Review Mode Plot ──────────────────────────────────────────────────


def _build_review_figure(rec, event: AnnotatedEvent, state,
                         bp_low=1.0, bp_high=50.0,
                         y_range=None, act_ymin=0.0, act_ymax=4.0,
                         show_baseline=False, show_threshold=False):
    """Build the EEG plot for review mode: +/- 10s around event."""
    context_sec = 10.0
    ch = event.channel
    onset, offset = event.onset_sec, event.offset_sec

    win_start = max(0, onset - context_sec)
    win_end = min(rec.duration_sec, offset + context_sec)

    start_idx = int(win_start * rec.fs)
    end_idx = min(int(win_end * rec.fs), rec.n_samples)
    data = rec.data[ch, start_idx:end_idx].astype(np.float64)

    # Bandpass filter
    data = bandpass_filter(data, rec.fs, bp_low, bp_high)

    time_axis = np.linspace(win_start, win_end, len(data))
    ds_time, ds_data = _minmax_downsample(time_axis, data)

    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
    unit_label = rec.units[ch] if ch < len(rec.units) else ""

    # Check for paired activity channel
    act_rec = state.activity_recordings.get("paired")
    pairings = state.channel_pairings or []
    has_act = False
    act_pairing = None
    if act_rec is not None and pairings:
        for p in pairings:
            if p.eeg_index == ch and p.activity_index is not None:
                has_act = True
                act_pairing = p
                break

    if has_act:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
    else:
        fig = go.Figure()

    trace = go.Scattergl(
        x=ds_time, y=ds_data,
        mode="lines", name=ch_name,
        line=dict(width=0.8, color="#58a6ff"),
    )
    if has_act:
        fig.add_trace(trace, row=1, col=1)
    else:
        fig.add_trace(trace)

    # Seizure region highlight — editable box so the user can drag edges
    colors = _LABEL_COLORS.get(event.label, _LABEL_COLORS["pending"])
    fig.add_shape(
        type="rect",
        x0=onset, x1=offset,
        y0=0, y1=1, yref="paper",
        fillcolor=colors["fill"],
        line=dict(color=colors["line"], width=1.5),
        layer="below",
        editable=True,
        name="highlight",
    )

    # Show detected spikes if available
    det_info = state.st_detection_info.get(ch, {})
    spike_times = det_info.get("all_spike_times", [])
    spike_samples = det_info.get("all_spike_samples", [])
    if spike_times:
        vis_t, vis_y = [], []
        for i, t in enumerate(spike_times):
            if win_start <= t <= win_end:
                local = spike_samples[i] - start_idx
                if 0 <= local < len(data):
                    vis_t.append(t)
                    vis_y.append(float(data[local]))
        if vis_t:
            sp_colors = []
            for t in vis_t:
                in_event = onset <= t <= offset
                sp_colors.append("#f85149" if in_event else "#ffb347")
            sp_trace = go.Scatter(
                x=vis_t, y=vis_y,
                mode="markers",
                marker=dict(color=sp_colors, size=5, symbol="circle"),
                showlegend=False,
                hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
            )
            if has_act:
                fig.add_trace(sp_trace, row=1, col=1)
            else:
                fig.add_trace(sp_trace)

    # Activity trace
    if has_act and act_rec is not None and act_pairing is not None:
        act_start = int(win_start * act_rec.fs)
        act_end = min(int(win_end * act_rec.fs), act_rec.n_samples)
        act_data = act_rec.data[act_pairing.activity_index, act_start:act_end]
        act_time = np.linspace(win_start, win_end, len(act_data))
        fig.add_trace(
            go.Scattergl(
                x=act_time, y=act_data,
                mode="lines", name=f"Act: {act_pairing.activity_label}",
                line=dict(width=1, color="#d29922"),
            ),
            row=2, col=1,
        )
        # Add seizure highlight on activity subplot too
        fig.add_shape(
            type="rect",
            x0=onset, x1=offset,
            y0=0, y1=1,
            yref="y2 domain",
            fillcolor=colors["fill"],
            line=dict(color=colors["line"], width=0.5),
            layer="below",
            editable=False,
            name="act_highlight",
        )

    # Baseline / threshold lines
    det_info = state.st_detection_info.get(ch, {})
    baseline_val = det_info.get("baseline_mean")
    threshold_val = det_info.get("threshold")
    _row_kw = dict(row=1, col=1) if has_act else {}
    if show_baseline and baseline_val is not None:
        fig.add_hline(
            y=baseline_val, **_row_kw,
            line=dict(color="#3fb950", width=1, dash="dot"),
            annotation_text="Baseline",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-baseline_val, **_row_kw,
            line=dict(color="#3fb950", width=1, dash="dot"),
        )
    if show_threshold and threshold_val is not None:
        fig.add_hline(
            y=threshold_val, **_row_kw,
            line=dict(color="#d29922", width=1, dash="dash"),
            annotation_text="Threshold",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-threshold_val, **_row_kw,
            line=dict(color="#d29922", width=1, dash="dash"),
        )

    # Y-axis range
    if y_range is not None and y_range > 0:
        half_yr = float(y_range) / 2.0
    else:
        y_ptp = float(np.ptp(data)) if len(data) > 0 else 1.0
        half_yr = y_ptp * 0.6
    y_center = float(np.mean(data)) if len(data) > 0 else 0.0

    total_height = 500 if has_act else 400

    if has_act:
        fig.update_xaxes(title_text="Time (s)", fixedrange=False,
                         uirevision="x_stable", row=2, col=1)
        fig.update_xaxes(fixedrange=False, uirevision="x_stable",
                         row=1, col=1)
        fig.update_yaxes(
            title_text=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
            fixedrange=False,
            range=[y_center - half_yr, y_center + half_yr],
            uirevision=f"y_review_{y_range}",
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
            xaxis=dict(title="Time (s)", fixedrange=False),
            yaxis=dict(
                title=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
                fixedrange=False,
                range=[y_center - half_yr, y_center + half_yr],
            ),
        )

    fig.update_layout(
        height=total_height,
        showlegend=False,
        dragmode="zoom",
        uirevision="review_stable",
    )

    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=10, b=40))

    return fig


def _build_spectral_plots(rec, event: AnnotatedEvent, state,
                          bp_low=1.0, bp_high=50.0):
    """Build spectrogram + band-power-over-time figures for training inspector.

    Returns (fig_spectrogram, fig_band_power).
    Mirrors the seizure inspector implementation exactly.
    """
    from scipy.signal import spectrogram as scipy_spectrogram, welch

    context_sec = 10.0
    ch = event.channel
    onset, offset = event.onset_sec, event.offset_sec

    win_start = max(0, onset - context_sec)
    win_end = min(rec.duration_sec, offset + context_sec)

    start_idx = int(win_start * rec.fs)
    end_idx = min(int(win_end * rec.fs), rec.n_samples)
    data = rec.data[ch, start_idx:end_idx].astype(np.float64)
    data = bandpass_filter(data, rec.fs, bp_low, bp_high)

    unit_label = rec.units[ch] if ch < len(rec.units) else ""

    # ── Spectrogram ─────────────────────────────────────────────
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
        showlegend=False, uirevision=f"tr_spec_{onset}_{ch}",
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

    win_samples = int(2.0 * rec.fs)
    step_samples = int(1.0 * rec.fs)
    band_power_data = {name: [] for name in bands}
    bp_times = []

    for start_s in range(0, max(1, len(data) - win_samples), step_samples):
        end_s = start_s + win_samples
        segment = data[start_s:end_s]
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
        uirevision=f"tr_bp_{onset}_{ch}",
    )
    apply_fig_theme(fig_bp)
    fig_bp.update_layout(margin=dict(l=60, r=20, t=30, b=40))

    return fig_spec, fig_bp


# ── Browse Mode Plot ──────────────────────────────────────────────────


def _build_browse_figure(rec, annotations: list[AnnotatedEvent], state,
                         start_sec=0.0, window_sec=60.0,
                         selected_channels=None,
                         bp_low=1.0, bp_high=50.0,
                         add_seizure_active=False,
                         remove_seizure_active=False):
    """Build the EEG plot for browse mode with annotation overlays."""
    if selected_channels is None or len(selected_channels) == 0:
        selected_channels = list(range(rec.n_channels))

    end_sec = min(start_sec + window_sec, rec.duration_sec)
    start_idx = int(start_sec * rec.fs)
    end_idx = min(int(end_sec * rec.fs), rec.n_samples)

    # Compute y spacing
    n_samp = min(int(10 * rec.fs), rec.n_samples)
    ptps = [float(np.ptp(rec.data[i, :n_samp])) for i in selected_channels]
    spacing = float(np.median(ptps)) * 1.5 if ptps else 1.0

    fig = go.Figure()
    channel_offsets = {}

    for i, ch_idx in enumerate(selected_channels):
        data = rec.data[ch_idx, start_idx:end_idx].astype(np.float64)
        data = bandpass_filter(data, rec.fs, bp_low, bp_high)

        offset = -i * spacing
        channel_offsets[ch_idx] = offset
        time_axis = np.linspace(start_sec, end_sec, len(data))
        ds_time, ds_data = _minmax_downsample(time_axis, data + offset)

        fig.add_trace(go.Scattergl(
            x=ds_time, y=ds_data,
            mode="lines", name=rec.channel_names[ch_idx],
            line=dict(width=0.8),
        ))

    # Annotation overlays with absolute event IDs
    for ann in annotations:
        if ann.offset_sec < start_sec or ann.onset_sec > end_sec:
            continue
        if ann.channel not in channel_offsets:
            continue

        ch_offset = channel_offsets[ann.channel]
        half = spacing / 2.0

        # Choose color based on source and label
        if ann.source == "manual":
            colors = _LABEL_COLORS["manual"]
        else:
            colors = _LABEL_COLORS.get(ann.label, _LABEL_COLORS["pending"])

        x0 = max(ann.onset_sec, start_sec)
        x1 = min(ann.offset_sec, end_sec)
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=ch_offset - half, y1=ch_offset + half,
            fillcolor=colors["fill"],
            line=dict(color=colors["line"], width=1),
            layer="below",
        )

        # Event ID label inside the shadow (absolute ID that never changes)
        if ann.event_id > 0:
            label_text = f"#{ann.event_id}"
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=ch_offset + half * 0.7,
                text=label_text,
                showarrow=False,
                font=dict(size=9, color=colors["text"]),
                opacity=0.85,
                xanchor="center", yanchor="middle",
            )

    # Layout
    n_ch = len(selected_channels)
    y_ticks = [channel_offsets[ch] for ch in selected_channels]
    y_labels = [rec.channel_names[ch] for ch in selected_channels]
    y_upper = spacing / 2
    y_lower = -(n_ch - 1) * spacing - spacing * 1.5

    fig.update_layout(
        xaxis=dict(title="Time (s)", fixedrange=False, uirevision="x_stable"),
        yaxis=dict(
            tickvals=y_ticks, ticktext=y_labels,
            zeroline=False, showgrid=False,
            fixedrange=False,
            range=[y_lower, y_upper],
            uirevision="y_stable",
        ),
        height=600,
        showlegend=False,
        dragmode="select" if add_seizure_active else "zoom",
        uirevision="browse_stable",
    )

    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=80, r=20, t=10, b=40))

    return fig


# ── Layout ────────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the training/annotation tab layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording

    # Load or initialise annotations
    annotations = _get_annotations(state)
    if not annotations:
        # Try loading from disk
        if rec.source_path:
            disk_annotations = load_annotations(rec.source_path)
            if disk_annotations:
                annotations = disk_annotations
                # Filter settings are loaded from the detection file
                # (done in _try_load_saved_detections at upload time)
            elif state.seizure_events:
                # Convert detections to annotations — use filtered set if available
                events_for_annotation = [
                    e for e in (state.detected_events or state.seizure_events)
                    if e.event_type == "seizure"
                ]
                if not events_for_annotation:
                    events_for_annotation = state.seizure_events
                annotations = detections_to_annotations(
                    events_for_annotation, rec.source_path or "",
                    animal_id=state.extra.get("tr_animal_id", ""),
                )
        if annotations:
            # Backfill event_ids from seizure_events for legacy annotations
            _backfill_event_ids(annotations, state.seizure_events or [])
            _set_annotations(state, annotations)
            # Save initial annotations to disk
            if rec.source_path:
                try:
                    save_annotations(rec.source_path, annotations)
                except Exception:
                    pass

    # Restore state
    current_idx = state.extra.get("tr_current_idx", 0)
    mode = state.extra.get("tr_mode", "review")
    annotator = state.extra.get("tr_annotator", "")
    animal_id = state.extra.get("tr_animal_id", "")
    channel_filter = state.extra.get("tr_channel_filter", None)
    browse_window = state.extra.get("tr_browse_window", 60)
    browse_start = state.extra.get("tr_browse_start", 0)

    # Restore filter settings from seizure tab (or training-specific overrides)
    tr_filter_on = state.extra.get("tr_filter_on", True)
    sz_fv = state.extra.get("sz_filter_values", {})
    tr_min_conf = state.extra.get("tr_min_conf", sz_fv.get("min_conf", 0))
    tr_min_dur = state.extra.get("tr_min_dur", sz_fv.get("min_dur", 0))
    tr_min_lbl = state.extra.get("tr_min_lbl", sz_fv.get("min_lbl", 0))
    tr_max_conf = state.extra.get("tr_max_conf", sz_fv.get("max_conf", None))
    tr_max_dur = state.extra.get("tr_max_dur", sz_fv.get("max_dur", None))
    tr_max_lbl = state.extra.get("tr_max_lbl", sz_fv.get("max_lbl", None))
    # Additional filters (matching seizure detection tab)
    tr_min_spikes = state.extra.get("tr_min_spikes", sz_fv.get("min_spikes", 0))
    tr_max_spikes = state.extra.get("tr_max_spikes", sz_fv.get("max_spikes", None))
    tr_min_amp = state.extra.get("tr_min_amp", sz_fv.get("min_amp", 0))
    tr_max_amp = state.extra.get("tr_max_amp", sz_fv.get("max_amp", None))
    tr_min_top_amp = state.extra.get("tr_min_top_amp", sz_fv.get("min_top_amp", 0))
    tr_max_top_amp = state.extra.get("tr_max_top_amp", sz_fv.get("max_top_amp", None))
    tr_min_freq = state.extra.get("tr_min_freq", sz_fv.get("min_freq", 0))
    tr_max_freq = state.extra.get("tr_max_freq", sz_fv.get("max_freq", None))

    # Y-range defaults (mirror viewer defaults)
    viewer_saved = state.extra.get("viewer_settings", {})
    default_yrange = state.extra.get("_viewer_default_yrange", None)
    if default_yrange is None:
        n_samp = min(int(10 * rec.fs), rec.n_samples)
        ptps = [float(np.ptp(rec.data[i, :n_samp])) for i in range(rec.n_channels)]
        default_yrange = float(np.median(ptps)) * 1.5 if ptps else 1.0
    tr_yrange = state.extra.get("tr_yrange", viewer_saved.get("yrange", default_yrange))
    tr_act_ymin = state.extra.get("tr_act_ymin", viewer_saved.get("act_ymin", 0.0))
    tr_act_ymax = state.extra.get("tr_act_ymax", viewer_saved.get("act_ymax", 4.0))

    # Check if activity channels exist
    has_activity = (state.activity_recordings.get("paired") is not None
                    and bool(state.channel_pairings))

    # Build filtered list for review mode
    filtered = _filter_by_channel(annotations, channel_filter)
    if tr_filter_on:
        filtered = _apply_annotation_filters(
            filtered, min_conf=tr_min_conf, min_dur=tr_min_dur, min_lbl=tr_min_lbl,
            max_conf=tr_max_conf, max_dur=tr_max_dur, max_lbl=tr_max_lbl,
            min_spikes=tr_min_spikes, max_spikes=tr_max_spikes,
            min_amp=tr_min_amp, max_amp=tr_max_amp,
            min_top_amp=tr_min_top_amp, max_top_amp=tr_max_top_amp,
            min_freq=tr_min_freq, max_freq=tr_max_freq)
    counts = _progress_counts(annotations)
    counts_filtered = _progress_counts(filtered) if tr_filter_on else counts

    # Per-channel counts
    ch_counts = {}
    for a in annotations:
        ch_name = rec.channel_names[a.channel] if a.channel < len(rec.channel_names) else f"Ch{a.channel}"
        if ch_name not in ch_counts:
            ch_counts[ch_name] = {"confirmed": 0, "rejected": 0, "pending": 0}
        if a.label in ch_counts[ch_name]:
            ch_counts[ch_name][a.label] += 1

    # Channel options
    ch_options = [
        {"label": rec.channel_names[i], "value": i}
        for i in range(rec.n_channels)
    ]

    # Clamp index
    if current_idx >= len(filtered):
        current_idx = max(0, len(filtered) - 1)

    # Progress text
    total = counts["total"]
    prog_text = (
        f"{counts['confirmed']} confirmed, "
        f"{counts['rejected']} rejected, "
        f"{counts['pending']} pending"
    )
    progress_pct = (
        int(100 * (counts["confirmed"] + counts["rejected"]) / total)
        if total > 0 else 0
    )

    # Current event info for review mode
    current_event = filtered[current_idx] if filtered else None
    if current_event:
        event_label_badge = _event_badge(current_event)
    else:
        event_label_badge = html.Span()
    if current_event and filtered:
        _ch = current_event.channel
        _ch_name = rec.channel_names[_ch] if _ch < len(rec.channel_names) else f"Ch{_ch}"
        _animal = state.extra.get("tr_animal_id", "")
        _id_str = f" [#{current_event.event_id}]" if current_event.event_id > 0 else ""
        _suffix = f" — {_ch_name}" + (f" ({_animal})" if _animal else "")
        event_nav_text = f"Event {current_idx + 1} of {len(filtered)}{_id_str}{_suffix}"
    else:
        event_nav_text = "No events"

    return html.Div(
        style={"padding": "24px"},
        children=[
            # Keyboard shortcut stores
            dcc.Store(id="tr-keyboard-store", data={"key": "", "ts": 0}),
            dcc.Store(id="tr-video-seek", data=0),
            html.Div(id="tr-keyboard-listener", style={"display": "none"}),

            # Header + Mode toggle (prominent)
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "16px"},
                children=[
                    html.H4("Seizure Annotation", style={"margin": "0"}),
                    html.Span(
                        "Training data",
                        style={"fontSize": "0.78rem", "color": "#8b949e",
                               "border": "1px solid #2d333b", "borderRadius": "12px",
                               "padding": "2px 10px"},
                    ),
                    html.Div(style={"flex": "1"}),
                    dbc.RadioItems(
                        id="tr-mode-toggle",
                        options=[
                            {"label": "\U0001F50D Review Mode", "value": "review"},
                            {"label": "\U0001F4C4 Browse Mode", "value": "browse"},
                        ],
                        value=mode,
                        inline=True,
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-secondary",
                        labelCheckedClassName="btn-ned-primary",
                        labelStyle={"fontSize": "0.92rem", "fontWeight": "600",
                                    "padding": "6px 18px"},
                    ),
                ],
            ),

            # Metadata row + filters
            dbc.Row([
                dbc.Col([
                    html.Label("Annotator",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Input(
                        id="tr-annotator", type="text",
                        value=annotator, debounce=True,
                        placeholder="Your name",
                        className="form-control",
                        style={"width": "100%"},
                    ),
                ], width=2),
                dbc.Col([
                    dcc.Input(
                        id="tr-animal-id", type="hidden",
                        value=animal_id,
                    ),
                ], style={"display": "none"}),
                dbc.Col([
                    html.Label("Channel",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Dropdown(
                        id="tr-channel-filter",
                        options=ch_options,
                        value=channel_filter,
                        placeholder="All channels",
                        clearable=True,
                        style={"fontSize": "0.82rem"},
                    ),
                ], width=2),
                # Filter toggle
                dbc.Col([
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "8px", "marginTop": "20px"},
                             children=[
                                 dbc.Switch(id="tr-filter-toggle", value=tr_filter_on,
                                            style={"fontSize": "0.78rem"}),
                                 html.Span("Filters",
                                           style={"fontSize": "0.78rem", "color": "#8b949e"}),
                             ]),
                ], width=1),
                # Hidden: browse annotate channel (used by browse callback)
                html.Div(
                    dcc.Dropdown(id="tr-browse-annotate-channel",
                                 options=ch_options,
                                 value=0 if ch_options else None,
                                 clearable=False),
                    style={"display": "none"},
                ),
            ], className="g-2 mb-2"),

            # Filter row 1: Confidence, Duration, Local BL, Spikes
            dbc.Row([
                dbc.Col([
                    html.Label("Confidence",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-conf", type="number", min=0, max=1,
                                  step=0.05, value=tr_min_conf, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-conf", type="number", min=0, max=1,
                                  step=0.05, value=tr_max_conf, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Duration (s)",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-dur", type="number", min=0, max=300,
                                  step=0.5, value=tr_min_dur, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-dur", type="number", min=0, max=300,
                                  step=0.5, value=tr_max_dur, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Local BL",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-lbl", type="number", min=0, max=20,
                                  step=0.1, value=tr_min_lbl, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-lbl", type="number", min=0, max=20,
                                  step=0.1, value=tr_max_lbl, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Spikes",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-spikes", type="number", min=0, max=100,
                                  step=1, value=tr_min_spikes, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-spikes", type="number", min=0, max=100,
                                  step=1, value=tr_max_spikes, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Amp (xBL)",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-amp", type="number", min=0, max=50,
                                  step=0.5, value=tr_min_amp, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-amp", type="number", min=0, max=50,
                                  step=0.5, value=tr_max_amp, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Top Amp",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-top-amp", type="number", min=0, max=20,
                                  step=0.5, value=tr_min_top_amp, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-top-amp", type="number", min=0, max=20,
                                  step=0.5, value=tr_max_top_amp, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
            ], className="g-2 mb-2"),

            # Filter row 2: Freq
            dbc.Row([
                dbc.Col([
                    html.Label("Freq (Hz)",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="tr-min-freq", type="number", min=0, max=50,
                                  step=0.5, value=tr_min_freq, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("–", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="tr-max-freq", type="number", min=0, max=50,
                                  step=0.5, value=tr_max_freq, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
            ], className="g-2 mb-3"),

            # ── Review Mode ──────────────────────────────────────────
            html.Div(
                id="tr-review-mode",
                style={"display": "block" if mode == "review" else "none"},
                children=[
                    # Event navigation
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px",
                               "marginBottom": "8px"},
                        children=[
                            dbc.Button(
                                "\u25C0 Prev (< ,)", id="tr-prev-btn", size="sm",
                                className="btn-ned-secondary",
                            ),
                            html.Div(
                                id="tr-event-nav-text",
                                style={"flex": "1", "textAlign": "center",
                                       "fontWeight": "600", "fontSize": "0.9rem"},
                                children=event_nav_text,
                            ),
                            dcc.Input(
                                id="tr-jump-to", type="number",
                                min=1, step=1, debounce=True,
                                placeholder="#",
                                className="form-control",
                                style={"width": "60px", "height": "30px",
                                       "fontSize": "0.78rem", "textAlign": "center"},
                            ),
                            dbc.Button(
                                "Next (> .) \u25B6", id="tr-next-btn", size="sm",
                                className="btn-ned-secondary",
                            ),
                        ],
                    ),

                    # Seizure property info boxes (updated by callback)
                    html.Div(
                        id="tr-event-properties",
                        children=_build_event_properties(rec, current_event),
                    ),

                    # Action buttons + Convulsive checkbox
                    html.Div(
                        style={"display": "flex", "gap": "12px",
                               "alignItems": "center",
                               "justifyContent": "center",
                               "marginBottom": "10px"},
                        children=[
                            dbc.Button(
                                [html.Span("\u2713 "), "Confirm (C)"],
                                id="tr-confirm-btn",
                                className="btn-ned-primary",
                                style={"minWidth": "120px"},
                            ),
                            dbc.Button(
                                [html.Span("\u2717 "), "Reject (R)"],
                                id="tr-reject-btn",
                                className="btn-ned-danger",
                                style={"minWidth": "120px"},
                            ),
                            dbc.Button(
                                [html.Span("\u2192 "), "Skip (S)"],
                                id="tr-skip-btn",
                                className="btn-ned-secondary",
                                style={"minWidth": "120px"},
                            ),
                            html.Div(
                                style={"borderLeft": "1px solid #2d333b",
                                       "paddingLeft": "12px", "marginLeft": "4px"},
                                children=[
                                    dbc.Checkbox(
                                        id="tr-convulsive-toggle",
                                        label="Convulsive (V)",
                                        value=bool((current_event.features or {}).get("convulsive", False))
                                        if current_event else False,
                                        style={"fontSize": "0.95rem"},
                                        label_style={"fontWeight": "600",
                                                     "fontSize": "0.95rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Status badge + Y-range controls
                    html.Div(
                        style={"display": "flex", "alignItems": "center",
                               "justifyContent": "space-between",
                               "marginBottom": "8px"},
                        children=[
                            html.Div(
                                id="tr-event-status",
                                children=event_label_badge,
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px",
                                       "alignItems": "center"},
                                children=[
                                    html.Label("Y range",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e"}),
                                    dcc.Input(
                                        id="tr-yrange", type="number",
                                        min=0, step=0.01,
                                        value=tr_yrange, debounce=True,
                                        className="form-control",
                                        style={"width": "80px", "height": "26px",
                                               "fontSize": "0.78rem"},
                                    ),
                                ] + ([
                                    html.Label("Act Y",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e",
                                                      "marginLeft": "8px"}),
                                    dcc.Input(
                                        id="tr-act-ymin", type="number",
                                        value=tr_act_ymin, step=0.1,
                                        debounce=True, className="form-control",
                                        style={"width": "60px", "height": "26px",
                                               "fontSize": "0.78rem"},
                                    ),
                                    html.Span("\u2013",
                                              style={"fontSize": "0.72rem",
                                                     "color": "#8b949e"}),
                                    dcc.Input(
                                        id="tr-act-ymax", type="number",
                                        value=tr_act_ymax, step=0.1,
                                        debounce=True, className="form-control",
                                        style={"width": "60px", "height": "26px",
                                               "fontSize": "0.78rem"},
                                    ),
                                ] if has_activity else [
                                    # Hidden placeholders when no activity
                                    dcc.Input(id="tr-act-ymin", type="number",
                                              value=0, style={"display": "none"}),
                                    dcc.Input(id="tr-act-ymax", type="number",
                                              value=4, style={"display": "none"}),
                                ]) + [
                                    dbc.Switch(
                                        id="tr-show-baseline",
                                        label="Baseline",
                                        value=state.extra.get("tr_show_baseline", False),
                                        style={"fontSize": "0.72rem",
                                               "marginLeft": "12px"},
                                    ),
                                    dbc.Switch(
                                        id="tr-show-threshold",
                                        label="Threshold",
                                        value=state.extra.get("tr_show_threshold", False),
                                        style={"fontSize": "0.72rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Review EEG plot — pre-render first event
                    dcc.Loading(
                        dcc.Graph(
                            id="tr-review-graph",
                            figure=_build_review_figure(
                                rec, current_event, state,
                                y_range=tr_yrange,
                                act_ymin=tr_act_ymin, act_ymax=tr_act_ymax,
                                show_baseline=state.extra.get("tr_show_baseline", False),
                                show_threshold=state.extra.get("tr_show_threshold", False),
                            ) if current_event else go.Figure(),
                            config={
                                "editable": False,
                                "edits": {"shapePosition": True},
                                "scrollZoom": True,
                                "displayModeBar": True,
                            },
                            style={"borderRadius": "8px"},
                        ),
                        type="circle", color="#58a6ff",
                    ),

                    # Boundary adjustment (under EEG)
                    html.Div(
                        style={"display": "flex", "gap": "12px",
                               "alignItems": "center", "justifyContent": "center",
                               "marginTop": "8px", "marginBottom": "4px"},
                        children=[
                            html.Label("Onset (s)",
                                       style={"fontSize": "0.72rem", "color": "#8b949e"}),
                            dcc.Input(
                                id="tr-onset-input", type="number",
                                value=round(current_event.onset_sec, 3) if current_event else 0,
                                step=0.01, debounce=True,
                                className="form-control",
                                style={"width": "100px", "height": "26px",
                                       "fontSize": "0.78rem"},
                            ),
                            html.Label("Offset (s)",
                                       style={"fontSize": "0.72rem", "color": "#8b949e",
                                              "marginLeft": "8px"}),
                            dcc.Input(
                                id="tr-offset-input", type="number",
                                value=round(current_event.offset_sec, 3) if current_event else 0,
                                step=0.01, debounce=True,
                                className="form-control",
                                style={"width": "100px", "height": "26px",
                                       "fontSize": "0.78rem"},
                            ),
                            html.Span(
                                id="tr-duration-display",
                                children=f"({current_event.offset_sec - current_event.onset_sec:.2f}s)"
                                if current_event else "",
                                style={"fontSize": "0.72rem", "color": "#8b949e"},
                            ),
                        ],
                    ),

                    # Video player (if available)
                    _training_video_player(state, sid,
                                          current_event.onset_sec if current_event else 0),

                    # Spectral plots (spectrogram + power over time)
                    dbc.Row(
                        id="tr-spectral-row",
                        children=_initial_spectral_row(rec, current_event, state),
                    ),

                    # Notes textarea
                    html.Div(
                        style={"marginBottom": "16px"},
                        children=[
                            html.Label("Notes",
                                       style={"fontSize": "0.78rem", "color": "#8b949e"}),
                            dcc.Textarea(
                                id="tr-notes",
                                value=current_event.notes if current_event else "",
                                placeholder="Optional notes for this event...",
                                style={
                                    "width": "100%", "height": "60px",
                                    "backgroundColor": "#1c2128",
                                    "border": "1px solid #2d333b",
                                    "borderRadius": "6px",
                                    "color": "#e6edf3",
                                    "fontSize": "0.85rem",
                                    "padding": "8px",
                                    "resize": "vertical",
                                },
                            ),
                        ],
                    ),

                    # Status message
                    html.Div(id="tr-review-status"),
                ],
            ),

            # ── Browse Mode ──────────────────────────────────────────
            html.Div(
                id="tr-browse-mode",
                style={"display": "block" if mode == "browse" else "none"},
                children=[
                    # Navigation controls
                    dbc.Row([
                        dbc.Col([
                            html.Label("Window (s)",
                                       style={"fontSize": "0.78rem", "color": "#8b949e"}),
                            dcc.Input(
                                id="tr-browse-window", type="number",
                                min=1, max=600, step=1, value=browse_window,
                                debounce=True, className="form-control",
                                style={"width": "100%"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Start (s)",
                                       style={"fontSize": "0.78rem", "color": "#8b949e"}),
                            dcc.Input(
                                id="tr-browse-start", type="number",
                                min=0, max=rec.duration_sec, step=1,
                                value=browse_start, debounce=True,
                                className="form-control",
                                style={"width": "100%"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Channel (for manual add)",
                                       style={"fontSize": "0.75rem", "color": "#8b949e"}),
                            dcc.Dropdown(
                                id="tr-browse-annotate-channel-vis",
                                options=ch_options,
                                value=0 if ch_options else None,
                                clearable=False,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("\u200B",
                                       style={"fontSize": "0.78rem", "display": "block"}),
                            html.Div(
                                style={"display": "flex", "gap": "6px"},
                                children=[
                                    dbc.Button(
                                        "Add Seizure",
                                        id="tr-add-seizure-btn",
                                        className="btn-ned-secondary",
                                        size="sm",
                                        active=False,
                                    ),
                                    dbc.Button(
                                        "Remove Seizure",
                                        id="tr-remove-seizure-btn",
                                        className="btn-ned-secondary",
                                        size="sm",
                                        active=False,
                                    ),
                                ],
                            ),
                        ], width=3),
                    ], className="g-2 mb-3"),
                    # Channel selection for browse view
                    html.Div(
                        style={"marginBottom": "8px", "display": "flex",
                               "alignItems": "center", "gap": "8px",
                               "flexWrap": "wrap"},
                        children=[
                            html.Label("Channels:",
                                       style={"fontSize": "0.78rem", "color": "#8b949e",
                                              "margin": "0", "fontWeight": "500"}),
                            dbc.Checklist(
                                id="tr-browse-channel-checks",
                                options=[
                                    {"label": rec.channel_names[i], "value": i}
                                    for i in range(rec.n_channels)
                                ],
                                value=list(range(rec.n_channels)),
                                inline=True,
                                style={"fontSize": "0.8rem"},
                            ),
                            html.A("All", id="tr-browse-ch-all", href="#",
                                   style={"fontSize": "0.75rem", "color": "#58a6ff",
                                          "cursor": "pointer", "marginLeft": "4px"}),
                            html.A("None", id="tr-browse-ch-none", href="#",
                                   style={"fontSize": "0.75rem", "color": "#58a6ff",
                                          "cursor": "pointer"}),
                        ],
                    ),

                    # Hidden store to trigger browse graph refresh after add/remove
                    dcc.Store(id="tr-annotations-version", data=0),

                    # Browse navigation bar
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px",
                               "marginBottom": "4px"},
                        children=[
                            dbc.Button("\u23EE", id="tr-nav-start", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            dbc.Button("\u23EA", id="tr-nav-back-big", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            dbc.Button("\u25C0", id="tr-nav-back", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            html.Div(
                                id="tr-nav-time-display",
                                style={"flex": "1", "textAlign": "center",
                                       "fontWeight": "600", "fontSize": "0.9rem"},
                            ),
                            dbc.Button("\u25B6", id="tr-nav-fwd", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            dbc.Button("\u23E9", id="tr-nav-fwd-big", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                        ],
                    ),

                    # Color legend
                    html.Div(
                        style={"display": "flex", "gap": "16px", "marginBottom": "6px",
                               "flexWrap": "wrap", "alignItems": "center"},
                        children=[
                            html.Span("Legend:",
                                      style={"fontSize": "0.72rem", "color": "#8b949e",
                                             "fontWeight": "600"}),
                            *[
                                html.Span(
                                    style={"display": "flex", "alignItems": "center",
                                           "gap": "4px"},
                                    children=[
                                        html.Span(
                                            style={
                                                "width": "12px", "height": "12px",
                                                "borderRadius": "2px",
                                                "background": colors["fill"],
                                                "border": f"1px solid {colors['line']}",
                                                "display": "inline-block",
                                            }
                                        ),
                                        html.Span(
                                            lbl.capitalize(),
                                            style={"fontSize": "0.72rem",
                                                   "color": colors["text"]},
                                        ),
                                    ],
                                )
                                for lbl, colors in _LABEL_COLORS.items()
                            ],
                        ],
                    ),

                    # Browse EEG plot — pre-render if in browse mode
                    dcc.Loading(
                        dcc.Graph(
                            id="tr-browse-graph",
                            figure=_build_browse_figure(
                                rec, _filter_by_channel(annotations, channel_filter),
                                state,
                                start_sec=float(browse_start),
                                window_sec=float(browse_window),
                            ) if mode == "browse" else go.Figure(),
                            config={"scrollZoom": True, "displayModeBar": True},
                            style={"borderRadius": "8px"},
                        ),
                        type="circle", color="#58a6ff",
                    ),

                    # Status message
                    html.Div(id="tr-browse-status"),
                ],
            ),

            # ── Annotation Counts (at end of page) ─────────────────────
            html.Hr(style={"borderColor": "#2d333b", "margin": "24px 0 12px 0"}),
            html.Div(
                id="tr-annotation-counts",
                children=_build_annotation_counts(
                    counts, counts_filtered, progress_pct,
                    ch_counts, tr_filter_on,
                ),
            ),
        ],
    )


# ── Keyboard shortcut clientside callback ─────────────────────────────

clientside_callback(
    """
    function(n) {
        if (!window._trKeyListenerActive) {
            window._trKeyListenerActive = true;
            document.addEventListener('keydown', function(e) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
                var key = e.key.toLowerCase();
                if (['c', 'r', 's', 'v', 'arrowleft', 'arrowright', ',', '.'].includes(key)) {
                    e.preventDefault();
                    if (window.dash_clientside && window.dash_clientside.set_props) {
                        window.dash_clientside.set_props('tr-keyboard-store', {data: {key: key, ts: Date.now()}});
                    }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("tr-keyboard-listener", "children"),
    Input("tr-keyboard-store", "data"),
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("tr-review-mode", "style"),
    Output("tr-browse-mode", "style"),
    Input("tr-mode-toggle", "value"),
    State("session-id", "data"),
)
def toggle_mode(mode, sid):
    """Show/hide review vs browse mode sections."""
    state = server_state.get_session(sid)
    state.extra["tr_mode"] = mode
    if mode == "review":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@callback(
    Output("tr-annotator", "value"),
    Input("tr-annotator", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_annotator(val, sid):
    """Persist annotator name."""
    state = server_state.get_session(sid)
    state.extra["tr_annotator"] = val or ""
    return val


@callback(
    Output("tr-animal-id", "value"),
    Input("tr-animal-id", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_animal_id(val, sid):
    """Persist animal ID."""
    state = server_state.get_session(sid)
    state.extra["tr_animal_id"] = val or ""
    return val


@callback(
    Output("tr-channel-filter", "value"),
    Input("tr-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_channel_filter(val, sid):
    """Persist channel filter."""
    state = server_state.get_session(sid)
    state.extra["tr_channel_filter"] = val
    # Reset index when filter changes
    state.extra["tr_current_idx"] = 0
    return val


# ── Review Mode Callbacks ─────────────────────────────────────────────


@callback(
    Output("tr-review-graph", "figure"),
    Output("tr-event-nav-text", "children"),
    Output("tr-event-status", "children"),
    Output("tr-notes", "value"),
    Output("tr-onset-input", "value"),
    Output("tr-offset-input", "value"),
    Output("tr-duration-display", "children"),
    Output("tr-video-seek", "data"),
    Output("tr-convulsive-toggle", "value"),
    Output("tr-event-properties", "children"),
    Output("tr-annotation-counts", "children"),
    Output("tr-spectrogram", "figure"),
    Output("tr-band-power", "figure"),
    Input("tr-mode-toggle", "value"),
    Input("tr-channel-filter", "value"),
    Input("tr-filter-toggle", "value"),
    Input("tr-min-conf", "value"),
    Input("tr-min-dur", "value"),
    Input("tr-min-lbl", "value"),
    Input("tr-max-conf", "value"),
    Input("tr-max-dur", "value"),
    Input("tr-max-lbl", "value"),
    Input("tr-min-spikes", "value"),
    Input("tr-max-spikes", "value"),
    Input("tr-min-amp", "value"),
    Input("tr-max-amp", "value"),
    Input("tr-min-top-amp", "value"),
    Input("tr-max-top-amp", "value"),
    Input("tr-min-freq", "value"),
    Input("tr-max-freq", "value"),
    Input("tr-prev-btn", "n_clicks"),
    Input("tr-next-btn", "n_clicks"),
    Input("tr-jump-to", "value"),
    Input("tr-confirm-btn", "n_clicks"),
    Input("tr-reject-btn", "n_clicks"),
    Input("tr-skip-btn", "n_clicks"),
    Input("tr-keyboard-store", "data"),
    Input("tr-yrange", "value"),
    Input("tr-act-ymin", "value"),
    Input("tr-act-ymax", "value"),
    Input("tr-show-baseline", "value"),
    Input("tr-show-threshold", "value"),
    Input("tr-convulsive-toggle", "value"),
    State("tr-onset-input", "value"),
    State("tr-offset-input", "value"),
    State("tr-notes", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_review(mode, ch_filter, filt_on, min_conf, min_dur, min_lbl,
                  max_conf, max_dur, max_lbl,
                  min_spikes, max_spikes, min_amp, max_amp,
                  min_top_amp, max_top_amp, min_freq, max_freq,
                  prev_clicks, next_clicks, jump_to,
                  confirm_clicks, reject_clicks, skip_clicks,
                  kb_data, tr_yrange, tr_act_ymin, tr_act_ymax,
                  show_baseline, show_threshold, convulsive_toggle,
                  onset_input, offset_input,
                  notes_val, sid):
    """Handle review mode: navigation, confirm, reject, skip, keyboard."""
    _N_OUT = 13  # number of outputs
    _no = no_update
    if mode != "review":
        return (_no,) * _N_OUT

    state = server_state.get_session(sid)
    if state.recording is None:
        return go.Figure(), "No events", html.Span(), "", 0, 0, "", 0, False, dbc.Row(), html.Div(), go.Figure(), go.Figure()

    trigger = ctx.triggered_id

    # Persist Y-range settings when their controls change
    if trigger == "tr-yrange" and tr_yrange is not None and tr_yrange > 0:
        state.extra["tr_yrange"] = float(tr_yrange)
    if trigger == "tr-act-ymin" and tr_act_ymin is not None:
        state.extra["tr_act_ymin"] = float(tr_act_ymin)
    if trigger == "tr-act-ymax" and tr_act_ymax is not None:
        state.extra["tr_act_ymax"] = float(tr_act_ymax)

    # Read Y-range from saved state (reliable)
    yr = state.extra.get("tr_yrange", state.extra.get(
        "viewer_settings", {}).get("yrange", None))
    act_ymin_val = state.extra.get("tr_act_ymin", 0.0)
    act_ymax_val = state.extra.get("tr_act_ymax", 4.0)

    # Only persist filter settings when the filter controls themselves changed.
    # For all other triggers (buttons, keyboard) read from saved state so that
    # Dash's initial-value-on-unmounted-component quirk cannot reset filters.
    _filter_triggers = {"tr-filter-toggle", "tr-min-conf", "tr-min-dur",
                        "tr-min-lbl", "tr-max-conf", "tr-max-dur", "tr-max-lbl",
                        "tr-min-spikes", "tr-max-spikes", "tr-min-amp", "tr-max-amp",
                        "tr-min-top-amp", "tr-max-top-amp", "tr-min-freq", "tr-max-freq"}
    if trigger in _filter_triggers:
        state.extra["tr_filter_on"] = bool(filt_on)
        if min_conf is not None:
            state.extra["tr_min_conf"] = min_conf
        if min_dur is not None:
            state.extra["tr_min_dur"] = min_dur
        if min_lbl is not None:
            state.extra["tr_min_lbl"] = min_lbl
        if min_spikes is not None:
            state.extra["tr_min_spikes"] = min_spikes
        if min_amp is not None:
            state.extra["tr_min_amp"] = min_amp
        if min_top_amp is not None:
            state.extra["tr_min_top_amp"] = min_top_amp
        if min_freq is not None:
            state.extra["tr_min_freq"] = min_freq
        # max values: None means "no limit" — always persist
        state.extra["tr_max_conf"] = max_conf
        state.extra["tr_max_dur"] = max_dur
        state.extra["tr_max_lbl"] = max_lbl
        state.extra["tr_max_spikes"] = max_spikes
        state.extra["tr_max_amp"] = max_amp
        state.extra["tr_max_top_amp"] = max_top_amp
        state.extra["tr_max_freq"] = max_freq
    else:
        # Use saved (reliable) filter values
        filt_on = state.extra.get("tr_filter_on", True)
        min_conf = state.extra.get("tr_min_conf", 0)
        min_dur = state.extra.get("tr_min_dur", 0)
        min_lbl = state.extra.get("tr_min_lbl", 0)
        max_conf = state.extra.get("tr_max_conf", None)
        max_dur = state.extra.get("tr_max_dur", None)
        max_lbl = state.extra.get("tr_max_lbl", None)
        min_spikes = state.extra.get("tr_min_spikes", 0)
        max_spikes = state.extra.get("tr_max_spikes", None)
        min_amp = state.extra.get("tr_min_amp", 0)
        max_amp = state.extra.get("tr_max_amp", None)
        min_top_amp = state.extra.get("tr_min_top_amp", 0)
        max_top_amp = state.extra.get("tr_max_top_amp", None)
        min_freq = state.extra.get("tr_min_freq", 0)
        max_freq = state.extra.get("tr_max_freq", None)

    rec = state.recording
    annotations = _get_annotations(state)
    filtered = _filter_by_channel(annotations, ch_filter)
    if filt_on:
        filtered = _apply_annotation_filters(
            filtered, min_conf=min_conf, min_dur=min_dur, min_lbl=min_lbl,
            max_conf=max_conf, max_dur=max_dur, max_lbl=max_lbl,
            min_spikes=min_spikes, max_spikes=max_spikes,
            min_amp=min_amp, max_amp=max_amp,
            min_top_amp=min_top_amp, max_top_amp=max_top_amp,
            min_freq=min_freq, max_freq=max_freq)

    if not filtered:
        fig = go.Figure()
        apply_fig_theme(fig)
        fig.update_layout(height=400)
        empty_counts = _build_annotation_counts(
            _progress_counts(annotations), _progress_counts([]),
            0, {}, filt_on)
        return fig, "No events to review", html.Span(), "", 0, 0, "", 0, False, dbc.Row(), empty_counts, go.Figure(), go.Figure()

    current_idx = state.extra.get("tr_current_idx", 0)

    # Determine action
    action = None
    if trigger == "tr-keyboard-store" and kb_data:
        key = kb_data.get("key", "")
        if key == "c":
            action = "confirm"
        elif key == "r":
            action = "reject"
        elif key == "s":
            action = "skip"
        elif key == "v":
            action = "toggle_convulsive"
        elif key in ("arrowleft", ","):
            action = "prev"
        elif key in ("arrowright", "."):
            action = "next"
    elif trigger == "tr-confirm-btn":
        action = "confirm"
    elif trigger == "tr-reject-btn":
        action = "reject"
    elif trigger == "tr-skip-btn":
        action = "skip"
    elif trigger == "tr-prev-btn":
        action = "prev"
    elif trigger == "tr-next-btn":
        action = "next"
    elif trigger == "tr-jump-to":
        action = "jump"
    elif trigger == "tr-convulsive-toggle":
        action = "set_convulsive"

    # Save notes for current event before moving
    if action in ("confirm", "reject", "skip", "prev", "next", "jump"):
        if 0 <= current_idx < len(filtered):
            event = filtered[current_idx]
            # Find in full annotations list and update notes
            for ann in annotations:
                if (ann.onset_sec == event.onset_sec and
                        ann.channel == event.channel and
                        ann.source == event.source):
                    ann.notes = notes_val or ""
                    break

    # Apply confirm/reject — also apply any boundary changes from inputs
    if action in ("confirm", "reject"):
        if 0 <= current_idx < len(filtered):
            event = filtered[current_idx]
            new_label = "confirmed" if action == "confirm" else "rejected"
            for ann in annotations:
                if (ann.onset_sec == event.onset_sec and
                        ann.channel == event.channel and
                        ann.source == event.source):
                    ann.label = new_label
                    ann.annotated_at = datetime.now(timezone.utc).isoformat()
                    ann.annotator = state.extra.get("tr_annotator", "")
                    # Apply boundary changes from onset/offset inputs
                    if onset_input is not None and offset_input is not None:
                        new_on = float(onset_input)
                        new_off = float(offset_input)
                        if new_on > new_off:
                            new_on, new_off = new_off, new_on
                        if (abs(new_on - ann.onset_sec) > 0.001 or
                                abs(new_off - ann.offset_sec) > 0.001):
                            # Store originals
                            if ann.original_onset_sec is None:
                                ann.original_onset_sec = event.onset_sec
                            if ann.original_offset_sec is None:
                                ann.original_offset_sec = event.offset_sec
                            ann.onset_sec = new_on
                            ann.offset_sec = new_off
                    # Recompute activity z-score for new boundaries
                    _recompute_activity_zscore(state, ann)
                    # Propagate boundary changes to seizure_events (Seizure tab)
                    _sync_boundary_to_seizure_events(
                        state, ann.channel, event.onset_sec, ann.onset_sec, ann.offset_sec)
                    break
            _auto_save(state, annotations)
            # Re-filter after label change (apply same filters)
            filtered = _filter_by_channel(annotations, ch_filter)
            if filt_on:
                filtered = _apply_annotation_filters(
                    filtered, min_conf=min_conf, min_dur=min_dur, min_lbl=min_lbl,
                    max_conf=max_conf, max_dur=max_dur, max_lbl=max_lbl,
                    min_spikes=min_spikes, max_spikes=max_spikes,
                    min_amp=min_amp, max_amp=max_amp,
                    min_top_amp=min_top_amp, max_top_amp=max_top_amp,
                    min_freq=min_freq, max_freq=max_freq)
            # Auto-advance to next pending
            next_pending = _find_next_pending(filtered, current_idx)
            if next_pending is not None:
                current_idx = next_pending
            elif current_idx < len(filtered) - 1:
                current_idx += 1

    # Handle convulsive tag
    if action in ("toggle_convulsive", "set_convulsive"):
        if 0 <= current_idx < len(filtered):
            event = filtered[current_idx]
            for ann in annotations:
                if (ann.onset_sec == event.onset_sec and
                        ann.channel == event.channel and
                        ann.source == event.source):
                    if action == "toggle_convulsive":
                        # Keyboard V: toggle
                        cur = (ann.features or {}).get("convulsive", False)
                        if ann.features is None:
                            ann.features = {}
                        ann.features["convulsive"] = not cur
                    else:
                        # Switch click: set to current toggle value
                        if ann.features is None:
                            ann.features = {}
                        ann.features["convulsive"] = bool(convulsive_toggle)
                    break
            _auto_save(state, annotations)

    # Navigate
    if action == "skip" or action == "next":
        current_idx = min(current_idx + 1, len(filtered) - 1)
    elif action == "prev":
        current_idx = max(current_idx - 1, 0)
    elif action == "jump" and jump_to is not None:
        current_idx = max(0, min(int(jump_to) - 1, len(filtered) - 1))

    # Clamp
    current_idx = max(0, min(current_idx, len(filtered) - 1))
    state.extra["tr_current_idx"] = current_idx

    # Build figure
    event = filtered[current_idx]
    # Persist baseline/threshold toggle state
    state.extra["tr_show_baseline"] = bool(show_baseline)
    state.extra["tr_show_threshold"] = bool(show_threshold)

    fig = _build_review_figure(rec, event, state,
                               y_range=yr, act_ymin=act_ymin_val,
                               act_ymax=act_ymax_val,
                               show_baseline=bool(show_baseline),
                               show_threshold=bool(show_threshold))

    _ch = event.channel
    _ch_name = rec.channel_names[_ch] if _ch < len(rec.channel_names) else f"Ch{_ch}"
    _animal = state.extra.get("tr_animal_id", "")
    _id_str = f" [#{event.event_id}]" if event.event_id > 0 else ""
    _suffix = f" — {_ch_name}" + (f" ({_animal})" if _animal else "")
    nav_text = f"Event {current_idx + 1} of {len(filtered)}{_id_str}{_suffix}"
    badge = _event_badge(event)
    notes = event.notes or ""
    ev_onset = round(event.onset_sec, 3)
    ev_offset = round(event.offset_sec, 3)
    ev_dur = f"({event.offset_sec - event.onset_sec:.2f}s)"

    video_seek = max(0, ev_onset - 10)
    is_convulsive = bool((event.features or {}).get("convulsive", False))
    props = _build_event_properties(rec, event)

    # Recompute annotation counts
    _all_counts = _progress_counts(annotations)
    _filt_counts = _progress_counts(filtered)
    _total = _all_counts["total"]
    _reviewed = _filt_counts["confirmed"] + _filt_counts["rejected"]
    _pct = int(100 * _reviewed / len(filtered)) if filtered else 0
    _ch_counts = {}
    for _a in filtered:
        _chn = rec.channel_names[_a.channel] if _a.channel < len(rec.channel_names) else f"Ch{_a.channel}"
        if _chn not in _ch_counts:
            _ch_counts[_chn] = {"confirmed": 0, "rejected": 0, "pending": 0}
        _lbl = _a.label if _a.label in ("confirmed", "rejected") else "pending"
        _ch_counts[_chn][_lbl] += 1
    ann_counts = _build_annotation_counts(
        _all_counts, _filt_counts, _pct, _ch_counts, filt_on)

    # Build spectral plots
    try:
        fig_spec, fig_bp = _build_spectral_plots(rec, event, state)
    except Exception:
        fig_spec, fig_bp = go.Figure(), go.Figure()

    return fig, nav_text, badge, notes, ev_onset, ev_offset, ev_dur, video_seek, is_convulsive, props, ann_counts, fig_spec, fig_bp


# Clientside callback: seek the training video when event changes
clientside_callback(
    """
    function(seekTime) {
        var video = document.getElementById('tr-review-video');
        if (video && seekTime != null && seekTime >= 0) {
            video.currentTime = seekTime;
        }
        return seekTime;
    }
    """,
    Output("tr-video-seek", "data", allow_duplicate=True),
    Input("tr-video-seek", "data"),
    prevent_initial_call=True,
)


def _find_next_pending(filtered: list[AnnotatedEvent], current_idx: int):
    """Find next pending event after current_idx, wrapping around."""
    n = len(filtered)
    if n == 0:
        return None
    for offset in range(1, n):
        idx = (current_idx + offset) % n
        if filtered[idx].label == "pending":
            return idx
    return None


# ── Boundary Adjustment ───────────────────────────────────────────────


@callback(
    Output("tr-review-status", "children"),
    Output("tr-onset-input", "value", allow_duplicate=True),
    Output("tr-offset-input", "value", allow_duplicate=True),
    Output("tr-duration-display", "children", allow_duplicate=True),
    Output("tr-event-status", "children", allow_duplicate=True),
    Output("tr-event-properties", "children", allow_duplicate=True),
    Input("tr-review-graph", "relayoutData"),
    State("tr-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_boundary_adjustment(relayout_data, ch_filter, sid):
    """Update annotation boundaries when shapes are dragged."""
    _n_out = 6
    if not relayout_data:
        return (no_update,) * _n_out

    state = server_state.get_session(sid)
    if state.recording is None:
        return (no_update,) * _n_out

    annotations = _get_annotations(state)
    # Apply SAME filters as update_review so current_idx matches
    filt_on = state.extra.get("tr_filter_on", False)
    filtered = _filter_by_channel(annotations, ch_filter)
    if filt_on:
        filtered = _apply_annotation_filters(
            filtered,
            min_conf=state.extra.get("tr_min_conf", 0),
            min_dur=state.extra.get("tr_min_dur", 0),
            min_lbl=state.extra.get("tr_min_lbl", 0))
    current_idx = state.extra.get("tr_current_idx", 0)

    if not filtered or current_idx >= len(filtered):
        return (no_update,) * _n_out

    event = filtered[current_idx]
    rec = state.recording

    # Parse shape edits from relayoutData.
    # The highlight rect is shapes[0]; x0=onset, x1=offset.
    import re
    new_onset = None
    new_offset = None

    for key, val in relayout_data.items():
        m = re.match(r"shapes\[(\d+)\]\.(x0|x1)$", key)
        if m:
            prop = m.group(2)
            if prop == "x0":
                new_onset = float(val)
            elif prop == "x1":
                new_offset = float(val)

    # Handle full shapes list update
    if "shapes" in relayout_data and new_onset is None and new_offset is None:
        for shape in relayout_data["shapes"]:
            if shape.get("name") == "highlight":
                new_onset = float(shape["x0"])
                new_offset = float(shape["x1"])
                break

    if new_onset is None and new_offset is None:
        return (no_update,) * _n_out

    # Update the matching annotation in the full list
    for ann in annotations:
        if (ann.onset_sec == event.onset_sec and
                ann.channel == event.channel and
                ann.source == event.source):
            if ann.original_onset_sec is None:
                ann.original_onset_sec = ann.onset_sec
            if ann.original_offset_sec is None:
                ann.original_offset_sec = ann.offset_sec

            if new_onset is not None:
                ann.onset_sec = new_onset
            if new_offset is not None:
                ann.offset_sec = new_offset

            if ann.onset_sec > ann.offset_sec:
                ann.onset_sec, ann.offset_sec = ann.offset_sec, ann.onset_sec

            ann.annotated_at = datetime.now(timezone.utc).isoformat()

            # Recompute activity z-score for new boundaries
            _recompute_activity_zscore(state, ann)

            # Return updated values for onset/offset inputs
            _auto_save(state, annotations)
            dur = f"({ann.offset_sec - ann.onset_sec:.2f}s)"
            badge = _event_badge(ann)
            props = _build_event_properties(rec, ann)
            return (
                alert("Boundaries updated", "success"),
                round(ann.onset_sec, 3),
                round(ann.offset_sec, 3),
                dur,
                badge,
                props,
            )

    return (no_update,) * _n_out


# ── Notes Save ────────────────────────────────────────────────────────


@callback(
    Output("tr-notes", "style"),
    Input("tr-notes", "value"),
    State("tr-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_notes(notes_val, ch_filter, sid):
    """Save notes on blur/change for the current event."""
    state = server_state.get_session(sid)
    annotations = _get_annotations(state)
    filtered = _filter_by_channel(annotations, ch_filter)
    current_idx = state.extra.get("tr_current_idx", 0)

    if filtered and 0 <= current_idx < len(filtered):
        event = filtered[current_idx]
        for ann in annotations:
            if (ann.onset_sec == event.onset_sec and
                    ann.channel == event.channel and
                    ann.source == event.source):
                ann.notes = notes_val or ""
                break
        _auto_save(state, annotations)

    # Return existing style unchanged
    return {
        "width": "100%", "height": "60px",
        "backgroundColor": "#1c2128",
        "border": "1px solid #2d333b",
        "borderRadius": "6px",
        "color": "#e6edf3",
        "fontSize": "0.85rem",
        "padding": "8px",
        "resize": "vertical",
    }


# ── Browse Mode Callbacks ─────────────────────────────────────────────


@callback(
    Output("tr-browse-start", "value"),
    Output("tr-nav-time-display", "children"),
    Input("tr-nav-start", "n_clicks"),
    Input("tr-nav-back-big", "n_clicks"),
    Input("tr-nav-back", "n_clicks"),
    Input("tr-nav-fwd", "n_clicks"),
    Input("tr-nav-fwd-big", "n_clicks"),
    Input("tr-browse-start", "value"),
    State("tr-browse-window", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_browse_navigation(ns, bb, b, f, fb, start_val, window, sid):
    """Handle browse mode navigation buttons."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update, no_update

    rec = state.recording
    window = float(window or 60)
    max_start = max(0.0, rec.duration_sec - window)
    current = float(start_val or 0)

    trigger = ctx.triggered_id
    if trigger == "tr-nav-start":
        current = 0.0
    elif trigger == "tr-nav-back-big":
        current -= window * 5
    elif trigger == "tr-nav-back":
        current -= window
    elif trigger == "tr-nav-fwd":
        current += window
    elif trigger == "tr-nav-fwd-big":
        current += window * 5

    current = max(0.0, min(max_start, current))
    state.extra["tr_browse_start"] = current

    time_display = f"{current:.1f}s \u2013 {current + window:.1f}s"
    return round(current, 2), time_display


@callback(
    Output("tr-browse-channel-checks", "value"),
    Input("tr-browse-ch-all", "n_clicks"),
    Input("tr-browse-ch-none", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def browse_channel_all_none(all_clicks, none_clicks, sid):
    """Handle All/None links for browse channel selection."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update
    if trigger == "tr-browse-ch-all":
        return list(range(state.recording.n_channels))
    if trigger == "tr-browse-ch-none":
        return []
    return no_update


@callback(
    Output("tr-browse-graph", "figure"),
    Input("tr-mode-toggle", "value"),
    Input("tr-browse-start", "value"),
    Input("tr-browse-window", "value"),
    Input("tr-add-seizure-btn", "active"),
    Input("tr-remove-seizure-btn", "active"),
    Input("tr-channel-filter", "value"),
    Input("tr-browse-channel-checks", "value"),
    Input("tr-filter-toggle", "value"),
    Input("tr-min-conf", "value"),
    Input("tr-min-dur", "value"),
    Input("tr-min-lbl", "value"),
    Input("tr-max-conf", "value"),
    Input("tr-max-dur", "value"),
    Input("tr-max-lbl", "value"),
    Input("tr-min-spikes", "value"),
    Input("tr-max-spikes", "value"),
    Input("tr-min-amp", "value"),
    Input("tr-max-amp", "value"),
    Input("tr-min-top-amp", "value"),
    Input("tr-max-top-amp", "value"),
    Input("tr-min-freq", "value"),
    Input("tr-max-freq", "value"),
    Input("tr-annotations-version", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_browse_graph(mode, start_val, window_val, add_active, remove_active,
                        ch_filter, selected_channels,
                        filt_on, min_conf, min_dur, min_lbl,
                        max_conf, max_dur, max_lbl,
                        min_spikes, max_spikes, min_amp, max_amp,
                        min_top_amp, max_top_amp, min_freq, max_freq,
                        ann_version, sid):
    """Render the browse mode EEG plot with annotation overlays."""
    if mode != "browse":
        return no_update

    state = server_state.get_session(sid)
    if state.recording is None:
        fig = go.Figure()
        apply_fig_theme(fig)
        fig.update_layout(height=600)
        return fig

    # Read filter values from server state (reliable, not affected by
    # Dash sending stale Input values after layout re-render)
    trigger = ctx.triggered_id
    _filter_triggers = {"tr-filter-toggle", "tr-min-conf", "tr-min-dur",
                        "tr-min-lbl", "tr-max-conf", "tr-max-dur", "tr-max-lbl",
                        "tr-min-spikes", "tr-max-spikes", "tr-min-amp", "tr-max-amp",
                        "tr-min-top-amp", "tr-max-top-amp", "tr-min-freq", "tr-max-freq"}
    if trigger in _filter_triggers:
        state.extra["tr_filter_on"] = bool(filt_on)
        if min_conf is not None:
            state.extra["tr_min_conf"] = min_conf
        if min_dur is not None:
            state.extra["tr_min_dur"] = min_dur
        if min_lbl is not None:
            state.extra["tr_min_lbl"] = min_lbl
        if min_spikes is not None:
            state.extra["tr_min_spikes"] = min_spikes
        if min_amp is not None:
            state.extra["tr_min_amp"] = min_amp
        if min_top_amp is not None:
            state.extra["tr_min_top_amp"] = min_top_amp
        if min_freq is not None:
            state.extra["tr_min_freq"] = min_freq
        state.extra["tr_max_conf"] = max_conf
        state.extra["tr_max_dur"] = max_dur
        state.extra["tr_max_lbl"] = max_lbl
        state.extra["tr_max_spikes"] = max_spikes
        state.extra["tr_max_amp"] = max_amp
        state.extra["tr_max_top_amp"] = max_top_amp
        state.extra["tr_max_freq"] = max_freq

    rec = state.recording
    annotations = _get_annotations(state)

    # Browse mode shows ALL annotations (channel-filtered only) with
    # different colors for each status.  Confidence/duration filters
    # only affect Review mode navigation.
    browse_annotations = _filter_by_channel(annotations, ch_filter)

    start_sec = float(start_val or 0)
    window_sec = float(window_val or 60)

    # Use browse channel selection
    if selected_channels and len(selected_channels) > 0:
        channels = [ch for ch in selected_channels if 0 <= ch < rec.n_channels]
    else:
        channels = list(range(rec.n_channels))
    if not channels:
        channels = list(range(rec.n_channels))

    state.extra["tr_browse_window"] = window_sec

    fig = _build_browse_figure(
        rec, browse_annotations, state,
        start_sec=start_sec, window_sec=window_sec,
        selected_channels=channels,
        add_seizure_active=bool(add_active),
        remove_seizure_active=bool(remove_active),
    )
    return fig


@callback(
    Output("tr-browse-annotate-channel", "value"),
    Input("tr-browse-annotate-channel-vis", "value"),
    prevent_initial_call=True,
)
def sync_annotate_channel(val):
    """Sync visible channel dropdown to hidden one used by add-seizure callback."""
    return val


@callback(
    Output("tr-add-seizure-btn", "active"),
    Output("tr-remove-seizure-btn", "active"),
    Output("tr-browse-status", "children"),
    Output("tr-browse-graph", "selectedData"),
    Output("tr-annotations-version", "data"),
    Input("tr-add-seizure-btn", "n_clicks"),
    Input("tr-remove-seizure-btn", "n_clicks"),
    Input("tr-browse-graph", "selectedData"),
    Input("tr-browse-graph", "clickData"),
    State("tr-add-seizure-btn", "active"),
    State("tr-remove-seizure-btn", "active"),
    State("tr-browse-annotate-channel", "value"),
    State("tr-annotations-version", "data"),
    State("tr-filter-toggle", "value"),
    State("tr-min-conf", "value"),
    State("tr-min-dur", "value"),
    State("tr-min-lbl", "value"),
    State("tr-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_add_remove_seizure(add_clicks, remove_clicks,
                               selected_data, click_data,
                               add_active, remove_active,
                               annotate_channel, ann_version,
                               filt_on, min_conf, min_dur, min_lbl,
                               ch_filter, sid):
    """Handle Add/Remove seizure button toggles and actions."""
    ann_version = ann_version or 0

    # Determine which prop triggered (needed because selectedData and clickData
    # share the same component id)
    triggered_props = [t["prop_id"] for t in ctx.triggered] if ctx.triggered else []
    trigger_id = ctx.triggered_id

    # ── Toggle buttons (mutual exclusion) ──
    if trigger_id == "tr-add-seizure-btn":
        new_add = not add_active
        return new_add, False, no_update, no_update, no_update

    if trigger_id == "tr-remove-seizure-btn":
        new_remove = not remove_active
        return False, new_remove, no_update, no_update, no_update

    # ── Add seizure via selection ──
    is_selection = any("selectedData" in p for p in triggered_props)
    is_click = any("clickData" in p for p in triggered_props)

    if is_selection and selected_data is not None:
        if not add_active:
            return no_update, no_update, no_update, no_update, no_update

        state = server_state.get_session(sid)
        if state.recording is None:
            return no_update, no_update, no_update, no_update, no_update

        rec = state.recording

        # Extract x-range from selection
        x_range = selected_data.get("range", {}).get("x", None)
        if not x_range or len(x_range) < 2:
            points = selected_data.get("points", [])
            if len(points) >= 2:
                xs = [p["x"] for p in points]
                x_range = [min(xs), max(xs)]
            else:
                return no_update, no_update, no_update, no_update, no_update

        onset = float(x_range[0])
        offset = float(x_range[1])
        if onset > offset:
            onset, offset = offset, onset
        if offset - onset < 0.1:
            return no_update, no_update, alert("Selection too small (< 0.1s)", "warning"), None, no_update

        channel = int(annotate_channel) if annotate_channel is not None else 0

        # Assign next available event_id (max of existing + 1)
        annotations = _get_annotations(state)
        existing_ids = [a.event_id for a in annotations if a.event_id > 0]
        # Also check seizure_events for IDs
        for ev in (state.seizure_events or []):
            if ev.event_id > 0:
                existing_ids.append(ev.event_id)
        next_id = max(existing_ids) + 1 if existing_ids else 1

        new_ann = AnnotatedEvent(
            file_path=rec.source_path or "",
            animal_id=state.extra.get("tr_animal_id", ""),
            annotator=state.extra.get("tr_annotator", ""),
            onset_sec=onset,
            offset_sec=offset,
            channel=channel,
            label="confirmed",
            source="manual",
            event_type="seizure",
            annotated_at=datetime.now(timezone.utc).isoformat(),
            event_id=next_id,
        )

        annotations.append(new_ann)
        annotations.sort(key=lambda e: (e.channel, e.onset_sec))
        _auto_save(state, annotations)

        # Propagate to seizure_events so it appears in the Seizure tab
        from eeg_seizure_analyzer.detection.base import DetectedEvent
        new_det = DetectedEvent(
            onset_sec=onset,
            offset_sec=offset,
            duration_sec=offset - onset,
            channel=channel,
            event_type="seizure",
            confidence=1.0,
            animal_id=state.extra.get("tr_animal_id", ""),
            event_id=next_id,
            source="manual",
        )

        # Compute features & quality metrics so the Seizure tab shows
        # meaningful values instead of zeros.
        try:
            new_det = _compute_manual_event_metrics(rec, new_det, state)
            # Copy computed metrics back to the annotation
            new_ann.detector_confidence = new_det.confidence
            new_ann.features = dict(new_det.features)
            new_ann.quality_metrics = dict(new_det.quality_metrics)
            # Re-save annotations with computed metrics
            _auto_save(state, annotations)
        except Exception:
            import traceback
            traceback.print_exc()

        state.seizure_events.append(new_det)
        state.seizure_events.sort(key=lambda e: (e.channel, e.onset_sec))
        state.detected_events = list(state.seizure_events) + state.spike_events
        # Re-save detection file with the new event
        _save_detection_file(state)

        msg = f"Added manual seizure: {onset:.2f}s \u2013 {offset:.2f}s on Ch {channel}"
        # Deactivate Add button after adding
        return False, False, alert(msg, "success"), None, ann_version + 1

    # ── Remove seizure via click ──
    if is_click and click_data is not None:
        if not remove_active:
            return no_update, no_update, no_update, no_update, no_update

        state = server_state.get_session(sid)
        if state.recording is None:
            return no_update, no_update, no_update, no_update, no_update

        # Get click x position
        points = click_data.get("points", [])
        if not points:
            return no_update, no_update, no_update, no_update, no_update

        click_x = float(points[0].get("x", 0))

        annotations = _get_annotations(state)
        # Browse shows all annotations (channel-filtered only)
        visible = _filter_by_channel(annotations, ch_filter)

        # Find annotation at click position
        best = None
        best_dur = float("inf")
        for ann in visible:
            if ann.onset_sec <= click_x <= ann.offset_sec:
                dur = ann.offset_sec - ann.onset_sec
                if dur < best_dur:
                    best = ann
                    best_dur = dur

        if best is None:
            return no_update, no_update, alert("No seizure found at click position", "warning"), no_update, no_update

        # Remove from full annotations list (match by onset + channel + source)
        annotations = [
            a for a in annotations
            if not (a.onset_sec == best.onset_sec and
                    a.channel == best.channel and
                    a.source == best.source and
                    a.offset_sec == best.offset_sec)
        ]
        _auto_save(state, annotations)

        # Also remove from seizure_events (Seizure tab)
        state.seizure_events = [
            ev for ev in state.seizure_events
            if not (abs(ev.onset_sec - best.onset_sec) < 0.01 and
                    ev.channel == best.channel)
        ]
        state.detected_events = list(state.seizure_events) + state.spike_events
        _save_detection_file(state)

        msg = f"Removed seizure: {best.onset_sec:.2f}s \u2013 {best.offset_sec:.2f}s on Ch {best.channel}"
        # Deactivate Remove button after removing
        return False, False, alert(msg, "info"), no_update, ann_version + 1

    return no_update, no_update, no_update, no_update, no_update
