"""Training tab for interictal spikes: annotate detected IS to build ML training data."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
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
    save_spike_annotations,
    load_spike_annotations,
    detections_to_annotations,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


# ── Helpers ───────────────────────────────────────────────────────────


def _get_annotations(state) -> list[AnnotatedEvent]:
    """Retrieve spike annotations from state.extra, deserialising if needed."""
    raw = state.extra.get("trs_annotations", [])
    if not raw:
        return []
    out: list[AnnotatedEvent] = []
    for item in raw:
        if isinstance(item, AnnotatedEvent):
            out.append(item)
        elif isinstance(item, dict):
            out.append(AnnotatedEvent.from_dict(item))
    return out


def _set_annotations(state, annotations: list[AnnotatedEvent]):
    """Store spike annotations into state.extra as dicts."""
    state.extra["trs_annotations"] = [a.to_dict() for a in annotations]


def _auto_save(state, annotations: list[AnnotatedEvent]):
    """Persist spike annotations to disk and to state."""
    _set_annotations(state, annotations)
    rec = state.recording
    if rec and rec.source_path:
        annotator = state.extra.get("trs_annotator", "")
        animal_id = state.extra.get("trs_animal_id", "")
        try:
            save_spike_annotations(rec.source_path, annotations,
                                   annotator=annotator, animal_id=animal_id)
        except Exception as e:
            import traceback
            traceback.print_exc()


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
                              min_amp=0, max_amp=None,
                              min_xbl=0, max_xbl=None,
                              min_dur_ms=0, max_dur_ms=None,
                              min_conf=0, max_conf=None,
                              min_snr=0, max_snr=None,
                              min_sharp=0, max_sharp=None):
    """Apply spike-specific min/max filters to annotation list.

    Filters mirror the Detection tab's spike filters.  Feature values
    are read from the ``features`` dict (where the spike detector stores
    amplitude, sharpness, etc.) and from ``detector_confidence``.
    """
    filtered = list(annotations)

    def _fmin(v):
        return float(v) if v is not None and v != "" else 0.0

    def _fmax(v):
        if v is None or v == "":
            return None
        return float(v)

    min_amp = _fmin(min_amp)
    min_xbl = _fmin(min_xbl)
    min_dur_ms = _fmin(min_dur_ms)
    min_conf = _fmin(min_conf)
    min_snr = _fmin(min_snr)
    min_sharp = _fmin(min_sharp)
    max_amp = _fmax(max_amp)
    max_xbl = _fmax(max_xbl)
    max_dur_ms = _fmax(max_dur_ms)
    max_conf = _fmax(max_conf)
    max_snr = _fmax(max_snr)
    max_sharp = _fmax(max_sharp)

    def _feat(a, key, default=0):
        return (a.features or {}).get(key, default) or default

    if min_amp > 0:
        filtered = [a for a in filtered if _feat(a, "amplitude") >= min_amp]
    if max_amp is not None:
        filtered = [a for a in filtered if _feat(a, "amplitude") <= max_amp]
    if min_xbl > 0:
        filtered = [a for a in filtered if _feat(a, "amplitude_x_baseline") >= min_xbl]
    if max_xbl is not None:
        filtered = [a for a in filtered if _feat(a, "amplitude_x_baseline") <= max_xbl]
    if min_dur_ms > 0:
        filtered = [a for a in filtered if _feat(a, "duration_ms") >= min_dur_ms]
    if max_dur_ms is not None:
        filtered = [a for a in filtered if _feat(a, "duration_ms") <= max_dur_ms]
    if min_conf > 0:
        filtered = [a for a in filtered if a.detector_confidence >= min_conf]
    if max_conf is not None:
        filtered = [a for a in filtered if a.detector_confidence <= max_conf]
    if min_snr > 0:
        filtered = [a for a in filtered if _feat(a, "local_snr") >= min_snr]
    if max_snr is not None:
        filtered = [a for a in filtered if _feat(a, "local_snr") <= max_snr]
    if min_sharp > 0:
        filtered = [a for a in filtered if _feat(a, "sharpness") >= min_sharp]
    if max_sharp is not None:
        filtered = [a for a in filtered if _feat(a, "sharpness") <= max_sharp]
    return filtered


def _sync_boundary_to_spike_events(state, channel: int,
                                   original_onset: float,
                                   new_onset: float, new_offset: float):
    """Push boundary changes from Training tab back to state.spike_events."""
    if not state.spike_events:
        return
    for ev in state.spike_events:
        if ev.channel == channel and abs(ev.onset_sec - original_onset) < 0.01:
            ev.onset_sec = new_onset
            ev.offset_sec = new_offset
            ev.duration_sec = new_offset - new_onset
            break
    # Update detected_events too
    state.detected_events = list(state.seizure_events) + state.spike_events
    _save_spike_detection_file(state)


def _backfill_event_ids(annotations: list[AnnotatedEvent],
                        spike_events) -> None:
    """Assign event_ids to annotations that don't have one yet."""
    for ann in annotations:
        if ann.event_id > 0:
            continue
        for ev in spike_events:
            if ev.channel == ann.channel and abs(ev.onset_sec - ann.onset_sec) < 0.5:
                if ev.event_id > 0:
                    ann.event_id = ev.event_id
                break

    all_ids = [a.event_id for a in annotations if a.event_id > 0]
    for ev in spike_events:
        if ev.event_id > 0:
            all_ids.append(ev.event_id)
    next_id = max(all_ids) + 1 if all_ids else 1

    for ann in annotations:
        if ann.event_id == 0:
            ann.event_id = next_id
            next_id += 1


def _save_spike_detection_file(state):
    """Re-save the spike detection JSON with current spike_events."""
    try:
        rec = state.recording
        _src = getattr(rec, "source_path", None) or "" if rec else ""
        if _src and _src.lower().endswith(".edf") and state.spike_events:
            from eeg_seizure_analyzer.io.persistence import save_spike_detections
            save_spike_detections(
                edf_path=_src,
                events=state.spike_events,
                detection_info=state.extra.get("sp_detection_info", {}),
                params_dict=state.extra.get("sp_params", {}),
                channels=state.extra.get("sp_selected_channels", []),
                filter_settings={
                    "filter_enabled": state.extra.get("sp_filter_enabled", True),
                    "filter_values": state.extra.get("sp_filter_values", {}),
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


# ── Review Mode Plot ──────────────────────────────────────────────────


def _build_spike_properties(rec, event) -> dbc.Row:
    """Build the spike property info boxes for the current event."""
    if event is None:
        return dbc.Row(className="g-2 mb-2")
    ch = event.channel
    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch{ch}"
    feat = event.features or {}
    amp = feat.get("amplitude")
    xbl = feat.get("amplitude_x_baseline")
    dur_ms = feat.get("duration_ms")
    sharp = feat.get("sharpness")
    return dbc.Row([
        dbc.Col(metric_card("Channel", ch_name), width=2),
        dbc.Col(metric_card("Amplitude",
                            f"{amp:.0f}" if amp is not None else "\u2014"), width=2),
        dbc.Col(metric_card("x Baseline",
                            f"{xbl:.1f}" if xbl is not None else "\u2014"), width=2),
        dbc.Col(metric_card("Duration",
                            f"{dur_ms:.1f} ms" if dur_ms is not None else "\u2014"), width=2),
        dbc.Col(metric_card("Sharpness",
                            f"{sharp:.1f}" if sharp is not None else "\u2014"), width=2),
        dbc.Col(metric_card("Confidence",
                            f"{event.detector_confidence:.2f}"), width=2),
    ], className="g-2 mb-2")


def _build_review_figure(rec, event: AnnotatedEvent, state,
                         bp_low=1.0, bp_high=100.0,
                         y_range=None, show_rect=True,
                         x_window=5.0,
                         show_baseline=False, show_threshold=False):
    """Build the EEG plot for review mode centred on spike.

    Parameters
    ----------
    x_window : float
        Total visible time in seconds (centred on spike midpoint).
        Default 1 s.
    show_rect : bool
        Whether to draw the spike duration rectangle.
    """
    # Load enough data for zooming out (2s each side), but set default
    # x-axis range to *x_window* centred on the spike.
    data_context = 2.0
    ch = event.channel
    onset, offset = event.onset_sec, event.offset_sec
    spike_mid = (onset + offset) / 2.0

    win_start = max(0, spike_mid - data_context)
    win_end = min(rec.duration_sec, spike_mid + data_context)

    start_idx = int(win_start * rec.fs)
    end_idx = min(int(win_end * rec.fs), rec.n_samples)
    data = rec.data[ch, start_idx:end_idx].astype(np.float64)

    # Bandpass filter
    data = bandpass_filter(data, rec.fs, bp_low, bp_high)

    time_axis = np.linspace(win_start, win_end, len(data))
    ds_time, ds_data = _minmax_downsample(time_axis, data)

    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
    unit_label = rec.units[ch] if ch < len(rec.units) else ""

    fig = go.Figure()

    trace = go.Scattergl(
        x=ds_time, y=ds_data,
        mode="lines", name=ch_name,
        line=dict(width=0.8, color="#58a6ff"),
    )
    fig.add_trace(trace)

    # Spike region highlight — editable box so the user can drag edges
    if show_rect:
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

    # Baseline / threshold lines (from detection features)
    baseline_val = (event.features or {}).get("baseline_mean")
    threshold_val = (event.features or {}).get("threshold")
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

    # Y-axis range
    if y_range is not None and y_range > 0:
        half_yr = float(y_range) / 2.0
    else:
        y_ptp = float(np.ptp(data)) if len(data) > 0 else 1.0
        half_yr = y_ptp * 0.6
    y_center = float(np.mean(data)) if len(data) > 0 else 0.0

    # X-axis: default range centred on spike with x_window width
    x_lo = spike_mid - x_window / 2.0
    x_hi = spike_mid + x_window / 2.0

    fig.update_layout(
        xaxis=dict(
            title="Time (s)", fixedrange=False,
            range=[x_lo, x_hi],
            uirevision="x_stable",
        ),
        yaxis=dict(
            title=f"Amplitude ({unit_label})" if unit_label else "Amplitude",
            fixedrange=False,
            range=[y_center - half_yr, y_center + half_yr],
        ),
        height=400,
        showlegend=False,
        dragmode="zoom",
        uirevision="review_stable",
    )

    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=10, b=40))

    return fig


# ── Browse Mode Plot ──────────────────────────────────────────────────


def _build_browse_figure(rec, annotations: list[AnnotatedEvent], state,
                         start_sec=0.0, window_sec=30.0,
                         selected_channels=None,
                         bp_low=1.0, bp_high=100.0,
                         add_spike_active=False,
                         remove_spike_active=False):
    """Build the EEG plot for browse mode with spike annotation overlays."""
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
        dragmode="select" if add_spike_active else "zoom",
        uirevision="browse_stable",
    )

    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=80, r=20, t=10, b=40))

    return fig


# ── Layout ────────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the interictal spike training/annotation tab layout."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()

    rec = state.recording

    # Load or initialise spike annotations
    annotations = _get_annotations(state)
    if not annotations:
        # Try loading from disk
        if rec.source_path:
            disk_annotations = load_spike_annotations(rec.source_path)
            if disk_annotations:
                annotations = disk_annotations
            elif state.spike_events:
                # Convert spike detections to annotations
                events_for_annotation = [
                    e for e in state.spike_events
                    if e.event_type == "spike"
                ]
                if not events_for_annotation:
                    events_for_annotation = state.spike_events
                annotations = detections_to_annotations(
                    events_for_annotation, rec.source_path or "",
                    animal_id=state.extra.get("trs_animal_id", ""),
                )
        if annotations:
            _backfill_event_ids(annotations, state.spike_events or [])
            _set_annotations(state, annotations)
            if rec.source_path:
                try:
                    save_spike_annotations(rec.source_path, annotations)
                except Exception:
                    pass

    # Restore state
    current_idx = state.extra.get("trs_current_idx", 0)
    mode = state.extra.get("trs_mode", "review")
    annotator = state.extra.get("trs_annotator", "")
    animal_id = state.extra.get("trs_animal_id", "")
    channel_filter = state.extra.get("trs_channel_filter", None)
    browse_window = state.extra.get("trs_browse_window", 30)
    browse_start = state.extra.get("trs_browse_start", 0)

    # Restore filter settings — inherit from detection tab if not yet set
    trs_filter_on = state.extra.get("trs_filter_on",
                                    state.extra.get("sp_filter_enabled", True))
    sp_fv = state.extra.get("sp_filter_values", {})
    trs_fv = state.extra.get("trs_filter_values", {})

    def _fv(key):
        """Get filter value: training override > detection value > 0."""
        return trs_fv.get(key, sp_fv.get(key, 0))

    def _fv_max(key):
        """Get max filter value: training override > detection value > None."""
        if key in trs_fv:
            return trs_fv[key]
        return sp_fv.get(key, None)

    trs_min_amp = _fv("min_amp")
    trs_max_amp = _fv_max("max_amp")
    trs_min_xbl = _fv("min_xbl")
    trs_max_xbl = _fv_max("max_xbl")
    trs_min_dur_ms = _fv("min_dur_ms")
    trs_max_dur_ms = _fv_max("max_dur_ms")
    trs_min_conf = _fv("min_conf")
    trs_max_conf = _fv_max("max_conf")
    trs_min_snr = _fv("min_snr")
    trs_max_snr = _fv_max("max_snr")
    trs_min_sharp = _fv("min_sharp")
    trs_max_sharp = _fv_max("max_sharp")

    # Y-range defaults
    viewer_saved = state.extra.get("viewer_settings", {})
    default_yrange = state.extra.get("_viewer_default_yrange", None)
    if default_yrange is None:
        n_samp = min(int(10 * rec.fs), rec.n_samples)
        ptps = [float(np.ptp(rec.data[i, :n_samp])) for i in range(rec.n_channels)]
        default_yrange = float(np.median(ptps)) * 1.5 if ptps else 1.0
    trs_yrange = state.extra.get("trs_yrange", viewer_saved.get("yrange", default_yrange))

    # Build filtered list for review mode
    filtered = _filter_by_channel(annotations, channel_filter)
    if trs_filter_on:
        filtered = _apply_annotation_filters(
            filtered,
            min_amp=trs_min_amp, max_amp=trs_max_amp,
            min_xbl=trs_min_xbl, max_xbl=trs_max_xbl,
            min_dur_ms=trs_min_dur_ms, max_dur_ms=trs_max_dur_ms,
            min_conf=trs_min_conf, max_conf=trs_max_conf,
            min_snr=trs_min_snr, max_snr=trs_max_snr,
            min_sharp=trs_min_sharp, max_sharp=trs_max_sharp)
    counts = _progress_counts(annotations)
    counts_filtered = _progress_counts(filtered) if trs_filter_on else counts

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
    progress_pct = (
        int(100 * (counts["confirmed"] + counts["rejected"]) / total)
        if total > 0 else 0
    )

    # Current event info for review mode
    current_event = filtered[current_idx] if filtered else None
    event_label_badge = _label_badge(current_event.label) if current_event else html.Span()
    if current_event and filtered:
        _ch = current_event.channel
        _ch_name = rec.channel_names[_ch] if _ch < len(rec.channel_names) else f"Ch{_ch}"
        _animal = state.extra.get("trs_animal_id", "")
        _id_str = f" [#{current_event.event_id}]" if current_event.event_id > 0 else ""
        _suffix = f" \u2014 {_ch_name}" + (f" ({_animal})" if _animal else "")
        event_nav_text = f"Spike {current_idx + 1} of {len(filtered)}{_id_str}{_suffix}"
    else:
        event_nav_text = "No spikes"

    return html.Div(
        style={"padding": "24px"},
        children=[
            # Keyboard shortcut stores
            dcc.Store(id="trs-keyboard-store", data={"key": "", "ts": 0}),
            html.Div(id="trs-keyboard-listener", style={"display": "none"}),

            # Header + Mode toggle (prominent)
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "marginBottom": "16px"},
                children=[
                    html.H4("Interictal Spike Annotation", style={"margin": "0"}),
                    html.Span(
                        "Training data",
                        style={"fontSize": "0.78rem", "color": "#8b949e",
                               "border": "1px solid #2d333b", "borderRadius": "12px",
                               "padding": "2px 10px"},
                    ),
                    html.Div(style={"flex": "1"}),
                    dbc.RadioItems(
                        id="trs-mode-toggle",
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

            # Metadata row
            dbc.Row([
                dbc.Col([
                    html.Label("Annotator",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Input(
                        id="trs-annotator", type="text",
                        value=annotator, debounce=True,
                        placeholder="Your name",
                        className="form-control",
                        style={"width": "100%"},
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Animal ID",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Input(
                        id="trs-animal-id", type="text",
                        value=animal_id, debounce=True,
                        placeholder="e.g. M001",
                        className="form-control",
                        style={"width": "100%"},
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Channel",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dcc.Dropdown(
                        id="trs-channel-filter",
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
                                 dbc.Switch(id="trs-filter-toggle", value=trs_filter_on,
                                            style={"fontSize": "0.78rem"}),
                                 html.Span("Filters",
                                           style={"fontSize": "0.78rem", "color": "#8b949e"}),
                             ]),
                ], width=1),
                # Hidden: browse annotate channel
                html.Div(
                    dcc.Dropdown(id="trs-browse-annotate-channel",
                                 options=ch_options,
                                 value=0 if ch_options else None,
                                 clearable=False),
                    style={"display": "none"},
                ),
            ], className="g-2 mb-2"),

            # Filter row — mirrors Detection tab spike filters
            dbc.Row([
                dbc.Col([
                    html.Label("Amplitude",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="trs-min-amp", type="number", min=0, max=10000,
                                  step=1, value=trs_min_amp, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("\u2013", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="trs-max-amp", type="number", min=0, max=10000,
                                  step=1, value=trs_max_amp, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("x Baseline",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="trs-min-xbl", type="number", min=0, max=100,
                                  step=0.1, value=trs_min_xbl, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("\u2013", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="trs-max-xbl", type="number", min=0, max=100,
                                  step=0.1, value=trs_max_xbl, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Duration (ms)",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="trs-min-dur-ms", type="number", min=0, max=500,
                                  step=1, value=trs_min_dur_ms, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("\u2013", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="trs-max-dur-ms", type="number", min=0, max=500,
                                  step=1, value=trs_max_dur_ms, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Confidence",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="trs-min-conf", type="number", min=0, max=1,
                                  step=0.05, value=trs_min_conf, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("\u2013", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="trs-max-conf", type="number", min=0, max=1,
                                  step=0.05, value=trs_max_conf, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Local SNR",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="trs-min-snr", type="number", min=0, max=50,
                                  step=0.1, value=trs_min_snr, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("\u2013", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="trs-max-snr", type="number", min=0, max=50,
                                  step=0.1, value=trs_max_snr, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
                dbc.Col([
                    html.Label("Sharpness",
                               style={"fontSize": "0.75rem", "color": "#8b949e"}),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px"}, children=[
                        dcc.Input(id="trs-min-sharp", type="number", min=0, max=20,
                                  step=0.1, value=trs_min_sharp, placeholder="min",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                        html.Span("\u2013", style={"color": "#8b949e",
                                              "fontSize": "0.8rem"}),
                        dcc.Input(id="trs-max-sharp", type="number", min=0, max=20,
                                  step=0.1, value=trs_max_sharp, placeholder="max",
                                  debounce=True, className="form-control",
                                  style={"width": "50%", "height": "28px",
                                         "fontSize": "0.78rem"}),
                    ]),
                ], width=2),
            ], className="g-2 mb-3"),

            # ── Review Mode ──────────────────────────────────────────
            html.Div(
                id="trs-review-mode",
                style={"display": "block" if mode == "review" else "none"},
                children=[
                    # Event navigation
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px",
                               "marginBottom": "8px"},
                        children=[
                            dbc.Button(
                                "\u25C0 Prev (< ,)", id="trs-prev-btn", size="sm",
                                className="btn-ned-secondary",
                            ),
                            html.Div(
                                id="trs-event-nav-text",
                                style={"flex": "1", "textAlign": "center",
                                       "fontWeight": "600", "fontSize": "0.9rem"},
                                children=event_nav_text,
                            ),
                            dcc.Input(
                                id="trs-jump-to", type="number",
                                min=1, step=1, debounce=True,
                                placeholder="#",
                                className="form-control",
                                style={"width": "60px", "height": "30px",
                                       "fontSize": "0.78rem", "textAlign": "center"},
                            ),
                            dbc.Button(
                                "Next (> .) \u25B6", id="trs-next-btn", size="sm",
                                className="btn-ned-secondary",
                            ),
                        ],
                    ),

                    # Spike property info boxes (updated by callback)
                    html.Div(
                        id="trs-event-properties",
                        children=_build_spike_properties(rec, current_event),
                    ),

                    # Action buttons
                    html.Div(
                        style={"display": "flex", "gap": "12px",
                               "alignItems": "center",
                               "justifyContent": "center",
                               "marginBottom": "10px"},
                        children=[
                            dbc.Button(
                                [html.Span("\u2713 "), "Confirm (C)"],
                                id="trs-confirm-btn",
                                className="btn-ned-primary",
                                style={"minWidth": "120px"},
                            ),
                            dbc.Button(
                                [html.Span("\u2717 "), "Reject (R)"],
                                id="trs-reject-btn",
                                className="btn-ned-danger",
                                style={"minWidth": "120px"},
                            ),
                            dbc.Button(
                                [html.Span("\u2192 "), "Skip (S)"],
                                id="trs-skip-btn",
                                className="btn-ned-secondary",
                                style={"minWidth": "120px"},
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
                                id="trs-event-status",
                                children=event_label_badge,
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px",
                                       "alignItems": "center"},
                                children=[
                                    html.Label("X (s)",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e"}),
                                    dcc.Input(
                                        id="trs-xwindow", type="number",
                                        min=0.1, max=10, step=0.1,
                                        value=state.extra.get("trs_xwindow", 5.0),
                                        debounce=True,
                                        className="form-control",
                                        style={"width": "60px", "height": "26px",
                                               "fontSize": "0.78rem"},
                                    ),
                                    html.Label("Y range",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e"}),
                                    dcc.Input(
                                        id="trs-yrange", type="number",
                                        min=0, step=0.01,
                                        value=trs_yrange, debounce=True,
                                        className="form-control",
                                        style={"width": "80px", "height": "26px",
                                               "fontSize": "0.78rem"},
                                    ),
                                    dbc.Switch(
                                        id="trs-show-rect",
                                        value=state.extra.get("trs_show_rect", True),
                                        style={"fontSize": "0.72rem",
                                               "marginLeft": "8px"},
                                    ),
                                    html.Label("Extent",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e"}),
                                    dbc.Switch(
                                        id="trs-show-baseline",
                                        value=state.extra.get("trs_show_baseline", False),
                                        style={"fontSize": "0.72rem",
                                               "marginLeft": "8px"},
                                    ),
                                    html.Label("Baseline",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e"}),
                                    dbc.Switch(
                                        id="trs-show-threshold",
                                        value=state.extra.get("trs_show_threshold", False),
                                        style={"fontSize": "0.72rem",
                                               "marginLeft": "8px"},
                                    ),
                                    html.Label("Threshold",
                                               style={"fontSize": "0.72rem",
                                                      "color": "#8b949e"}),
                                ],
                            ),
                        ],
                    ),

                    # Review EEG plot
                    dcc.Loading(
                        dcc.Graph(
                            id="trs-review-graph",
                            figure=_build_review_figure(
                                rec, current_event, state,
                                bp_low=float(state.extra.get("sp_params", {}).get("sp-bp-low", 10.0)),
                                bp_high=float(state.extra.get("sp_params", {}).get("sp-bp-high", 70.0)),
                                y_range=trs_yrange,
                                show_rect=state.extra.get("trs_show_rect", True),
                                x_window=state.extra.get("trs_xwindow", 5.0),
                                show_baseline=state.extra.get("trs_show_baseline", False),
                                show_threshold=state.extra.get("trs_show_threshold", False),
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
                                id="trs-onset-input", type="number",
                                value=round(current_event.onset_sec, 3) if current_event else 0,
                                step=0.001, debounce=True,
                                className="form-control",
                                style={"width": "100px", "height": "26px",
                                       "fontSize": "0.78rem"},
                            ),
                            html.Label("Offset (s)",
                                       style={"fontSize": "0.72rem", "color": "#8b949e",
                                              "marginLeft": "8px"}),
                            dcc.Input(
                                id="trs-offset-input", type="number",
                                value=round(current_event.offset_sec, 3) if current_event else 0,
                                step=0.001, debounce=True,
                                className="form-control",
                                style={"width": "100px", "height": "26px",
                                       "fontSize": "0.78rem"},
                            ),
                            html.Span(
                                id="trs-duration-display",
                                children=f"({(current_event.offset_sec - current_event.onset_sec) * 1000:.1f}ms)"
                                if current_event else "",
                                style={"fontSize": "0.72rem", "color": "#8b949e"},
                            ),
                        ],
                    ),

                    # Notes textarea
                    html.Div(
                        style={"marginBottom": "16px"},
                        children=[
                            html.Label("Notes",
                                       style={"fontSize": "0.78rem", "color": "#8b949e"}),
                            dcc.Textarea(
                                id="trs-notes",
                                value=current_event.notes if current_event else "",
                                placeholder="Optional notes for this spike...",
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
                    html.Div(id="trs-review-status"),
                ],
            ),

            # ── Browse Mode ──────────────────────────────────────────
            html.Div(
                id="trs-browse-mode",
                style={"display": "block" if mode == "browse" else "none"},
                children=[
                    # Navigation controls
                    dbc.Row([
                        dbc.Col([
                            html.Label("Window (s)",
                                       style={"fontSize": "0.78rem", "color": "#8b949e"}),
                            dcc.Input(
                                id="trs-browse-window", type="number",
                                min=1, max=600, step=1, value=browse_window,
                                debounce=True, className="form-control",
                                style={"width": "100%"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Start (s)",
                                       style={"fontSize": "0.78rem", "color": "#8b949e"}),
                            dcc.Input(
                                id="trs-browse-start", type="number",
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
                                id="trs-browse-annotate-channel-vis",
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
                                        "Add Spike",
                                        id="trs-add-spike-btn",
                                        className="btn-ned-secondary",
                                        size="sm",
                                        active=False,
                                    ),
                                    dbc.Button(
                                        "Remove Spike",
                                        id="trs-remove-spike-btn",
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
                                id="trs-browse-channel-checks",
                                options=[
                                    {"label": rec.channel_names[i], "value": i}
                                    for i in range(rec.n_channels)
                                ],
                                value=list(range(rec.n_channels)),
                                inline=True,
                                style={"fontSize": "0.8rem"},
                            ),
                            html.A("All", id="trs-browse-ch-all", href="#",
                                   style={"fontSize": "0.75rem", "color": "#58a6ff",
                                          "cursor": "pointer", "marginLeft": "4px"}),
                            html.A("None", id="trs-browse-ch-none", href="#",
                                   style={"fontSize": "0.75rem", "color": "#58a6ff",
                                          "cursor": "pointer"}),
                        ],
                    ),

                    # Hidden store to trigger browse graph refresh after add/remove
                    dcc.Store(id="trs-annotations-version", data=0),

                    # Browse navigation bar
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px",
                               "marginBottom": "4px"},
                        children=[
                            dbc.Button("\u23EE", id="trs-nav-start", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            dbc.Button("\u23EA", id="trs-nav-back-big", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            dbc.Button("\u25C0", id="trs-nav-back", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            html.Div(
                                id="trs-nav-time-display",
                                style={"flex": "1", "textAlign": "center",
                                       "fontWeight": "600", "fontSize": "0.9rem"},
                            ),
                            dbc.Button("\u25B6", id="trs-nav-fwd", size="sm",
                                       className="btn-ned-secondary",
                                       style={"fontSize": "1rem"}),
                            dbc.Button("\u23E9", id="trs-nav-fwd-big", size="sm",
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
                            id="trs-browse-graph",
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
                    html.Div(id="trs-browse-status"),
                ],
            ),

            # ── Annotation Counts (at end of page) ─────────────────────
            html.Hr(style={"borderColor": "#2d333b", "margin": "24px 0 12px 0"}),
            html.Div(
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
                                   if trs_filter_on else ""),
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
            ),
        ],
    )


# ── Keyboard shortcut clientside callback ─────────────────────────────

clientside_callback(
    """
    function(n) {
        if (!window._trsKeyListenerActive) {
            window._trsKeyListenerActive = true;
            document.addEventListener('keydown', function(e) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
                var key = e.key.toLowerCase();
                if (['c', 'r', 's', 'arrowleft', 'arrowright', ',', '.'].includes(key)) {
                    e.preventDefault();
                    if (window.dash_clientside && window.dash_clientside.set_props) {
                        window.dash_clientside.set_props('trs-keyboard-store', {data: {key: key, ts: Date.now()}});
                    }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("trs-keyboard-listener", "children"),
    Input("trs-keyboard-store", "data"),
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("trs-review-mode", "style"),
    Output("trs-browse-mode", "style"),
    Input("trs-mode-toggle", "value"),
    State("session-id", "data"),
)
def trs_toggle_mode(mode, sid):
    """Show/hide review vs browse mode sections."""
    state = server_state.get_session(sid)
    state.extra["trs_mode"] = mode
    if mode == "review":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@callback(
    Output("trs-annotator", "value"),
    Input("trs-annotator", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_save_annotator(val, sid):
    """Persist annotator name."""
    state = server_state.get_session(sid)
    state.extra["trs_annotator"] = val or ""
    return val


@callback(
    Output("trs-animal-id", "value"),
    Input("trs-animal-id", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_save_animal_id(val, sid):
    """Persist animal ID."""
    state = server_state.get_session(sid)
    state.extra["trs_animal_id"] = val or ""
    return val


@callback(
    Output("trs-channel-filter", "value"),
    Input("trs-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_save_channel_filter(val, sid):
    """Persist channel filter."""
    state = server_state.get_session(sid)
    state.extra["trs_channel_filter"] = val
    state.extra["trs_current_idx"] = 0
    return val


# ── Review Mode Callbacks ─────────────────────────────────────────────


@callback(
    Output("trs-review-graph", "figure"),
    Output("trs-event-nav-text", "children"),
    Output("trs-event-status", "children"),
    Output("trs-notes", "value"),
    Output("trs-onset-input", "value"),
    Output("trs-offset-input", "value"),
    Output("trs-duration-display", "children"),
    Output("trs-event-properties", "children"),
    Input("trs-mode-toggle", "value"),
    Input("trs-channel-filter", "value"),
    Input("trs-filter-toggle", "value"),
    Input("trs-min-amp", "value"),
    Input("trs-max-amp", "value"),
    Input("trs-min-xbl", "value"),
    Input("trs-max-xbl", "value"),
    Input("trs-min-dur-ms", "value"),
    Input("trs-max-dur-ms", "value"),
    Input("trs-min-conf", "value"),
    Input("trs-max-conf", "value"),
    Input("trs-min-snr", "value"),
    Input("trs-max-snr", "value"),
    Input("trs-min-sharp", "value"),
    Input("trs-max-sharp", "value"),
    Input("trs-prev-btn", "n_clicks"),
    Input("trs-next-btn", "n_clicks"),
    Input("trs-jump-to", "value"),
    Input("trs-confirm-btn", "n_clicks"),
    Input("trs-reject-btn", "n_clicks"),
    Input("trs-skip-btn", "n_clicks"),
    Input("trs-keyboard-store", "data"),
    Input("trs-yrange", "value"),
    Input("trs-show-rect", "value"),
    Input("trs-xwindow", "value"),
    Input("trs-show-baseline", "value"),
    Input("trs-show-threshold", "value"),
    State("trs-onset-input", "value"),
    State("trs-offset-input", "value"),
    State("trs-notes", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_update_review(mode, ch_filter, filt_on,
                      min_amp, max_amp, min_xbl, max_xbl,
                      min_dur_ms, max_dur_ms, min_conf, max_conf,
                      min_snr, max_snr, min_sharp, max_sharp,
                      prev_clicks, next_clicks, jump_to,
                      confirm_clicks, reject_clicks, skip_clicks,
                      kb_data, trs_yrange, trs_show_rect, trs_xwindow,
                      trs_show_baseline, trs_show_threshold,
                      onset_input, offset_input,
                      notes_val, sid):
    """Handle review mode: navigation, confirm, reject, skip, keyboard."""
    _no = no_update
    if mode != "review":
        return _no, _no, _no, _no, _no, _no, _no, _no

    state = server_state.get_session(sid)
    if state.recording is None:
        return go.Figure(), "No spikes", html.Span(), "", 0, 0, ""

    trigger = ctx.triggered_id

    # Persist Y-range, show_rect, xwindow
    if trigger == "trs-yrange" and trs_yrange is not None and trs_yrange > 0:
        state.extra["trs_yrange"] = float(trs_yrange)
    if trigger == "trs-show-rect":
        state.extra["trs_show_rect"] = bool(trs_show_rect)
    if trigger == "trs-xwindow" and trs_xwindow is not None and trs_xwindow > 0:
        state.extra["trs_xwindow"] = float(trs_xwindow)
    if trigger == "trs-show-baseline":
        state.extra["trs_show_baseline"] = bool(trs_show_baseline)
    if trigger == "trs-show-threshold":
        state.extra["trs_show_threshold"] = bool(trs_show_threshold)

    yr = state.extra.get("trs_yrange", state.extra.get(
        "viewer_settings", {}).get("yrange", None))
    show_rect = state.extra.get("trs_show_rect", True)
    x_win = state.extra.get("trs_xwindow", 5.0)
    show_bl = state.extra.get("trs_show_baseline", False)
    show_th = state.extra.get("trs_show_threshold", False)

    # Persist filter settings only when filter controls change
    _filter_triggers = {
        "trs-filter-toggle",
        "trs-min-amp", "trs-max-amp", "trs-min-xbl", "trs-max-xbl",
        "trs-min-dur-ms", "trs-max-dur-ms", "trs-min-conf", "trs-max-conf",
        "trs-min-snr", "trs-max-snr", "trs-min-sharp", "trs-max-sharp",
    }
    if trigger in _filter_triggers:
        state.extra["trs_filter_on"] = bool(filt_on)
        fv = {}
        for k, v in [("min_amp", min_amp), ("min_xbl", min_xbl),
                      ("min_dur_ms", min_dur_ms), ("min_conf", min_conf),
                      ("min_snr", min_snr), ("min_sharp", min_sharp)]:
            if v is not None:
                fv[k] = v
            else:
                fv[k] = 0
        for k, v in [("max_amp", max_amp), ("max_xbl", max_xbl),
                      ("max_dur_ms", max_dur_ms), ("max_conf", max_conf),
                      ("max_snr", max_snr), ("max_sharp", max_sharp)]:
            fv[k] = v  # None means no limit
        state.extra["trs_filter_values"] = fv
    else:
        # Read from saved state (reliable)
        filt_on = state.extra.get("trs_filter_on", True)
        fv = state.extra.get("trs_filter_values",
                             state.extra.get("sp_filter_values", {}))

    rec = state.recording
    annotations = _get_annotations(state)
    filtered = _filter_by_channel(annotations, ch_filter)
    if filt_on:
        filtered = _apply_annotation_filters(filtered, **fv)

    if not filtered:
        fig = go.Figure()
        apply_fig_theme(fig)
        fig.update_layout(height=400)
        return fig, "No spikes to review", html.Span(), "", 0, 0, "", dbc.Row()

    current_idx = state.extra.get("trs_current_idx", 0)

    # Determine action
    action = None
    if trigger == "trs-keyboard-store" and kb_data:
        key = kb_data.get("key", "")
        if key == "c":
            action = "confirm"
        elif key == "r":
            action = "reject"
        elif key == "s":
            action = "skip"
        elif key in ("arrowleft", ","):
            action = "prev"
        elif key in ("arrowright", "."):
            action = "next"
    elif trigger == "trs-confirm-btn":
        action = "confirm"
    elif trigger == "trs-reject-btn":
        action = "reject"
    elif trigger == "trs-skip-btn":
        action = "skip"
    elif trigger == "trs-prev-btn":
        action = "prev"
    elif trigger == "trs-next-btn":
        action = "next"
    elif trigger == "trs-jump-to":
        action = "jump"

    # Save notes for current event before moving
    if action in ("confirm", "reject", "skip", "prev", "next", "jump"):
        if 0 <= current_idx < len(filtered):
            event = filtered[current_idx]
            for ann in annotations:
                if (ann.onset_sec == event.onset_sec and
                        ann.channel == event.channel and
                        ann.source == event.source):
                    ann.notes = notes_val or ""
                    break

    # Apply confirm/reject
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
                    ann.annotator = state.extra.get("trs_annotator", "")
                    # Apply boundary changes
                    if onset_input is not None and offset_input is not None:
                        new_on = float(onset_input)
                        new_off = float(offset_input)
                        if new_on > new_off:
                            new_on, new_off = new_off, new_on
                        if (abs(new_on - ann.onset_sec) > 0.001 or
                                abs(new_off - ann.offset_sec) > 0.001):
                            if ann.original_onset_sec is None:
                                ann.original_onset_sec = event.onset_sec
                            if ann.original_offset_sec is None:
                                ann.original_offset_sec = event.offset_sec
                            ann.onset_sec = new_on
                            ann.offset_sec = new_off
                    _sync_boundary_to_spike_events(
                        state, ann.channel, event.onset_sec, ann.onset_sec, ann.offset_sec)
                    break
            _auto_save(state, annotations)
            # Re-filter after label change
            filtered = _filter_by_channel(annotations, ch_filter)
            if filt_on:
                filtered = _apply_annotation_filters(filtered, **fv)
            # Auto-advance to next pending
            next_pending = _find_next_pending(filtered, current_idx)
            if next_pending is not None:
                current_idx = next_pending
            elif current_idx < len(filtered) - 1:
                current_idx += 1

    # Navigate
    if action == "skip" or action == "next":
        current_idx = min(current_idx + 1, len(filtered) - 1)
    elif action == "prev":
        current_idx = max(current_idx - 1, 0)
    elif action == "jump" and jump_to is not None:
        current_idx = max(0, min(int(jump_to) - 1, len(filtered) - 1))

    # Clamp
    current_idx = max(0, min(current_idx, len(filtered) - 1))
    state.extra["trs_current_idx"] = current_idx

    # Build figure
    event = filtered[current_idx]
    _sp = state.extra.get("sp_params", {})
    fig = _build_review_figure(rec, event, state,
                               bp_low=float(_sp.get("sp-bp-low", 10.0)),
                               bp_high=float(_sp.get("sp-bp-high", 70.0)),
                               y_range=yr,
                               show_rect=show_rect, x_window=x_win,
                               show_baseline=show_bl, show_threshold=show_th)

    _ch = event.channel
    _ch_name = rec.channel_names[_ch] if _ch < len(rec.channel_names) else f"Ch{_ch}"
    _animal = state.extra.get("trs_animal_id", "")
    _id_str = f" [#{event.event_id}]" if event.event_id > 0 else ""
    _suffix = f" \u2014 {_ch_name}" + (f" ({_animal})" if _animal else "")
    nav_text = f"Spike {current_idx + 1} of {len(filtered)}{_id_str}{_suffix}"
    badge = _label_badge(event.label)
    notes = event.notes or ""
    ev_onset = round(event.onset_sec, 3)
    ev_offset = round(event.offset_sec, 3)
    ev_dur = f"({(event.offset_sec - event.onset_sec) * 1000:.1f}ms)"

    props = _build_spike_properties(rec, event)
    return fig, nav_text, badge, notes, ev_onset, ev_offset, ev_dur, props


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
    Output("trs-review-status", "children"),
    Output("trs-onset-input", "value", allow_duplicate=True),
    Output("trs-offset-input", "value", allow_duplicate=True),
    Output("trs-duration-display", "children", allow_duplicate=True),
    Input("trs-review-graph", "relayoutData"),
    State("trs-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_handle_boundary_adjustment(relayout_data, ch_filter, sid):
    """Update annotation boundaries when shapes are dragged."""
    if not relayout_data:
        return no_update, no_update, no_update, no_update

    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update, no_update, no_update, no_update

    annotations = _get_annotations(state)
    filt_on = state.extra.get("trs_filter_on", False)
    filtered = _filter_by_channel(annotations, ch_filter)
    if filt_on:
        fv = state.extra.get("trs_filter_values",
                             state.extra.get("sp_filter_values", {}))
        filtered = _apply_annotation_filters(filtered, **fv)
    current_idx = state.extra.get("trs_current_idx", 0)

    if not filtered or current_idx >= len(filtered):
        return no_update, no_update, no_update, no_update

    event = filtered[current_idx]

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

    if "shapes" in relayout_data and new_onset is None and new_offset is None:
        for shape in relayout_data["shapes"]:
            if shape.get("name") == "highlight":
                new_onset = float(shape["x0"])
                new_offset = float(shape["x1"])
                break

    if new_onset is None and new_offset is None:
        return no_update, no_update, no_update, no_update

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

            _auto_save(state, annotations)
            dur = f"({(ann.offset_sec - ann.onset_sec) * 1000:.1f}ms)"
            return (
                alert("Boundaries updated", "success"),
                round(ann.onset_sec, 3),
                round(ann.offset_sec, 3),
                dur,
            )

    return no_update, no_update, no_update, no_update


# ── Notes Save ────────────────────────────────────────────────────────


@callback(
    Output("trs-notes", "style"),
    Input("trs-notes", "value"),
    State("trs-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_save_notes(notes_val, ch_filter, sid):
    """Save notes on blur/change for the current event."""
    state = server_state.get_session(sid)
    annotations = _get_annotations(state)
    filtered = _filter_by_channel(annotations, ch_filter)
    current_idx = state.extra.get("trs_current_idx", 0)

    if filtered and 0 <= current_idx < len(filtered):
        event = filtered[current_idx]
        for ann in annotations:
            if (ann.onset_sec == event.onset_sec and
                    ann.channel == event.channel and
                    ann.source == event.source):
                ann.notes = notes_val or ""
                break
        _auto_save(state, annotations)

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
    Output("trs-browse-start", "value"),
    Output("trs-nav-time-display", "children"),
    Input("trs-nav-start", "n_clicks"),
    Input("trs-nav-back-big", "n_clicks"),
    Input("trs-nav-back", "n_clicks"),
    Input("trs-nav-fwd", "n_clicks"),
    Input("trs-nav-fwd-big", "n_clicks"),
    Input("trs-browse-start", "value"),
    State("trs-browse-window", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_handle_browse_navigation(ns, bb, b, f, fb, start_val, window, sid):
    """Handle browse mode navigation buttons."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update, no_update

    rec = state.recording
    window = float(window or 30)
    max_start = max(0.0, rec.duration_sec - window)
    current = float(start_val or 0)

    trigger = ctx.triggered_id
    if trigger == "trs-nav-start":
        current = 0.0
    elif trigger == "trs-nav-back-big":
        current -= window * 5
    elif trigger == "trs-nav-back":
        current -= window
    elif trigger == "trs-nav-fwd":
        current += window
    elif trigger == "trs-nav-fwd-big":
        current += window * 5

    current = max(0.0, min(max_start, current))
    state.extra["trs_browse_start"] = current

    time_display = f"{current:.1f}s \u2013 {current + window:.1f}s"
    return round(current, 2), time_display


@callback(
    Output("trs-browse-channel-checks", "value"),
    Input("trs-browse-ch-all", "n_clicks"),
    Input("trs-browse-ch-none", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_browse_channel_all_none(all_clicks, none_clicks, sid):
    """Handle All/None links for browse channel selection."""
    trigger = ctx.triggered_id
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update
    if trigger == "trs-browse-ch-all":
        return list(range(state.recording.n_channels))
    if trigger == "trs-browse-ch-none":
        return []
    return no_update


@callback(
    Output("trs-browse-graph", "figure"),
    Input("trs-mode-toggle", "value"),
    Input("trs-browse-start", "value"),
    Input("trs-browse-window", "value"),
    Input("trs-add-spike-btn", "active"),
    Input("trs-remove-spike-btn", "active"),
    Input("trs-channel-filter", "value"),
    Input("trs-browse-channel-checks", "value"),
    Input("trs-filter-toggle", "value"),
    Input("trs-min-amp", "value"),
    Input("trs-max-amp", "value"),
    Input("trs-min-xbl", "value"),
    Input("trs-max-xbl", "value"),
    Input("trs-min-dur-ms", "value"),
    Input("trs-max-dur-ms", "value"),
    Input("trs-min-conf", "value"),
    Input("trs-max-conf", "value"),
    Input("trs-min-snr", "value"),
    Input("trs-max-snr", "value"),
    Input("trs-min-sharp", "value"),
    Input("trs-max-sharp", "value"),
    Input("trs-annotations-version", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_update_browse_graph(mode, start_val, window_val, add_active, remove_active,
                            ch_filter, selected_channels,
                            filt_on,
                            min_amp, max_amp, min_xbl, max_xbl,
                            min_dur_ms, max_dur_ms, min_conf, max_conf,
                            min_snr, max_snr, min_sharp, max_sharp,
                            ann_version, sid):
    """Render the browse mode EEG plot with spike annotation overlays."""
    if mode != "browse":
        return no_update

    state = server_state.get_session(sid)
    if state.recording is None:
        fig = go.Figure()
        apply_fig_theme(fig)
        fig.update_layout(height=600)
        return fig

    trigger = ctx.triggered_id
    _filter_triggers = {
        "trs-filter-toggle",
        "trs-min-amp", "trs-max-amp", "trs-min-xbl", "trs-max-xbl",
        "trs-min-dur-ms", "trs-max-dur-ms", "trs-min-conf", "trs-max-conf",
        "trs-min-snr", "trs-max-snr", "trs-min-sharp", "trs-max-sharp",
    }
    if trigger in _filter_triggers:
        state.extra["trs_filter_on"] = bool(filt_on)
        fv = {}
        for k, v in [("min_amp", min_amp), ("min_xbl", min_xbl),
                      ("min_dur_ms", min_dur_ms), ("min_conf", min_conf),
                      ("min_snr", min_snr), ("min_sharp", min_sharp)]:
            fv[k] = v if v is not None else 0
        for k, v in [("max_amp", max_amp), ("max_xbl", max_xbl),
                      ("max_dur_ms", max_dur_ms), ("max_conf", max_conf),
                      ("max_snr", max_snr), ("max_sharp", max_sharp)]:
            fv[k] = v
        state.extra["trs_filter_values"] = fv
    else:
        filt_on = state.extra.get("trs_filter_on", True)

    rec = state.recording
    annotations = _get_annotations(state)
    browse_annotations = _filter_by_channel(annotations, ch_filter)

    start_sec = float(start_val or 0)
    window_sec = float(window_val or 30)

    if selected_channels and len(selected_channels) > 0:
        channels = [ch for ch in selected_channels if 0 <= ch < rec.n_channels]
    else:
        channels = list(range(rec.n_channels))
    if not channels:
        channels = list(range(rec.n_channels))

    state.extra["trs_browse_window"] = window_sec

    fig = _build_browse_figure(
        rec, browse_annotations, state,
        start_sec=start_sec, window_sec=window_sec,
        selected_channels=channels,
        add_spike_active=bool(add_active),
        remove_spike_active=bool(remove_active),
    )
    return fig


@callback(
    Output("trs-browse-annotate-channel", "value"),
    Input("trs-browse-annotate-channel-vis", "value"),
    prevent_initial_call=True,
)
def trs_sync_annotate_channel(val):
    """Sync visible channel dropdown to hidden one used by add-spike callback."""
    return val


@callback(
    Output("trs-add-spike-btn", "active"),
    Output("trs-remove-spike-btn", "active"),
    Output("trs-browse-status", "children"),
    Output("trs-browse-graph", "selectedData"),
    Output("trs-annotations-version", "data"),
    Input("trs-add-spike-btn", "n_clicks"),
    Input("trs-remove-spike-btn", "n_clicks"),
    Input("trs-browse-graph", "selectedData"),
    Input("trs-browse-graph", "clickData"),
    State("trs-add-spike-btn", "active"),
    State("trs-remove-spike-btn", "active"),
    State("trs-browse-annotate-channel", "value"),
    State("trs-annotations-version", "data"),
    State("trs-filter-toggle", "value"),
    State("trs-min-conf", "value"),
    State("trs-min-amp", "value"),
    State("trs-min-snr", "value"),
    State("trs-channel-filter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def trs_handle_add_remove_spike(add_clicks, remove_clicks,
                                selected_data, click_data,
                                add_active, remove_active,
                                annotate_channel, ann_version,
                                filt_on, min_conf, min_amp, min_snr,
                                ch_filter, sid):
    """Handle Add/Remove spike button toggles and actions."""
    ann_version = ann_version or 0

    triggered_props = [t["prop_id"] for t in ctx.triggered] if ctx.triggered else []
    trigger_id = ctx.triggered_id

    # Toggle buttons (mutual exclusion)
    if trigger_id == "trs-add-spike-btn":
        new_add = not add_active
        return new_add, False, no_update, no_update, no_update

    if trigger_id == "trs-remove-spike-btn":
        new_remove = not remove_active
        return False, new_remove, no_update, no_update, no_update

    # Add spike via selection
    is_selection = any("selectedData" in p for p in triggered_props)
    is_click = any("clickData" in p for p in triggered_props)

    if is_selection and selected_data is not None:
        if not add_active:
            return no_update, no_update, no_update, no_update, no_update

        state = server_state.get_session(sid)
        if state.recording is None:
            return no_update, no_update, no_update, no_update, no_update

        rec = state.recording

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
        if offset - onset < 0.005:
            return no_update, no_update, alert("Selection too small (< 5ms)", "warning"), None, no_update

        channel = int(annotate_channel) if annotate_channel is not None else 0

        # Assign next available event_id
        annotations = _get_annotations(state)
        existing_ids = [a.event_id for a in annotations if a.event_id > 0]
        for ev in (state.spike_events or []):
            if ev.event_id > 0:
                existing_ids.append(ev.event_id)
        next_id = max(existing_ids) + 1 if existing_ids else 1

        new_ann = AnnotatedEvent(
            file_path=rec.source_path or "",
            animal_id=state.extra.get("trs_animal_id", ""),
            annotator=state.extra.get("trs_annotator", ""),
            onset_sec=onset,
            offset_sec=offset,
            channel=channel,
            label="confirmed",
            source="manual",
            event_type="spike",
            annotated_at=datetime.now(timezone.utc).isoformat(),
            event_id=next_id,
        )

        annotations.append(new_ann)
        annotations.sort(key=lambda e: (e.channel, e.onset_sec))
        _auto_save(state, annotations)

        # Propagate to spike_events
        from eeg_seizure_analyzer.detection.base import DetectedEvent
        new_det = DetectedEvent(
            onset_sec=onset,
            offset_sec=offset,
            duration_sec=offset - onset,
            channel=channel,
            event_type="spike",
            confidence=1.0,
            animal_id=state.extra.get("trs_animal_id", ""),
            event_id=next_id,
            source="manual",
        )

        state.spike_events.append(new_det)
        state.spike_events.sort(key=lambda e: (e.channel, e.onset_sec))
        state.detected_events = list(state.seizure_events) + state.spike_events
        _save_spike_detection_file(state)

        msg = f"Added manual spike: {onset:.3f}s \u2013 {offset:.3f}s on Ch {channel}"
        return False, False, alert(msg, "success"), None, ann_version + 1

    # Remove spike via click
    if is_click and click_data is not None:
        if not remove_active:
            return no_update, no_update, no_update, no_update, no_update

        state = server_state.get_session(sid)
        if state.recording is None:
            return no_update, no_update, no_update, no_update, no_update

        points = click_data.get("points", [])
        if not points:
            return no_update, no_update, no_update, no_update, no_update

        click_x = float(points[0].get("x", 0))

        annotations = _get_annotations(state)
        visible = _filter_by_channel(annotations, ch_filter)

        best = None
        best_dur = float("inf")
        for ann in visible:
            if ann.onset_sec <= click_x <= ann.offset_sec:
                dur = ann.offset_sec - ann.onset_sec
                if dur < best_dur:
                    best = ann
                    best_dur = dur

        if best is None:
            return no_update, no_update, alert("No spike found at click position", "warning"), no_update, no_update

        annotations = [
            a for a in annotations
            if not (a.onset_sec == best.onset_sec and
                    a.channel == best.channel and
                    a.source == best.source and
                    a.offset_sec == best.offset_sec)
        ]
        _auto_save(state, annotations)

        # Also remove from spike_events
        state.spike_events = [
            ev for ev in state.spike_events
            if not (abs(ev.onset_sec - best.onset_sec) < 0.01 and
                    ev.channel == best.channel)
        ]
        state.detected_events = list(state.seizure_events) + state.spike_events
        _save_spike_detection_file(state)

        msg = f"Removed spike: {best.onset_sec:.3f}s \u2013 {best.offset_sec:.3f}s on Ch {best.channel}"
        return False, False, alert(msg, "info"), no_update, ann_version + 1

    return no_update, no_update, no_update, no_update, no_update
