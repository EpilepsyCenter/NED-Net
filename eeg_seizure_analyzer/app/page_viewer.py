"""EEG Viewer page: scrollable multi-channel time series with activity overlay."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from eeg_seizure_analyzer.app.components import (
    _get_store,
    channel_selector,
    persist_restore,
    persist_save,
    persist_widget_callback,
    sidebar_param,
    require_recording,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter, notch_filter

# Widget keys managed by persist_restore / persist_save (non-sidebar_param widgets)
_VIEWER_KEYS = [
    "viewer_start", "viewer_filter", "viewer_filt_low", "viewer_filt_high",
    "viewer_notch", "viewer_notch_freq", "viewer_show_events",
    "viewer_show_spikes", "viewer_show_baseline", "viewer_show_threshold",
    "viewer_act_yrange",
]


def _minmax_downsample(time_arr: np.ndarray, data: np.ndarray, target_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max downsampling that preserves spike morphology.

    For each bucket of samples, keeps both the min and max values,
    so spikes are never lost regardless of zoom level.
    """
    n = len(data)
    if n <= target_points:
        return time_arr, data

    # Each bucket produces 2 points (min and max)
    n_buckets = target_points // 2
    if n_buckets < 1:
        n_buckets = 1
    bucket_size = n // n_buckets

    times_out = []
    data_out = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = min(start + bucket_size, n)
        chunk = data[start:end]
        t_chunk = time_arr[start:end]

        min_idx = np.argmin(chunk)
        max_idx = np.argmax(chunk)

        # Preserve temporal order (min before max or vice versa)
        if min_idx <= max_idx:
            times_out.extend([t_chunk[min_idx], t_chunk[max_idx]])
            data_out.extend([chunk[min_idx], chunk[max_idx]])
        else:
            times_out.extend([t_chunk[max_idx], t_chunk[min_idx]])
            data_out.extend([chunk[max_idx], chunk[min_idx]])

    return np.array(times_out), np.array(data_out)


def render():
    st.header("EEG Viewer")
    recording = require_recording()
    act_rec = st.session_state.get("activity_recording")
    pairings = st.session_state.get("channel_pairings")

    # Restore persisted values before rendering widgets
    persist_restore(_VIEWER_KEYS)

    # ── Compute a sensible default Y range from the recording ────────
    store = _get_store()
    if "viewer_yrange" not in store or store["viewer_yrange"] is None:
        n_samp = min(int(10 * recording.fs), recording.n_samples)
        ptps = [
            float(np.ptp(recording.data[i, :n_samp]))
            for i in range(recording.n_channels)
        ]
        auto = float(np.median(ptps)) * 1.5 if ptps else 1.0
        # Round to nearest "nice" value
        nice = [float(m * 10 ** e) for e in range(-3, 6) for m in [1, 2, 5]]
        auto = float(min(nice, key=lambda v: abs(v - auto)))
        store["viewer_yrange"] = auto
        st.session_state["viewer_yrange"] = auto

    # Determine the signal unit from the recording
    unit_label = ""
    if recording.units:
        unit_label = recording.units[0] if len(recording.units) > 0 else ""

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.subheader("Viewer Settings")
    channels = channel_selector(recording, key="viewer_channels")

    window_sec = sidebar_param(
        "Window width (s)", 1.0, 600.0, 1.0, "viewer_window",
        help_text="Horizontal time span visible in the plot.",
    )

    # Clamp max_start based on window
    max_start = max(0.5, recording.duration_sec - window_sec)
    current_start = st.session_state.get("viewer_start", 0.0)
    if current_start > max_start:
        st.session_state["viewer_start"] = max_start

    start_sec = st.sidebar.number_input(
        "Start time (s)", 0.0, max_start, step=1.0, key="viewer_start",
        on_change=persist_widget_callback, args=("viewer_start",),
    )

    # Y range per channel (real units)
    y_range_max = float(store["viewer_yrange"]) * 20.0  # generous upper limit
    y_range_step = float(store["viewer_yrange"]) / 20.0
    y_range = sidebar_param(
        f"Y range per channel ({unit_label})" if unit_label else "Y range per channel",
        0.01, y_range_max, y_range_step, "viewer_yrange",
        help_text="Vertical amplitude range per channel strip. "
                  "Stays fixed while navigating. Decrease to zoom in, increase to zoom out.",
    )

    # Activity Y range (shown only if activity channels exist)
    if act_rec is not None:
        act_unit = act_rec.units[0] if act_rec.units else ""
        act_label_str = f"Activity Y max ({act_unit})" if act_unit else "Activity Y max"
        # Auto-compute default if not set
        if "viewer_act_yrange" not in store or store["viewer_act_yrange"] is None:
            n_samp_act = min(int(10 * act_rec.fs), act_rec.n_samples)
            act_ptp = float(np.max([
                np.ptp(act_rec.data[i, :n_samp_act])
                for i in range(act_rec.n_channels)
            ])) * 1.5 if act_rec.n_channels > 0 else 1.0
            store["viewer_act_yrange"] = max(act_ptp, 0.01)
        sidebar_param(
            act_label_str, 0.001, 100.0, 0.01, "viewer_act_yrange",
            help_text="Maximum Y value for the activity/movement channel panel.",
        )

    plot_height = sidebar_param(
        "Plot height (px)", 300, 3000, 50, "viewer_height",
    )
    plot_height = int(plot_height)

    # Optional filtering for display
    apply_filter = st.sidebar.checkbox(
        "Apply bandpass filter", key="viewer_filter",
        on_change=persist_widget_callback, args=("viewer_filter",),
    )
    if apply_filter:
        filter_low = st.sidebar.number_input(
            "Low (Hz)", 0.5, 500.0, key="viewer_filt_low",
            on_change=persist_widget_callback, args=("viewer_filt_low",),
        )
        filter_high = st.sidebar.number_input(
            "High (Hz)", 1.0, 1000.0, key="viewer_filt_high",
            on_change=persist_widget_callback, args=("viewer_filt_high",),
        )

    apply_notch = st.sidebar.checkbox(
        "Apply notch filter", key="viewer_notch",
        on_change=persist_widget_callback, args=("viewer_notch",),
    )
    notch_freq = 50.0
    if apply_notch:
        notch_freq = st.sidebar.selectbox(
            "Notch freq (Hz)", [50.0, 60.0], key="viewer_notch_freq",
            on_change=persist_widget_callback, args=("viewer_notch_freq",),
        )

    # Show detected events if available
    show_events = st.sidebar.checkbox(
        "Show detected events", key="viewer_show_events",
        on_change=persist_widget_callback, args=("viewer_show_events",),
    )

    # Overlay toggles (only if detection data exists)
    has_detection = "st_detection_info" in st.session_state
    show_spikes = False
    show_baseline = False
    show_threshold = False
    if has_detection:
        show_spikes = st.sidebar.checkbox(
            "Show detected spikes", key="viewer_show_spikes",
            on_change=persist_widget_callback, args=("viewer_show_spikes",),
        )
        show_baseline = st.sidebar.checkbox(
            "Show baseline", key="viewer_show_baseline",
            on_change=persist_widget_callback, args=("viewer_show_baseline",),
        )
        show_threshold = st.sidebar.checkbox(
            "Show threshold", key="viewer_show_threshold",
            on_change=persist_widget_callback, args=("viewer_show_threshold",),
        )

    # Save / recall defaults
    st.sidebar.divider()
    def_col1, def_col2 = st.sidebar.columns(2)
    if def_col1.button("💾 Save defaults", key="viewer_save_defaults"):
        from eeg_seizure_analyzer.app.components import save_user_defaults
        path = save_user_defaults()
        st.sidebar.success(f"Saved to {path}")
    if def_col2.button("📂 Recall defaults", key="viewer_recall_defaults"):
        from eeg_seizure_analyzer.app.components import apply_user_defaults
        if apply_user_defaults():
            st.sidebar.success("Defaults loaded")
            st.rerun()
        else:
            st.sidebar.warning("No saved defaults found")

    # Save non-sidebar_param keys
    persist_save(_VIEWER_KEYS)

    if not channels:
        st.info("Select at least one channel.")
        return

    # ── Navigation buttons ────────────────────────────────────────────

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        st.button("\u23ea", key="nav_back_big", on_click=_navigate,
                  args=(recording, window_sec, -5))
    with col2:
        st.button("\u25c0", key="nav_back", on_click=_navigate,
                  args=(recording, window_sec, -1))
    with col3:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px;'>"
            f"<b>{start_sec:.1f}s \u2013 {start_sec + window_sec:.1f}s</b></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.button("\u25b6", key="nav_fwd", on_click=_navigate,
                  args=(recording, window_sec, 1))
    with col5:
        st.button("\u23e9", key="nav_fwd_big", on_click=_navigate,
                  args=(recording, window_sec, 5))

    # ── Determine target points for downsampling ──────────────────────
    # Tie to plot width — assume ~1200px effective width, 2 points/px
    target_points = 2400

    # ── Check if we have activity channels to show ────────────────────
    show_activity = act_rec is not None and pairings is not None
    has_paired_act = False
    if show_activity:
        # Check if any displayed EEG channel has a paired activity channel
        for ch_idx in channels:
            for p in pairings:
                if p.eeg_index == ch_idx and p.activity_index is not None:
                    has_paired_act = True
                    break
            if has_paired_act:
                break

    # ── Build the figure ──────────────────────────────────────────────
    end_sec = start_sec + window_sec

    if has_paired_act:
        # Two-panel layout: EEG on top, activity on bottom
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
        )
    else:
        fig = go.Figure()

    n_ch = len(channels)
    spacing = float(y_range)
    channel_offsets = {}
    channel_displayed_data = {}  # cache filtered data per channel for spike overlay

    # ── EEG traces ────────────────────────────────────────────────────
    start_idx = int(start_sec * recording.fs)
    end_idx = min(int(end_sec * recording.fs), recording.n_samples)

    for i, ch_idx in enumerate(channels):
        data = recording.data[ch_idx, start_idx:end_idx].copy()

        if apply_filter:
            data = bandpass_filter(data, recording.fs, filter_low, filter_high)
        if apply_notch:
            data = notch_filter(data, recording.fs, notch_freq)

        offset = -i * spacing
        channel_offsets[ch_idx] = offset
        channel_displayed_data[ch_idx] = data  # store for spike Y lookup
        time_axis = np.linspace(start_sec, end_sec, len(data))

        # Min/max downsample for performance
        ds_time, ds_data = _minmax_downsample(time_axis, data + offset, target_points)

        trace = go.Scattergl(
            x=ds_time,
            y=ds_data,
            mode="lines",
            name=recording.channel_names[ch_idx],
            line=dict(width=0.8),
        )

        if has_paired_act:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    # ── Scale bar (real units) ────────────────────────────────────────
    half_spacing = spacing / 2.0
    nice_values = [m * 10 ** e for e in range(-3, 6) for m in [1, 2, 5]]
    scale_val = min(nice_values, key=lambda v: abs(v - half_spacing))
    if scale_val <= 0:
        scale_val = half_spacing

    bar_x = end_sec - window_sec * 0.02
    last_offset = -(n_ch - 1) * spacing
    bar_y_center = last_offset - spacing * 0.7
    bar_y_top = bar_y_center + scale_val / 2
    bar_y_bot = bar_y_center - scale_val / 2

    scale_trace = go.Scatter(
        x=[bar_x, bar_x],
        y=[bar_y_bot, bar_y_top],
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False,
        hoverinfo="skip",
    )
    if has_paired_act:
        fig.add_trace(scale_trace, row=1, col=1)
    else:
        fig.add_trace(scale_trace)

    if scale_val >= 1:
        scale_text = f"{scale_val:.0f} {unit_label}" if unit_label else f"{scale_val:.0f}"
    else:
        scale_text = f"{scale_val:.2g} {unit_label}" if unit_label else f"{scale_val:.2g}"

    fig.add_annotation(
        x=bar_x, y=bar_y_center, text=scale_text,
        showarrow=False, xanchor="right", xshift=-8,
        font=dict(size=11, color="black"),
        row=1 if has_paired_act else None,
        col=1 if has_paired_act else None,
    )

    # ── Overlay annotations ──────────────────────────────────────────
    for ann in recording.annotations:
        if start_sec <= ann.onset_sec <= end_sec:
            fig.add_vline(
                x=ann.onset_sec,
                line=dict(color="green", width=1, dash="dash"),
                annotation_text=ann.text[:20],
                annotation_position="top",
            )
            if ann.duration_sec:
                fig.add_vrect(
                    x0=ann.onset_sec,
                    x1=ann.onset_sec + ann.duration_sec,
                    fillcolor="green", opacity=0.1, line_width=0,
                )

    # ── Overlay detected events (per-channel strips) ────────────────
    # Use distinct colors per channel so multi-channel events are clear
    _event_colors = [
        "rgba(255, 60, 60, 0.18)",   # red
        "rgba(60, 60, 255, 0.18)",   # blue
        "rgba(60, 180, 60, 0.18)",   # green
        "rgba(200, 120, 0, 0.18)",   # orange
        "rgba(160, 60, 200, 0.18)",  # purple
        "rgba(0, 180, 180, 0.18)",   # teal
    ]
    _event_border_colors = [
        "red", "blue", "green", "darkorange", "purple", "teal",
    ]

    if show_events and "detected_events" in st.session_state:
        events = st.session_state["detected_events"]
        for event in events:
            if event.offset_sec < start_sec or event.onset_sec > end_sec:
                continue
            # Only show if the event's channel is currently displayed
            if event.channel not in channel_offsets:
                continue

            ch_offset = channel_offsets[event.channel]
            half = spacing / 2.0
            y0 = ch_offset - half
            y1 = ch_offset + half

            # Color by channel position
            ch_pos = list(channel_offsets.keys()).index(event.channel)
            fill_c = _event_colors[ch_pos % len(_event_colors)]
            border_c = _event_border_colors[ch_pos % len(_event_border_colors)]

            # Use add_shape with yref="y" for bounded rectangle
            yref = "y" if not has_paired_act else "y"
            fig.add_shape(
                type="rect",
                x0=max(event.onset_sec, start_sec),
                x1=min(event.offset_sec, end_sec),
                y0=y0, y1=y1,
                fillcolor=fill_c,
                line=dict(color=border_c, width=1),
                layer="below",
                yref=yref,
            )

    # ── Overlay spikes / baseline / threshold ────────────────────────
    if has_detection and (show_spikes or show_baseline or show_threshold):
        det_info_all = st.session_state.get("st_detection_info", {})
        for ch_idx in channels:
            det_info = det_info_all.get(ch_idx)
            if det_info is None:
                continue
            offset = channel_offsets[ch_idx]
            bl_mean = det_info.get("baseline_mean", 0)
            bl_std = det_info.get("baseline_std", 0)
            threshold = det_info.get("threshold", 0)

            if show_baseline:
                # Baseline mean as horizontal band (±1 std shown as shaded)
                for sign in [1, -1]:
                    bl_val = sign * bl_mean + offset
                    bl_trace = go.Scatter(
                        x=[start_sec, end_sec], y=[bl_val, bl_val],
                        mode="lines",
                        line=dict(color="green", width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    )
                    if has_paired_act:
                        fig.add_trace(bl_trace, row=1, col=1)
                    else:
                        fig.add_trace(bl_trace)

            if show_threshold:
                for sign in [1, -1]:
                    thr_val = sign * threshold + offset
                    thr_trace = go.Scatter(
                        x=[start_sec, end_sec], y=[thr_val, thr_val],
                        mode="lines",
                        line=dict(color="orange", width=1.5, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    )
                    if has_paired_act:
                        fig.add_trace(thr_trace, row=1, col=1)
                    else:
                        fig.add_trace(thr_trace)

            if show_spikes:
                spike_times = det_info.get("all_spike_times", [])
                spike_samples = det_info.get("all_spike_samples", [])
                displayed = channel_displayed_data.get(ch_idx)
                # Filter spikes in current view window
                visible_idx = [
                    i for i, t in enumerate(spike_times)
                    if start_sec <= t <= end_sec
                ]
                if visible_idx and displayed is not None:
                    sp_times = [spike_times[i] for i in visible_idx]
                    # Read Y from displayed (filtered) data so dots sit on peaks
                    sp_y = []
                    for i in visible_idx:
                        samp = spike_samples[i]
                        local_idx = samp - start_idx
                        if 0 <= local_idx < len(displayed):
                            sp_y.append(float(displayed[local_idx]) + offset)
                        else:
                            sp_y.append(offset)

                    spike_trace = go.Scatter(
                        x=sp_times, y=sp_y,
                        mode="markers",
                        marker=dict(color="red", size=5, symbol="circle"),
                        showlegend=False,
                        hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
                    )
                    if has_paired_act:
                        fig.add_trace(spike_trace, row=1, col=1)
                    else:
                        fig.add_trace(spike_trace)

    # ── Activity traces (bottom panel, native resolution) ────────────
    if has_paired_act:
        act_start_idx = int(start_sec * act_rec.fs)
        act_end_idx = min(int(end_sec * act_rec.fs), act_rec.n_samples)

        act_labels = []
        for ch_idx in channels:
            for p in pairings:
                if p.eeg_index == ch_idx and p.activity_index is not None:
                    act_data = act_rec.data[p.activity_index, act_start_idx:act_end_idx]
                    act_time = np.linspace(start_sec, end_sec, len(act_data))

                    fig.add_trace(
                        go.Scattergl(
                            x=act_time,
                            y=act_data,
                            mode="lines",
                            name=f"Act: {p.activity_label}",
                            line=dict(width=1),
                        ),
                        row=2, col=1,
                    )
                    act_labels.append(p.activity_label)

        # Activity panel unit label
        act_unit = ""
        if act_rec.units:
            act_unit = act_rec.units[0]

    # ── Layout ────────────────────────────────────────────────────────
    y_tickvals = [channel_offsets[ch] for ch in channels]
    y_ticktext = [recording.channel_names[ch] for ch in channels]
    y_upper = spacing / 2
    y_lower = -(n_ch - 1) * spacing - spacing * 1.5

    # uirevision: tied to navigation so zoom resets on scroll,
    # but persists across overlay checkbox toggles
    viewer_ui_rev = f"{start_sec:.2f}_{window_sec:.2f}"

    if has_paired_act:
        fig.update_xaxes(title_text="Time (s)", fixedrange=False, row=2, col=1)
        fig.update_xaxes(fixedrange=False, row=1, col=1)

        fig.update_yaxes(
            tickvals=y_tickvals, ticktext=y_ticktext,
            zeroline=False, showgrid=False, fixedrange=False,
            range=[y_lower, y_upper],
            row=1, col=1,
        )
        act_y_max = float(store.get("viewer_act_yrange", 1.0))
        fig.update_yaxes(
            title_text=f"Activity ({act_unit})" if act_unit else "Activity",
            zeroline=False, fixedrange=False,
            range=[0, act_y_max],
            row=2, col=1,
        )

        fig.update_layout(
            height=plot_height,
            uirevision=viewer_ui_rev,
            showlegend=False,
            margin=dict(l=80, r=10, t=30, b=40),
            dragmode="zoom",
        )
    else:
        fig.update_layout(
            height=plot_height,
            uirevision=viewer_ui_rev,
            xaxis=dict(title="Time (s)", fixedrange=False),
            yaxis=dict(
                tickvals=y_tickvals,
                ticktext=y_ticktext,
                zeroline=False,
                showgrid=False,
                fixedrange=False,
                range=[y_lower, y_upper],
            ),
            showlegend=False,
            margin=dict(l=80, r=10, t=30, b=40),
            dragmode="zoom",
        )

    st.plotly_chart(fig, use_container_width=True, key="main_viewer_chart",
                    config={"scrollZoom": True, "displayModeBar": True})


def _navigate(recording, window_sec: float, multiplier: int):
    """Navigation callback — updates viewer_start before widgets render on rerun."""
    current = st.session_state.get("viewer_start", 0.0)
    max_start = max(0.0, recording.duration_sec - window_sec)
    new_val = current + window_sec * multiplier
    new_val = max(0.0, min(max_start, new_val))
    st.session_state["viewer_start"] = new_val
    _get_store()["viewer_start"] = new_val
