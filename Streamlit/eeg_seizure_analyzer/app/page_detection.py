"""Seizure Detection page: run detectors with tunable parameters, inspect events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from eeg_seizure_analyzer.app.components import (
    channel_selector,
    persist_restore,
    persist_save,
    persist_widget_callback,
    sidebar_param,
    require_recording,
)
from eeg_seizure_analyzer.config import SeizureDetectionParams, SpikeDetectionParams
from eeg_seizure_analyzer.detection.burden import compute_burden, compute_spike_rate
from eeg_seizure_analyzer.detection.seizure import SeizureDetector
from eeg_seizure_analyzer.detection.spike import SpikeDetector
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


# ── Parameter help text ──────────────────────────────────────────────

SEIZURE_PARAM_HELP = {
    "bandpass_low": (
        "**Bandpass low (Hz)** — Lower edge of the bandpass filter applied before "
        "feature extraction. Removes slow drift. Default 1 Hz is standard for mouse "
        "EEG. Raise to 2-3 Hz if you see baseline wander causing false detections."
    ),
    "bandpass_high": (
        "**Bandpass high (Hz)** — Upper edge of the bandpass filter. Default 50 Hz "
        "captures the main seizure frequency range in mice (theta–gamma). Raise to "
        "70-100 Hz if your seizures have prominent high-gamma components."
    ),
    "line_length_window": (
        "**Line-length window (s)** — Duration of the sliding window for computing "
        "line-length (sum of sample-to-sample differences). Longer windows smooth "
        "the signal and reduce sensitivity to brief bursts. Shorter windows catch "
        "faster onsets. Default: 2s."
    ),
    "line_length_threshold": (
        "**Line-length threshold (z-score)** — How many standard deviations above "
        "baseline the line-length must exceed to flag a candidate seizure. "
        "**Lower = more sensitive** (catches more events, but more false positives). "
        "**Higher = more specific** (fewer false alarms, may miss mild seizures). "
        "Default: 5.0."
    ),
    "energy_threshold": (
        "**Energy threshold (z-score)** — Same concept as line-length threshold, "
        "but for signal energy (sum of squared amplitudes). Both line-length AND "
        "energy must exceed their thresholds simultaneously, which reduces false "
        "positives. Default: 4.0."
    ),
    "onset_offset": (
        "**Onset/offset boundary (z-score)** — After finding a candidate seizure "
        "above the main thresholds, the algorithm walks the onset backward and "
        "offset forward until both features drop below this lower threshold. "
        "This refines the exact seizure boundaries. Lower values = wider events. "
        "Default: 2.0."
    ),
    "min_duration": (
        "**Min duration (s)** — Events shorter than this are discarded. Filters out "
        "brief artefacts or muscle bursts. For mouse seizures, 5s is typical. Set "
        "lower (2-3s) for absence-like events, higher (10s+) if only interested in "
        "generalized seizures."
    ),
    "merge_gap": (
        "**Merge gap (s)** — If two candidate events are separated by less than this "
        "gap, they are merged into one. Prevents a single seizure with brief "
        "amplitude fluctuations from being split. Default: 1s."
    ),
    "baseline_method": (
        "**Baseline method** — How the 'normal' signal statistics are estimated:\n"
        "- **robust** (default): Uses the median and MAD of the entire recording. "
        "Robust to seizure contamination since median ignores outliers.\n"
        "- **first_n**: Uses the mean and std of the first N minutes. Assumes the "
        "recording starts with a clean baseline.\n"
        "- **manual**: Uses statistics from a user-specified time range."
    ),
    "baseline_duration": (
        "**Baseline duration (min)** — How many minutes of recording to use for "
        "baseline calculation when method is 'first_n'. Only the first N minutes "
        "are used. Default: 5 min."
    ),
}

SPIKE_PARAM_HELP = {
    "bandpass": (
        "**Spike bandpass (Hz)** — Filter range for spike detection. 10-70 Hz "
        "captures the sharp transient (high freq) and slow wave (low freq) "
        "components of interictal spikes. Narrowing the range can help if "
        "there is high-frequency noise."
    ),
    "amplitude_threshold": (
        "**Spike amplitude threshold (z-score)** — Teager energy must exceed this "
        "many standard deviations above baseline. Teager energy is proportional to "
        "amplitude × frequency, making it ideal for detecting sharp spikes. "
        "**Lower = more spikes detected.** Default: 4.0."
    ),
    "max_duration": (
        "**Max spike duration (ms)** — Spikes wider than this are rejected. "
        "True interictal spikes are typically 20-70ms. Increase if your spikes "
        "have prominent slow-wave components."
    ),
    "refractory": (
        "**Refractory period (ms)** — Minimum time between two spikes. Prevents "
        "double-counting. 200ms is standard; decrease if you see high-frequency "
        "spike bursts being under-counted."
    ),
}


# Keys only for widgets NOT handled by sidebar_param (selectboxes, etc.)
_SELECT_KEYS = ["sz_baseline", "sp_baseline"]
_CB = persist_widget_callback  # shorthand


def render():
    st.header("Seizure & Spike Detection")
    recording = require_recording()

    # Restore selectbox keys
    persist_restore(_SELECT_KEYS)

    channels = channel_selector(recording, key="det_channels")

    # ── Parameter help expander ───────────────────────────────────────

    with st.expander("Parameter guide", expanded=False):
        tab_sz, tab_sp = st.tabs(["Seizure parameters", "Spike parameters"])
        with tab_sz:
            for help_text in SEIZURE_PARAM_HELP.values():
                st.markdown(help_text)
                st.markdown("---")
        with tab_sp:
            for help_text in SPIKE_PARAM_HELP.values():
                st.markdown(help_text)
                st.markdown("---")

    # ── Seizure detection parameters (sidebar) ────────────────────────

    st.sidebar.subheader("Seizure Parameters")

    sz_bp_low = sidebar_param(
        "Bandpass low (Hz)", 0.5, 100.0, 0.5, "sz_bp_low",
        help_text="Lower filter edge. Removes slow drift.",
    )
    sz_bp_high = sidebar_param(
        "Bandpass high (Hz)", 1.0, 500.0, 1.0, "sz_bp_high",
        help_text="Upper filter edge. Captures seizure frequencies.",
    )
    sz_ll_win = sidebar_param(
        "Line-length window (s)", 0.5, 10.0, 0.5, "sz_ll_win",
        help_text="Sliding window for line-length. Longer = smoother.",
    )
    sz_ll_thr = sidebar_param(
        "Line-length threshold (z)", 1.0, 30.0, 0.5, "sz_ll_thr",
        help_text="Lower = more sensitive, higher = more specific.",
    )
    sz_en_thr = sidebar_param(
        "Energy threshold (z)", 1.0, 30.0, 0.5, "sz_en_thr",
        help_text="Both LL and energy must exceed thresholds.",
    )
    sz_oo_thr = sidebar_param(
        "Onset/offset boundary (z)", 0.5, 15.0, 0.5, "sz_oo_thr",
        help_text="Lower boundary for refining seizure edges.",
    )
    sz_min_dur = sidebar_param(
        "Min duration (s)", 0.5, 60.0, 0.5, "sz_min_dur",
        help_text="Events shorter than this are discarded.",
    )
    sz_merge = sidebar_param(
        "Merge gap (s)", 0.0, 30.0, 0.5, "sz_merge",
        help_text="Events closer than this gap are merged.",
    )

    sz_baseline = st.sidebar.selectbox(
        "Baseline method",
        ["robust", "first_n", "manual"],
        key="sz_baseline",
        help="How 'normal' statistics are estimated.",
        on_change=_CB, args=("sz_baseline",),
    )
    sz_bl_dur = sidebar_param(
        "Baseline duration (min)", 1.0, 120.0, 1.0, "sz_bl_dur",
        help_text="Used only with 'first_n' method.",
    )

    sz_params = SeizureDetectionParams(
        bandpass_low=sz_bp_low,
        bandpass_high=sz_bp_high,
        line_length_window_sec=sz_ll_win,
        line_length_threshold_zscore=sz_ll_thr,
        energy_threshold_zscore=sz_en_thr,
        onset_offset_zscore=sz_oo_thr,
        min_duration_sec=sz_min_dur,
        merge_gap_sec=sz_merge,
        baseline_method=sz_baseline,
        baseline_duration_min=sz_bl_dur,
    )

    # ── Spike detection parameters (sidebar) ──────────────────────────

    st.sidebar.subheader("Spike Parameters")

    sp_bp_low = sidebar_param(
        "Spike bandpass low (Hz)", 1.0, 100.0, 1.0, "sp_bp_low",
        help_text="Lower edge for spike bandpass filter.",
    )
    sp_bp_high = sidebar_param(
        "Spike bandpass high (Hz)", 10.0, 500.0, 1.0, "sp_bp_high",
        help_text="Upper edge for spike bandpass filter.",
    )
    sp_amp_thr = sidebar_param(
        "Spike amplitude threshold (z)", 1.0, 30.0, 0.5, "sp_amp_thr",
        help_text="Teager energy z-score threshold.",
    )
    sp_max_dur = sidebar_param(
        "Max spike duration (ms)", 10.0, 500.0, 5.0, "sp_max_dur",
        help_text="Spikes wider than this are rejected.",
    )
    sp_refract = sidebar_param(
        "Refractory period (ms)", 10.0, 2000.0, 10.0, "sp_refract",
        help_text="Min time between consecutive spikes.",
    )

    sp_baseline = st.sidebar.selectbox(
        "Spike baseline method",
        ["robust", "first_n", "manual"],
        key="sp_baseline",
        help="How baseline is computed for spike Teager energy.",
        on_change=_CB, args=("sp_baseline",),
    )

    sp_params = SpikeDetectionParams(
        bandpass_low=sp_bp_low,
        bandpass_high=sp_bp_high,
        amplitude_threshold_zscore=sp_amp_thr,
        max_duration_ms=sp_max_dur,
        refractory_ms=sp_refract,
        baseline_method=sp_baseline,
    )

    # Save selectbox keys
    persist_save(_SELECT_KEYS)

    # ── Run detection ─────────────────────────────────────────────────

    col1, col2 = st.columns(2)
    run_seizure = col1.button("Detect Seizures", type="primary", key="run_sz")
    run_spikes = col2.button("Detect Spikes", type="primary", key="run_sp")

    if run_seizure:
        with st.spinner("Detecting seizures..."):
            detector = SeizureDetector()
            seizures = detector.detect_all_channels(recording, channels, params=sz_params)
            st.session_state["seizure_events"] = seizures
            spikes = st.session_state.get("spike_events", [])
            st.session_state["detected_events"] = seizures + spikes
        st.success(f"Found {len(seizures)} seizure(s)")

    if run_spikes:
        with st.spinner("Detecting spikes..."):
            detector = SpikeDetector()
            spikes = detector.detect_all_channels(recording, channels, params=sp_params)
            st.session_state["spike_events"] = spikes
            seizures = st.session_state.get("seizure_events", [])
            st.session_state["detected_events"] = seizures + spikes
        st.success(f"Found {len(spikes)} spike(s)")

    # ── Display results ───────────────────────────────────────────────

    _display_seizure_results(recording)
    _display_spike_results(recording)


# ── Seizure results with clickable inspection ─────────────────────────


def _display_seizure_results(recording):
    """Display seizure results with clickable event inspection."""
    seizures = st.session_state.get("seizure_events", [])
    if not seizures:
        return

    st.subheader("Seizure Results")

    # Burden summary
    burden = compute_burden(seizures, recording.duration_sec)

    cols = st.columns(5)
    cols[0].metric("Total Seizures", burden.n_seizures)
    cols[1].metric("Total Seizure Time", f"{burden.total_seizure_time_sec:.1f}s")
    cols[2].metric("Seizures/Hour", f"{burden.seizure_frequency_per_hour:.2f}")
    cols[3].metric("% Time in Seizure", f"{burden.percent_time_in_seizure:.2f}%")
    cols[4].metric("Mean Duration", f"{burden.mean_duration_sec:.1f}s")

    # Severity breakdown
    st.markdown(
        f"**Severity:** Mild: {burden.severity_counts.get('mild', 0)} | "
        f"Moderate: {burden.severity_counts.get('moderate', 0)} | "
        f"Severe: {burden.severity_counts.get('severe', 0)}"
    )

    # Clickable events table
    st.markdown("**Click a seizure to inspect it on the EEG trace:**")

    # Build table data
    table_data = []
    for i, e in enumerate(seizures):
        table_data.append({
            "#": i + 1,
            "Channel": recording.channel_names[e.channel],
            "Onset (s)": round(e.onset_sec, 2),
            "Offset (s)": round(e.offset_sec, 2),
            "Duration (s)": round(e.duration_sec, 2),
            "Severity": e.severity,
            "Peak LL (z)": round(e.features.get("peak_line_length_zscore", 0), 1),
            "Peak Energy (z)": round(e.features.get("peak_energy_zscore", 0), 1),
        })

    df = pd.DataFrame(table_data)

    selection = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="seizure_table_select",
    )

    # Show EEG trace for the selected seizure
    selected_rows = selection.get("selection", {}).get("rows", [])
    if selected_rows:
        sel_idx = selected_rows[0]
        event = seizures[sel_idx]
        _render_event_trace(recording, event, context_sec=10.0, title=f"Seizure #{sel_idx + 1}")

    # Hourly seizure count plot
    if burden.hourly_counts:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(burden.hourly_counts))),
            y=burden.hourly_counts,
            name="Seizures per hour",
        ))
        fig.update_layout(
            title="Seizures Per Hour",
            xaxis_title="Hour",
            yaxis_title="Count",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Spike results with clickable inspection ───────────────────────────


def _display_spike_results(recording):
    """Display spike results with clickable event inspection."""
    spikes = st.session_state.get("spike_events", [])
    if not spikes:
        return

    st.subheader("Interictal Spike Results")

    cols = st.columns(3)
    cols[0].metric("Total Spikes", len(spikes))
    rate_per_min = len(spikes) / (recording.duration_sec / 60) if recording.duration_sec > 0 else 0
    cols[1].metric("Rate", f"{rate_per_min:.2f}/min")
    mean_amp = sum(e.features.get("amplitude", 0) for e in spikes) / len(spikes)
    cols[2].metric("Mean Amplitude", f"{mean_amp:.1f}")

    # Clickable spikes table (show first 200 to keep UI responsive)
    display_spikes = spikes[:200]
    if len(spikes) > 200:
        st.caption(f"Showing first 200 of {len(spikes)} spikes.")

    st.markdown("**Click a spike to inspect it on the EEG trace:**")

    table_data = []
    for i, e in enumerate(display_spikes):
        table_data.append({
            "#": i + 1,
            "Channel": recording.channel_names[e.channel],
            "Time (s)": round(e.features.get("peak_time_sec", e.onset_sec), 2),
            "Amplitude": round(e.features.get("amplitude", 0), 1),
            "Teager (z)": round(e.features.get("teager_zscore", 0), 1),
            "Duration (ms)": round(e.features.get("duration_ms", 0), 1) if e.features.get("duration_ms") else "-",
        })

    df = pd.DataFrame(table_data)

    selection = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="spike_table_select",
    )

    selected_rows = selection.get("selection", {}).get("rows", [])
    if selected_rows:
        sel_idx = selected_rows[0]
        event = display_spikes[sel_idx]
        _render_event_trace(recording, event, context_sec=1.0, title=f"Spike #{sel_idx + 1}")

    # Spike rate over time
    time_bins, rate = compute_spike_rate(spikes, recording.duration_sec, bin_sec=60.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_bins / 60,
        y=rate,
        mode="lines",
        name="Spike rate",
        fill="tozeroy",
    ))
    fig.update_layout(
        title="Spike Rate Over Time",
        xaxis_title="Time (min)",
        yaxis_title="Spikes/min",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Shared event trace renderer ───────────────────────────────────────


def _render_event_trace(recording, event, context_sec: float = 10.0, title: str = "Event"):
    """Render an EEG trace centered on a detected event with highlighting."""
    st.markdown(f"#### {title}")

    # Time window: event ± context
    mid = (event.onset_sec + event.offset_sec) / 2
    window_start = max(0, event.onset_sec - context_sec)
    window_end = min(recording.duration_sec, event.offset_sec + context_sec)

    start_idx = int(window_start * recording.fs)
    end_idx = min(int(window_end * recording.fs), recording.n_samples)

    ch = event.channel
    data = recording.data[ch, start_idx:end_idx].copy()
    time_axis = np.linspace(window_start, window_end, len(data))

    # Create figure with raw trace
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scattergl(
        x=time_axis,
        y=data,
        mode="lines",
        name=recording.channel_names[ch],
        line=dict(color="#1f77b4", width=1),
    ))

    # Highlight the event region
    color = "rgba(255, 60, 60, 0.2)" if event.event_type == "seizure" else "rgba(60, 60, 255, 0.2)"
    border_color = "red" if event.event_type == "seizure" else "blue"

    fig.add_vrect(
        x0=event.onset_sec,
        x1=event.offset_sec,
        fillcolor=color,
        line=dict(color=border_color, width=2),
        annotation_text=event.event_type.capitalize(),
        annotation_position="top left",
        annotation_font_color=border_color,
    )

    # Mark onset and offset with vertical lines
    fig.add_vline(x=event.onset_sec, line=dict(color=border_color, width=1.5, dash="dash"))
    fig.add_vline(x=event.offset_sec, line=dict(color=border_color, width=1.5, dash="dash"))

    fig.update_layout(
        height=300,
        xaxis_title="Time (s)",
        yaxis_title=recording.units[ch] if ch < len(recording.units) else "",
        title=dict(
            text=f"{recording.channel_names[ch]} — {event.onset_sec:.2f}s to {event.offset_sec:.2f}s "
                 f"({event.duration_sec:.2f}s)",
            font_size=13,
        ),
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Event details
    detail_cols = st.columns(4)
    detail_cols[0].markdown(f"**Onset:** {event.onset_sec:.3f}s")
    detail_cols[1].markdown(f"**Offset:** {event.offset_sec:.3f}s")
    detail_cols[2].markdown(f"**Duration:** {event.duration_sec:.3f}s")
    if event.severity:
        detail_cols[3].markdown(f"**Severity:** {event.severity}")
    elif event.features.get("amplitude"):
        detail_cols[3].markdown(f"**Amplitude:** {event.features['amplitude']:.1f}")
