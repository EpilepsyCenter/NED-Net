"""Interictal Spike Detection page: run detector with tunable parameters, inspect events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eeg_seizure_analyzer.app.components import (
    channel_selector,
    persist_restore,
    persist_save,
    persist_widget_callback,
    render_event_trace,
    sidebar_param,
    require_recording,
)
from eeg_seizure_analyzer.config import SpikeDetectionParams
from eeg_seizure_analyzer.detection.burden import compute_spike_rate
from eeg_seizure_analyzer.detection.spike import SpikeDetector


# ── Parameter help text ──────────────────────────────────────────────

SPIKE_PARAM_HELP = {
    "bandpass": (
        "**Spike bandpass (Hz)** — Filter range for spike detection. 10-70 Hz "
        "captures the sharp transient (high freq) and slow wave (low freq) "
        "components of interictal spikes. Narrowing the range can help if "
        "there is high-frequency noise."
    ),
    "amplitude_threshold": (
        "**Spike amplitude threshold (z-score)** — Spike amplitude must exceed "
        "baseline mean + z × std. The baseline is computed from quiet windows "
        "(percentile method). **Lower = more spikes detected.** Default: 4.0."
    ),
    "prominence": (
        "**Spike prominence (× baseline)** — Each spike must stand out from "
        "its local surroundings by at least this multiple of the baseline mean. "
        "Rejects noise peaks riding on oscillations. Default: 1.5."
    ),
    "width": (
        "**Spike width (ms)** — Min and max spike half-width. Rejects "
        "single-sample noise (too narrow) and slow waves (too wide). "
        "Typical interictal spikes are 2-70 ms."
    ),
    "refractory": (
        "**Refractory period (ms)** — Minimum time between two spikes. Prevents "
        "double-counting of biphasic spikes. 200 ms is standard; decrease if "
        "you see high-frequency spike bursts being under-counted."
    ),
    "min_amplitude_uv": (
        "**Min amplitude floor (µV)** — Absolute minimum spike amplitude. "
        "Set to 0 to disable. Useful if your baseline is very low and you "
        "want to reject tiny deflections."
    ),
}


_SELECT_KEYS = ["sp_baseline"]
_CB = persist_widget_callback


def render():
    st.header("Interictal Spike Detection")
    recording = require_recording()

    persist_restore(_SELECT_KEYS)

    channels = channel_selector(recording, key="spike_det_channels")

    # ── Parameter guide ──────────────────────────────────────────────
    with st.expander("Parameter guide", expanded=False):
        for help_text in SPIKE_PARAM_HELP.values():
            st.markdown(help_text)
            st.markdown("---")

    # ── Sidebar parameters ───────────────────────────────────────────
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
        "Amplitude threshold (z-score)", 1.0, 30.0, 0.5, "sp_amp_thr",
        help_text="Spike must exceed mean + z × std of baseline.",
    )
    sp_min_amp_uv = sidebar_param(
        "Min amplitude floor (µV)", 0.0, 500.0, 5.0, "sp_min_amp_uv",
        help_text="Absolute minimum spike amplitude. 0 = disabled.",
    )
    sp_prom_x = sidebar_param(
        "Prominence (× baseline)", 0.5, 10.0, 0.1, "sp_prom_x",
        help_text="Spike must stand out from local context by this × baseline.",
    )
    sp_max_dur = sidebar_param(
        "Max spike width (ms)", 10.0, 500.0, 5.0, "sp_max_dur",
        help_text="Spikes wider than this are rejected.",
    )
    sp_min_dur = sidebar_param(
        "Min spike width (ms)", 0.5, 50.0, 0.5, "sp_min_dur",
        help_text="Spikes narrower than this are rejected (noise).",
    )
    sp_refract = sidebar_param(
        "Refractory period (ms)", 10.0, 2000.0, 10.0, "sp_refract",
        help_text="Min time between consecutive spikes.",
    )

    # ── Baseline section ─────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Baseline")

    sp_baseline = st.sidebar.selectbox(
        "Baseline method",
        ["percentile", "rolling", "first_n"],
        key="sp_baseline",
        help="How baseline (mean, std) is computed for spike detection.",
        on_change=_CB, args=("sp_baseline",),
    )

    sp_bl_percentile = sidebar_param(
        "Baseline percentile", 1, 50, 1, "sp_bl_percentile",
        help_text="Nth percentile of RMS windows used for quiet baseline.",
    )
    sp_bl_rms_win = sidebar_param(
        "RMS window (s)", 1.0, 60.0, 1.0, "sp_bl_rms_win",
        help_text="Window size for RMS computation.",
    )

    sp_rolling_lookback = 30.0
    sp_rolling_step = 5.0
    if sp_baseline == "rolling":
        sp_rolling_lookback = sidebar_param(
            "Rolling lookback (min)", 5.0, 120.0, 5.0, "sp_rolling_lookback",
            help_text="Lookback window for rolling baseline.",
        )
        sp_rolling_step = sidebar_param(
            "Rolling step (min)", 1.0, 30.0, 1.0, "sp_rolling_step",
            help_text="Step size for rolling baseline updates.",
        )

    # Save / recall defaults
    st.sidebar.divider()
    def_col1, def_col2 = st.sidebar.columns(2)
    if def_col1.button("Save defaults", key="spike_save_defaults"):
        from eeg_seizure_analyzer.app.components import save_user_defaults
        path = save_user_defaults()
        st.sidebar.success(f"Saved to {path}")
    if def_col2.button("Recall defaults", key="spike_recall_defaults"):
        from eeg_seizure_analyzer.app.components import apply_user_defaults
        if apply_user_defaults():
            st.sidebar.success("Defaults loaded")
            st.rerun()
        else:
            st.sidebar.warning("No saved defaults found")

    persist_save(_SELECT_KEYS)

    sp_params = SpikeDetectionParams(
        bandpass_low=sp_bp_low,
        bandpass_high=sp_bp_high,
        amplitude_threshold_zscore=sp_amp_thr,
        spike_min_amplitude_uv=sp_min_amp_uv,
        spike_prominence_x_baseline=sp_prom_x,
        max_duration_ms=sp_max_dur,
        min_duration_ms=sp_min_dur,
        refractory_ms=sp_refract,
        baseline_method=sp_baseline,
        baseline_percentile=sp_bl_percentile,
        baseline_rms_window_sec=sp_bl_rms_win,
        rolling_lookback_sec=sp_rolling_lookback * 60,  # min → sec
        rolling_step_sec=sp_rolling_step * 60,          # min → sec
    )

    # ── Run detection ────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col2:
        if st.session_state.get("spike_events"):
            if st.button("Clear all spikes", key="clear_spikes"):
                st.session_state["spike_events"] = []
                st.session_state.pop("sp_detection_info", None)
                seizures = st.session_state.get("seizure_events", [])
                st.session_state["detected_events"] = seizures
                st.rerun()

    with btn_col1:
        if st.button("Detect Spikes", type="primary", key="run_sp"):
            with st.spinner("Detecting spikes..."):
                detector = SpikeDetector()
                all_spikes = []
                detection_info: dict[int, dict] = {}

                for ch in channels:
                    ch_spikes = detector.detect(recording, ch, params=sp_params)
                    all_spikes.extend(ch_spikes)
                    if hasattr(detector, "_last_detection_info"):
                        detection_info[ch] = dict(detector._last_detection_info)

                st.session_state["spike_events"] = all_spikes
                st.session_state["sp_detection_info"] = detection_info
                seizures = st.session_state.get("seizure_events", [])
                st.session_state["detected_events"] = seizures + all_spikes
            st.success(f"Found {len(all_spikes)} spike(s)")

    # ── Display results ──────────────────────────────────────────────
    _display_spike_results(recording)


def _display_spike_results(recording):
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

    # Clickable spikes table (first 200 for performance)
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
            "x Baseline": round(e.features.get("amplitude_x_baseline", 0), 1),
            "Duration (ms)": round(e.features.get("duration_ms", 0), 1)
                if e.features.get("duration_ms") else "-",
            "Confidence": round(e.confidence, 2),
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
        render_event_trace(recording, event, context_sec=1.0,
                           title=f"Spike #{sel_idx + 1}")

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
