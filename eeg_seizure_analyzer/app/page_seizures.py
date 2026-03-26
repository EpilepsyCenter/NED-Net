"""Seizure Detection page: spike-train method."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eeg_seizure_analyzer.app.components import (
    apply_user_defaults,
    channel_selector,
    persist_restore,
    persist_save,
    persist_widget_callback,
    render_event_trace,
    save_user_defaults,
    sidebar_param,
    require_recording,
)
from eeg_seizure_analyzer.config import SpikeTrainSeizureParams
from eeg_seizure_analyzer.detection.burden import compute_burden
from eeg_seizure_analyzer.detection.spike_train_seizure import SpikeTrainSeizureDetector


# ── Help text dicts ──────────────────────────────────────────────────

ST_HELP = {
    "spike_amplitude": (
        "**Spike amplitude (× baseline)** — Individual spike peaks must exceed "
        "this multiple of the recording's baseline amplitude. Based on Twele et al.: "
        "≥2× for HPD evolved phase, ≥3× for HVSWs."
    ),
    "max_isi": (
        "**Max inter-spike interval (ms)** — Spikes further apart than this "
        "are assigned to separate trains. Default 500 ms."
    ),
    "min_train_spikes": (
        "**Min spikes in train** — Minimum number of spikes to form a train."
    ),
    "min_train_duration": (
        "**Min train duration (s)** — Shortest acceptable spike train."
    ),
    "hvsw_criteria": (
        "**HVSW** (high-voltage sharp wave): monomorphic high-amplitude (≥3×) "
        "spikes at ≥2 Hz for ≥5 s with low evolution (low CV of ISI)."
    ),
    "hpd_criteria": (
        "**HPD** (hippocampal paroxysmal discharge): evolving pattern, starts "
        "with HVSWs then transitions to faster (≥5 Hz) lower-amplitude (≥2×) "
        "spikes. Typically >20 s."
    ),
    "convulsive_criteria": (
        "**Electroclinical/convulsive**: very high amplitude (≥5×), long duration "
        "(≥20 s), followed by post-ictal suppression."
    ),
    "boundary_refinement": (
        "**Boundary refinement** — After grouping spikes into trains, the onset "
        "and offset are refined.\n\n"
        "• **Signal (RMS)**: Computes a short-window RMS envelope of the filtered "
        "signal and walks backward from the first spike (forward from the last) to "
        "find where the envelope first crosses above the threshold. This finds the "
        "actual electrographic transition, not just where spikes happen to cluster.\n\n"
        "• **Spike density**: Trims edges based on local spike rate and amplitude. "
        "Walks inward until both the rate and amplitude meet the thresholds.\n\n"
        "• **None**: Uses the raw first/last spike times as boundaries."
    ),
}


# Keys for non-sidebar_param widgets
_SELECT_KEYS = [
    "st_baseline",
    "activity_enabled", "activity_channel",
    "quality_enabled", "quality_individual_filters",
    "st_bnd_method", "st_classify_subtypes",
]
_CB = persist_widget_callback


def render():
    st.header("Seizure Detection")
    recording = require_recording()

    persist_restore(_SELECT_KEYS)

    channels = channel_selector(recording, key="det_channels")

    # ── Parameter guide ──────────────────────────────────────────────
    with st.expander("Parameter guide", expanded=False):
        for help_text in ST_HELP.values():
            st.markdown(help_text)
            st.markdown("---")

    # ── Sidebar parameters ───────────────────────────────────────────
    _render_st_sidebar()

    # ── Activity channel ─────────────────────────────────────────────
    _render_activity_sidebar(recording)

    # ── Quality metrics / confidence scoring ─────────────────────────
    _render_quality_sidebar()

    # Save / recall defaults
    st.sidebar.divider()
    def_col1, def_col2 = st.sidebar.columns(2)
    if def_col1.button("Save defaults"):
        path = save_user_defaults()
        st.sidebar.success(f"Saved to {path}")
    if def_col2.button("Recall defaults"):
        if apply_user_defaults():
            st.sidebar.success("Defaults loaded")
            st.rerun()
        else:
            st.sidebar.warning("No saved defaults found")

    persist_save(_SELECT_KEYS)

    # ── Run detection ────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        _run_st_detection(recording, channels)
    with btn_col2:
        if st.session_state.get("seizure_events"):
            if st.button("Clear all seizures", key="clear_seizures"):
                st.session_state["seizure_events"] = []
                st.session_state.pop("st_detection_info", None)
                spikes = st.session_state.get("spike_events", [])
                st.session_state["detected_events"] = spikes
                st.rerun()

    # ── Post-detection: activity flagging & quality scoring ──────────
    _apply_post_detection(recording)

    # ── Display results ──────────────────────────────────────────────
    _display_seizure_results(recording)


# ── Spike-train sidebar & runner ─────────────────────────────────────


def _render_st_sidebar():
    st.sidebar.subheader("Spike-Train Parameters")

    st.sidebar.caption("**Spike front-end**")
    sidebar_param("Bandpass low (Hz)", 0.5, 100.0, 0.5, "st_bp_low")
    sidebar_param("Bandpass high (Hz)", 10.0, 500.0, 1.0, "st_bp_high")
    sidebar_param("Spike threshold (z-score)", 1.0, 10.0, 0.5, "st_spike_amp_x",
                  help_text="Z-score multiplier for spike detection. Threshold = "
                            "baseline_mean + z × baseline_std. Higher = fewer spikes.")
    sidebar_param("Spike min amplitude (µV)", 0.0, 500.0, 5.0, "st_spike_min_uv",
                  help_text="Absolute amplitude floor in µV. Spikes below this are "
                            "rejected regardless of the relative threshold. Set to 0 to disable. "
                            "Useful for quiet recordings where the relative threshold becomes too low.")
    sidebar_param("Spike prominence (× baseline)", 0.5, 10.0, 0.5, "st_spike_prom_x",
                  help_text="Minimum prominence as a multiple of baseline amplitude. "
                            "A spike must stand out from its local surroundings by at least this much. "
                            "Rejects noise peaks riding on slow oscillations.")
    sidebar_param("Spike max width (ms)", 10.0, 200.0, 5.0, "st_spike_max_width",
                  help_text="Maximum spike half-width in ms. Rejects slow waves "
                            "that exceed amplitude threshold but are not sharp.")
    sidebar_param("Spike min width (ms)", 0.5, 20.0, 0.5, "st_spike_min_width",
                  help_text="Minimum spike half-width in ms. Rejects single-sample "
                            "noise artifacts that are too narrow to be real spikes.")
    sidebar_param("Spike refractory (ms)", 5.0, 200.0, 5.0, "st_spike_refract")

    st.sidebar.caption("**Train grouping**")
    sidebar_param("Max inter-spike interval (ms)", 50.0, 2000.0, 50.0, "st_max_isi")
    sidebar_param("Min spikes in train", 2, 50, 1, "st_min_spikes")
    sidebar_param("Min train duration (s)", 1.0, 60.0, 0.5, "st_min_train_dur")
    sidebar_param("Min inter-event interval (s)", 0.5, 30.0, 0.5, "st_min_iei")

    st.sidebar.caption("**Boundary refinement**")
    bnd_methods = ["signal", "spike_density", "none"]
    st.sidebar.selectbox(
        "Boundary method", bnd_methods,
        key="st_bnd_method", on_change=_CB, args=("st_bnd_method",),
        help="**signal**: uses RMS envelope of the raw signal to find the "
             "electrographic onset/offset.  "
             "**spike_density**: trims edges based on local spike rate and amplitude.  "
             "**none**: uses raw spike train boundaries (first/last spike).",
    )

    bnd_method = st.session_state.get("st_bnd_method", "signal")

    if bnd_method == "signal":
        sidebar_param("RMS window (ms)", 10.0, 500.0, 10.0, "st_bnd_rms_win",
                      help_text="Short window for computing the RMS envelope of the signal.")
        sidebar_param("RMS threshold (× baseline)", 0.5, 10.0, 0.5, "st_bnd_rms_thr",
                      help_text="Onset/offset is where the RMS envelope crosses this "
                                "multiple of baseline amplitude.")
        sidebar_param("Max trim from spike edge (s)", 0.5, 20.0, 0.5, "st_bnd_max_trim",
                      help_text="Maximum seconds to search beyond first/last spike "
                                "for the actual electrographic boundary.")
    elif bnd_method == "spike_density":
        sidebar_param("Boundary window (s)", 0.5, 10.0, 0.5, "st_bnd_window",
                      help_text="Sliding window for computing local spike rate at edges.")
        sidebar_param("Boundary min rate (Hz)", 0.5, 20.0, 0.5, "st_bnd_rate",
                      help_text="Edges are trimmed until local rate meets this threshold.")
        sidebar_param("Boundary min amplitude (× BL)", 0.5, 10.0, 0.5, "st_bnd_amp_x",
                      help_text="Edge spikes must exceed this amplitude to keep the boundary.")

    st.sidebar.selectbox(
        "Baseline method", ["percentile", "rolling", "first_n"],
        key="st_baseline", on_change=_CB, args=("st_baseline",),
    )

    st_baseline = st.session_state.get("st_baseline", "percentile")
    if st_baseline in ("percentile", "rolling"):
        sidebar_param("Baseline percentile", 1, 50, 1, "st_bl_percentile",
                      help_text="Which percentile of RMS windows to use for baseline.")
        sidebar_param("RMS window (s)", 1.0, 60.0, 1.0, "st_bl_rms_win",
                      help_text="Window size for computing RMS values.")
    if st_baseline == "rolling":
        sidebar_param("Rolling lookback (min)", 5.0, 120.0, 5.0, "st_rolling_lookback",
                      help_text="How far back to look for baseline computation.")
        sidebar_param("Rolling step (min)", 1.0, 30.0, 1.0, "st_rolling_step",
                      help_text="How often to recompute the baseline.")

    # ── Subtype classification (optional) ────────────────────────────
    st.sidebar.divider()
    st.sidebar.checkbox(
        "Classify subtypes (HVSW/HPD/convulsive)", key="st_classify_subtypes",
        on_change=_CB, args=("st_classify_subtypes",),
        help="When disabled, all spike trains are reported as generic 'seizure' "
             "events without HVSW/HPD/convulsive classification.",
    )

    if st.session_state.get("st_classify_subtypes", True):
        st.sidebar.caption("**HVSW criteria**")
        sidebar_param("HVSW amplitude (× baseline)", 1.0, 10.0, 0.5, "st_hvsw_amp_x")
        sidebar_param("HVSW min frequency (Hz)", 0.5, 10.0, 0.5, "st_hvsw_freq")
        sidebar_param("HVSW min duration (s)", 1.0, 30.0, 0.5, "st_hvsw_dur")
        sidebar_param("HVSW max evolution (CV)", 0.1, 1.0, 0.05, "st_hvsw_max_ev")

        st.sidebar.caption("**HPD criteria**")
        sidebar_param("HPD amplitude (× baseline)", 1.0, 10.0, 0.5, "st_hpd_amp_x")
        sidebar_param("HPD min frequency (Hz)", 1.0, 20.0, 0.5, "st_hpd_freq")
        sidebar_param("HPD min duration (s)", 1.0, 60.0, 1.0, "st_hpd_dur")

        st.sidebar.caption("**Electroclinical/convulsive**")
        sidebar_param("Convulsive min duration (s)", 5.0, 120.0, 1.0, "st_conv_dur")
        sidebar_param("Convulsive amplitude (× baseline)", 2.0, 20.0, 0.5, "st_conv_amp_x")
        sidebar_param("Post-ictal suppression (s)", 1.0, 30.0, 1.0, "st_conv_postictal")


def _get_st_params():
    from eeg_seizure_analyzer.app.components import _get_store
    s = _get_store()
    return SpikeTrainSeizureParams(
        classify_subtypes=s.get("st_classify_subtypes", True),
        bandpass_low=s.get("st_bp_low", 1.0),
        bandpass_high=s.get("st_bp_high", 100.0),
        spike_amplitude_x_baseline=s.get("st_spike_amp_x", 3.0),
        spike_min_amplitude_uv=s.get("st_spike_min_uv", 0.0),
        spike_prominence_x_baseline=s.get("st_spike_prom_x", 1.5),
        spike_max_width_ms=s.get("st_spike_max_width", 70.0),
        spike_min_width_ms=s.get("st_spike_min_width", 2.0),
        spike_refractory_ms=s.get("st_spike_refract", 50.0),
        max_interspike_interval_ms=s.get("st_max_isi", 500.0),
        min_train_spikes=int(s.get("st_min_spikes", 5)),
        min_train_duration_sec=s.get("st_min_train_dur", 5.0),
        min_interevent_interval_sec=s.get("st_min_iei", 3.0),
        hvsw_min_amplitude_x=s.get("st_hvsw_amp_x", 3.0),
        hvsw_min_frequency_hz=s.get("st_hvsw_freq", 2.0),
        hvsw_min_duration_sec=s.get("st_hvsw_dur", 5.0),
        hvsw_max_evolution=s.get("st_hvsw_max_ev", 0.4),
        hpd_min_amplitude_x=s.get("st_hpd_amp_x", 2.0),
        hpd_min_frequency_hz=s.get("st_hpd_freq", 5.0),
        hpd_min_duration_sec=s.get("st_hpd_dur", 10.0),
        convulsive_min_duration_sec=s.get("st_conv_dur", 20.0),
        convulsive_min_amplitude_x=s.get("st_conv_amp_x", 5.0),
        convulsive_postictal_suppression_sec=s.get("st_conv_postictal", 5.0),
        boundary_method=s.get("st_bnd_method", "signal"),
        boundary_window_sec=s.get("st_bnd_window", 2.0),
        boundary_min_rate_hz=s.get("st_bnd_rate", 2.0),
        boundary_min_amplitude_x=s.get("st_bnd_amp_x", 2.0),
        boundary_rms_window_ms=s.get("st_bnd_rms_win", 100.0),
        boundary_rms_threshold_x=s.get("st_bnd_rms_thr", 2.0),
        boundary_max_trim_sec=s.get("st_bnd_max_trim", 5.0),
        baseline_method=s.get("st_baseline", "percentile"),
        baseline_percentile=int(s.get("st_bl_percentile", 15)),
        baseline_rms_window_sec=s.get("st_bl_rms_win", 10.0),
        rolling_lookback_sec=s.get("st_rolling_lookback", 30.0) * 60,  # min → sec
        rolling_step_sec=s.get("st_rolling_step", 5.0) * 60,            # min → sec
    )


def _run_st_detection(recording, channels):
    if st.button("Detect Seizures", type="primary", key="run_sz_st"):
        params = _get_st_params()
        with st.spinner("Detecting seizures (spike-train method)…"):
            detector = SpikeTrainSeizureDetector()
            # Run per-channel to capture detection metadata (spikes, baseline)
            seizures = []
            detection_info = {}
            for ch in channels:
                ch_events = detector.detect(recording, ch, params=params)
                seizures.extend(ch_events)
                # Store per-channel detection info for visualization
                if hasattr(detector, "_last_detection_info"):
                    detection_info[ch] = detector._last_detection_info.copy()
            seizures.sort(key=lambda e: e.onset_sec)

            st.session_state["seizure_events"] = seizures
            st.session_state["st_detection_info"] = detection_info
            spikes = st.session_state.get("spike_events", [])
            st.session_state["detected_events"] = seizures + spikes
        st.success(f"Found {len(seizures)} seizure(s)")


# ── Activity channel sidebar ──────────────────────────────────────────


def _render_activity_sidebar(recording):
    st.sidebar.divider()
    st.sidebar.subheader("Activity Channel")

    st.sidebar.checkbox(
        "Enable movement flagging", key="activity_enabled",
        on_change=_CB, args=("activity_enabled",),
        help="Flag detected events that co-occur with elevated activity "
             "(e.g. EMG/accelerometer). Events are flagged, not removed.",
    )

    if st.session_state.get("activity_enabled", False):
        # Show all channels (including different-rate ones) for selection
        all_info = st.session_state.get("all_channels_info", [])
        if all_info:
            ch_labels = [f"{ch['index']}: {ch['label']} ({ch['fs']} Hz)" for ch in all_info]
        else:
            ch_labels = [f"{i}: {name}" for i, name in enumerate(recording.channel_names)]

        st.sidebar.selectbox(
            "Activity channel", ch_labels,
            key="activity_channel", on_change=_CB, args=("activity_channel",),
            help="Select the channel to use for movement detection.",
        )
        sidebar_param("Threshold percentile", 50.0, 99.0, 1.0, "activity_threshold_pct",
                      help_text="Percentile of activity signal above which movement is detected.")
        sidebar_param("Padding (s)", 0.0, 10.0, 0.5, "activity_pad_sec",
                      help_text="Seconds of padding around events for activity check.")


# ── Quality metrics sidebar ──────────────────────────────────────────


def _render_quality_sidebar():
    st.sidebar.divider()
    st.sidebar.subheader("Quality Scoring")

    st.sidebar.checkbox(
        "Enable confidence scoring", key="quality_enabled",
        on_change=_CB, args=("quality_enabled",),
        help="Compute post-detection quality metrics (LL/energy z-scores, "
             "spectral entropy, band ratios) and filter by confidence.",
    )

    if st.session_state.get("quality_enabled", False):
        sidebar_param("Min confidence (overall)", 0.0, 1.0, 0.05, "quality_min_confidence",
                      help_text="Events with overall confidence below this are removed. "
                                "Set to 0 to keep all but still compute metrics.")

        st.sidebar.checkbox(
            "Enable individual metric filters", key="quality_individual_filters",
            on_change=_CB, args=("quality_individual_filters",),
            help="Filter events by individual quality metrics in addition to "
                 "the overall confidence score.",
        )

        if st.session_state.get("quality_individual_filters", False):
            sidebar_param("Min LL z-score", 0.0, 20.0, 0.5, "quality_min_ll_z",
                          help_text="Minimum peak line-length z-score. "
                                    "Events with LL z-score below this are removed.")
            sidebar_param("Min energy z-score", 0.0, 20.0, 0.5, "quality_min_en_z",
                          help_text="Minimum peak energy z-score.")
            sidebar_param("Min signal/baseline ratio", 0.0, 10.0, 0.5, "quality_min_sbr",
                          help_text="Minimum signal-to-baseline RMS ratio.")
            sidebar_param("Min spike frequency (Hz)", 0.0, 50.0, 0.5, "quality_min_spike_freq",
                          help_text="Minimum mean spike frequency within the seizure. "
                                    "Only applies to spike-train detection.")
            sidebar_param("Min spectral entropy", 0.0, 10.0, 0.5, "quality_min_se",
                          help_text="Minimum spectral entropy. Very low values "
                                    "suggest monotone artifacts.")
            sidebar_param("Max spectral entropy", 0.0, 15.0, 0.5, "quality_max_se",
                          help_text="Maximum spectral entropy. Very high values "
                                    "suggest broadband noise.")


# ── Post-detection processing ────────────────────────────────────────


def _apply_post_detection(recording):
    """Apply activity flagging and quality scoring to detected seizures."""
    from eeg_seizure_analyzer.app.components import _get_store
    store = _get_store()
    seizures = st.session_state.get("seizure_events", [])
    if not seizures:
        return

    # ── Activity flagging ────────────────────────────────────────────
    if store.get("activity_enabled", False):
        try:
            from eeg_seizure_analyzer.processing.activity import (
                load_activity_channel,
                compute_movement_threshold,
                flag_movement_artifacts,
            )
            ch_label = store.get("activity_channel", "0")
            # Extract channel index from label like "3: EMG (2.0 Hz)"
            ch_idx = int(str(ch_label).split(":")[0]) if isinstance(ch_label, str) else int(ch_label)
            threshold_pct = float(store.get("activity_threshold_pct", 85.0))
            pad = float(store.get("activity_pad_sec", 2.0))
            source = recording.source_path

            if source and os.path.isfile(source):
                activity, _ = load_activity_channel(source, ch_idx, recording.fs)
                threshold = compute_movement_threshold(activity, threshold_pct)
                flag_movement_artifacts(seizures, activity, recording.fs, threshold, pad)
        except Exception as e:
            st.warning(f"Activity flagging failed: {e}")

    # ── Quality scoring ──────────────────────────────────────────────
    if store.get("quality_enabled", False):
        try:
            from eeg_seizure_analyzer.detection.confidence import apply_quality_filter
            min_conf = float(store.get("quality_min_confidence", 0.3))

            # Build per-metric filters if enabled
            metric_filters = {}
            if store.get("quality_individual_filters", False):
                min_ll = float(store.get("quality_min_ll_z", 0.0))
                min_en = float(store.get("quality_min_en_z", 0.0))
                min_sbr = float(store.get("quality_min_sbr", 0.0))
                min_se = float(store.get("quality_min_se", 0.0))
                max_se = float(store.get("quality_max_se", 0.0))
                min_sf = float(store.get("quality_min_spike_freq", 0.0))
                if min_ll > 0:
                    metric_filters["min_ll_zscore"] = min_ll
                if min_en > 0:
                    metric_filters["min_energy_zscore"] = min_en
                if min_sbr > 0:
                    metric_filters["min_signal_to_baseline_ratio"] = min_sbr
                if min_sf > 0:
                    metric_filters["min_spike_frequency"] = min_sf
                if min_se > 0:
                    metric_filters["min_spectral_entropy"] = min_se
                if max_se > 0:
                    metric_filters["max_spectral_entropy"] = max_se

            seizures = apply_quality_filter(
                seizures, recording,
                min_confidence=min_conf,
                metric_filters=metric_filters if metric_filters else None,
            )
            st.session_state["seizure_events"] = seizures
            # Update combined events list too
            spikes = st.session_state.get("spike_events", [])
            st.session_state["detected_events"] = seizures + spikes
        except Exception as e:
            st.warning(f"Quality scoring failed: {e}")


# ── Results display (shared by both methods) ─────────────────────────


def _display_seizure_results(recording):
    seizures = st.session_state.get("seizure_events", [])
    if not seizures:
        return

    st.subheader("Seizure Results")

    burden = compute_burden(seizures, recording.duration_sec)

    cols = st.columns(5)
    cols[0].metric("Total Seizures", burden.n_seizures)
    cols[1].metric("Total Seizure Time", f"{burden.total_seizure_time_sec:.1f}s")
    cols[2].metric("Seizures/Hour", f"{burden.seizure_frequency_per_hour:.2f}")
    cols[3].metric("% Time in Seizure", f"{burden.percent_time_in_seizure:.2f}%")
    cols[4].metric("Mean Duration", f"{burden.mean_duration_sec:.1f}s")

    st.markdown(
        f"**Severity:** Mild: {burden.severity_counts.get('mild', 0)} | "
        f"Moderate: {burden.severity_counts.get('moderate', 0)} | "
        f"Severe: {burden.severity_counts.get('severe', 0)}"
    )

    # ── Build table ──────────────────────────────────────────────────
    from eeg_seizure_analyzer.app.components import _get_store
    store = _get_store()
    show_quality = store.get("quality_enabled", False)
    show_activity = store.get("activity_enabled", False)
    classify_on = store.get("st_classify_subtypes", True)

    table_data = []
    for i, e in enumerate(seizures):
        row = {
            "#": i + 1,
            "Channel": recording.channel_names[e.channel],
            "Onset (s)": round(e.onset_sec, 2),
            "Offset (s)": round(e.offset_sec, 2),
            "Duration (s)": round(e.duration_sec, 2),
            "Severity": e.severity,
        }
        if classify_on:
            row["Type"] = e.features.get("seizure_subtype", "—")
        row["Spikes"] = e.features.get("n_spikes", 0)
        row["Max Amp (×BL)"] = round(e.features.get("max_amplitude_x_baseline", 0), 1)
        row["Freq (Hz)"] = round(e.features.get("mean_spike_frequency_hz", 0), 1)
        # Quality metrics columns (always shown when scoring is enabled)
        if show_quality:
            qm = e.quality_metrics or {}
            row["Confidence"] = round(e.confidence, 2)
            row["LL (z)"] = round(qm.get("peak_ll_zscore", 0), 1)
            row["Energy (z)"] = round(qm.get("peak_energy_zscore", 0), 1)
            row["Sig/BL"] = round(qm.get("signal_to_baseline_ratio", 0), 1)
            row["Spec Ent"] = round(qm.get("spectral_entropy", 0), 1)
            row["Peak Freq (Hz)"] = round(qm.get("dominant_freq_hz", 0), 1)
            row["θ/δ"] = round(qm.get("theta_delta_ratio", 0), 2)
        if show_activity:
            row["Movement"] = "Yes" if e.movement_flag else ""
        table_data.append(row)

    df = pd.DataFrame(table_data)

    st.markdown("**Click a seizure to inspect it on the EEG trace:**")

    selection = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="seizure_table_select",
    )

    selected_rows = selection.get("selection", {}).get("rows", [])
    if selected_rows:
        sel_idx = selected_rows[0]
        event = seizures[sel_idx]
        subtype = event.features.get("seizure_subtype", "")
        title_suffix = f" ({subtype})" if subtype else ""
        render_event_trace(recording, event, context_sec=10.0,
                           title=f"Seizure #{sel_idx + 1}{title_suffix}")
