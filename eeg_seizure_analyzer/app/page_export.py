"""Export page: download detection results, reports, and data."""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from eeg_seizure_analyzer.app.components import require_recording
from eeg_seizure_analyzer.detection.burden import compute_burden, compute_spike_rate


def render():
    st.header("Export Results")
    recording = require_recording()

    seizures = st.session_state.get("seizure_events", [])
    spikes = st.session_state.get("spike_events", [])
    validation = st.session_state.get("validation_result")

    if not seizures and not spikes:
        st.warning("No detection results to export. Run detection first.")
        return

    st.subheader("Download Options")

    col1, col2, col3 = st.columns(3)

    # Seizure events CSV
    with col1:
        if seizures:
            df = _seizures_to_df(seizures, recording)
            csv = df.to_csv(index=False)
            st.download_button(
                "Seizure Events (CSV)",
                csv,
                file_name=f"seizures_{_timestamp()}.csv",
                mime="text/csv",
            )

    # Spike events CSV
    with col2:
        if spikes:
            df = _spikes_to_df(spikes, recording)
            csv = df.to_csv(index=False)
            st.download_button(
                "Spike Events (CSV)",
                csv,
                file_name=f"spikes_{_timestamp()}.csv",
                mime="text/csv",
            )

    # Full summary JSON
    with col3:
        summary = _build_summary(recording, seizures, spikes, validation)
        json_str = json.dumps(summary, indent=2, default=str)
        st.download_button(
            "Full Summary (JSON)",
            json_str,
            file_name=f"summary_{_timestamp()}.json",
            mime="application/json",
        )

    # Validation report
    if validation:
        st.subheader("Validation Report")
        val_df = _validation_to_df(validation)
        csv = val_df.to_csv(index=False)
        st.download_button(
            "Validation Report (CSV)",
            csv,
            file_name=f"validation_{_timestamp()}.csv",
            mime="text/csv",
        )

    # Preview
    st.subheader("Preview")
    tab1, tab2, tab3 = st.tabs(["Seizures", "Spikes", "Summary"])

    with tab1:
        if seizures:
            st.dataframe(_seizures_to_df(seizures, recording), use_container_width=True)
        else:
            st.info("No seizures detected.")

    with tab2:
        if spikes:
            st.dataframe(_spikes_to_df(spikes, recording), use_container_width=True, height=400)
        else:
            st.info("No spikes detected.")

    with tab3:
        summary = _build_summary(recording, seizures, spikes, validation)
        st.json(summary)


def _seizures_to_df(seizures, recording) -> pd.DataFrame:
    rows = []
    for e in seizures:
        row = e.to_dict()
        row["channel_name"] = recording.channel_names[e.channel]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Reorder: put human-readable columns first
    front = ["channel_name", "channel", "onset_sec", "offset_sec", "duration_sec",
             "severity", "confidence", "movement_flag", "animal_id", "event_type"]
    front = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def _spikes_to_df(spikes, recording) -> pd.DataFrame:
    return pd.DataFrame({
        "channel": [recording.channel_names[e.channel] for e in spikes],
        "channel_idx": [e.channel for e in spikes],
        "onset_sec": [e.onset_sec for e in spikes],
        "peak_time_sec": [e.features.get("peak_time_sec", 0) for e in spikes],
        "amplitude": [e.features.get("amplitude", 0) for e in spikes],
        "teager_zscore": [e.features.get("teager_zscore", 0) for e in spikes],
        "duration_ms": [e.features.get("duration_ms") for e in spikes],
        "confidence": [e.confidence for e in spikes],
    })


def _validation_to_df(validation) -> pd.DataFrame:
    rows = []
    for det, ann in validation.matched_pairs:
        rows.append({
            "type": "true_positive",
            "detected_onset": det.onset_sec,
            "detected_offset": det.offset_sec,
            "annotated_onset": ann.onset_sec,
            "annotated_duration": ann.duration_sec,
            "onset_error": det.onset_sec - ann.onset_sec,
        })
    for det in validation.false_positives:
        rows.append({
            "type": "false_positive",
            "detected_onset": det.onset_sec,
            "detected_offset": det.offset_sec,
        })
    for ann in validation.false_negatives:
        rows.append({
            "type": "false_negative",
            "annotated_onset": ann.onset_sec,
            "annotated_duration": ann.duration_sec,
        })
    return pd.DataFrame(rows)


def _build_summary(recording, seizures, spikes, validation) -> dict:
    summary = {
        "file": recording.source_path,
        "duration_sec": recording.duration_sec,
        "duration_hours": recording.duration_sec / 3600,
        "n_channels": recording.n_channels,
        "channel_names": recording.channel_names,
        "sampling_rate_hz": recording.fs,
        "start_time": str(recording.start_time) if recording.start_time else None,
        "n_annotations": len(recording.annotations),
    }

    if seizures:
        burden = compute_burden(seizures, recording.duration_sec)
        summary["seizure_burden"] = {
            "n_seizures": burden.n_seizures,
            "total_seizure_time_sec": burden.total_seizure_time_sec,
            "seizure_frequency_per_hour": burden.seizure_frequency_per_hour,
            "mean_duration_sec": burden.mean_duration_sec,
            "median_duration_sec": burden.median_duration_sec,
            "max_duration_sec": burden.max_duration_sec,
            "percent_time_in_seizure": burden.percent_time_in_seizure,
            "severity_counts": burden.severity_counts,
            "hourly_counts": burden.hourly_counts,
        }

    if spikes:
        summary["spikes"] = {
            "n_spikes": len(spikes),
            "rate_per_min": len(spikes) / (recording.duration_sec / 60) if recording.duration_sec > 0 else 0,
        }

    if validation:
        summary["validation"] = {
            "sensitivity": validation.sensitivity,
            "precision": validation.precision,
            "f1_score": validation.f1_score,
            "n_true_positives": validation.n_true_positives,
            "n_false_positives": validation.n_false_positives,
            "n_false_negatives": validation.n_false_negatives,
            "mean_onset_error_sec": validation.mean_onset_error_sec,
            "mean_offset_error_sec": validation.mean_offset_error_sec,
        }

    return summary


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
