"""Validation page: compare detected events against manual annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eeg_seizure_analyzer.app.components import (
    persist_restore,
    persist_save,
    persist_widget_callback,
    require_recording,
)
from eeg_seizure_analyzer.io.annotations import (
    find_seizure_annotations,
    pair_onset_offset_annotations,
)
from eeg_seizure_analyzer.validation.metrics import validate_detections


_VAL_KEYS = ["val_keywords", "val_onset_kw", "val_offset_kw", "val_iou", "val_onset_tol"]
_CB = persist_widget_callback


def render():
    st.header("Validation")
    recording = require_recording()

    persist_restore(_VAL_KEYS)

    seizures = st.session_state.get("seizure_events", [])
    if not seizures:
        st.warning("No seizure detections found. Run detection first.")
        return

    if not recording.annotations:
        st.warning("No manual annotations found in this recording.")
        return

    # Annotation parsing settings
    st.sidebar.subheader("Annotation Settings")

    keywords_str = st.sidebar.text_input(
        "Seizure keywords (comma-separated)",
        key="val_keywords",
        on_change=_CB, args=("val_keywords",),
    )
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

    onset_kw_str = st.sidebar.text_input(
        "Onset keywords", key="val_onset_kw",
        on_change=_CB, args=("val_onset_kw",),
    )
    onset_keywords = [k.strip() for k in onset_kw_str.split(",") if k.strip()]

    offset_kw_str = st.sidebar.text_input(
        "Offset keywords", key="val_offset_kw",
        on_change=_CB, args=("val_offset_kw",),
    )
    offset_keywords = [k.strip() for k in offset_kw_str.split(",") if k.strip()]

    overlap_threshold = st.sidebar.number_input(
        "Overlap threshold (IoU)", 0.0, 1.0, step=0.05, key="val_iou",
        on_change=_CB, args=("val_iou",),
    )
    onset_tolerance = st.sidebar.number_input(
        "Onset tolerance (s)", 0.5, 60.0, step=0.5, key="val_onset_tol",
        on_change=_CB, args=("val_onset_tol",),
    )

    persist_save(_VAL_KEYS)

    if st.button("Run Validation", type="primary"):
        # Parse annotations
        seizure_anns = find_seizure_annotations(recording.annotations, keywords)
        paired_anns = pair_onset_offset_annotations(
            seizure_anns, onset_keywords, offset_keywords
        )

        # If no paired annotations, use point annotations
        if not paired_anns:
            paired_anns = seizure_anns

        if not paired_anns:
            st.error("No seizure annotations found with the specified keywords.")
            return

        # Run validation
        result = validate_detections(
            seizures, paired_anns, overlap_threshold, onset_tolerance
        )
        st.session_state["validation_result"] = result
        st.session_state["validation_annotations"] = paired_anns

    result = st.session_state.get("validation_result")
    if result is None:
        st.info("Click 'Run Validation' to compare detections against annotations.")
        return

    paired_anns = st.session_state.get("validation_annotations", [])

    # Metrics display
    st.subheader("Detection Performance")
    cols = st.columns(6)
    cols[0].metric("Sensitivity", f"{result.sensitivity:.2%}")
    cols[1].metric("Precision", f"{result.precision:.2%}")
    cols[2].metric("F1 Score", f"{result.f1_score:.2%}")
    cols[3].metric("True Positives", result.n_true_positives)
    cols[4].metric("False Positives", result.n_false_positives)
    cols[5].metric("False Negatives", result.n_false_negatives)

    # Onset/offset error stats
    if result.onset_errors_sec:
        st.subheader("Timing Errors")
        cols2 = st.columns(2)
        cols2[0].metric("Mean Onset Error", f"{result.mean_onset_error_sec:.2f}s")
        if result.offset_errors_sec:
            cols2[1].metric("Mean Offset Error", f"{result.mean_offset_error_sec:.2f}s")

    # Timeline visualization
    st.subheader("Event Timeline")
    fig = go.Figure()

    # Ground truth annotations
    for i, ann in enumerate(paired_anns):
        end_sec = ann.onset_sec + (ann.duration_sec or 5.0)
        is_matched = any(ann is pair[1] for pair in result.matched_pairs)
        color = "green" if is_matched else "orange"
        fig.add_trace(go.Scatter(
            x=[ann.onset_sec, end_sec, end_sec, ann.onset_sec, ann.onset_sec],
            y=[1.2, 1.2, 1.8, 1.8, 1.2],
            fill="toself",
            fillcolor=color,
            opacity=0.4,
            line=dict(color=color),
            name="TP (annotated)" if is_matched else "FN (missed)",
            showlegend=(i == 0),
        ))

    # Detections
    for i, det in enumerate(seizures):
        is_tp = any(det is pair[0] for pair in result.matched_pairs)
        color = "green" if is_tp else "red"
        fig.add_trace(go.Scatter(
            x=[det.onset_sec, det.offset_sec, det.offset_sec, det.onset_sec, det.onset_sec],
            y=[0.2, 0.2, 0.8, 0.8, 0.2],
            fill="toself",
            fillcolor=color,
            opacity=0.4,
            line=dict(color=color),
            name="TP (detected)" if is_tp else "FP (false alarm)",
            showlegend=(i == 0),
        ))

    fig.update_layout(
        yaxis=dict(
            tickvals=[0.5, 1.5],
            ticktext=["Detected", "Annotated"],
            range=[0, 2],
        ),
        xaxis_title="Time (s)",
        height=250,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error distributions
    if result.onset_errors_sec:
        st.subheader("Error Distributions")
        col1, col2 = st.columns(2)

        with col1:
            fig_onset = go.Figure()
            fig_onset.add_trace(go.Histogram(
                x=result.onset_errors_sec,
                nbinsx=20,
                name="Onset error",
            ))
            fig_onset.update_layout(
                title="Onset Error Distribution",
                xaxis_title="Error (s)",
                height=300,
            )
            st.plotly_chart(fig_onset, use_container_width=True)

        with col2:
            if result.offset_errors_sec:
                fig_offset = go.Figure()
                fig_offset.add_trace(go.Histogram(
                    x=result.offset_errors_sec,
                    nbinsx=20,
                    name="Offset error",
                ))
                fig_offset.update_layout(
                    title="Offset Error Distribution",
                    xaxis_title="Error (s)",
                    height=300,
                )
                st.plotly_chart(fig_offset, use_container_width=True)
