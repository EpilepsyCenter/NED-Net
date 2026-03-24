"""Spectral Analysis page: spectrograms, band power, PSD."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eeg_seizure_analyzer.app.components import (
    persist_restore,
    persist_save,
    persist_widget_callback,
    require_recording,
)
from eeg_seizure_analyzer.processing.spectral import (
    compute_band_powers,
    compute_psd,
    compute_spectrogram,
)


_SPEC_KEYS = ["spec_full", "spec_start", "spec_max_freq"]
_CB = persist_widget_callback


def render():
    st.header("Spectral Analysis")
    recording = require_recording()

    persist_restore(_SPEC_KEYS)

    # Channel selector
    ch_options = {f"{i}: {name}": i for i, name in enumerate(recording.channel_names)}
    selected_ch = st.sidebar.selectbox(
        "Channel", list(ch_options.keys()), key="spec_channel"
    )
    ch_idx = ch_options[selected_ch]

    # Time range
    st.sidebar.subheader("Time Range")
    use_full = st.sidebar.checkbox("Full recording", key="spec_full",
                                   on_change=_CB, args=("spec_full",))
    if use_full:
        start_sec = 0.0
        end_sec = recording.duration_sec
    else:
        start_sec = st.sidebar.number_input("Start (s)", 0.0, recording.duration_sec, key="spec_start",
                                            on_change=_CB, args=("spec_start",))
        end_sec = st.sidebar.number_input("End (s)", 0.0, recording.duration_sec, min(300.0, recording.duration_sec), key="spec_end")

    max_freq = st.sidebar.number_input("Max frequency (Hz)", 10.0, 500.0, step=10.0, key="spec_max_freq",
                                      on_change=_CB, args=("spec_max_freq",))

    persist_save(_SPEC_KEYS)

    # Get data
    start_idx = int(start_sec * recording.fs)
    end_idx = min(int(end_sec * recording.fs), recording.n_samples)
    data = recording.data[ch_idx, start_idx:end_idx]

    if len(data) < recording.fs:
        st.warning("Selected time range too short for spectral analysis.")
        return

    run_analysis = st.button("Run Spectral Analysis", type="primary")

    if run_analysis:
        with st.spinner("Computing spectral analysis..."):
            # Spectrogram
            times, freqs, sxx_db = compute_spectrogram(
                data, recording.fs, max_freq=max_freq
            )
            times += start_sec  # offset by start time

            # Band powers
            band_powers = compute_band_powers(data, recording.fs)
            band_powers["time_sec"] += start_sec

            # PSD
            psd_freqs, psd_values = compute_psd(data, recording.fs)

            st.session_state["spectral_results"] = {
                "times": times,
                "freqs": freqs,
                "sxx_db": sxx_db,
                "band_powers": band_powers,
                "psd_freqs": psd_freqs,
                "psd_values": psd_values,
            }

    results = st.session_state.get("spectral_results")
    if results is None:
        st.info("Click 'Run Spectral Analysis' to compute.")
        return

    # Spectrogram
    st.subheader("Spectrogram")
    fig = go.Figure(data=go.Heatmap(
        x=results["times"],
        y=results["freqs"],
        z=results["sxx_db"],
        colorscale="Viridis",
        colorbar=dict(title="Power (dB)"),
    ))
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=400,
    )

    # Mark seizure events on spectrogram
    if "seizure_events" in st.session_state:
        for event in st.session_state["seizure_events"]:
            if event.channel == ch_idx and event.offset_sec >= start_sec and event.onset_sec <= end_sec:
                fig.add_vrect(
                    x0=event.onset_sec,
                    x1=event.offset_sec,
                    fillcolor="red",
                    opacity=0.2,
                    line_width=1,
                    line_color="red",
                )

    st.plotly_chart(fig, use_container_width=True)

    # Band Power Time Series
    st.subheader("Band Power Over Time")
    bp = results["band_powers"]
    fig_bp = go.Figure()
    band_colors = {
        "delta": "#1f77b4",
        "theta": "#ff7f0e",
        "alpha": "#2ca02c",
        "beta": "#d62728",
        "gamma_low": "#9467bd",
        "gamma_high": "#8c564b",
    }
    for col in bp.columns:
        if col == "time_sec":
            continue
        fig_bp.add_trace(go.Scatter(
            x=bp["time_sec"],
            y=bp[col],
            mode="lines",
            name=col,
            line=dict(color=band_colors.get(col)),
            stackgroup="bands",
        ))
    fig_bp.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Power",
        height=350,
    )
    st.plotly_chart(fig_bp, use_container_width=True)

    # PSD
    st.subheader("Power Spectral Density")
    psd_freqs = results["psd_freqs"]
    psd_values = results["psd_values"]
    freq_mask = psd_freqs <= max_freq
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(
        x=psd_freqs[freq_mask],
        y=10 * np.log10(psd_values[freq_mask] + 1e-12),
        mode="lines",
        name="PSD",
    ))
    fig_psd.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB/Hz)",
        height=350,
    )
    st.plotly_chart(fig_psd, use_container_width=True)
