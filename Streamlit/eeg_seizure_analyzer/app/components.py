"""Shared Streamlit components and utilities."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
import numpy as np

from eeg_seizure_analyzer.io.base import EEGRecording

# ── Save / load user defaults ─────────────────────────────────────────

_DEFAULTS_DIR = Path.home() / ".eeg_seizure_analyzer"
_DEFAULTS_FILE = _DEFAULTS_DIR / "defaults.json"


def save_user_defaults():
    """Save all current persistent parameters to a JSON file."""
    store = _get_store()
    # Filter out None values and non-serialisable entries
    serialisable = {}
    for k, v in store.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            serialisable[k] = v
        elif isinstance(v, list):
            serialisable[k] = v
    _DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)
    return str(_DEFAULTS_FILE)


def load_user_defaults() -> dict | None:
    """Load user defaults from disk. Returns the dict or None."""
    if not _DEFAULTS_FILE.exists():
        return None
    try:
        with open(_DEFAULTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def apply_user_defaults():
    """Load saved defaults into the persist store and flag for rerun.

    Because Streamlit forbids writing to session_state keys that are
    already bound to rendered widgets, we only update the shadow
    ``_persist`` store here.  The caller must then ``st.rerun()`` so
    that ``persist_restore()`` pushes the values back *before* the
    widgets are instantiated on the next cycle.
    """
    saved = load_user_defaults()
    if saved is None:
        return False
    store = _get_store()
    for k, v in saved.items():
        store[k] = v
    return True


# ── Persistent parameter store ────────────────────────────────────────
#
# Streamlit removes widget keys from session_state when their widget is
# not rendered (e.g. on a different page). We keep a shadow copy in
# st.session_state["_persist"] that survives page switches.
# Call `persist_save(key)` after rendering a widget to snapshot its value,
# and `persist_restore(keys)` before rendering to write them back.


_DEFAULTS = {
    # Viewer page
    "viewer_start": 0.0,
    "viewer_window": 10.0,
    "viewer_yrange": None,      # auto-computed from data on first render
    "viewer_height": 600,
    "viewer_filter": False,
    "viewer_filt_low": 1.0,
    "viewer_filt_high": 50.0,
    "viewer_notch": False,
    "viewer_notch_freq": 50.0,
    "viewer_show_events": True,
    # Spike detection params
    "sp_bp_low": 10.0,
    "sp_bp_high": 70.0,
    "sp_amp_thr": 4.0,
    "sp_min_amp_uv": 0.0,
    "sp_prom_x": 1.5,
    "sp_max_dur": 70.0,
    "sp_min_dur": 2.0,
    "sp_refract": 200.0,
    "sp_baseline": "percentile",
    "sp_bl_percentile": 15,
    "sp_bl_rms_win": 10.0,
    "sp_rolling_lookback": 30.0,
    "sp_rolling_step": 5.0,
    # Spike-train seizure detection params
    "st_classify_subtypes": True,
    "st_bp_low": 1.0,
    "st_bp_high": 100.0,
    "st_spike_amp_x": 3.0,
    "st_spike_min_uv": 0.0,
    "st_spike_prom_x": 1.5,
    "st_spike_max_width": 70.0,
    "st_spike_min_width": 2.0,
    "st_spike_refract": 50.0,
    "st_max_isi": 500.0,
    "st_min_spikes": 5,
    "st_min_train_dur": 5.0,
    "st_min_iei": 3.0,
    "st_hvsw_amp_x": 3.0,
    "st_hvsw_freq": 2.0,
    "st_hvsw_dur": 5.0,
    "st_hvsw_max_ev": 0.4,
    "st_hpd_amp_x": 2.0,
    "st_hpd_freq": 5.0,
    "st_hpd_dur": 10.0,
    "st_conv_dur": 20.0,
    "st_conv_amp_x": 5.0,
    "st_conv_postictal": 5.0,
    "st_bnd_method": "signal",
    "st_bnd_window": 2.0,
    "st_bnd_rate": 2.0,
    "st_bnd_amp_x": 2.0,
    "st_bnd_rms_win": 100.0,
    "st_bnd_rms_thr": 2.0,
    "st_bnd_max_trim": 5.0,
    "st_baseline": "percentile",
    "st_bl_percentile": 15,
    "st_bl_rms_win": 10.0,
    "st_rolling_lookback": 30.0,
    "st_rolling_step": 5.0,
    # Activity channel
    "activity_enabled": False,
    "activity_channel": 0,
    "activity_threshold_pct": 85.0,
    "activity_pad_sec": 2.0,
    # Quality metrics / confidence scoring
    "quality_enabled": False,
    "quality_min_confidence": 0.3,
    "quality_individual_filters": False,
    "quality_min_ll_z": 0.0,
    "quality_min_en_z": 0.0,
    "quality_min_sbr": 0.0,
    "quality_min_spike_freq": 0.0,
    "quality_min_se": 0.0,
    "quality_max_se": 0.0,
    # Spectral page
    "spec_full": True,
    "spec_start": 0.0,
    "spec_max_freq": 100.0,
    # Validation page
    "val_keywords": "seizure, sz, ictal, onset, offset, start, end",
    "val_onset_kw": "onset, start, begin, sz start",
    "val_offset_kw": "offset, end, stop, sz end",
    "val_iou": 0.3,
    "val_onset_tol": 10.0,
}


def _get_store() -> dict:
    """Get or create the persistent parameter store."""
    if "_persist" not in st.session_state:
        st.session_state["_persist"] = dict(_DEFAULTS)
    return st.session_state["_persist"]


def persist_restore(keys: list[str]):
    """Restore widget keys from persistent store before rendering widgets."""
    store = _get_store()
    for key in keys:
        if key in store:
            st.session_state[key] = store[key]
        elif key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]


def persist_save(keys: list[str]):
    """Save widget keys to persistent store after rendering widgets."""
    store = _get_store()
    for key in keys:
        if key in st.session_state:
            store[key] = st.session_state[key]


def persist_widget_callback(key: str):
    """Callback that saves a single widget value to the persistent store."""
    store = _get_store()
    if key in st.session_state:
        store[key] = st.session_state[key]


def sidebar_param(
    label: str,
    min_val,
    max_val,
    step,
    key: str,
    help_text: str | None = None,
):
    """Render a synced slider + number-input pair in the sidebar.

    Both widgets stay synchronised via *key* in ``st.session_state`` and the
    persistent ``_persist`` store so values survive page switches.
    """
    store = _get_store()

    # ── Determine current value ──────────────────────────────────────
    if key in store:
        current = store[key]
    elif key in st.session_state:
        current = st.session_state[key]
    elif key in _DEFAULTS:
        current = _DEFAULTS[key]
    else:
        current = min_val

    # Ensure types match (Streamlit is strict about int vs float)
    is_int = isinstance(min_val, int) and isinstance(max_val, int)
    if is_int:
        current = int(max(min_val, min(max_val, int(current))))
        step = int(step) if step else 1
    else:
        current = float(max(min_val, min(max_val, float(current))))
        step = float(step) if step else 0.1

    sl_key = f"__sl_{key}"
    ni_key = f"__ni_{key}"

    # Pre-set widget keys so they render with the correct value
    st.session_state[sl_key] = current
    st.session_state[ni_key] = current

    def _on_slider():
        val = st.session_state[sl_key]
        st.session_state[key] = val
        st.session_state[ni_key] = val
        store[key] = val

    def _on_input():
        val = st.session_state[ni_key]
        st.session_state[key] = val
        st.session_state[sl_key] = val
        store[key] = val

    st.sidebar.caption(label)
    c1, c2 = st.sidebar.columns([3, 1])
    with c1:
        st.slider(
            label, min_val, max_val, step=step, key=sl_key,
            on_change=_on_slider, label_visibility="collapsed",
            help=help_text,
        )
    with c2:
        st.number_input(
            label, min_val, max_val, step=step, key=ni_key,
            on_change=_on_input, label_visibility="collapsed",
        )

    # Ensure session_state[key] and store are up-to-date
    final = st.session_state.get(sl_key, current)
    st.session_state[key] = final
    store[key] = final
    return final


def init_session_defaults():
    """Initialize the persistent store (called once from main.py)."""
    store = _get_store()  # creates the store with defaults if not exists
    # On first launch, apply user-saved defaults from disk
    if "_user_defaults_applied" not in st.session_state:
        apply_user_defaults()
        st.session_state["_user_defaults_applied"] = True


# ── Recording helpers ─────────────────────────────────────────────────


def get_recording() -> EEGRecording | None:
    """Get the current recording from session state."""
    return st.session_state.get("recording")


def require_recording() -> EEGRecording:
    """Get recording or show error and stop."""
    rec = get_recording()
    if rec is None:
        st.warning("No recording loaded. Go to **Upload** first.")
        st.stop()
    return rec


@st.cache_data
def scan_edf_channels_from_bytes(file_bytes: bytes, filename: str) -> list[dict]:
    """Scan EDF channel metadata from uploaded bytes without loading data."""
    import tempfile
    import os
    from eeg_seizure_analyzer.io.edf_reader import scan_edf_channels

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return scan_edf_channels(tmp_path)
    finally:
        os.unlink(tmp_path)


@st.cache_data
def load_edf_file(file_bytes: bytes, filename: str, channels: tuple[int, ...] | None = None) -> EEGRecording:
    """Load an EDF file from uploaded bytes, optionally selecting channels."""
    import tempfile
    import os
    from eeg_seizure_analyzer.io.edf_reader import read_edf

    ch_list = list(channels) if channels is not None else None

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        recording = read_edf(tmp_path, channels=ch_list)
        recording.source_path = filename
        return recording
    finally:
        os.unlink(tmp_path)


@st.cache_data
def load_adicht_file(file_bytes: bytes, filename: str, channels: tuple[int, ...] | None = None) -> EEGRecording:
    """Load an adicht file from uploaded bytes (Windows only)."""
    import tempfile
    import os
    from eeg_seizure_analyzer.io.adicht_reader import read_adicht

    ch_list = list(channels) if channels is not None else None

    with tempfile.NamedTemporaryFile(suffix=".adicht", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        recording = read_adicht(tmp_path, channels=ch_list)
        recording.source_path = filename
        return recording
    finally:
        os.unlink(tmp_path)


def recording_info_card(recording: EEGRecording):
    """Display recording metadata in a card layout."""
    cols = st.columns(4)
    cols[0].metric("Channels", recording.n_channels)
    cols[1].metric("Duration", f"{recording.duration_sec:.1f}s ({recording.duration_sec / 3600:.2f}h)")
    cols[2].metric("Sampling Rate", f"{recording.fs:.0f} Hz")
    cols[3].metric("Annotations", len(recording.annotations))


def channel_selector(recording: EEGRecording, key: str = "channels") -> list[int]:
    """Multi-select widget for channels. Remembers selection across page switches."""
    options = {f"{i}: {name}": i for i, name in enumerate(recording.channel_names)}
    all_labels = list(options.keys())

    # Restore from persistent store
    store = _get_store()
    if key in store:
        stored = store[key]
        if isinstance(stored, list):
            # Validate stored labels still exist (recording may have changed)
            valid = [lbl for lbl in stored if lbl in options]
            # Respect explicit empty selection — only default to all if the
            # stored value contained labels that no longer exist (new recording)
            if not valid and stored:
                # Had labels but none match current recording → reset to all
                st.session_state[key] = all_labels
            else:
                st.session_state[key] = valid
        else:
            st.session_state[key] = all_labels
    elif key not in st.session_state:
        st.session_state[key] = all_labels

    selected = st.sidebar.multiselect(
        "Channels",
        options=all_labels,
        key=key,
        on_change=persist_widget_callback,
        args=(key,),
    )

    # Save to persistent store
    store[key] = selected

    return [options[s] for s in selected]


def is_windows() -> bool:
    return sys.platform == "win32"


# ── Shared event trace renderer ──────────────────────────────────────


def render_event_trace(
    recording: EEGRecording,
    event,
    context_sec: float = 10.0,
    title: str = "Event",
):
    """Render an EEG trace centred on a detected event.

    * Independent bandpass/notch filter controls (default ON, using
      the spike-train detection bandpass by default).
    * Uses the Viewer's Y-range for consistent amplitude scaling.
    * Both axes are freely zoomable; ``uirevision`` preserves zoom
      across checkbox toggles.
    * Optionally overlays detected spikes, baseline, threshold, and activity.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter, notch_filter

    st.markdown(f"#### {title}")

    store = _get_store()

    # ── Overlay toggles (stable keys — survive switching events) ──────
    ovl_cols = st.columns(4)
    apply_bp = ovl_cols[0].checkbox("Bandpass filter", value=True,
                                     key="evt_apply_bp")
    show_spikes = ovl_cols[1].checkbox("Show spikes", value=True,
                                        key="evt_show_spikes")
    show_baseline = ovl_cols[2].checkbox("Show baseline", value=False,
                                          key="evt_show_baseline")
    show_threshold = ovl_cols[3].checkbox("Show threshold", value=True,
                                           key="evt_show_threshold")

    # ── Time window: event ± context ──────────────────────────────────
    window_start = max(0, event.onset_sec - context_sec)
    window_end = min(recording.duration_sec, event.offset_sec + context_sec)

    start_idx = int(window_start * recording.fs)
    end_idx = min(int(window_end * recording.fs), recording.n_samples)

    ch = event.channel
    data = recording.data[ch, start_idx:end_idx].copy()
    time_axis = np.linspace(window_start, window_end, len(data))

    # ── Apply bandpass / notch if enabled ─────────────────────────────
    # Default to the spike-train detection bandpass, fall back to viewer
    if apply_bp:
        low = float(store.get("st_bp_low", store.get("viewer_filt_low", 1.0)))
        high = float(store.get("st_bp_high", store.get("viewer_filt_high", 100.0)))
        data = bandpass_filter(data, recording.fs, low, high)
    if store.get("viewer_notch", False):
        nf = float(store.get("viewer_notch_freq", 50.0))
        data = notch_filter(data, recording.fs, nf)

    # ── Y-range from Viewer ───────────────────────────────────────────
    y_range = float(store.get("viewer_yrange", float(np.ptp(data)) * 1.2))
    unit_label = ""
    if recording.units and ch < len(recording.units):
        unit_label = recording.units[ch]

    data_mean = float(np.mean(data))
    y_lower = data_mean - y_range / 2
    y_upper = data_mean + y_range / 2

    # ── Check for activity channel ────────────────────────────────────
    act_rec = st.session_state.get("activity_recording")
    pairings = st.session_state.get("channel_pairings")
    paired_act_idx = None
    if act_rec is not None and pairings is not None:
        for p in pairings:
            if p.eeg_index == ch and p.activity_index is not None:
                paired_act_idx = p.activity_index
                break

    has_act = paired_act_idx is not None

    # ── Build figure ──────────────────────────────────────────────────
    if has_act:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
        )
    else:
        fig = go.Figure()

    eeg_trace = go.Scattergl(
        x=time_axis,
        y=data,
        mode="lines",
        name=recording.channel_names[ch],
        line=dict(color="#1f77b4", width=1),
    )
    if has_act:
        fig.add_trace(eeg_trace, row=1, col=1)
    else:
        fig.add_trace(eeg_trace)

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

    # Onset / offset vertical lines
    fig.add_vline(x=event.onset_sec, line=dict(color=border_color, width=1.5, dash="dash"))
    fig.add_vline(x=event.offset_sec, line=dict(color=border_color, width=1.5, dash="dash"))

    # ── Overlay spikes / baseline / threshold ────────────────────────
    # Get from per-channel detection info (all spikes) or from event features
    det_info_all = st.session_state.get("st_detection_info", {})
    # Try both int and string keys for robustness
    det_info = det_info_all.get(ch, det_info_all.get(str(ch), {}))
    bl_mean = det_info.get("baseline_mean", event.features.get("baseline_mean", 0))
    bl_std = det_info.get("baseline_std", event.features.get("baseline_std", 0))
    threshold = det_info.get("threshold", event.features.get("threshold", 0))

    if show_baseline and bl_mean > 0:
        for sign in [1, -1]:
            bl_val = sign * bl_mean + data_mean
            bl_trace = go.Scatter(
                x=[window_start, window_end], y=[bl_val, bl_val],
                mode="lines",
                line=dict(color="green", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            )
            if has_act:
                fig.add_trace(bl_trace, row=1, col=1)
            else:
                fig.add_trace(bl_trace)

    if show_threshold and threshold > 0:
        for sign in [1, -1]:
            thr_val = sign * threshold + data_mean
            thr_trace = go.Scatter(
                x=[window_start, window_end], y=[thr_val, thr_val],
                mode="lines",
                line=dict(color="orange", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            )
            if has_act:
                fig.add_trace(thr_trace, row=1, col=1)
            else:
                fig.add_trace(thr_trace)

    if show_spikes:
        # Show ALL spikes in the window (from global detection info),
        # with different colors for in-event vs out-of-event spikes.
        # Global info has all spikes; event features only has in-event spikes.
        all_spike_times = det_info.get("all_spike_times", [])
        all_spike_samples = det_info.get("all_spike_samples", [])

        # Fallback: if no global detection info, use event-level data
        if not all_spike_times:
            all_spike_times = event.features.get("spike_times", [])
            all_spike_samples = event.features.get("spike_samples", [])

        # Split into in-event and out-of-event
        in_event_pts = []
        out_event_pts = []
        for t, s in zip(all_spike_times, all_spike_samples):
            if not (window_start <= t <= window_end):
                continue
            local = s - start_idx
            y_val = float(data[local]) if 0 <= local < len(data) else data_mean
            if event.onset_sec <= t <= event.offset_sec:
                in_event_pts.append((t, y_val))
            else:
                out_event_pts.append((t, y_val))

        # In-event spikes: red
        if in_event_pts:
            spike_trace = go.Scatter(
                x=[v[0] for v in in_event_pts],
                y=[v[1] for v in in_event_pts],
                mode="markers",
                marker=dict(color="red", size=5, symbol="circle"),
                showlegend=False,
                hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
            )
            if has_act:
                fig.add_trace(spike_trace, row=1, col=1)
            else:
                fig.add_trace(spike_trace)
        # Out-of-event spikes: orange, slightly smaller
        if out_event_pts:
            spike_trace_out = go.Scatter(
                x=[v[0] for v in out_event_pts],
                y=[v[1] for v in out_event_pts],
                mode="markers",
                marker=dict(color="orange", size=5, symbol="circle",
                            opacity=0.85),
                showlegend=False,
                hovertemplate="Spike @ %{x:.3f}s<extra></extra>",
            )
            if has_act:
                fig.add_trace(spike_trace_out, row=1, col=1)
            else:
                fig.add_trace(spike_trace_out)

    # ── Activity channel (bottom panel) ──────────────────────────────
    if has_act:
        act_start = int(window_start * act_rec.fs)
        act_end = min(int(window_end * act_rec.fs), act_rec.n_samples)
        act_data = act_rec.data[paired_act_idx, act_start:act_end]
        act_time = np.linspace(window_start, window_end, len(act_data))
        act_label = ""
        if pairings:
            for p in pairings:
                if p.activity_index == paired_act_idx:
                    act_label = p.activity_label or "Activity"
                    break
        fig.add_trace(
            go.Scattergl(
                x=act_time, y=act_data,
                mode="lines", name=act_label,
                line=dict(width=1, color="#e377c2"),
            ),
            row=2, col=1,
        )
        act_unit = act_rec.units[0] if act_rec.units else ""

    # ── Scale bar ─────────────────────────────────────────────────────
    nice_values = [float(m * 10 ** e) for e in range(-3, 6) for m in [1, 2, 5]]
    scale_val = min(nice_values, key=lambda v: abs(v - y_range / 4))
    if scale_val <= 0:
        scale_val = y_range / 4

    bar_x = window_end - (window_end - window_start) * 0.02
    bar_y_center = y_lower + y_range * 0.15
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
    if has_act:
        fig.add_trace(scale_trace, row=1, col=1)
    else:
        fig.add_trace(scale_trace)

    scale_text = f"{scale_val:.0f} {unit_label}" if scale_val >= 1 else f"{scale_val:.2g} {unit_label}"
    fig.add_annotation(
        x=bar_x, y=bar_y_center, text=scale_text.strip(),
        showarrow=False, xanchor="right", xshift=-8,
        font=dict(size=11, color="black"),
        row=1 if has_act else None,
        col=1 if has_act else None,
    )

    # ── Layout ────────────────────────────────────────────────────────
    # uirevision: tied to event identity so zoom resets on event switch
    # but persists across checkbox toggles within the same event
    ui_rev = f"{event.onset_sec:.4f}_{event.channel}"

    if has_act:
        fig.update_xaxes(title_text="Time (s)", fixedrange=False, row=2, col=1)
        fig.update_xaxes(fixedrange=False, row=1, col=1)
        fig.update_yaxes(
            title=unit_label, fixedrange=False,
            range=[y_lower, y_upper], zeroline=False,
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
            height=450,
            uirevision=ui_rev,
            title=dict(
                text=f"{recording.channel_names[ch]} — {event.onset_sec:.2f}s to "
                     f"{event.offset_sec:.2f}s ({event.duration_sec:.2f}s)",
                font_size=13,
            ),
            margin=dict(l=60, r=10, t=40, b=40),
            showlegend=False,
            dragmode="zoom",
        )
    else:
        fig.update_layout(
            height=350,
            uirevision=ui_rev,
            xaxis=dict(title="Time (s)", fixedrange=False),
            yaxis=dict(
                title=unit_label,
                fixedrange=False,
                range=[y_lower, y_upper],
                zeroline=False,
            ),
            title=dict(
                text=f"{recording.channel_names[ch]} — {event.onset_sec:.2f}s to "
                     f"{event.offset_sec:.2f}s ({event.duration_sec:.2f}s)",
                font_size=13,
            ),
            margin=dict(l=60, r=10, t=40, b=40),
            showlegend=False,
            dragmode="zoom",
        )

    st.plotly_chart(fig, use_container_width=True, key="evt_trace_chart",
                    config={"scrollZoom": True, "displayModeBar": True})

    # Event detail metrics
    detail_cols = st.columns(4)
    detail_cols[0].markdown(f"**Onset:** {event.onset_sec:.3f}s")
    detail_cols[1].markdown(f"**Offset:** {event.offset_sec:.3f}s")
    detail_cols[2].markdown(f"**Duration:** {event.duration_sec:.3f}s")
    if event.severity:
        detail_cols[3].markdown(f"**Severity:** {event.severity}")
    elif event.features.get("amplitude"):
        detail_cols[3].markdown(f"**Amplitude:** {event.features['amplitude']:.1f}")

    # ── Spectral analysis of the seizure segment ────────────────────
    if event.event_type == "seizure" and event.duration_sec > 0.5:
        _render_event_power_spectrum(
            recording, event, unit_label,
            window_start=window_start, window_end=window_end,
        )


def _render_event_power_spectrum(
    recording: EEGRecording,
    event,
    unit_label: str = "",
    window_start: float = 0.0,
    window_end: float | None = None,
):
    """Render spectrogram, band-power-over-time, and PSD for a seizure.

    The spectrogram and band-power plots share the same x-axis range as
    the EEG trace (window_start … window_end) for easy visual comparison.
    """
    import plotly.graph_objects as go
    from scipy.signal import welch as scipy_welch, spectrogram as scipy_spectrogram
    from eeg_seizure_analyzer.config import BANDS
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter, notch_filter

    fs = recording.fs
    ch = event.channel

    if window_end is None:
        window_end = min(recording.duration_sec, event.offset_sec + 10.0)

    # Use the same window as the EEG trace
    start_idx = int(window_start * fs)
    end_idx = min(int(window_end * fs), recording.n_samples)
    segment = recording.data[ch, start_idx:end_idx].copy()

    if len(segment) < int(fs * 0.5):
        return

    # Apply same filters as viewer
    store = _get_store()
    if store.get("viewer_filter", False):
        low = float(store.get("viewer_filt_low", 1.0))
        high = float(store.get("viewer_filt_high", 50.0))
        segment = bandpass_filter(segment, fs, low, high)
    if store.get("viewer_notch", False):
        nf = float(store.get("viewer_notch_freq", 50.0))
        segment = notch_filter(segment, fs, nf)

    psd_unit = f"{unit_label}²/Hz" if unit_label else "Power/Hz"

    band_colors_line = {
        "delta": "#1f77b4",
        "theta": "#ff7f0e",
        "alpha": "#2ca02c",
        "beta": "#d62728",
        "gamma_low": "#9467bd",
        "gamma_high": "#8c564b",
    }
    band_colors_fill = {
        "delta": "rgba(31, 119, 180, 0.15)",
        "theta": "rgba(255, 127, 14, 0.15)",
        "alpha": "rgba(44, 160, 44, 0.15)",
        "beta": "rgba(214, 39, 40, 0.15)",
        "gamma_low": "rgba(148, 103, 189, 0.15)",
        "gamma_high": "rgba(140, 86, 75, 0.15)",
    }

    # ── Shared: compute spectrogram over the full window ───────────────
    nperseg_spec = min(int(1.0 * fs), len(segment) // 4)
    nperseg_spec = max(nperseg_spec, int(0.25 * fs))
    noverlap = int(nperseg_spec * 0.9)

    f_spec, t_spec, Sxx = scipy_spectrogram(
        segment, fs=fs, nperseg=nperseg_spec, noverlap=noverlap,
    )

    # Limit to 0–100 Hz
    f_mask = f_spec <= 100.0
    f_spec = f_spec[f_mask]
    Sxx = Sxx[f_mask, :]

    # Absolute time axis (matching the EEG trace)
    t_spec_abs = t_spec + window_start

    # ── 1. Spectrogram ─────────────────────────────────────────────────
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Heatmap(
        x=t_spec_abs,
        y=f_spec,
        z=Sxx_db,
        colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.1f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
    ))

    fig_spec.add_vline(x=event.onset_sec, line=dict(color="red", width=2, dash="dash"))
    fig_spec.add_vline(x=event.offset_sec, line=dict(color="red", width=2, dash="dash"))

    fig_spec.update_layout(
        height=280,
        title=dict(text="Spectrogram", font_size=13),
        xaxis=dict(title="Time (s)", fixedrange=False, range=[window_start, window_end]),
        yaxis=dict(title="Frequency (Hz)", fixedrange=False),
        margin=dict(l=60, r=10, t=40, b=40),
        dragmode="zoom",
    )

    st.plotly_chart(fig_spec, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True,
    })

    # ── 2. Band power over time (same method as Spectral tab) ────────
    from eeg_seizure_analyzer.processing.spectral import compute_band_powers

    bp = compute_band_powers(segment, fs, window_sec=2.0, step_sec=1.0)
    # Shift time to absolute (matching EEG trace x-axis)
    bp_time = bp["time_sec"].values + window_start

    fig_bp = go.Figure()
    for col in bp.columns:
        if col == "time_sec":
            continue
        color = band_colors_line.get(col, "#888888")
        fig_bp.add_trace(go.Scatter(
            x=bp_time,
            y=bp[col],
            mode="lines",
            name=col,
            line=dict(color=color),
            stackgroup="bands",
        ))

    fig_bp.add_vline(x=event.onset_sec, line=dict(color="red", width=2, dash="dash"))
    fig_bp.add_vline(x=event.offset_sec, line=dict(color="red", width=2, dash="dash"))

    fig_bp.update_layout(
        height=280,
        title=dict(text="Band Power Over Time", font_size=13),
        xaxis=dict(title="Time (s)", fixedrange=False, range=[window_start, window_end]),
        yaxis=dict(title="Power", fixedrange=False),
        margin=dict(l=60, r=10, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font_size=10,
        ),
        dragmode="zoom",
    )

    st.plotly_chart(fig_bp, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True,
    })

    # ── 3. Power Spectral Density (seizure segment only) ───────────────
    sz_start = int((event.onset_sec - window_start) * fs)
    sz_end = int((event.offset_sec - window_start) * fs)
    sz_segment = segment[max(0, sz_start):min(len(segment), sz_end)]

    if len(sz_segment) < int(fs * 0.5):
        sz_segment = segment

    nperseg_psd = min(int(2 * fs), len(sz_segment))
    freqs, psd = scipy_welch(sz_segment, fs=fs, nperseg=nperseg_psd)

    freq_mask = freqs <= 100.0
    freqs = freqs[freq_mask]
    psd = psd[freq_mask]

    fig_psd = go.Figure()

    for band_name, (f_lo, f_hi) in BANDS.items():
        color = band_colors_fill.get(band_name, "rgba(128, 128, 128, 0.1)")
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        if np.any(band_mask):
            band_freqs = freqs[band_mask]
            band_psd = psd[band_mask]
            fig_psd.add_trace(go.Scatter(
                x=np.concatenate([band_freqs, band_freqs[::-1]]),
                y=np.concatenate([band_psd, np.zeros(len(band_psd))]),
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                name=f"{band_name} ({f_lo}–{f_hi} Hz)",
                hoverinfo="skip",
                showlegend=True,
            ))

    fig_psd.add_trace(go.Scattergl(
        x=freqs, y=psd,
        mode="lines",
        line=dict(color="#1f77b4", width=1.5),
        name="PSD", showlegend=False,
    ))

    band_powers = {}
    total_power = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else 1.0
    for band_name, (f_lo, f_hi) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if np.any(mask):
            band_powers[band_name] = float(np.trapezoid(psd[mask], freqs[mask]))

    fig_psd.update_layout(
        height=280,
        title=dict(text="Power Spectrum (seizure segment)", font_size=13),
        xaxis=dict(title="Frequency (Hz)", fixedrange=False),
        yaxis=dict(title=psd_unit, fixedrange=False, type="log"),
        margin=dict(l=60, r=10, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font_size=10,
        ),
        dragmode="zoom",
    )

    st.plotly_chart(fig_psd, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True,
    })

    # Band power percentages
    if band_powers:
        bp_cols = st.columns(len(band_powers))
        for col, (band_name, bp) in zip(bp_cols, band_powers.items()):
            pct = (bp / total_power * 100) if total_power > 0 else 0
            col.metric(band_name, f"{pct:.1f}%")
