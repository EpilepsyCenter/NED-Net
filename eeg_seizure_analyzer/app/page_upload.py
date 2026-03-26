"""Upload & Preview page — two-step flow: scan channels, then select and load."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from eeg_seizure_analyzer.app.components import (
    is_windows,
    load_adicht_file,
    load_edf_file,
    recording_info_card,
    scan_edf_channels_from_bytes,
)
from eeg_seizure_analyzer.io.edf_reader import auto_pair_channels


def render():
    st.header("Upload & Preview")

    # File type support depends on OS
    accepted_types = ["edf"]
    if is_windows():
        accepted_types.append("adicht")

    st.markdown(
        f"Supported formats: **{', '.join('.' + t for t in accepted_types)}**"
    )

    uploaded = st.file_uploader(
        "Upload EEG recording",
        type=accepted_types,
        help="Upload a .edf file (any OS) or .adicht file (Windows only)",
    )

    # Also support loading from a file path (useful for large files)
    with st.expander("Or load from file path"):
        file_path = st.text_input("File path", placeholder="/path/to/recording.edf")
        load_path_btn = st.button("Load from path")

    # ── Step 1: Scan channels ──────────────────────────────────────────

    file_bytes = None
    filename = None
    source_path = None

    if uploaded is not None:
        file_bytes = uploaded.read()
        filename = uploaded.name
    elif load_path_btn and file_path:
        source_path = file_path

    # Scan channels from uploaded file
    if file_bytes is not None and filename and filename.lower().endswith(".edf"):
        channel_info = scan_edf_channels_from_bytes(file_bytes, filename)
        st.session_state["_upload_channel_info"] = channel_info
        st.session_state["_upload_file_bytes"] = file_bytes
        st.session_state["_upload_filename"] = filename
        st.session_state["_upload_source_path"] = None

    # Scan channels from file path
    elif source_path and source_path.lower().endswith(".edf"):
        from eeg_seizure_analyzer.io.edf_reader import scan_edf_channels
        try:
            channel_info = scan_edf_channels(source_path)
            st.session_state["_upload_channel_info"] = channel_info
            st.session_state["_upload_file_bytes"] = None
            st.session_state["_upload_filename"] = source_path.split("/")[-1]
            st.session_state["_upload_source_path"] = source_path
        except Exception as e:
            st.error(f"Error scanning file: {e}")

    # ── Step 2: Display channel table & selection ──────────────────────

    channel_info = st.session_state.get("_upload_channel_info")

    if channel_info is not None and "recording" not in st.session_state:
        st.subheader("Channel Selection")

        # Build a display table
        df = pd.DataFrame(channel_info)
        df.columns = ["Index", "Label", "Unit", "Sampling Rate (Hz)", "Samples"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Try auto-pairing
        eeg_indices, act_indices, pairings = auto_pair_channels(channel_info)
        has_pairs = any(p.activity_index is not None for p in pairings)

        if has_pairs:
            st.success(
                f"**Auto-paired {len(pairings)} EEG channel(s) with "
                f"{len(act_indices)} activity channel(s).**"
            )
            pair_data = []
            for p in pairings:
                pair_data.append({
                    "EEG Channel": p.eeg_label,
                    "Activity Channel": p.activity_label if p.activity_index is not None else "—",
                })
            st.dataframe(pd.DataFrame(pair_data), use_container_width=True, hide_index=True)

            # Let user modify EEG channel selection
            eeg_labels = {f"{channel_info[idx]['index']}: {channel_info[idx]['label']}": idx
                          for idx in eeg_indices}
            selected_eeg = st.multiselect(
                "EEG channels to load",
                options=list(eeg_labels.keys()),
                default=list(eeg_labels.keys()),
                key="upload_eeg_select",
            )
            selected_eeg_indices = [eeg_labels[lbl] for lbl in selected_eeg]

            if not selected_eeg_indices:
                st.warning("Select at least one EEG channel.")
                return

            if st.button("Load channels", type="primary", key="load_paired_btn"):
                _load_paired_channels(selected_eeg_indices, act_indices, pairings)
        else:
            # No pairing found — fall back to rate-based selection
            st.caption("Select which channels to load. Channels must share the same sampling rate.")
            rates = sorted(set(ch["fs"] for ch in channel_info))
            rate_groups = {}
            for rate in rates:
                rate_groups[rate] = [ch for ch in channel_info if ch["fs"] == rate]

            if len(rates) > 1:
                st.info(
                    f"This file has **{len(rates)} different sampling rates**: "
                    f"{', '.join(f'{r:.0f} Hz ({len(rate_groups[r])} ch)' for r in rates)}. "
                    f"Only channels with the same rate can be loaded together."
                )

            selected_rate = st.selectbox(
                "Sampling rate group",
                options=rates,
                index=rates.index(max(rates)),
                format_func=lambda r: f"{r:.0f} Hz — {len(rate_groups[r])} channels ({', '.join(ch['label'] for ch in rate_groups[r][:4])}{'...' if len(rate_groups[r]) > 4 else ''})",
                key="upload_rate_select",
            )

            available = rate_groups[selected_rate]
            options_map = {f"{ch['index']}: {ch['label']}": ch["index"] for ch in available}
            default_selected = list(options_map.keys())

            selected_labels = st.multiselect(
                f"Channels to load ({selected_rate:.0f} Hz)",
                options=list(options_map.keys()),
                default=default_selected,
                key="upload_channel_select",
            )
            selected_indices = [options_map[lbl] for lbl in selected_labels]

            if not selected_indices:
                st.warning("Select at least one channel.")
                return

            if st.button("Load selected channels", type="primary", key="load_channels_btn"):
                _load_selected_channels(selected_indices)

    # ── Handle non-EDF files (adicht) — direct load ───────────────────

    if file_bytes is not None and filename and filename.lower().endswith(".adicht"):
        st.info("Loading .adicht file — all channels will be scanned.")
        if st.button("Load file", type="primary", key="load_adicht_btn"):
            with st.spinner(f"Loading {filename}..."):
                recording = load_adicht_file(file_bytes, filename)
            st.session_state["recording"] = recording
            st.session_state.pop("_upload_channel_info", None)
            st.rerun()

    if source_path and source_path.lower().endswith(".adicht"):
        if st.button("Load file", type="primary", key="load_adicht_path_btn"):
            with st.spinner(f"Loading {source_path}..."):
                from eeg_seizure_analyzer.io.adicht_reader import read_adicht
                recording = read_adicht(source_path)
                st.session_state["recording"] = recording
                st.session_state.pop("_upload_channel_info", None)
                st.rerun()

    # ── Display loaded recording info ─────────────────────────────────

    rec = st.session_state.get("recording")
    if rec is not None:
        st.divider()
        st.success(f"Loaded: **{rec.source_path}** — {rec.n_channels} channels @ {rec.fs:.0f} Hz")
        recording_info_card(rec)

        # Channel info table
        st.subheader("Loaded Channels")
        channel_data = {
            "Index": list(range(rec.n_channels)),
            "Name": rec.channel_names,
            "Units": rec.units,
        }
        st.dataframe(pd.DataFrame(channel_data), use_container_width=True)

        # Annotations table
        if rec.annotations:
            st.subheader(f"Annotations ({len(rec.annotations)})")
            ann_data = {
                "Time (s)": [f"{a.onset_sec:.2f}" for a in rec.annotations],
                "Duration (s)": [f"{a.duration_sec:.2f}" if a.duration_sec else "-" for a in rec.annotations],
                "Text": [a.text for a in rec.annotations],
                "Channel": [a.channel if a.channel is not None else "All" for a in rec.annotations],
            }
            st.dataframe(pd.DataFrame(ann_data), use_container_width=True, height=300)

        # Record boundaries
        if rec.records:
            st.subheader("Record Blocks")
            rec_data = {
                "Block": [r.index for r in rec.records],
                "Start Sample": [r.start_sample for r in rec.records],
                "Samples": [r.n_samples for r in rec.records],
                "Start Time": [str(r.start_time) if r.start_time else "-" for r in rec.records],
            }
            st.dataframe(pd.DataFrame(rec_data), use_container_width=True)

        # Option to reload with different channels
        if st.button("Change channel selection", key="change_channels_btn"):
            st.session_state.pop("recording", None)
            st.rerun()


def _load_paired_channels(
    eeg_indices: list[int],
    act_indices: list[int],
    pairings: list,
):
    """Load EEG + Activity channels as paired recordings."""
    file_bytes = st.session_state.get("_upload_file_bytes")
    filename = st.session_state.get("_upload_filename", "")
    source_path = st.session_state.get("_upload_source_path")

    with st.spinner(f"Loading {len(eeg_indices)} EEG + {len(act_indices)} activity channels..."):
        try:
            import tempfile
            import os

            if file_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                load_path = tmp_path
            elif source_path:
                load_path = source_path
                tmp_path = None
            else:
                st.error("No file data available.")
                return

            from eeg_seizure_analyzer.io.edf_reader import read_edf_paired
            eeg_rec, act_rec = read_edf_paired(load_path, eeg_indices, act_indices)
            eeg_rec.source_path = source_path or filename

            if tmp_path:
                os.unlink(tmp_path)

            st.session_state["recording"] = eeg_rec
            st.session_state["activity_recording"] = act_rec
            st.session_state["channel_pairings"] = pairings
            channel_info = st.session_state.get("_upload_channel_info")
            if channel_info:
                st.session_state["all_channels_info"] = channel_info
            st.session_state.pop("_upload_channel_info", None)
            st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")


def _load_selected_channels(selected_indices: list[int]):
    """Load the recording with the selected channel indices."""
    file_bytes = st.session_state.get("_upload_file_bytes")
    filename = st.session_state.get("_upload_filename", "")
    source_path = st.session_state.get("_upload_source_path")

    channels_tuple = tuple(selected_indices)  # tuple for cache key

    with st.spinner(f"Loading {len(selected_indices)} channels..."):
        try:
            if file_bytes is not None:
                recording = load_edf_file(file_bytes, filename, channels=channels_tuple)
            elif source_path:
                from eeg_seizure_analyzer.io.edf_reader import read_edf
                recording = read_edf(source_path, channels=list(selected_indices))
                recording.source_path = source_path
            else:
                st.error("No file data available.")
                return

            st.session_state["recording"] = recording
            # Store full channel info for activity channel selection
            channel_info = st.session_state.get("_upload_channel_info")
            if channel_info:
                st.session_state["all_channels_info"] = channel_info
            st.session_state.pop("_upload_channel_info", None)
            st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")
