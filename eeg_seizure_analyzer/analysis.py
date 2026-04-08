"""Unified analysis pipeline — wraps CNN inference for all analysis modes.

All three modes (single, batch, live) call ``process_chunk()`` which:
  1. Loads the EDF file
  2. Runs CNN sliding-window detection via the existing ``predict_seizures()``
  3. Classifies events (convulsive/non-convulsive, HVSW/HPD subtypes)
  4. Writes results to SQLite via ``db`` module

No CNN logic is reimplemented here — we only wrap ``ml.predict.predict_seizures``.
"""

from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import welch

from eeg_seizure_analyzer import db
from eeg_seizure_analyzer.detection.base import DetectedEvent
from eeg_seizure_analyzer.io.edf_reader import (
    scan_edf_channels,
    auto_pair_channels,
    read_edf_window,
)
from eeg_seizure_analyzer.io.channel_ids import load_channel_ids
from eeg_seizure_analyzer.ml.predict import predict_seizures
from eeg_seizure_analyzer.ml.train import list_models


# ---------------------------------------------------------------------------
# Thread-safe analysis status — polled by dcc.Interval callbacks
# ---------------------------------------------------------------------------

_analysis_status: dict = {
    "running": False,
    "paused": False,
    "cancel_requested": False,
    "mode": None,
    "total_files": 0,
    "processed_files": 0,
    "current_file": None,
    "start_time": None,
    "mean_file_sec": None,
    "last_error": None,
    # Per-file progress (for single mode)
    "file_progress_current": 0,
    "file_progress_total": 0,
}

_status_lock = threading.Lock()


def get_status() -> dict:
    """Return a snapshot of the analysis status dict."""
    with _status_lock:
        return dict(_analysis_status)


def _update_status(**kwargs):
    with _status_lock:
        _analysis_status.update(kwargs)


def _is_cancel_requested() -> bool:
    with _status_lock:
        return _analysis_status.get("cancel_requested", False)


def _is_paused() -> bool:
    with _status_lock:
        return _analysis_status.get("paused", False)


def request_cancel():
    """Request cancellation — current file finishes, then stop."""
    _update_status(cancel_requested=True)


def request_pause():
    """Pause processing after current file completes."""
    _update_status(paused=True)


def request_resume():
    """Resume paused processing."""
    _update_status(paused=False)


def reset_status():
    """Clear all status fields."""
    _update_status(
        running=False,
        paused=False,
        cancel_requested=False,
        mode=None,
        total_files=0,
        processed_files=0,
        current_file=None,
        start_time=None,
        mean_file_sec=None,
        last_error=None,
        file_progress_current=0,
        file_progress_total=0,
    )


# ---------------------------------------------------------------------------
# Event type classification — HVSW / HPD subtype thresholds
# ---------------------------------------------------------------------------

@dataclass
class ClassificationParams:
    """User-configurable thresholds for HVSW/HPD classification."""
    hvsw_max_freq_hz: float = 4.0
    hvsw_min_slow_wave_index: float = 0.5
    hpd_min_freq_hz: float = 15.0
    hpd_min_hf_index: float = 0.3


def classify_event_types(
    events: list[DetectedEvent],
    edf_path: str,
    eeg_channels: list[int],
    fs: float,
    params: ClassificationParams | None = None,
) -> list[DetectedEvent]:
    """Classify each event as convulsive/non-convulsive, then HVSW/HPD.

    If the model already predicts convulsive probability (2-class output),
    we use that. Otherwise we fall back to spectral heuristics.
    For non-convulsive events, spectral features determine HVSW vs HPD.

    Parameters
    ----------
    events : list[DetectedEvent]
        Events from predict_seizures (already have features dict).
    edf_path : str
        Path to the EDF file (for reading event windows).
    eeg_channels : list[int]
        EEG channel indices in the file.
    fs : float
        Sampling rate of the EEG data.
    params : ClassificationParams, optional
        Thresholds for HVSW/HPD classification.
    """
    if params is None:
        params = ClassificationParams()

    for ev in events:
        feat = ev.features or {}

        # --- Convulsive classification ---
        # Use model's convulsive prediction if available
        if "convulsive" in feat:
            if feat["convulsive"]:
                ev.features["seizure_subtype"] = "convulsive"
            else:
                ev.features["seizure_subtype"] = "non_convulsive"
        elif "convulsive_probability" in feat:
            if feat["convulsive_probability"] > 0.5:
                ev.features["seizure_subtype"] = "convulsive"
            else:
                ev.features["seizure_subtype"] = "non_convulsive"
        else:
            # No convulsive prediction — default to non-convulsive
            ev.features["seizure_subtype"] = "non_convulsive"

        # --- HVSW / HPD subtype for non-convulsive ---
        if ev.features.get("seizure_subtype") == "non_convulsive":
            subtype = _classify_nonconvulsive(
                edf_path, ev, eeg_channels, fs, params,
            )
            if subtype:
                ev.features["nonconvulsive_subtype"] = subtype

    return events


def _classify_nonconvulsive(
    edf_path: str,
    event: DetectedEvent,
    eeg_channels: list[int],
    fs: float,
    params: ClassificationParams,
) -> str | None:
    """Classify a non-convulsive event as HVSW, HPD, or None (unknown)."""
    try:
        # Read just the event window
        ch_idx = event.channel
        if ch_idx not in eeg_channels:
            ch_idx = eeg_channels[0] if eeg_channels else 0

        duration = max(event.duration_sec, 1.0)
        rec = read_edf_window(
            edf_path,
            channels=[ch_idx],
            start_sec=event.onset_sec,
            duration_sec=duration,
        )
        signal = rec.data[0]
        actual_fs = rec.fs

        # Compute PSD
        nperseg = min(int(2 * actual_fs), len(signal))
        if nperseg < 4:
            return None
        freqs, psd = welch(signal, fs=actual_fs, nperseg=nperseg)

        # Dominant frequency
        if len(psd) == 0:
            return None
        dominant_freq = freqs[np.argmax(psd)]

        # Slow-wave index: fraction of power below hvsw_max_freq_hz
        total_power = np.sum(psd)
        if total_power < 1e-15:
            return None
        slow_mask = freqs <= params.hvsw_max_freq_hz
        slow_wave_index = np.sum(psd[slow_mask]) / total_power

        # High-frequency index: fraction of power above hpd_min_freq_hz
        hf_mask = freqs >= params.hpd_min_freq_hz
        hf_index = np.sum(psd[hf_mask]) / total_power

        # Store computed features
        event.features["dominant_freq_hz"] = round(float(dominant_freq), 2)
        event.features["slow_wave_index"] = round(float(slow_wave_index), 3)
        event.features["hf_index"] = round(float(hf_index), 3)

        # Classify
        if dominant_freq < params.hvsw_max_freq_hz and slow_wave_index > params.hvsw_min_slow_wave_index:
            return "HVSW"
        elif dominant_freq > params.hpd_min_freq_hz and hf_index > params.hpd_min_hf_index:
            return "HPD"
        else:
            return None  # Unknown subtype

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Date extraction from file path
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # YYYYMMDD
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    # DDMMYYYY
    (re.compile(r"(\d{2})(\d{2})(\d{4})"), lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
    # YYYY-MM-DD or YYYY_MM_DD
    (re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})"), lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
]


def parse_date_from_path(path: str) -> str:
    """Try to extract a date string (YYYY-MM-DD) from filename.

    Returns empty string if no date pattern found.
    """
    stem = Path(path).stem
    for pattern, formatter in _DATE_PATTERNS:
        match = pattern.search(stem)
        if match:
            try:
                date_str = formatter(match)
                # Validate it's a real date
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                continue
    return ""


# ---------------------------------------------------------------------------
# Core: process_chunk — used by all three modes
# ---------------------------------------------------------------------------

def process_chunk(
    edf_path: str,
    model_name: str,
    confidence_threshold: float = 0.5,
    min_duration_sec: float = 5.0,
    merge_gap_sec: float = 2.0,
    mode: str = "single",
    cohort: str = "",
    group_id: str = "",
    classification_params: ClassificationParams | None = None,
    progress_callback=None,
    file_metadata: dict | None = None,
) -> dict:
    """Master detection function shared by all three analysis modes.

    Loads EDF, runs CNN detection, classifies events, writes to SQLite.
    Does not know or care how the file arrived.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file.
    model_name : str
        Name of the trained model to use.
    confidence_threshold : float
        CNN probability threshold.
    min_duration_sec : float
        Minimum seizure duration (seconds).
    merge_gap_sec : float
        Merge events closer than this (seconds).
    mode : str
        'single', 'batch', or 'live'.
    cohort : str
        Cohort identifier for the chunk.
    group_id : str
        Group identifier.
    classification_params : ClassificationParams, optional
        HVSW/HPD thresholds.
    progress_callback : callable, optional
        Called with (current_step, total_steps) for progress updates.
    file_metadata : dict, optional
        From batch_metadata Excel: {cohort, group_id, channel_ids}.
        Overrides cohort/group_id params and supplements channel IDs.

    Returns
    -------
    dict
        Summary: n_events, n_convulsive, n_nonconvulsive, n_hvsw, n_hpd,
        processing_sec, chunk_id.
    """
    t_start = time.time()

    # Apply batch metadata if provided
    if file_metadata:
        cohort = file_metadata.get("cohort", "") or cohort
        group_id = file_metadata.get("group_id", "") or group_id

    # Skip if already processed
    if str(edf_path) in db.get_processed_paths():
        return {"skipped": True, "reason": "already_processed"}

    # Get channel info
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, act_idx, pairings = auto_pair_channels(ch_info)

    if not eeg_idx:
        raise ValueError(f"No EEG channels found in {edf_path}")

    eeg_fs = ch_info[eeg_idx[0]]["fs"]
    rec_duration = ch_info[eeg_idx[0]]["n_samples"] / eeg_fs

    # Load channel→animal ID mapping (sidecar first, then batch metadata)
    ch_ids = load_channel_ids(edf_path) or {}
    if file_metadata and file_metadata.get("channel_ids"):
        # Batch metadata supplements — sidecar takes priority
        for ch_idx, animal_id in file_metadata["channel_ids"].items():
            ch_idx = int(ch_idx)
            if ch_idx not in ch_ids:
                ch_ids[ch_idx] = animal_id

    # Write chunk record
    chunk_id = db.write_chunk(edf_path, {
        "cohort": cohort,
        "group_id": group_id,
        "date": parse_date_from_path(edf_path),
        "chunk_start_sec": 0,
        "chunk_end_sec": rec_duration,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
    }, mode)

    try:
        # Run CNN detection (wraps existing predict_seizures)
        events = predict_seizures(
            edf_path=edf_path,
            model_name=model_name,
            channels=None,  # auto-detect all EEG channels
            threshold=confidence_threshold,
            min_duration_sec=min_duration_sec,
            merge_gap_sec=merge_gap_sec,
            progress_callback=progress_callback,
        )

        # Classify event types (convulsive, HVSW, HPD)
        events = classify_event_types(
            events, edf_path, eeg_idx, eeg_fs, classification_params,
        )

        # Determine hour-of-day for each event from file start time
        file_start_hour = _get_file_start_hour(edf_path)

        # Group events by animal and write to DB
        events_by_animal: dict[str, list[DetectedEvent]] = {}
        for ev in events:
            aid = ev.animal_id or ch_ids.get(ev.channel, "")
            events_by_animal.setdefault(aid, []).append(ev)

        # Write all events
        event_dicts = []
        for ev in events:
            feat = ev.features or {}
            subtype = feat.get("seizure_subtype", "non_convulsive")
            nonconv_subtype = feat.get("nonconvulsive_subtype")

            # Determine type and subtype for DB
            ev_type = "convulsive" if subtype == "convulsive" else "non_convulsive"
            ev_subtype = nonconv_subtype  # "HVSW", "HPD", or None

            # Compute hour of day
            hour = None
            if file_start_hour is not None:
                total_sec = ev.onset_sec
                hour = (file_start_hour + int(total_sec // 3600)) % 24

            event_dicts.append({
                "animal_id": ev.animal_id or ch_ids.get(ev.channel, ""),
                "date": parse_date_from_path(edf_path),
                "start_sec": ev.onset_sec,
                "end_sec": ev.offset_sec,
                "duration_sec": ev.duration_sec,
                "type": ev_type,
                "subtype": ev_subtype,
                "cnn_confidence": ev.confidence,
                "convulsive_confidence": feat.get("convulsive_probability", 0.0),
                "movement_flag": ev.movement_flag,
                "recording_day": None,
                "hour_of_day": hour,
            })

        db.write_events(chunk_id, event_dicts, source="seizure_cnn")

        # Write per-animal summaries
        for animal_id, animal_events in events_by_animal.items():
            n_conv = sum(
                1 for e in animal_events
                if (e.features or {}).get("seizure_subtype") == "convulsive"
            )
            n_nonconv = len(animal_events) - n_conv
            n_flagged = sum(1 for e in animal_events if e.movement_flag)
            total_dur = sum(e.duration_sec for e in animal_events)

            db.write_summary(chunk_id, animal_id or "", {
                "n_convulsive": n_conv,
                "n_nonconvulsive": n_nonconv,
                "n_flagged": n_flagged,
                "total_duration_sec": total_dur,
            })

        processing_sec = time.time() - t_start
        db.update_chunk_timing(chunk_id, processing_sec)

        # Build summary
        n_conv = sum(1 for d in event_dicts if d["type"] == "convulsive")
        n_nonconv = sum(1 for d in event_dicts if d["type"] == "non_convulsive")
        n_hvsw = sum(1 for d in event_dicts if d["subtype"] == "HVSW")
        n_hpd = sum(1 for d in event_dicts if d["subtype"] == "HPD")

        return {
            "skipped": False,
            "chunk_id": chunk_id,
            "n_events": len(event_dicts),
            "n_convulsive": n_conv,
            "n_nonconvulsive": n_nonconv,
            "n_hvsw": n_hvsw,
            "n_hpd": n_hpd,
            "n_flagged": sum(1 for d in event_dicts if d["movement_flag"]),
            "processing_sec": round(processing_sec, 1),
        }

    except Exception as e:
        db.mark_chunk_error(chunk_id, str(e))
        raise


def process_spike_chunk(
    edf_path: str,
    model_name: str,
    confidence_threshold: float = 0.5,
    min_duration_sec: float = 0.002,
    merge_gap_sec: float = 0.05,
    mode: str = "single",
    cohort: str = "",
    group_id: str = "",
    progress_callback=None,
    file_metadata: dict | None = None,
) -> dict:
    """Run IS CNN detection on an EDF file and write results to SQLite.

    Parallel to process_chunk() but for interictal spikes.
    """
    from eeg_seizure_analyzer.ml.spike_predict import predict_spikes

    t_start = time.time()

    if file_metadata:
        cohort = file_metadata.get("cohort", "") or cohort
        group_id = file_metadata.get("group_id", "") or group_id

    ch_info = scan_edf_channels(edf_path)
    eeg_idx, act_idx, pairings = auto_pair_channels(ch_info)

    if not eeg_idx:
        raise ValueError(f"No EEG channels found in {edf_path}")

    eeg_fs = ch_info[eeg_idx[0]]["fs"]
    rec_duration = ch_info[eeg_idx[0]]["n_samples"] / eeg_fs

    ch_ids = load_channel_ids(edf_path) or {}
    if file_metadata and file_metadata.get("channel_ids"):
        for ch_idx, animal_id in file_metadata["channel_ids"].items():
            ch_idx = int(ch_idx)
            if ch_idx not in ch_ids:
                ch_ids[ch_idx] = animal_id

    chunk_id = db.write_chunk(edf_path, {
        "cohort": cohort,
        "group_id": group_id,
        "date": parse_date_from_path(edf_path),
        "chunk_start_sec": 0,
        "chunk_end_sec": rec_duration,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
    }, mode)

    try:
        events = predict_spikes(
            edf_path=edf_path,
            model_name=model_name,
            channels=None,
            threshold=confidence_threshold,
            min_duration_sec=min_duration_sec,
            merge_gap_sec=merge_gap_sec,
            progress_callback=progress_callback,
        )

        file_start_hour = _get_file_start_hour(edf_path)

        event_dicts = []
        for ev in events:
            hour = None
            if file_start_hour is not None:
                hour = (file_start_hour + int(ev.onset_sec // 3600)) % 24

            event_dicts.append({
                "animal_id": ev.animal_id or ch_ids.get(ev.channel, ""),
                "date": parse_date_from_path(edf_path),
                "start_sec": ev.onset_sec,
                "end_sec": ev.offset_sec,
                "duration_sec": ev.duration_sec,
                "type": "interictal_spike",
                "subtype": None,
                "cnn_confidence": ev.confidence,
                "convulsive_confidence": 0.0,
                "movement_flag": ev.movement_flag,
                "recording_day": None,
                "hour_of_day": hour,
                "source": "spike_cnn",
            })

        db.write_events(chunk_id, event_dicts, source="spike_cnn")

        # Per-animal summaries
        events_by_animal: dict[str, list] = {}
        for d in event_dicts:
            events_by_animal.setdefault(d["animal_id"], []).append(d)

        for animal_id, aevents in events_by_animal.items():
            db.write_summary(chunk_id, animal_id or "", {
                "n_convulsive": 0,
                "n_nonconvulsive": len(aevents),
                "n_flagged": 0,
                "total_duration_sec": sum(e["duration_sec"] for e in aevents),
            })

        processing_sec = time.time() - t_start
        db.update_chunk_timing(chunk_id, processing_sec)

        return {
            "skipped": False,
            "chunk_id": chunk_id,
            "n_events": len(event_dicts),
            "n_spikes": len(event_dicts),
            "processing_sec": round(processing_sec, 1),
        }

    except Exception as e:
        db.mark_chunk_error(chunk_id, str(e))
        raise


def _get_file_start_hour(edf_path: str) -> int | None:
    """Try to extract file start hour from EDF header."""
    try:
        import pyedflib
        f = pyedflib.EdfReader(edf_path)
        try:
            start = f.getStartdatetime()
            return start.hour
        finally:
            f.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def scan_folder(
    folder: str,
    include_subfolders: bool = True,
) -> dict:
    """Scan a folder for EDF files and check which are already processed.

    Returns
    -------
    dict
        total: int, already_processed: int, to_process: int,
        files: list[str] (sorted chronologically by filename)
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return {"total": 0, "already_processed": 0, "to_process": 0, "files": []}

    pattern = "**/*.edf" if include_subfolders else "*.edf"
    edf_files = sorted(
        str(p) for p in folder_path.glob(pattern) if p.is_file()
    )

    processed = db.get_processed_paths()
    already = sum(1 for f in edf_files if str(f) in processed)

    return {
        "total": len(edf_files),
        "already_processed": already,
        "to_process": len(edf_files) - already,
        "files": edf_files,
    }


def run_batch(
    folder: str,
    model_name: str,
    confidence_threshold: float = 0.5,
    min_duration_sec: float = 5.0,
    merge_gap_sec: float = 2.0,
    include_subfolders: bool = True,
    cohort: str = "",
    group_id: str = "",
    classification_params: ClassificationParams | None = None,
    metadata_path: str | None = None,
    detection_type: str = "seizure",
):
    """Run batch analysis in the current thread.

    Updates _analysis_status as it progresses. Called from a background thread
    by the Dash callback. Respects pause/cancel requests.

    Parameters
    ----------
    metadata_path : str, optional
        Path to batch_metadata.xlsx. If provided, cohort/group_id/channel_ids
        are read from this file per-EDF.
    """
    # Load batch metadata if provided
    batch_meta: dict[str, dict] = {}
    if metadata_path:
        try:
            from eeg_seizure_analyzer.io.batch_metadata import load_metadata
            batch_meta = load_metadata(metadata_path)
        except Exception:
            pass

    scan = scan_folder(folder, include_subfolders)
    files = scan["files"]
    processed_paths = db.get_processed_paths()
    to_process = [f for f in files if str(f) not in processed_paths]

    _update_status(
        running=True,
        paused=False,
        cancel_requested=False,
        mode="batch",
        total_files=len(to_process),
        processed_files=0,
        current_file=None,
        start_time=time.time(),
        mean_file_sec=None,
        last_error=None,
    )

    elapsed_times = []

    for i, edf_path in enumerate(to_process):
        # Check cancel
        if _is_cancel_requested():
            break

        # Check pause — wait until resumed
        while _is_paused() and not _is_cancel_requested():
            time.sleep(0.5)

        if _is_cancel_requested():
            break

        _update_status(
            current_file=Path(edf_path).name,
            processed_files=i,
            file_progress_current=0,
            file_progress_total=0,
        )

        t0 = time.time()
        try:
            def _batch_progress(current, total):
                _update_status(
                    file_progress_current=current,
                    file_progress_total=total,
                )

            # Look up per-file metadata
            fname = Path(edf_path).name
            file_meta = batch_meta.get(fname)

            if detection_type == "spike":
                process_spike_chunk(
                    edf_path=edf_path,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    min_duration_sec=min_duration_sec,
                    merge_gap_sec=merge_gap_sec,
                    mode="batch",
                    cohort=cohort,
                    group_id=group_id,
                    progress_callback=_batch_progress,
                    file_metadata=file_meta,
                )
            else:
                process_chunk(
                    edf_path=edf_path,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    min_duration_sec=min_duration_sec,
                    merge_gap_sec=merge_gap_sec,
                    mode="batch",
                    cohort=cohort,
                    group_id=group_id,
                    classification_params=classification_params,
                    progress_callback=_batch_progress,
                    file_metadata=file_meta,
                )
        except Exception as e:
            _update_status(last_error=f"{Path(edf_path).name}: {e}")

        elapsed_times.append(time.time() - t0)
        mean_sec = sum(elapsed_times) / len(elapsed_times)
        _update_status(
            processed_files=i + 1,
            mean_file_sec=round(mean_sec, 1),
        )

    _update_status(
        running=False,
        current_file=None,
    )


# ---------------------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------------------

_live_thread: threading.Thread | None = None
_live_stop_event = threading.Event()


def start_live_monitoring(
    watch_folder: str,
    model_name: str,
    confidence_threshold: float = 0.5,
    min_duration_sec: float = 5.0,
    merge_gap_sec: float = 2.0,
    wait_sec: int = 30,
    process_backlog: bool = True,
    cohort: str = "",
    group_id: str = "",
    classification_params: ClassificationParams | None = None,
):
    """Start live monitoring in a background thread."""
    global _live_thread

    if _live_thread is not None and _live_thread.is_alive():
        return  # Already running

    _live_stop_event.clear()

    _live_thread = threading.Thread(
        target=_live_monitor_worker,
        args=(
            watch_folder, model_name, confidence_threshold,
            min_duration_sec, merge_gap_sec, wait_sec,
            process_backlog, cohort, group_id, classification_params,
        ),
        daemon=True,
    )
    _live_thread.start()


def stop_live_monitoring():
    """Stop live monitoring — waits for current file to finish."""
    _live_stop_event.set()
    _update_status(running=False)


def is_live_running() -> bool:
    return _live_thread is not None and _live_thread.is_alive()


def _live_monitor_worker(
    watch_folder, model_name, confidence_threshold,
    min_duration_sec, merge_gap_sec, wait_sec,
    process_backlog, cohort, group_id, classification_params,
):
    """Background thread for live monitoring."""
    _update_status(
        running=True,
        mode="live",
        start_time=time.time(),
        processed_files=0,
        total_files=0,
        last_error=None,
    )

    # Process backlog first
    if process_backlog:
        _update_status(current_file="Processing backlog...")
        scan = scan_folder(watch_folder, include_subfolders=True)
        processed_paths = db.get_processed_paths()
        backlog = [f for f in scan["files"] if str(f) not in processed_paths]

        for edf_path in backlog:
            if _live_stop_event.is_set():
                _update_status(running=False)
                return

            _update_status(current_file=Path(edf_path).name)
            try:
                process_chunk(
                    edf_path=edf_path,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    min_duration_sec=min_duration_sec,
                    merge_gap_sec=merge_gap_sec,
                    mode="live",
                    cohort=cohort,
                    group_id=group_id,
                    classification_params=classification_params,
                )
                with _status_lock:
                    _analysis_status["processed_files"] += 1
            except Exception as e:
                _update_status(last_error=f"{Path(edf_path).name}: {e}")

    # Start watchdog
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    except ImportError:
        _update_status(
            running=False,
            last_error="watchdog not installed. pip install watchdog",
        )
        return

    class _EDFHandler(FileSystemEventHandler):
        def on_created(self, event):
            if _live_stop_event.is_set():
                return
            if not isinstance(event, FileCreatedEvent):
                return
            if not event.src_path.lower().endswith(".edf"):
                return

            # Wait before processing to let LabChart finish writing
            _update_status(
                current_file=f"Waiting {wait_sec}s: {Path(event.src_path).name}",
            )
            for _ in range(wait_sec):
                if _live_stop_event.is_set():
                    return
                time.sleep(1)

            _update_status(current_file=Path(event.src_path).name)
            try:
                process_chunk(
                    edf_path=event.src_path,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    min_duration_sec=min_duration_sec,
                    merge_gap_sec=merge_gap_sec,
                    mode="live",
                    cohort=cohort,
                    group_id=group_id,
                    classification_params=classification_params,
                )
                with _status_lock:
                    _analysis_status["processed_files"] += 1
            except Exception as e:
                _update_status(last_error=f"{Path(event.src_path).name}: {e}")

    observer = Observer()
    observer.schedule(_EDFHandler(), watch_folder, recursive=True)
    observer.start()

    _update_status(current_file="Watching for new files...")

    try:
        while not _live_stop_event.is_set():
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
        _update_status(running=False, current_file=None)
