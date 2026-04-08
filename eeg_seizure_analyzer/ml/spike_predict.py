"""Inference — run a trained IS model on EEG recordings.

Same sliding-window approach as seizure prediction but with:
- Shorter windows (4s default)
- Much shorter min_duration (2ms) and merge_gap (50ms)
- Events tagged as source="spike_cnn" / event_type="interictal_spike"
"""

from __future__ import annotations

import numpy as np
import torch

from eeg_seizure_analyzer.detection.base import DetectedEvent
from eeg_seizure_analyzer.io.edf_reader import (
    read_edf_window,
    scan_edf_channels,
    auto_pair_channels,
)
from eeg_seizure_analyzer.io.channel_ids import load_channel_ids
from eeg_seizure_analyzer.ml.dataset import (
    _downsample,
    _normalize_channels,
    _pad_or_trim,
)
from eeg_seizure_analyzer.ml.train import load_trained_model


def predict_spikes(
    edf_path: str,
    model_name: str,
    channels: list[int] | None = None,
    threshold: float = 0.5,
    min_duration_sec: float = 0.002,
    merge_gap_sec: float = 0.05,
    overlap_sec: float = 1.0,
    progress_callback=None,
) -> list[DetectedEvent]:
    """Run interictal spike detection on an EDF file using a trained model.

    Parameters
    ----------
    edf_path : str
        Path to EDF file.
    model_name : str
        Name of the trained IS model.
    channels : list[int], optional
        EEG channel indices. None = auto-detect.
    threshold : float
        Probability threshold for spike detection.
    min_duration_sec : float
        Minimum spike duration (default 2ms).
    merge_gap_sec : float
        Merge spikes closer than this (default 50ms).
    overlap_sec : float
        Overlap between sliding windows (default 1s).
    progress_callback : callable, optional
        Called with (current_step, total_steps).

    Returns
    -------
    list[DetectedEvent]
    """
    model, metadata = load_trained_model(model_name)

    target_fs = metadata.get("target_fs", 250)
    window_sec = metadata.get("window_sec", 4)
    include_activity = metadata.get("include_activity", False)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Channel info
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, act_idx, pairings = auto_pair_channels(ch_info)

    if channels is not None:
        eeg_idx = channels
    if not eeg_idx:
        raise ValueError("No EEG channels found in the file.")

    eeg_to_act: dict[int, int | None] = {}
    for p in pairings:
        eeg_to_act[p.eeg_index] = p.activity_index

    ch_ids = load_channel_ids(edf_path) or {}

    eeg_fs = ch_info[eeg_idx[0]]["fs"]
    rec_duration = ch_info[eeg_idx[0]]["n_samples"] / eeg_fs

    target_samples = target_fs * window_sec
    stride_sec = window_sec - overlap_sec
    n_chunks = max(1, int(np.ceil((rec_duration - window_sec) / stride_sec)) + 1)

    total_steps = len(eeg_idx) * n_chunks
    current_step = 0

    all_events: list[DetectedEvent] = []
    event_id = 1

    for eeg_ch in eeg_idx:
        act_ch = eeg_to_act.get(eeg_ch) if include_activity else None
        animal_id = ch_ids.get(eeg_ch, "")

        total_target_samples = int(rec_duration * target_fs)
        pred_sum = np.zeros(total_target_samples, dtype=np.float64)
        pred_count = np.zeros(total_target_samples, dtype=np.float64)

        for chunk_idx in range(n_chunks):
            start_sec = chunk_idx * stride_sec
            current_step += 1

            if progress_callback:
                progress_callback(current_step, total_steps)

            # Load EEG
            rec = read_edf_window(
                edf_path, channels=[eeg_ch],
                start_sec=start_sec, duration_sec=window_sec,
            )
            eeg = _downsample(rec.data, rec.fs, target_fs)
            eeg = _normalize_channels(eeg)
            eeg = _pad_or_trim(eeg, target_samples)

            # Activity channel
            if act_ch is not None:
                try:
                    act_rec = read_edf_window(
                        edf_path, channels=[act_ch],
                        start_sec=start_sec, duration_sec=window_sec,
                    )
                    act = _downsample(act_rec.data, act_rec.fs, target_fs)
                    act = _normalize_channels(act)
                    act = _pad_or_trim(act, target_samples)
                    eeg = np.concatenate([eeg, act], axis=0)
                except Exception:
                    zeros = np.zeros((1, target_samples), dtype=np.float32)
                    eeg = np.concatenate([eeg, zeros], axis=0)

            # Run model
            with torch.no_grad():
                x = torch.from_numpy(eeg).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            # Handle shape — n_classes=1 for IS
            if probs.ndim == 2:
                spike_probs = probs[0]
            else:
                spike_probs = probs

            out_start = int(start_sec * target_fs)
            out_len = min(len(spike_probs), total_target_samples - out_start)
            pred_sum[out_start:out_start + out_len] += spike_probs[:out_len]
            pred_count[out_start:out_start + out_len] += 1.0

        # Average overlapping predictions
        pred_count[pred_count == 0] = 1.0
        avg_probs = pred_sum / pred_count

        # Extract spike events
        binary = (avg_probs > threshold).astype(int)
        ch_events = _extract_spike_events(
            binary, avg_probs, target_fs,
            min_duration_sec=min_duration_sec,
            merge_gap_sec=merge_gap_sec,
            channel=eeg_ch,
            animal_id=animal_id,
            start_event_id=event_id,
        )
        event_id += len(ch_events)
        all_events.extend(ch_events)

    all_events.sort(key=lambda e: (e.channel, e.onset_sec))
    return all_events


def _extract_spike_events(
    binary: np.ndarray,
    probs: np.ndarray,
    fs: int,
    min_duration_sec: float = 0.002,
    merge_gap_sec: float = 0.05,
    channel: int = 0,
    animal_id: str = "",
    start_event_id: int = 1,
) -> list[DetectedEvent]:
    """Convert binary prediction array to DetectedEvent list for IS."""
    # Find contiguous segments
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary)))

    # Merge close segments
    merge_gap_samples = int(merge_gap_sec * fs)
    if merge_gap_samples > 0 and len(segments) > 1:
        merged = [segments[0]]
        for s, e in segments[1:]:
            if s - merged[-1][1] <= merge_gap_samples:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        segments = merged

    events = []
    event_id = start_event_id

    for seg_start, seg_end in segments:
        onset_sec = seg_start / fs
        offset_sec = seg_end / fs
        duration_sec = offset_sec - onset_sec

        if duration_sec < min_duration_sec:
            continue

        seg_probs = probs[seg_start:seg_end]
        confidence = float(np.mean(seg_probs))

        events.append(DetectedEvent(
            onset_sec=round(onset_sec, 4),
            offset_sec=round(offset_sec, 4),
            duration_sec=round(duration_sec, 4),
            channel=channel,
            event_type="interictal_spike",
            confidence=round(confidence, 3),
            features={
                "detection_method": "ml_unet_spike",
                "peak_probability": round(float(np.max(seg_probs)), 3),
                "mean_probability": round(confidence, 3),
                "duration_ms": round(duration_sec * 1000, 1),
            },
            animal_id=animal_id,
            event_id=event_id,
            source="spike_cnn",
        ))
        event_id += 1

    return events
