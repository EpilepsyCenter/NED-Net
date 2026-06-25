"""Inference — run a trained model on new EEG recordings.

Processes each EEG channel independently (since each channel = one animal),
slides a window across the recording, runs the model, merges
overlapping predictions, and returns detected seizure events.
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
    _read_window_signal,
)
from eeg_seizure_analyzer.ml.train import load_trained_model


def predict_seizures(
    edf_path: str,
    model_name: str,
    channels: list[int] | None = None,
    threshold: float = 0.5,
    boundary_threshold: float | None = None,
    convulsive_threshold: float = 0.5,
    min_duration_sec: float = 3.0,
    merge_gap_sec: float = 2.0,
    overlap_sec: float = 15.0,
    progress_callback=None,
    convulsive_model_name: str | None = None,
) -> list[DetectedEvent]:
    """Run seizure detection on an EDF file using a trained model.

    Each EEG channel is processed independently (one channel = one animal).
    The model runs on single-channel input, optionally with paired activity.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file.
    model_name : str
        Name of the trained model to use.
    channels : list[int], optional
        EEG channel indices to process. None = auto-detect all EEG channels.
    threshold : float
        Probability threshold for seizure detection (channel 0, all seizures).
        An event must have a core whose probability exceeds this — keeps
        detection precise.
    boundary_threshold : float, optional
        Lower threshold used to *grow* each detected event's onset/offset
        outward along the probability curve (hysteresis). Captures the seizure's
        ramp-up/decay that sits below the detection ``threshold``, so boundaries
        aren't clipped short. None or >= ``threshold`` disables growing (the
        boundaries are exactly the detection-threshold crossings, as before).
    convulsive_threshold : float
        Threshold on the convulsive channel (1) mean probability for labelling a
        detected event convulsive vs non-convulsive.
    min_duration_sec : float
        Minimum seizure duration (shorter events are discarded).
    merge_gap_sec : float
        Merge predicted segments closer than this (seconds).
    overlap_sec : float
        Overlap between sliding windows (seconds).
    progress_callback : callable, optional
        Called with (current_step, total_steps) for progress reporting.
    convulsive_model_name : str, optional
        Name of a Stage-2 convulsive classifier. When set, each detected seizure
        crop is run through this classifier and its output *overrides* the
        detector's ch1 convulsive flag. When None, the ch1 fallback is used
        unchanged.

    Returns
    -------
    list[DetectedEvent]
    """
    # Load model
    model, metadata = load_trained_model(model_name)

    architecture = metadata.get("architecture", "unet")
    target_fs = metadata.get("target_fs", 250)
    window_sec = metadata.get("window_sec", 60)
    include_activity = metadata.get("include_activity", False)
    n_classes = metadata.get("n_classes", 1)
    has_convulsive = n_classes >= 2

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Get channel info and pairings
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, act_idx, pairings = auto_pair_channels(ch_info)

    if channels is not None:
        eeg_idx = channels

    if not eeg_idx:
        raise ValueError("No EEG channels found in the file.")

    # Build EEG→activity mapping
    eeg_to_act: dict[int, int | None] = {}
    for p in pairings:
        eeg_to_act[p.eeg_index] = p.activity_index

    # Load channel→animal ID mapping
    ch_ids = load_channel_ids(edf_path) or {}

    # Recording duration
    eeg_fs = ch_info[eeg_idx[0]]["fs"]
    rec_duration = ch_info[eeg_idx[0]]["n_samples"] / eeg_fs

    # Sliding window parameters
    target_samples = target_fs * window_sec
    stride_sec = window_sec - overlap_sec
    n_chunks = max(1, int(np.ceil((rec_duration - window_sec) / stride_sec)) + 1)

    total_steps = len(eeg_idx) * n_chunks
    current_step = 0

    all_events: list[DetectedEvent] = []
    event_id = 1

    # Process each channel independently
    for eeg_ch in eeg_idx:
        act_ch = eeg_to_act.get(eeg_ch) if include_activity else None
        animal_id = ch_ids.get(eeg_ch, "")

        # Accumulate predictions for this channel
        total_target_samples = int(rec_duration * target_fs)
        pred_sum = np.zeros(total_target_samples, dtype=np.float64)
        pred_count = np.zeros(total_target_samples, dtype=np.float64)
        # Convulsive channel accumulator (if model supports it)
        conv_sum = np.zeros(total_target_samples, dtype=np.float64) if has_convulsive else None

        for chunk_idx in range(n_chunks):
            start_sec = chunk_idx * stride_sec
            current_step += 1

            if progress_callback:
                progress_callback(current_step, total_steps)

            # Load single EEG channel
            rec = read_edf_window(
                edf_path, channels=[eeg_ch],
                start_sec=start_sec, duration_sec=window_sec,
            )
            eeg = _downsample(rec.data, rec.fs, target_fs)
            eeg = _normalize_channels(eeg)
            eeg = _pad_or_trim(eeg, target_samples)

            # Load paired activity if needed
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
                # probs shape: (n_classes, n_samples) or (n_samples,) for legacy

            # Accumulate — seizure channel
            if probs.ndim == 1:
                seizure_probs = probs
                conv_probs = None
            else:
                seizure_probs = probs[0]
                conv_probs = probs[1] if has_convulsive else None

            out_start = int(start_sec * target_fs)
            out_len = min(len(seizure_probs), total_target_samples - out_start)
            pred_sum[out_start:out_start + out_len] += seizure_probs[:out_len]
            pred_count[out_start:out_start + out_len] += 1.0
            if conv_probs is not None and conv_sum is not None:
                conv_sum[out_start:out_start + out_len] += conv_probs[:out_len]

        # Average overlapping predictions
        pred_count[pred_count == 0] = 1.0
        avg_probs = pred_sum / pred_count
        avg_conv = (conv_sum / pred_count) if conv_sum is not None else None

        # Extract events for this channel
        binary = (avg_probs > threshold).astype(int)
        ch_events = _extract_events(
            binary, avg_probs, target_fs,
            boundary_threshold=boundary_threshold,
            min_duration_sec=min_duration_sec,
            merge_gap_sec=merge_gap_sec,
            channel=eeg_ch,
            animal_id=animal_id,
            start_event_id=event_id,
            convulsive_probs=avg_conv,
            convulsive_threshold=convulsive_threshold,
            architecture=architecture,
        )
        event_id += len(ch_events)
        all_events.extend(ch_events)

    # ── Stage 2: convulsive classifier (cascade) ────────────────
    # If a dedicated convulsive classifier is selected, run it on each detected
    # seizure crop and override the detector's ch1 flag. Convulsive ⊂ seizure,
    # so the classifier only ever sees seizure crops — a balanced, well-posed
    # task compared with ch1 firing against the whole recording.
    if convulsive_model_name and all_events:
        _apply_convulsive_classifier(
            all_events, edf_path, convulsive_model_name,
            convulsive_threshold, device,
        )

    # Sort by onset
    all_events.sort(key=lambda e: (e.channel, e.onset_sec))

    return all_events


def _apply_convulsive_classifier(
    events: list[DetectedEvent],
    edf_path: str,
    convulsive_model_name: str,
    convulsive_threshold: float,
    device,
) -> None:
    """Override each event's convulsive flag using a Stage-2 classifier.

    Mutates ``events`` in place: sets ``features["convulsive_probability"]`` and
    ``features["convulsive"]``. The crop is the event's centred ``window_sec``
    window on its own channel (EEG only — the classifier ignores activity).
    """
    conv_model, conv_meta = load_trained_model(convulsive_model_name)
    conv_model = conv_model.to(device)
    c_fs = conv_meta.get("target_fs", 250)
    c_win = conv_meta.get("window_sec", 60)
    c_samples = int(c_fs * c_win)
    # Default the threshold to the model's trained optimum unless the caller
    # explicitly overrode it (i.e. passed something other than the 0.5 default).
    thr = convulsive_threshold
    if abs(convulsive_threshold - 0.5) < 1e-9:
        thr = float(conv_meta.get("best_threshold", 0.5))

    # Crop each event's centred window (EEG only).
    crops = []
    for ev in events:
        centre = (ev.onset_sec + ev.offset_sec) / 2.0
        start_sec = max(0.0, centre - c_win / 2.0)
        try:
            sig = _read_window_signal(
                edf_path, ev.channel, None, start_sec, c_win, c_fs, c_samples)
        except Exception:
            sig = np.zeros((1, c_samples), dtype=np.float32)
        crops.append(sig)

    if not crops:
        return

    batch = torch.from_numpy(np.stack(crops)).to(device)
    with torch.no_grad():
        logits = conv_model(batch)
        probs = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)

    for ev, prob in zip(events, probs):
        ev.features["convulsive_probability"] = round(float(prob), 3)
        ev.features["convulsive"] = bool(prob > thr)


def _extract_events(
    binary: np.ndarray,
    probs: np.ndarray,
    fs: int,
    boundary_threshold: float | None = None,
    min_duration_sec: float = 3.0,
    merge_gap_sec: float = 2.0,
    channel: int = 0,
    animal_id: str = "",
    start_event_id: int = 1,
    convulsive_probs: np.ndarray | None = None,
    convulsive_threshold: float = 0.5,
    architecture: str = "unet",
) -> list[DetectedEvent]:
    """Convert binary prediction array to DetectedEvent list.

    Parameters
    ----------
    binary : (n_samples,) binary array — the detection-threshold mask (cores)
    probs : (n_samples,) seizure probability array
    fs : sampling rate
    boundary_threshold : lower prob to grow each core's onset/offset out to
        (hysteresis). None disables growing.
    min_duration_sec : discard events shorter than this
    merge_gap_sec : merge events closer than this
    channel : EEG channel index
    animal_id : animal ID for this channel
    start_event_id : starting event ID
    convulsive_probs : (n_samples,) convulsive probability array, optional

    Returns
    -------
    list[DetectedEvent]
    """
    # Find contiguous detection cores (runs above the detection threshold)
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

    # Hysteresis: grow each core outward to where the probability falls below
    # the (lower) boundary threshold, capturing the seizure's ramp-up/decay that
    # sits under the detection threshold. A core only exists where prob exceeded
    # the (higher) detection threshold, so this never invents new events — it
    # only widens the ones already found. Disabled when boundary_threshold is
    # None or not below the core edges (then onset/offset stay at the crossings).
    if boundary_threshold is not None and segments:
        n = len(probs)
        grown = []
        for s, e in segments:
            ns = s
            while ns > 0 and probs[ns - 1] > boundary_threshold:
                ns -= 1
            ne = e
            while ne < n and probs[ne] > boundary_threshold:
                ne += 1
            grown.append((ns, ne))
        # Growing can reorder/overlap neighbours; sort so the merge below (which
        # assumes ascending starts) collapses any overlaps correctly.
        segments = sorted(grown)

    # Merge close segments (and always coalesce overlaps, which hysteresis
    # growing can create even when merge_gap is 0).
    merge_gap_samples = max(0, int(merge_gap_sec * fs))
    if len(segments) > 1:
        merged = [segments[0]]
        for s, e in segments[1:]:
            if s - merged[-1][1] <= merge_gap_samples:
                # extend; keep the later end (a nested segment must not shrink it)
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        segments = merged

    # Convert to events
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

        # Convulsive prediction for this segment
        method = "ml_bendr" if architecture == "bendr" else "ml_unet"
        feat = {
            "detection_method": method,
            "peak_probability": round(float(np.max(seg_probs)), 3),
            "mean_probability": round(confidence, 3),
        }
        if convulsive_probs is not None:
            seg_conv = convulsive_probs[seg_start:seg_end]
            conv_confidence = float(np.mean(seg_conv))
            feat["convulsive_probability"] = round(conv_confidence, 3)
            feat["convulsive"] = conv_confidence > convulsive_threshold

        events.append(DetectedEvent(
            onset_sec=round(onset_sec, 3),
            offset_sec=round(offset_sec, 3),
            duration_sec=round(duration_sec, 3),
            channel=channel,
            event_type="seizure",
            confidence=round(confidence, 3),
            features=feat,
            animal_id=animal_id,
            event_id=event_id,
            source="detector",
        ))
        event_id += 1

    return events
