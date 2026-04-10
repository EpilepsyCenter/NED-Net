"""PyTorch Dataset for EEG seizure detection training.

Loads annotated EDF files from a dataset definition, extracts signal
windows, and builds training tensors.  Each sample is a fixed-length
chunk of multi-channel EEG (optionally + activity) with a per-sample
binary seizure mask as the label.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_seizure_analyzer.io.edf_reader import (
    read_edf_window,
    auto_pair_channels,
    scan_edf_channels,
)
from eeg_seizure_analyzer.io.channel_ids import load_channel_ids

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_FS = 250  # Hz — downsample everything to this
WINDOW_SEC = 60  # seconds per training window
CONTEXT_SEC = 10  # seconds of context before/after each event


@dataclass
class DatasetConfig:
    """Configuration for dataset construction."""

    target_fs: int = TARGET_FS
    window_sec: int = WINDOW_SEC
    context_sec: int = CONTEXT_SEC
    include_activity: bool = False
    neg_pos_ratio: float = 2.0  # negative windows per positive
    min_seizure_overlap: float = 0.5  # fraction of window that must be seizure
    augment: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

_SEIZURE_ANN_SUFFIX = "_ned_annotations.json"


def _load_annotations(edf_path: str) -> list[dict]:
    """Load seizure annotations for an EDF file."""
    stem = Path(edf_path).stem
    ann_path = Path(edf_path).parent / (stem + _SEIZURE_ANN_SUFFIX)
    if not ann_path.exists():
        return []
    with open(ann_path) as f:
        data = json.load(f)
    return data.get("annotations", [])


def _get_channel_info(edf_path: str) -> tuple[list[int], list[int]]:
    """Identify EEG and activity channel indices for an EDF file."""
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, act_idx, _ = auto_pair_channels(ch_info)
    return eeg_idx, act_idx


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------


def _downsample(data: np.ndarray, orig_fs: float, target_fs: int) -> np.ndarray:
    """Downsample multi-channel data using decimation.

    Parameters
    ----------
    data : (n_channels, n_samples)
    orig_fs : original sampling rate
    target_fs : target sampling rate

    Returns
    -------
    np.ndarray : (n_channels, n_samples_new)
    """
    if abs(orig_fs - target_fs) < 1.0:
        return data

    from scipy.signal import decimate

    factor = int(round(orig_fs / target_fs))
    if factor <= 1:
        return data

    # decimate works on last axis
    return decimate(data, factor, axis=1, zero_phase=True).astype(np.float32)


def _normalize_channels(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently.

    Parameters
    ----------
    data : (n_channels, n_samples)

    Returns
    -------
    np.ndarray : normalized data, same shape
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return ((data - mean) / std).astype(np.float32)


def _pad_or_trim(data: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to exact length along time axis.

    Parameters
    ----------
    data : (n_channels, n_samples)
    target_len : desired number of samples

    Returns
    -------
    np.ndarray : (n_channels, target_len)
    """
    n_ch, n_samp = data.shape
    if n_samp >= target_len:
        return data[:, :target_len]
    padded = np.zeros((n_ch, target_len), dtype=data.dtype)
    padded[:, :n_samp] = data
    return padded


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def _augment(eeg: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Apply random augmentations to a single training sample.

    Parameters
    ----------
    eeg : (n_channels, n_samples) — already normalized
    mask : (n_classes, n_samples) — multi-channel target mask
    rng : numpy random generator

    Returns
    -------
    (eeg, mask) — augmented, same shapes
    """
    # 1. Amplitude scaling (per-channel) — ±20%
    scales = rng.uniform(0.8, 1.2, size=(eeg.shape[0], 1)).astype(np.float32)
    eeg = eeg * scales

    # 2. Gaussian noise — small
    noise_std = rng.uniform(0.0, 0.05)
    eeg = eeg + rng.normal(0, noise_std, size=eeg.shape).astype(np.float32)

    # 3. Time shift — up to ±2 seconds (shift both signal and mask)
    max_shift = min(500, eeg.shape[1] // 10)  # samples at 250 Hz
    shift = rng.integers(-max_shift, max_shift + 1)
    if shift != 0:
        eeg = np.roll(eeg, shift, axis=1)
        mask = np.roll(mask, shift, axis=-1)
        # Zero-fill the wrapped portion
        if shift > 0:
            eeg[:, :shift] = 0
            mask[..., :shift] = 0
        else:
            eeg[:, shift:] = 0
            mask[..., shift:] = 0

    # 4. Channel dropout — zero out one channel with 15% probability
    if eeg.shape[0] > 1 and rng.random() < 0.15:
        drop_ch = rng.integers(0, eeg.shape[0])
        eeg[drop_ch] = 0

    return eeg, mask


# ---------------------------------------------------------------------------
# Window extraction plan
# ---------------------------------------------------------------------------


@dataclass
class WindowSpec:
    """Describes one single-channel training window to extract.

    Each channel is a separate animal, so every window is one EEG channel
    (+ optionally its paired activity channel).
    """

    edf_path: str
    start_sec: float
    duration_sec: float
    eeg_channel: int  # single EEG channel index
    act_channel: int | None  # paired activity channel (or None)
    is_positive: bool
    # For positive windows: list of (onset_sec, offset_sec) of seizures
    # (relative to window start)
    seizure_intervals: list[tuple[float, float]] = field(default_factory=list)
    # Subset of seizure_intervals that are convulsive
    convulsive_intervals: list[tuple[float, float]] = field(default_factory=list)
    animal_id: str = ""


def build_window_specs(
    dataset_def: dict,
    config: DatasetConfig,
) -> list[WindowSpec]:
    """Plan all training windows from a dataset definition.

    Each channel is treated as a separate animal.  For each channel
    in each file:
    - Positive windows: centred on each confirmed seizure on that channel
    - Negative windows: rejected events on that channel + random background

    Parameters
    ----------
    dataset_def : dict
        Dataset definition from dataset_store (has 'files' list).
    config : DatasetConfig

    Returns
    -------
    list[WindowSpec]
    """
    rng = np.random.default_rng(config.seed)
    specs: list[WindowSpec] = []

    for file_entry in dataset_def.get("files", []):
        edf_path = file_entry["edf_path"]
        if not os.path.exists(edf_path):
            continue

        # Load annotation data
        annotations = _load_annotations(edf_path)
        if not annotations:
            continue

        # Get channel info and pairings
        ch_info = scan_edf_channels(edf_path)
        eeg_idx, act_idx, pairings = auto_pair_channels(ch_info)
        if not eeg_idx:
            continue

        # Build EEG→activity mapping
        eeg_to_act: dict[int, int | None] = {}
        for p in pairings:
            eeg_to_act[p.eeg_index] = p.activity_index

        # Load channel→animal ID mapping
        ch_ids = load_channel_ids(edf_path) or {}

        # Recording duration
        rec_duration = ch_info[eeg_idx[0]]["n_samples"] / ch_info[eeg_idx[0]]["fs"]

        # Process each EEG channel independently (each = one animal)
        for eeg_ch in eeg_idx:
            animal_id = ch_ids.get(eeg_ch, f"ch{eeg_ch}")
            act_ch = eeg_to_act.get(eeg_ch) if config.include_activity else None

            # Filter annotations for this channel only
            ch_confirmed = [
                a for a in annotations
                if a.get("label") == "confirmed"
                and a.get("event_type") == "seizure"
                and a.get("channel") == eeg_ch
            ]
            ch_rejected = [
                a for a in annotations
                if a.get("label") == "rejected"
                and a.get("event_type") == "seizure"
                and a.get("channel") == eeg_ch
            ]

            if not ch_confirmed and not ch_rejected:
                continue  # no annotations on this channel

            # Seizure intervals for this channel (for overlap checks)
            ch_seizure_intervals = [
                (a["onset_sec"], a["offset_sec"]) for a in ch_confirmed
            ]
            # Convulsive subset
            ch_convulsive_intervals = [
                (a["onset_sec"], a["offset_sec"]) for a in ch_confirmed
                if (a.get("features") or {}).get("convulsive", False)
            ]

            # --- Positive windows (centred on confirmed seizures) ---
            for ann in ch_confirmed:
                onset = ann["onset_sec"]
                offset = ann["offset_sec"]

                centre = (onset + offset) / 2
                half_win = config.window_sec / 2
                win_start = max(0, centre - half_win)
                win_end = min(rec_duration, win_start + config.window_sec)
                win_start = max(0, win_end - config.window_sec)

                # All seizures on this channel within the window
                win_seizures = [
                    (max(s[0], win_start) - win_start,
                     min(s[1], win_end) - win_start)
                    for s in ch_seizure_intervals
                    if s[1] > win_start and s[0] < win_end
                ]
                # Convulsive seizures within the window
                win_convulsive = [
                    (max(s[0], win_start) - win_start,
                     min(s[1], win_end) - win_start)
                    for s in ch_convulsive_intervals
                    if s[1] > win_start and s[0] < win_end
                ]

                specs.append(WindowSpec(
                    edf_path=edf_path,
                    start_sec=win_start,
                    duration_sec=config.window_sec,
                    eeg_channel=eeg_ch,
                    act_channel=act_ch,
                    is_positive=True,
                    seizure_intervals=win_seizures,
                    convulsive_intervals=win_convulsive,
                    animal_id=animal_id,
                ))

            # --- Negative windows ---
            n_neg_needed = max(1, int(len(ch_confirmed) * config.neg_pos_ratio))

            # Hard negatives from rejected events
            for ann in ch_rejected[:n_neg_needed]:
                onset = ann["onset_sec"]
                offset = ann["offset_sec"]
                centre = (onset + offset) / 2
                half_win = config.window_sec / 2
                win_start = max(0, centre - half_win)
                win_end = min(rec_duration, win_start + config.window_sec)
                win_start = max(0, win_end - config.window_sec)

                specs.append(WindowSpec(
                    edf_path=edf_path,
                    start_sec=win_start,
                    duration_sec=config.window_sec,
                    eeg_channel=eeg_ch,
                    act_channel=act_ch,
                    is_positive=False,
                    seizure_intervals=[],
                    animal_id=animal_id,
                ))

            # Random background negatives
            n_random = n_neg_needed - len(ch_rejected[:n_neg_needed])
            if n_random > 0 and rec_duration > config.window_sec:
                max_start = rec_duration - config.window_sec
                for _ in range(n_random):
                    for _attempt in range(50):
                        start = rng.uniform(0, max_start)
                        end = start + config.window_sec
                        overlaps = any(
                            s[1] > start and s[0] < end
                            for s in ch_seizure_intervals
                        )
                        if not overlaps:
                            break

                    specs.append(WindowSpec(
                        edf_path=edf_path,
                        start_sec=start,
                        duration_sec=config.window_sec,
                        eeg_channel=eeg_ch,
                        act_channel=act_ch,
                        is_positive=False,
                        seizure_intervals=[],
                        animal_id=animal_id,
                    ))

    return specs


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SeizureDataset(Dataset):
    """PyTorch Dataset for seizure detection training.

    Each sample is a single EEG channel (= one animal) optionally
    paired with its activity channel.

    Each item returns:
        eeg : Tensor of shape (n_input_channels, n_samples)
              where n_input_channels = 1 (EEG) or 2 (EEG + activity)
        mask : Tensor of shape (n_samples,) — binary seizure mask
        meta : dict with file path, animal_id, channel, etc.
    """

    def __init__(
        self,
        specs: list[WindowSpec],
        config: DatasetConfig,
        augment: bool = True,
    ):
        self.specs = specs
        self.config = config
        self.augment = augment
        self.target_samples = config.target_fs * config.window_sec
        self.rng = np.random.default_rng(config.seed)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]

        # Load single EEG channel window from disk
        rec = read_edf_window(
            spec.edf_path,
            channels=[spec.eeg_channel],
            start_sec=spec.start_sec,
            duration_sec=spec.duration_sec,
        )

        # Downsample
        eeg = _downsample(rec.data, rec.fs, self.config.target_fs)

        # Normalize
        eeg = _normalize_channels(eeg)

        # Pad or trim to exact size — shape: (1, target_samples)
        eeg = _pad_or_trim(eeg, self.target_samples)

        # Build multi-channel mask: (n_classes, target_samples)
        # Channel 0 = seizure, channel 1 = convulsive
        n_classes = 2
        mask = np.zeros((n_classes, self.target_samples), dtype=np.float32)
        for onset, offset in spec.seizure_intervals:
            start_idx = int(onset * self.config.target_fs)
            end_idx = int(offset * self.config.target_fs)
            start_idx = max(0, min(start_idx, self.target_samples))
            end_idx = max(0, min(end_idx, self.target_samples))
            mask[0, start_idx:end_idx] = 1.0
        for onset, offset in spec.convulsive_intervals:
            start_idx = int(onset * self.config.target_fs)
            end_idx = int(offset * self.config.target_fs)
            start_idx = max(0, min(start_idx, self.target_samples))
            end_idx = max(0, min(end_idx, self.target_samples))
            mask[1, start_idx:end_idx] = 1.0

        # Load paired activity channel if requested
        if spec.act_channel is not None:
            try:
                act_rec = read_edf_window(
                    spec.edf_path,
                    channels=[spec.act_channel],
                    start_sec=spec.start_sec,
                    duration_sec=spec.duration_sec,
                )
                act = _downsample(act_rec.data, act_rec.fs, self.config.target_fs)
                act = _normalize_channels(act)
                act = _pad_or_trim(act, self.target_samples)
                # Stack: (2, target_samples) — EEG + activity
                eeg = np.concatenate([eeg, act], axis=0)
            except Exception:
                # Activity load failed — pad with zeros
                zeros = np.zeros((1, self.target_samples), dtype=np.float32)
                eeg = np.concatenate([eeg, zeros], axis=0)

        # Augmentation (no channel dropout for single-channel)
        if self.augment:
            eeg, mask = _augment(eeg, mask, self.rng)

        eeg_tensor = torch.from_numpy(eeg)
        mask_tensor = torch.from_numpy(mask)

        meta = {
            "edf_path": spec.edf_path,
            "start_sec": spec.start_sec,
            "animal_id": spec.animal_id,
            "eeg_channel": spec.eeg_channel,
            "is_positive": spec.is_positive,
        }

        return eeg_tensor, mask_tensor, meta


# ---------------------------------------------------------------------------
# Split by animal
# ---------------------------------------------------------------------------


def split_by_animal(
    specs: list[WindowSpec],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[WindowSpec], list[WindowSpec]]:
    """Split window specs into train/val sets by animal ID.

    All windows from the same animal go into the same set to prevent
    data leakage.

    Parameters
    ----------
    specs : all window specs
    val_fraction : approximate fraction for validation
    seed : random seed

    Returns
    -------
    (train_specs, val_specs)
    """
    rng = random.Random(seed)

    # Group by animal
    animals: dict[str, list[WindowSpec]] = {}
    for s in specs:
        # animal_id is set from channel mapping; fallback to file+channel
        aid = s.animal_id or f"{s.edf_path}:ch{s.eeg_channel}"
        animals.setdefault(aid, []).append(s)

    animal_ids = sorted(animals.keys())
    rng.shuffle(animal_ids)

    if len(animal_ids) < 2:
        raise ValueError(
            "Training requires annotations from at least 2 different animals "
            "so the data can be split into train and validation sets without "
            "data leakage. Found only 1 animal. Add recordings from more "
            "animals to the dataset, or assign distinct Animal IDs to "
            "different channels on the Load tab."
        )

    # Allocate animals to val until we hit the target fraction
    total = len(specs)
    val_count = 0
    val_animals = set()

    for aid in animal_ids:
        if val_count / max(total, 1) >= val_fraction:
            break
        val_animals.add(aid)
        val_count += len(animals[aid])

    # Ensure at least one animal in val
    if not val_animals:
        val_animals.add(animal_ids[0])

    # Ensure at least one animal in train
    if val_animals == set(animal_ids):
        val_animals.discard(animal_ids[-1])

    train_specs = []
    val_specs = []
    for aid, window_specs in animals.items():
        if aid in val_animals:
            val_specs.extend(window_specs)
        else:
            train_specs.extend(window_specs)

    return train_specs, val_specs


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------


def build_datasets(
    dataset_def: dict,
    config: DatasetConfig | None = None,
) -> tuple[SeizureDataset, SeizureDataset, DatasetConfig]:
    """Build train and validation PyTorch Datasets from a dataset definition.

    Parameters
    ----------
    dataset_def : dict
        From dataset_store.load_dataset().
    config : DatasetConfig, optional

    Returns
    -------
    (train_dataset, val_dataset, config)
    """
    if config is None:
        config = DatasetConfig()

    specs = build_window_specs(dataset_def, config)

    if not specs:
        raise ValueError("No training windows could be extracted. "
                         "Check that the dataset has confirmed annotations.")

    train_specs, val_specs = split_by_animal(specs, seed=config.seed)

    train_ds = SeizureDataset(train_specs, config, augment=config.augment)
    val_ds = SeizureDataset(val_specs, config, augment=False)

    return train_ds, val_ds, config
