"""PyTorch Dataset for interictal spike (IS) detection training.

Same architecture as seizure detection (1D U-Net with per-sample masks),
but with shorter windows (4s) and spike-duration labels (2–70ms events).

Loads annotated EDF files, extracts windows centred on confirmed/rejected
spikes, and builds training tensors.
"""

from __future__ import annotations

import json
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
from eeg_seizure_analyzer.ml.dataset import (
    _downsample,
    _normalize_channels,
    _pad_or_trim,
    _augment,
    split_by_animal,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_FS = 250
WINDOW_SEC = 4  # shorter windows for spike context
CONTEXT_SEC = 1  # context before/after spike

_SPIKE_ANN_SUFFIX = "_ned_spike_annotations.json"


@dataclass
class SpikeDatasetConfig:
    """Configuration for IS dataset construction."""

    target_fs: int = TARGET_FS
    window_sec: int = WINDOW_SEC
    context_sec: int = CONTEXT_SEC
    include_activity: bool = False
    neg_pos_ratio: float = 2.0
    augment: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _load_spike_annotations(edf_path: str) -> list[dict]:
    """Load spike annotations for an EDF file."""
    stem = Path(edf_path).stem
    ann_path = Path(edf_path).parent / (stem + _SPIKE_ANN_SUFFIX)
    if not ann_path.exists():
        return []
    with open(ann_path) as f:
        data = json.load(f)
    return data.get("annotations", [])


# ---------------------------------------------------------------------------
# Window specs
# ---------------------------------------------------------------------------


@dataclass
class SpikeWindowSpec:
    """Describes one single-channel training window for IS detection."""

    edf_path: str
    start_sec: float
    duration_sec: float
    eeg_channel: int
    act_channel: int | None
    is_positive: bool
    spike_intervals: list[tuple[float, float]] = field(default_factory=list)
    animal_id: str = ""


def build_spike_window_specs(
    dataset_def: dict,
    config: SpikeDatasetConfig,
) -> list[SpikeWindowSpec]:
    """Plan training windows from spike annotations.

    For each channel in each file:
    - Positive windows: centred on each confirmed spike
    - Negative windows: rejected spikes + random background
    """
    rng = np.random.default_rng(config.seed)
    specs: list[SpikeWindowSpec] = []

    for file_entry in dataset_def.get("files", []):
        edf_path = file_entry["edf_path"]
        if not os.path.exists(edf_path):
            continue

        annotations = _load_spike_annotations(edf_path)
        if not annotations:
            continue

        ch_info = scan_edf_channels(edf_path)
        eeg_idx, act_idx, pairings = auto_pair_channels(ch_info)
        if not eeg_idx:
            continue

        eeg_to_act: dict[int, int | None] = {}
        for p in pairings:
            eeg_to_act[p.eeg_index] = p.activity_index

        ch_ids = load_channel_ids(edf_path) or {}
        rec_duration = ch_info[eeg_idx[0]]["n_samples"] / ch_info[eeg_idx[0]]["fs"]

        for eeg_ch in eeg_idx:
            animal_id = ch_ids.get(eeg_ch, f"ch{eeg_ch}")
            act_ch = eeg_to_act.get(eeg_ch) if config.include_activity else None

            ch_confirmed = [
                a for a in annotations
                if a.get("label") == "confirmed"
                and a.get("channel") == eeg_ch
            ]
            ch_rejected = [
                a for a in annotations
                if a.get("label") == "rejected"
                and a.get("channel") == eeg_ch
            ]

            if not ch_confirmed and not ch_rejected:
                continue

            ch_spike_intervals = [
                (a["onset_sec"], a["offset_sec"]) for a in ch_confirmed
            ]

            # --- Positive windows (centred on confirmed spikes) ---
            for ann in ch_confirmed:
                onset = ann["onset_sec"]
                offset = ann["offset_sec"]
                centre = (onset + offset) / 2
                half_win = config.window_sec / 2

                win_start = max(0, centre - half_win)
                win_end = min(rec_duration, win_start + config.window_sec)
                win_start = max(0, win_end - config.window_sec)

                # All confirmed spikes within this window
                win_spikes = [
                    (max(s[0], win_start) - win_start,
                     min(s[1], win_end) - win_start)
                    for s in ch_spike_intervals
                    if s[1] > win_start and s[0] < win_end
                ]

                specs.append(SpikeWindowSpec(
                    edf_path=edf_path,
                    start_sec=win_start,
                    duration_sec=config.window_sec,
                    eeg_channel=eeg_ch,
                    act_channel=act_ch,
                    is_positive=True,
                    spike_intervals=win_spikes,
                    animal_id=animal_id,
                ))

            # --- Negative windows ---
            n_neg_needed = max(1, int(len(ch_confirmed) * config.neg_pos_ratio))

            # Hard negatives from rejected spikes
            for ann in ch_rejected[:n_neg_needed]:
                onset = ann["onset_sec"]
                offset = ann["offset_sec"]
                centre = (onset + offset) / 2
                half_win = config.window_sec / 2

                win_start = max(0, centre - half_win)
                win_end = min(rec_duration, win_start + config.window_sec)
                win_start = max(0, win_end - config.window_sec)

                specs.append(SpikeWindowSpec(
                    edf_path=edf_path,
                    start_sec=win_start,
                    duration_sec=config.window_sec,
                    eeg_channel=eeg_ch,
                    act_channel=act_ch,
                    is_positive=False,
                    spike_intervals=[],
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
                            for s in ch_spike_intervals
                        )
                        if not overlaps:
                            break
                    specs.append(SpikeWindowSpec(
                        edf_path=edf_path,
                        start_sec=start,
                        duration_sec=config.window_sec,
                        eeg_channel=eeg_ch,
                        act_channel=act_ch,
                        is_positive=False,
                        spike_intervals=[],
                        animal_id=animal_id,
                    ))

    return specs


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SpikeDataset(Dataset):
    """PyTorch dataset for IS detection training.

    Each sample is a short EEG window (default 4s at 250Hz = 1000 samples)
    with a binary spike mask as the label.
    """

    def __init__(
        self,
        specs: list[SpikeWindowSpec],
        config: SpikeDatasetConfig,
        augment: bool = True,
    ):
        self.specs = specs
        self.config = config
        self.augment = augment
        self._rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        target_samples = self.config.target_fs * self.config.window_sec

        # Load single EEG channel
        channels = [spec.eeg_channel]
        if spec.act_channel is not None:
            channels.append(spec.act_channel)

        try:
            rec = read_edf_window(
                spec.edf_path,
                channels=[spec.eeg_channel],
                start_sec=spec.start_sec,
                duration_sec=spec.duration_sec,
            )
            eeg = _downsample(rec.data, rec.fs, self.config.target_fs)
            eeg = _normalize_channels(eeg)
            eeg = _pad_or_trim(eeg, target_samples)
        except Exception:
            eeg = np.zeros((1, target_samples), dtype=np.float32)

        # Activity channel
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
                act = _pad_or_trim(act, target_samples)
                eeg = np.concatenate([eeg, act], axis=0)
            except Exception:
                eeg = np.concatenate(
                    [eeg, np.zeros((1, target_samples), dtype=np.float32)],
                    axis=0,
                )

        # Build spike mask — single channel (spike/no-spike)
        mask = np.zeros((1, target_samples), dtype=np.float32)
        for onset, offset in spec.spike_intervals:
            s_start = int(onset * self.config.target_fs)
            s_end = int(offset * self.config.target_fs)
            # Ensure at least 1 sample is marked
            s_end = max(s_end, s_start + 1)
            s_start = max(0, min(s_start, target_samples))
            s_end = max(0, min(s_end, target_samples))
            mask[0, s_start:s_end] = 1.0

        # Augment
        if self.augment and self._rng.random() > 0.2:
            eeg, mask = _augment(eeg, mask, self._rng)

        meta = {
            "edf_path": spec.edf_path,
            "start_sec": spec.start_sec,
            "channel": spec.eeg_channel,
            "is_positive": spec.is_positive,
            "animal_id": spec.animal_id,
        }

        return (
            torch.from_numpy(eeg),
            torch.from_numpy(mask),
            meta,
        )


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def build_spike_datasets(
    dataset_def: dict,
    config: SpikeDatasetConfig | None = None,
) -> tuple[SpikeDataset, SpikeDataset, SpikeDatasetConfig]:
    """Build train and validation IS datasets from a dataset definition.

    Returns (train_dataset, val_dataset, config).
    """
    if config is None:
        config = SpikeDatasetConfig()

    specs = build_spike_window_specs(dataset_def, config)

    # Split by animal to avoid data leakage
    # Reuse the seizure split helper — it works on any spec with animal_id
    train_specs, val_specs = split_by_animal(specs)

    train_ds = SpikeDataset(train_specs, config, augment=config.augment)
    val_ds = SpikeDataset(val_specs, config, augment=False)

    return train_ds, val_ds, config
