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
    neg_pos_ratio: float = 2.0  # negative windows per positive (global target);
    #                             hard negatives preferred, random tops up a
    #                             shortfall; <= 0 keeps every hard negative.
    use_hard_negatives: bool = True  # True: rejected events become negatives
    #                             (hard-negative mining). False: ignore rejected
    #                             labels and sample all negatives as random
    #                             background from the recordings.
    min_seizure_overlap: float = 0.5  # fraction of window that must be seizure
    augment: bool = True
    seed: int = 42
    exclude_animals: tuple = ()  # animal IDs to drop from the dataset entirely
    #   (no train/val windows) — e.g. noisy recordings.
    cache_windows: bool = True  # keep each window's read+resampled signal in RAM
    #   after the first read, so no epoch re-reads from disk. (Compute, not I/O,
    #   was the real bottleneck — see mixed precision in train.py — but the cache
    #   is cheap insurance.) Masks are rebuilt and augmentation re-applied per epoch.


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
    exclude_set = set(config.exclude_animals or ())
    specs: list[WindowSpec] = []  # positive windows
    hard_neg_specs: list[WindowSpec] = []  # all rejected-event windows
    # Per-channel context for generating random background negatives on demand.
    neg_ctx: list[dict] = []

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
            # Skip animals the user excluded (e.g. noisy recordings) — they
            # contribute no training/val windows at all.
            if animal_id in exclude_set:
                continue
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

            # --- Hard negatives (every rejected candidate) ---
            # Collect them all here; the global balancing step below decides how
            # many to keep based on config.neg_pos_ratio.  Skipped entirely in
            # random-background mode, where rejected labels are ignored and all
            # negatives are sampled from the recording instead.
            for ann in (ch_rejected if config.use_hard_negatives else []):
                onset = ann["onset_sec"]
                offset = ann["offset_sec"]
                centre = (onset + offset) / 2
                half_win = config.window_sec / 2
                win_start = max(0, centre - half_win)
                win_end = min(rec_duration, win_start + config.window_sec)
                win_start = max(0, win_end - config.window_sec)

                hard_neg_specs.append(WindowSpec(
                    edf_path=edf_path,
                    start_sec=win_start,
                    duration_sec=config.window_sec,
                    eeg_channel=eeg_ch,
                    act_channel=act_ch,
                    is_positive=False,
                    seizure_intervals=[],
                    animal_id=animal_id,
                ))

            # Remember this channel so random background can be sampled later if
            # hard negatives alone don't meet the requested ratio.
            if rec_duration > config.window_sec:
                neg_ctx.append({
                    "edf_path": edf_path,
                    "eeg_channel": eeg_ch,
                    "act_channel": act_ch,
                    "animal_id": animal_id,
                    "rec_duration": rec_duration,
                    "seizure_intervals": list(ch_seizure_intervals),
                })

    # ── Global negative balancing ────────────────────────────────────
    # neg_pos_ratio is the target number of negative windows per positive
    # window across the whole dataset.  Hard negatives (reviewed-and-rejected
    # events) are always preferred; random background only tops up a shortfall.
    # A ratio <= 0 means "use every hard negative" (no cap).
    n_pos = len(specs)
    specs.extend(
        _balance_negatives(hard_neg_specs, neg_ctx, n_pos, config, rng)
    )
    return specs


def _balance_negatives(
    hard_neg_specs: list[WindowSpec],
    neg_ctx: list[dict],
    n_pos: int,
    config: DatasetConfig,
    rng: np.random.Generator,
) -> list[WindowSpec]:
    """Select negative windows to meet ``config.neg_pos_ratio`` globally.

    Prefers hard negatives (random-subsampled if they exceed the target),
    then tops up with random background windows if there's a shortfall.
    ``neg_pos_ratio <= 0`` keeps every hard negative.
    """
    ratio = config.neg_pos_ratio
    if ratio <= 0 or n_pos == 0:
        return list(hard_neg_specs)  # use all hard negatives

    target = int(round(n_pos * ratio))
    out: list[WindowSpec] = []

    if len(hard_neg_specs) >= target:
        idx = rng.choice(len(hard_neg_specs), size=target, replace=False)
        return [hard_neg_specs[i] for i in idx]

    # Keep all hard negatives, then add random background to reach the target.
    out.extend(hard_neg_specs)
    n_random = target - len(hard_neg_specs)
    if n_random > 0 and neg_ctx:
        for _ in range(n_random):
            ctx = neg_ctx[rng.integers(len(neg_ctx))]
            max_start = ctx["rec_duration"] - config.window_sec
            start = 0.0
            for _attempt in range(50):
                start = float(rng.uniform(0, max_start))
                end = start + config.window_sec
                if not any(
                    s[1] > start and s[0] < end
                    for s in ctx["seizure_intervals"]
                ):
                    break
            out.append(WindowSpec(
                edf_path=ctx["edf_path"],
                start_sec=start,
                duration_sec=config.window_sec,
                eeg_channel=ctx["eeg_channel"],
                act_channel=ctx["act_channel"],
                is_positive=False,
                seizure_intervals=[],
                animal_id=ctx["animal_id"],
            ))
    return out


# ---------------------------------------------------------------------------
# Window signal loading (module-level so it's picklable for process pools)
# ---------------------------------------------------------------------------


def _read_window_signal(edf_path, eeg_channel, act_channel,
                        start_sec, duration_sec, target_fs, target_samples):
    """Read + downsample + normalize one window's signal (EEG, optionally with
    a paired activity channel). Shared by SeizureDataset and the preload pool."""
    rec = read_edf_window(
        edf_path, channels=[eeg_channel],
        start_sec=start_sec, duration_sec=duration_sec,
    )
    eeg = _downsample(rec.data, rec.fs, target_fs)
    eeg = _normalize_channels(eeg)
    eeg = _pad_or_trim(eeg, target_samples)
    if act_channel is not None:
        try:
            act_rec = read_edf_window(
                edf_path, channels=[act_channel],
                start_sec=start_sec, duration_sec=duration_sec,
            )
            act = _downsample(act_rec.data, act_rec.fs, target_fs)
            act = _normalize_channels(act)
            act = _pad_or_trim(act, target_samples)
            eeg = np.concatenate([eeg, act], axis=0)
        except Exception:
            eeg = np.concatenate(
                [eeg, np.zeros((1, target_samples), dtype=np.float32)], axis=0)
    return eeg


def _preload_one(args):
    """Process-pool worker: load one window, returning (index, signal) — or
    (index, None) if it can't be read, so a single bad window doesn't abort the
    whole preload (it falls back to a lazy read later)."""
    idx, edf_path, eeg_channel, act_channel, start_sec, duration_sec, \
        target_fs, target_samples = args
    try:
        return idx, _read_window_signal(
            edf_path, eeg_channel, act_channel,
            start_sec, duration_sec, target_fs, target_samples)
    except Exception:
        return idx, None


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
        # idx -> read+resampled signal (pre-augmentation). Populated lazily, or
        # up front via preload(). Masks/augmentation are NOT cached (cheap /
        # must vary per epoch).
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.specs)

    def _load_signal(self, spec: "WindowSpec") -> np.ndarray:
        """Read + downsample + normalize the window's signal from disk — the
        expensive, deterministic part that the cache stores."""
        return _read_window_signal(
            spec.edf_path, spec.eeg_channel, spec.act_channel,
            spec.start_sec, spec.duration_sec,
            self.config.target_fs, self.target_samples)

    def preload(self, max_workers: int = 8) -> None:
        """Fill the window cache up front, before creating the DataLoader.

        Uses a PROCESS pool (pyedflib's reader is not thread-safe — concurrent
        threads corrupt each other's reads), each process loads its windows and
        ships the arrays back to this main process. On Linux the DataLoader
        workers then fork and inherit the cache copy-on-write, so epochs 2+ do
        no disk I/O and there's no per-worker duplication. Falls back to a
        serial read if the pool can't start. No-op if caching is disabled."""
        if not self.config.cache_windows:
            return
        todo = [i for i in range(len(self.specs)) if i not in self._cache]
        if not todo:
            return
        args = [
            (i, s.edf_path, s.eeg_channel, s.act_channel, s.start_sec,
             s.duration_sec, self.config.target_fs, self.target_samples)
            for i, s in ((i, self.specs[i]) for i in todo)
        ]
        n = max(1, max_workers)
        if n > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=n) as ex:
                    for idx, sig in ex.map(_preload_one, args, chunksize=8):
                        if sig is not None:
                            self._cache[idx] = sig
                return
            except Exception:
                pass  # fall through to serial
        # Serial fallback (also used when max_workers <= 1).
        for i in todo:
            try:
                self._cache[i] = self._load_signal(self.specs[i])
            except Exception:
                pass

    def __getitem__(self, idx: int):
        spec = self.specs[idx]

        # Signal: from cache if present, else read from disk (and cache it).
        if self.config.cache_windows and idx in self._cache:
            eeg = self._cache[idx]
        else:
            eeg = self._load_signal(spec)
            if self.config.cache_windows:
                self._cache[idx] = eeg

        # Build multi-channel mask: (n_classes, target_samples) — rebuilt each
        # call (cheap; channel 0 = seizure, channel 1 = convulsive).
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

        # Augmentation — on a COPY so the cached signal is never mutated.
        if self.augment:
            eeg, mask = _augment(eeg.copy(), mask, self.rng)

        eeg_tensor = torch.from_numpy(np.ascontiguousarray(eeg))
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

    All windows from the same animal go into the same set to prevent data
    leakage.

    The split is **convulsive-aware**: animals that have convulsive events form
    one stratum and the rest another, and ~``val_fraction`` of *each* stratum is
    allocated to validation. So when ≥2 animals had convulsive seizures, the
    validation set is guaranteed to contain convulsive events to score against
    and the training set keeps convulsive events to learn from — instead of
    leaving convulsive coverage to the luck of a random split. (Convulsive
    seizures are typically concentrated in a minority of animals, so a plain
    random split can land them all on one side.) With only one convulsive animal
    it is kept in train, since the model can't both learn and be validated on a
    single subject's convulsive events.

    Parameters
    ----------
    specs : all window specs
    val_fraction : approximate fraction for validation (applied per stratum)
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

    if len(animal_ids) < 2:
        raise ValueError(
            "Training requires annotations from at least 2 different animals "
            "so the data can be split into train and validation sets without "
            "data leakage. Found only 1 animal. Add recordings from more "
            "animals to the dataset, or assign distinct Animal IDs to "
            "different channels on the Load tab."
        )

    def _has_convulsive(aid: str) -> bool:
        return any(s.convulsive_intervals for s in animals[aid])

    conv_ids = [a for a in animal_ids if _has_convulsive(a)]
    other_ids = [a for a in animal_ids if not _has_convulsive(a)]

    def _n_conv(aid: str) -> int:
        return sum(1 for s in animals[aid] if s.convulsive_intervals)

    val_animals: set[str] = set()

    def _allocate(group: list[str], weight, ensure_both_sides: bool) -> None:
        """Move whole animals into validation until ~val_fraction of this
        stratum's ``weight`` is reached. ``weight(aid)`` is the quantity being
        balanced — total windows for the background stratum, convulsive windows
        for the convulsive stratum (so validation gets a representative share of
        convulsive events, not just any animal that happens to have some). When
        ``ensure_both_sides`` and the stratum has ≥2 animals, guarantee at least
        one animal on each side."""
        group = list(group)
        rng.shuffle(group)
        grp_w = sum(weight(a) for a in group)
        taken = 0
        for aid in group:
            if grp_w and taken / grp_w >= val_fraction:
                break
            val_animals.add(aid)
            taken += weight(aid)
        if ensure_both_sides and len(group) >= 2:
            in_val = [a for a in group if a in val_animals]
            if not in_val:  # none picked → add the smallest
                val_animals.add(min(group, key=lambda a: len(animals[a])))
            elif len(in_val) == len(group):  # all picked → keep largest in train
                val_animals.discard(max(group, key=lambda a: len(animals[a])))

    # Convulsive stratum: balance by convulsive-window count, and only split
    # across train/val when ≥2 such animals exist; a lone convulsive animal
    # stays in train so the model can learn it.
    if len(conv_ids) >= 2:
        _allocate(conv_ids, _n_conv, ensure_both_sides=True)
    _allocate(other_ids, lambda a: len(animals[a]), ensure_both_sides=True)

    # Overall safety: at least one animal on each side.
    if not val_animals:
        val_animals.add(animal_ids[0])
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
