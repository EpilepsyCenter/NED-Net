"""Self-supervised pre-training for BENDR on unlabelled EEG data.

Trains the convolutional encoder and transformer contextualizer with a
masked-prediction objective. The default ``--method data2vec`` uses an EMA
teacher (data2vec / Baevski et al., 2022) — it learns and generalises to
unseen recordings without the representation collapse that sank the legacy
``--method contrastive`` (wav2vec 2.0 style) objective. See
``BENDR_PRETRAIN_FIX_TODO.md`` for the diagnosis. No annotations required —
only raw EDF files.

Designed to run on a GPU cluster (e.g., LUNARC COSMOS with A100 GPUs).
Not part of the Dash GUI.

Usage
-----
From the command line::

    python -m eeg_seizure_analyzer.ml.bendr_pretrain \\
        --data-dir /path/to/edf/files \\
        --output-dir /path/to/output \\
        --epochs 30 \\
        --batch-size 64

Or resume a previous run::

    python -m eeg_seizure_analyzer.ml.bendr_pretrain \\
        --data-dir /path/to/edf/files \\
        --output-dir /path/to/output \\
        --resume /path/to/output/checkpoint_epoch_15.pt

The output directory will contain:

- ``checkpoint_epoch_N.pt`` — periodic checkpoints (encoder + contextualizer
  + optimizer state + epoch)
- ``best_model.pt`` — combined checkpoint with lowest validation loss
- ``pretrain_log.json`` — per-epoch metrics (loss, accuracy, lr)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import torch.nn.functional as F

from eeg_seizure_analyzer.ml.bendr_model import (
    build_data2vec_pretrain_model,
    build_pretrain_model,
)


# ── Streaming EDF Dataset ────────────────────────────────────────────


class EdfStreamDataset(IterableDataset):
    """Stream random EEG segments from EDF files without full preload.

    Uses pyedflib's ``readSignal(ch, start, n)`` for memory-efficient
    random-access reads.  Each worker handles a shard of the file list.

    Each channel is treated as an **independent** training example, yielding
    tensors of shape ``(1, segment_samples)``.  This means an 8-channel EDF
    file effectively produces 8× the training data, each channel contributing
    its own single-channel segments.  This matches BENDR's single-channel
    encoder architecture and maximises the use of multi-channel recordings.

    Parameters
    ----------
    edf_paths : list[str]
        Paths to EDF files.
    channels : list[int]
        Channel indices to read (typically EEG channels only).
        Each channel is yielded as a separate ``(1, segment_samples)``
        training example.
    segment_sec : float
        Length of each training segment in seconds.
    target_fs : float
        Target sampling rate.  Files at different rates are resampled.
    segments_per_file : int or None
        Number of random segments to draw per file **per channel** per epoch.
        If None, computed from file duration to cover ~80% of data.
    shuffle : bool
        Shuffle file order within each worker's shard.
    """

    def __init__(
        self,
        edf_paths: list[str],
        channels: list[int],
        segment_sec: float = 60.0,
        target_fs: float = 250.0,
        segments_per_file: int | None = None,
        shuffle: bool = True,
        bad_channels: dict[str, list[int]] | None = None,
    ):
        super().__init__()
        self.edf_paths = sorted(edf_paths)
        self.channels = channels
        self.segment_sec = segment_sec
        self.target_fs = target_fs
        self.segment_samples = int(segment_sec * target_fs)
        self.segments_per_file = segments_per_file
        self.shuffle = shuffle
        # Map filename (basename) -> list of channel indices to exclude
        self.bad_channels = bad_channels or {}

    def _channels_for_file(self, path: str) -> list[int]:
        """Return the channel list for a file with bad channels removed."""
        basename = os.path.basename(path)
        bad = set(self.bad_channels.get(basename, []))
        if not bad:
            return list(self.channels)
        return [c for c in self.channels if c not in bad]

    def _scan_file(self, path: str) -> dict | None:
        """Get file metadata without loading data."""
        try:
            import pyedflib
            f = pyedflib.EdfReader(path)
            try:
                fs = f.getSampleFrequency(self.channels[0])
                n_samples = int(f.getNSamples()[self.channels[0]])
                duration_sec = n_samples / fs
            finally:
                f.close()
            return {"path": path, "fs": fs, "n_samples": n_samples, "duration_sec": duration_sec}
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            return None

    def _read_single_channel(
        self, path: str, ch: int, fs: float, start_sample: int,
    ) -> np.ndarray | None:
        """Read a single-channel segment from an EDF file.

        Returns array of shape ``(1, segment_samples)`` at
        ``target_fs``, or None if reading fails or the signal is bad.
        """
        try:
            import pyedflib
            n_read = int(self.segment_sec * fs)

            f = pyedflib.EdfReader(path)
            try:
                raw = f.readSignal(ch, start_sample, n_read).astype(np.float32)
            finally:
                f.close()

            data = raw.reshape(1, -1)

            # Resample if needed. Prefer integer-factor decimate to mirror
            # dataset.py / predict.py (zero-phase, anti-aliased); fall back to
            # Fourier resample only for non-integer ratios.
            if abs(fs - self.target_fs) > 0.5:
                factor = int(round(fs / self.target_fs))
                if factor >= 2 and abs(fs / factor - self.target_fs) < 0.5:
                    from scipy.signal import decimate
                    data = decimate(data, factor, axis=1, zero_phase=True).astype(np.float32)
                else:
                    from scipy.signal import resample
                    data = resample(data, self.segment_samples, axis=1).astype(np.float32)

            if data.shape[1] != self.segment_samples:
                # Trim or pad to exact length
                if data.shape[1] > self.segment_samples:
                    data = data[:, :self.segment_samples]
                else:
                    pad = np.zeros(
                        (1, self.segment_samples - data.shape[1]),
                        dtype=np.float32,
                    )
                    data = np.concatenate([data, pad], axis=1)

            # Reject segments with NaN, inf, or flat signal
            if not np.isfinite(data).all():
                return None
            std = float(np.std(data))
            if std < 1e-8:
                return None

            # Z-score normalize, matching dataset.py / predict.py
            # (_normalize_channels). The pre-training stream was the only
            # path that fed raw amplitudes, so the model would have been
            # pre-trained on a different input distribution than it sees at
            # fine-tune / inference time.
            data = ((data - np.mean(data)) / std).astype(np.float32)
            return data

        except Exception:
            return None

    def __iter__(self):
        """Yield ``(1, segment_samples)`` tensors — one per channel per segment."""
        # Handle multi-worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            paths = self.edf_paths[worker_id::n_workers]
        else:
            paths = self.edf_paths

        if self.shuffle:
            paths = list(paths)
            np.random.shuffle(paths)

        for path in paths:
            info = self._scan_file(path)
            if info is None:
                continue

            fs = info["fs"]
            n_samples = info["n_samples"]
            segment_samples_native = int(self.segment_sec * fs)

            # How many segments can we draw from this file per channel?
            max_segments = max(1, (n_samples - segment_samples_native) // segment_samples_native)
            if self.segments_per_file is not None:
                n_segments = min(self.segments_per_file, max_segments)
            else:
                # Cover ~80% of the file
                n_segments = max(1, int(max_segments * 0.8))

            # Random start positions (shared across channels for this file)
            max_start = n_samples - segment_samples_native
            if max_start <= 0:
                continue
            starts = np.random.randint(0, max_start, size=n_segments)

            # Yield each channel independently as a (1, samples) example.
            # Shuffle channels within each segment for better mixing.
            # Skip channels marked bad for this specific file.
            ch_order = self._channels_for_file(path)
            if not ch_order:
                continue  # all channels bad in this file
            for start in starts:
                if self.shuffle:
                    np.random.shuffle(ch_order)
                for ch in ch_order:
                    segment = self._read_single_channel(path, ch, fs, int(start))
                    if segment is not None:
                        yield torch.from_numpy(segment)


def find_edf_files(data_dir: str, recursive: bool = True) -> list[str]:
    """Find all .edf files in a directory."""
    data_path = Path(data_dir)
    pattern = "**/*.edf" if recursive else "*.edf"
    paths = sorted(str(p) for p in data_path.glob(pattern))
    # Also check .EDF extension
    paths += sorted(str(p) for p in data_path.glob(pattern.replace(".edf", ".EDF")))
    return list(dict.fromkeys(paths))  # deduplicate preserving order


# ── Pre-training Loop ────────────────────────────────────────────────


def pretrain_bendr(
    data_dir: str,
    output_dir: str,
    channels: list[int] | None = None,
    segment_sec: float = 60.0,
    target_fs: float = 250.0,
    encoder_h: int = 512,
    context_layers: int = 8,
    context_heads: int = 8,
    method: str = "data2vec",
    mask_rate: float = 0.15,
    mask_span: int = 6,
    temp: float = 0.5,
    num_negatives: int = 100,
    ema_decay: float = 0.999,
    ema_end_decay: float = 0.9999,
    ema_anneal_steps: int = 5000,
    top_k_layers: int = 4,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    num_workers: int = 4,
    checkpoint_every: int = 5,
    segments_per_file: int | None = None,
    val_fraction: float = 0.05,
    resume_from: str | None = None,
    bad_channels_file: str | None = None,
) -> dict:
    """Self-supervised pre-training of BENDR on unlabelled EDF files.

    Parameters
    ----------
    data_dir : str
        Directory containing EDF files (searched recursively).
    output_dir : str
        Directory for checkpoints and logs.
    channels : list[int], optional
        EEG channel indices to use.  If None, uses channel 0.
        Each channel is treated as an independent single-channel
        training example (8 channels = 8× the training data).
    segment_sec : float
        Training segment length in seconds.
    target_fs : float
        Target sampling rate. Defaults to 250 Hz to match fine-tuning
        and inference (dataset.py / predict.py).
    encoder_h : int
        Encoder hidden dimension.
    context_layers, context_heads : int
        Transformer configuration.
    mask_rate : float
        Probability of masking each temporal position.
    mask_span : int
        Length of contiguous mask spans.
    temp : float
        Contrastive loss temperature.
    num_negatives : int
        Number of negative samples per masked position.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Peak learning rate (with warmup).
    weight_decay : float
        AdamW weight decay.
    num_workers : int
        DataLoader worker processes.
    checkpoint_every : int
        Save checkpoint every N epochs.
    segments_per_file : int, optional
        Random segments per file per channel per epoch.  None = auto
        (~80% coverage).  With 8 channels, each file yields
        ``segments_per_file × 8`` training examples per epoch.
    val_fraction : float
        Fraction of files held out for validation.
    resume_from : str, optional
        Path to checkpoint to resume from.
    bad_channels_file : str, optional
        Path to a JSON file mapping EDF filenames (basenames) to lists
        of channel indices to exclude for that file.  Example:
        ``{"animal01.edf": [2, 5], "animal02.edf": [0]}``.
        Files not listed use all channels.  Files where every channel
        is excluded are skipped entirely.

    Returns
    -------
    dict with training results and paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if channels is None:
        channels = [0]

    # Load per-file bad channel exclusion map (if provided)
    bad_channels: dict[str, list[int]] = {}
    if bad_channels_file:
        with open(bad_channels_file) as _f:
            raw = json.load(_f)
        # Normalise keys (basenames) and values (list of ints)
        for k, v in raw.items():
            bad_channels[os.path.basename(k)] = [int(c) for c in v]
        n_excluded_files = sum(1 for v in bad_channels.values() if v)
        n_total_excluded = sum(len(v) for v in bad_channels.values())
        print(f"Loaded bad-channels map: {n_excluded_files} file(s) with "
              f"{n_total_excluded} channel exclusion(s) total")

    # Each channel is treated independently — the model always sees
    # single-channel input (1, samples).  In typical rodent EEG setups,
    # each channel corresponds to a separate animal, so they are
    # genuinely independent recordings.
    n_input_channels = 1

    print(f"Using {len(channels)} channel(s) per file "
          f"(each treated as independent single-channel input)")

    # ── Find EDF files ───────────────────────────────────────────
    all_paths = find_edf_files(data_dir)
    if not all_paths:
        raise FileNotFoundError(f"No EDF files found in {data_dir}")

    print(f"Found {len(all_paths)} EDF files in {data_dir}")

    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(all_paths))
    n_val = max(1, int(len(all_paths) * val_fraction))
    val_indices = set(indices[:n_val])
    train_paths = [p for i, p in enumerate(all_paths) if i not in val_indices]
    val_paths = [p for i, p in enumerate(all_paths) if i in val_indices]

    print(f"Train files: {len(train_paths)}, Validation files: {len(val_paths)}")

    # ── Datasets and loaders ─────────────────────────────────────
    train_ds = EdfStreamDataset(
        train_paths, channels,
        segment_sec=segment_sec,
        target_fs=target_fs,
        segments_per_file=segments_per_file,
        shuffle=True,
        bad_channels=bad_channels,
    )
    val_ds = EdfStreamDataset(
        val_paths, channels,
        segment_sec=segment_sec,
        target_fs=target_fs,
        segments_per_file=max(1, (segments_per_file or 10) // 5),
        shuffle=False,
        bad_channels=bad_channels,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if method == "data2vec":
        # EMA-teacher objective — does not collapse on continuous EEG targets
        # (see BENDR_PRETRAIN_FIX_TODO.md). This is the default.
        model = build_data2vec_pretrain_model(
            n_eeg_channels=n_input_channels,
            encoder_h=encoder_h,
            context_layers=context_layers,
            context_heads=context_heads,
            mask_rate=mask_rate,
            mask_span=mask_span,
            ema_decay=ema_decay,
            ema_end_decay=ema_end_decay,
            ema_anneal_steps=ema_anneal_steps,
            top_k_layers=top_k_layers,
        )
    elif method == "contrastive":
        # Legacy wav2vec-2.0-style objective. Kept for reference; it suffers a
        # slow representation collapse on this data — prefer data2vec.
        model = build_pretrain_model(
            n_eeg_channels=n_input_channels,
            encoder_h=encoder_h,
            context_layers=context_layers,
            context_heads=context_heads,
            mask_rate=mask_rate,
            mask_span=mask_span,
            temp=temp,
            num_negatives=num_negatives,
        )
    else:
        raise ValueError(f"Unknown method {method!r} (use 'data2vec' or 'contrastive')")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer + scheduler ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01,
    )

    # ── Resume from checkpoint ───────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict] = []

    if resume_from:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", [])

        # Rebuild the LR schedule for the *current* ``epochs`` horizon instead of
        # restoring the checkpoint's scheduler state. The saved scheduler was
        # built with the original run's ``T_max`` (e.g. 5 for the short run);
        # ``load_state_dict`` would pin ``T_max`` back to that value, so a resume
        # that extends ``--epochs`` (e.g. to 30) would NOT anneal once over 30 —
        # the cosine (period 2·T_max) would instead warm-restart, oscillating the
        # LR between floor (5e-6) and peak (5e-4) every ~10 epochs, with the first
        # resumed epoch wasted at the floor. Reconstructing with
        # ``last_epoch=start_epoch-1`` stretches a single cosine across all
        # ``epochs`` and positions the LR at the correct point of it.
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01,
            last_epoch=start_epoch - 1,
        )
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        print(f"LR schedule rebuilt: cosine T_max={epochs}, "
              f"lr now {optimizer.param_groups[0]['lr']:.2e}")

    # ── Attention backend ────────────────────────────────────────
    # Force the math SDPA kernel on CUDA. The fused flash / mem-efficient
    # attention kernels emit NaN *gradients* on the A100 for this model
    # (diagnosed: the forward is finite, but the fused backward produces nan
    # grads within a few steps; clip_grad_norm then propagates the nan and
    # the optimizer step poisons every weight). The math kernel — the same
    # one CPU uses, where training is always clean — is correct here, and is
    # free for us: the post-encoder sequence is only ~150 tokens, so flash
    # attention's long-sequence advantage does not apply.
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # ── Precision ────────────────────────────────────────────────
    # bf16 autocast on CUDA for speed. The earlier mixed-precision NaNs were
    # the fused-attention backward (disabled above), not precision itself —
    # bf16 keeps fp32's exponent range, so no overflow and no GradScaler
    # needed. fp32 remains the proven-clean fallback: set use_amp=False to
    # revert. The contrastive cosine similarity is still computed in fp32
    # inside the model regardless of this setting.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    # ── Training loop ────────────────────────────────────────────
    log_path = output_path / "pretrain_log.json"
    print(f"\nStarting pre-training for {epochs} epochs")
    print(f"Precision: {'AMP bf16' if use_amp else 'full fp32'} on {device.type}")
    if device.type == "cuda":
        print("SDPA backend: math (fused flash/mem-efficient disabled)")
    print(f"Segment: {segment_sec}s at {target_fs} Hz = {int(segment_sec * target_fs)} samples")
    if method == "data2vec":
        print(f"Method: data2vec (EMA teacher) | mask rate={mask_rate} span={mask_span} "
              f"| ema {ema_decay}→{ema_end_decay} over {ema_anneal_steps} steps "
              f"| top-{top_k_layers} layers")
    else:
        print(f"Method: contrastive | mask rate={mask_rate} span={mask_span} "
              f"negatives={num_negatives} temp={temp}")
    print("-" * 70)

    # Per-method quality metric (label-0 contrastive accuracy, or data2vec
    # prediction–target cosine on masked positions). Both are stored under the
    # train_acc/val_acc log keys; ``metric_label`` names what they mean.
    metric_label = "acc" if method == "contrastive" else "cos"

    def quality_sum_count(out) -> tuple[float, int]:
        """Sample-weighted (sum, count) for the method's quality metric."""
        if method == "contrastive":
            logits, _z, mask = out
            ml = model.select_masked_logits(logits, mask)
            return float((ml.argmax(dim=1) == 0).sum().item()), int(ml.shape[0])
        pred, target, mask = out
        p = F.normalize(pred[mask], dim=1)
        t = F.normalize(target[mask], dim=1)
        return float((p * t).sum(dim=1).sum().item()), int(p.shape[0])

    # data2vec's continuous-regression loss can diverge if launched straight at
    # peak LR (validated locally); a short linear warmup prevents it. Skip it on
    # resume — the model is already past the fragile early phase.
    clip_norm = 3.0 if method == "data2vec" else 10.0
    global_step = 0 if start_epoch == 0 else warmup_steps

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────
        model.train()
        train_losses = []
        train_metric_sum = 0.0
        train_metric_count = 0
        n_batches = 0
        n_skipped = 0

        # Peak LR for this epoch as set by the cosine scheduler; warmup scales
        # below it during the first warmup_steps global steps only.
        epoch_peak_lr = optimizer.param_groups[0]["lr"]

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if global_step < warmup_steps:
                warm_lr = epoch_peak_lr * (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = warm_lr

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss = model.compute_loss(*out)

            # Skip non-finite batches instead of stepping on them — one bad
            # batch's gradient would otherwise corrupt every weight (this is
            # the guard the fp16 GradScaler used to provide implicitly).
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            if method == "data2vec":
                model.update_ema()
            global_step += 1

            train_losses.append(loss.item())

            with torch.no_grad():
                s, c = quality_sum_count(out)
                train_metric_sum += s
                train_metric_count += c

            n_batches += 1
            if n_batches % 100 == 0:
                avg_loss = np.mean(train_losses[-100:])
                metric = train_metric_sum / max(1, train_metric_count)
                print(f"  Epoch {epoch+1} | batch {n_batches} | "
                      f"loss={avg_loss:.4f} | {metric_label}={metric:.3f}")

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = train_metric_sum / max(1, train_metric_count)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_losses = []
        val_metric_sum = 0.0
        val_metric_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(batch)
                    loss = model.compute_loss(*out)
                if not torch.isfinite(loss):
                    continue
                val_losses.append(loss.item())
                s, c = quality_sum_count(out)
                val_metric_sum += s
                val_metric_count += c

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = val_metric_sum / max(1, val_metric_count)

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # ── Log ──────────────────────────────────────────────
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "lr": lr,
            "elapsed_sec": round(elapsed, 1),
            "n_batches": n_batches,
            "n_skipped": n_skipped,
        }
        history.append(epoch_log)

        skipped_note = (
            f" | SKIPPED {n_skipped}/{n_batches} non-finite batches"
            if n_skipped else ""
        )
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_{metric_label}={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_{metric_label}={val_acc:.3f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s{skipped_note}"
        )

        # Save log
        with open(log_path, "w") as f:
            json.dump({"config": _config_dict(locals()), "history": history}, f, indent=2)

        # ── Best model ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_path / "best_model.pt"
            torch.save({
                "encoder": model.encoder.state_dict(),
                "contextualizer": model.contextualizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_path)
            print(f"  → New best model saved (val_loss={val_loss:.4f})")

        # ── Periodic checkpoint ──────────────────────────────
        if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
            ckpt_path = output_path / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path.name}")

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Pre-training complete. {epochs} epochs, best val_loss={best_val_loss:.4f}")
    print(f"Best model: {output_path / 'best_model.pt'}")
    print(f"Log: {log_path}")

    return {
        "output_dir": str(output_path),
        "best_model_path": str(output_path / "best_model.pt"),
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs,
        "history": history,
    }


def _config_dict(local_vars: dict) -> dict:
    """Extract serialisable config from local variables."""
    keys = [
        "data_dir", "output_dir", "channels", "segment_sec", "target_fs",
        "encoder_h", "context_layers", "context_heads", "method", "mask_rate",
        "mask_span", "temp", "num_negatives", "ema_decay", "ema_end_decay",
        "ema_anneal_steps", "top_k_layers", "epochs", "batch_size",
        "learning_rate", "weight_decay", "warmup_steps", "num_workers",
        "val_fraction", "metric_label",
    ]
    return {k: local_vars[k] for k in keys if k in local_vars}


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised BENDR pre-training on unlabelled EEG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing EDF files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--channels", type=int, nargs="+", default=None,
        help="EEG channel indices to use (default: channel 0). "
             "Each channel is treated independently as a separate "
             "training example, multiplying the effective dataset size. "
             "E.g. --channels 0 1 2 3 4 5 6 7 gives 8× the data.",
    )
    parser.add_argument(
        "--segment-sec", type=float, default=60.0,
        help="Training segment length in seconds",
    )
    parser.add_argument("--target-fs", type=float, default=250.0,
                        help="Target sampling rate (must match fine-tuning at 250 Hz)")
    parser.add_argument("--encoder-h", type=int, default=512,
                        help="Encoder hidden dimension")
    parser.add_argument("--context-layers", type=int, default=8,
                        help="Transformer layers")
    parser.add_argument("--context-heads", type=int, default=8,
                        help="Attention heads")
    parser.add_argument("--method", choices=["data2vec", "contrastive"],
                        default="data2vec",
                        help="Pre-training objective. data2vec (EMA teacher) is "
                             "the default and does not collapse; contrastive is "
                             "the legacy wav2vec-2.0 objective kept for reference.")
    parser.add_argument("--mask-rate", type=float, default=0.15,
                        help="Masking probability")
    parser.add_argument("--mask-span", type=int, default=6,
                        help="Contiguous mask span length")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="[contrastive only] cosine similarity temperature")
    parser.add_argument("--num-negatives", type=int, default=100,
                        help="[contrastive only] negative samples per position")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="[data2vec] starting EMA teacher momentum")
    parser.add_argument("--ema-end-decay", type=float, default=0.9999,
                        help="[data2vec] final EMA teacher momentum")
    parser.add_argument("--ema-anneal-steps", type=int, default=5000,
                        help="[data2vec] steps to anneal EMA momentum over")
    parser.add_argument("--top-k-layers", type=int, default=4,
                        help="[data2vec] teacher layers averaged into the target")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear LR warmup steps (stabilises data2vec start)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--segments-per-file", type=int, default=None,
                        help="Random segments per file per epoch (None=auto)")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of files for validation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument(
        "--bad-channels", type=str, default=None,
        help="JSON file mapping EDF filenames to lists of channel indices "
             "to exclude. Example: {\"animal01.edf\": [2, 5]}. Files not "
             "listed use all channels.",
    )

    args = parser.parse_args()

    pretrain_bendr(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        segment_sec=args.segment_sec,
        target_fs=args.target_fs,
        encoder_h=args.encoder_h,
        context_layers=args.context_layers,
        context_heads=args.context_heads,
        method=args.method,
        mask_rate=args.mask_rate,
        mask_span=args.mask_span,
        temp=args.temp,
        num_negatives=args.num_negatives,
        ema_decay=args.ema_decay,
        ema_end_decay=args.ema_end_decay,
        ema_anneal_steps=args.ema_anneal_steps,
        top_k_layers=args.top_k_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        segments_per_file=args.segments_per_file,
        val_fraction=args.val_fraction,
        resume_from=args.resume,
        bad_channels_file=args.bad_channels,
    )


if __name__ == "__main__":
    main()
