"""Self-supervised pre-training for BENDR on unlabelled EEG data.

Trains the convolutional encoder and transformer contextualizer using
a contrastive masked prediction objective (wav2vec 2.0 style).  No
annotations required — only raw EDF files.

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

from eeg_seizure_analyzer.ml.bendr_model import build_pretrain_model


# ── Streaming EDF Dataset ────────────────────────────────────────────


class EdfStreamDataset(IterableDataset):
    """Stream random EEG segments from EDF files without full preload.

    Uses pyedflib's ``readSignal(ch, start, n)`` for memory-efficient
    random-access reads.  Each worker handles a shard of the file list.

    Parameters
    ----------
    edf_paths : list[str]
        Paths to EDF files.
    channels : list[int]
        Channel indices to read (typically EEG channels only).
    segment_sec : float
        Length of each training segment in seconds.
    target_fs : float
        Target sampling rate.  Files at different rates are resampled.
    segments_per_file : int or None
        Number of random segments to draw per file per epoch.
        If None, computed from file duration to cover ~80% of data.
    shuffle : bool
        Shuffle file order within each worker's shard.
    """

    def __init__(
        self,
        edf_paths: list[str],
        channels: list[int],
        segment_sec: float = 60.0,
        target_fs: float = 256.0,
        segments_per_file: int | None = None,
        shuffle: bool = True,
    ):
        super().__init__()
        self.edf_paths = sorted(edf_paths)
        self.channels = channels
        self.segment_sec = segment_sec
        self.target_fs = target_fs
        self.segment_samples = int(segment_sec * target_fs)
        self.segments_per_file = segments_per_file
        self.shuffle = shuffle

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

    def _read_segment(
        self, path: str, fs: float, start_sample: int,
    ) -> np.ndarray | None:
        """Read a single segment from an EDF file.

        Returns array of shape ``(n_channels, segment_samples)`` at
        ``target_fs``, or None if reading fails.
        """
        try:
            import pyedflib
            n_read = int(self.segment_sec * fs)

            f = pyedflib.EdfReader(path)
            try:
                data = np.zeros((len(self.channels), n_read), dtype=np.float32)
                for i, ch in enumerate(self.channels):
                    data[i] = f.readSignal(ch, start_sample, n_read).astype(np.float32)
            finally:
                f.close()

            # Resample if needed
            if abs(fs - self.target_fs) > 0.5:
                from scipy.signal import resample
                target_n = self.segment_samples
                data = resample(data, target_n, axis=1).astype(np.float32)
            elif data.shape[1] != self.segment_samples:
                # Trim or pad to exact length
                if data.shape[1] > self.segment_samples:
                    data = data[:, :self.segment_samples]
                else:
                    pad = np.zeros(
                        (data.shape[0], self.segment_samples - data.shape[1]),
                        dtype=np.float32,
                    )
                    data = np.concatenate([data, pad], axis=1)

            # Reject segments with NaN, inf, or flat signal
            if not np.isfinite(data).all():
                return None
            if np.std(data) < 1e-8:
                return None

            return data

        except Exception:
            return None

    def __iter__(self):
        """Yield ``(n_channels, segment_samples)`` tensors."""
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

            # How many segments can we draw from this file?
            max_segments = max(1, (n_samples - segment_samples_native) // segment_samples_native)
            if self.segments_per_file is not None:
                n_segments = min(self.segments_per_file, max_segments)
            else:
                # Cover ~80% of the file
                n_segments = max(1, int(max_segments * 0.8))

            # Random start positions
            max_start = n_samples - segment_samples_native
            if max_start <= 0:
                continue
            starts = np.random.randint(0, max_start, size=n_segments)

            for start in starts:
                segment = self._read_segment(path, fs, int(start))
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
    target_fs: float = 256.0,
    encoder_h: int = 512,
    context_layers: int = 8,
    context_heads: int = 8,
    mask_rate: float = 0.1,
    mask_span: int = 6,
    temp: float = 0.5,
    num_negatives: int = 100,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    checkpoint_every: int = 5,
    segments_per_file: int | None = None,
    val_fraction: float = 0.05,
    resume_from: str | None = None,
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
    segment_sec : float
        Training segment length in seconds.
    target_fs : float
        Target sampling rate (256 Hz for BENDR).
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
        Random segments per file per epoch.  None = auto (~80% coverage).
    val_fraction : float
        Fraction of files held out for validation.
    resume_from : str, optional
        Path to checkpoint to resume from.

    Returns
    -------
    dict with training results and paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if channels is None:
        channels = [0]
    n_channels = len(channels)

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
    )
    val_ds = EdfStreamDataset(
        val_paths, channels,
        segment_sec=segment_sec,
        target_fs=target_fs,
        segments_per_file=max(1, (segments_per_file or 10) // 5),
        shuffle=False,
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

    model = build_pretrain_model(
        n_eeg_channels=n_channels,
        encoder_h=encoder_h,
        context_layers=context_layers,
        context_heads=context_heads,
        mask_rate=mask_rate,
        mask_span=mask_span,
        temp=temp,
        num_negatives=num_negatives,
    )
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
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", [])
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Mixed precision ──────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────
    log_path = output_path / "pretrain_log.json"
    print(f"\nStarting pre-training for {epochs} epochs")
    print(f"Segment: {segment_sec}s at {target_fs} Hz = {int(segment_sec * target_fs)} samples")
    print(f"Mask: rate={mask_rate}, span={mask_span}, negatives={num_negatives}")
    print("-" * 70)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits, z, mask = model(batch)
                loss = model.compute_loss(logits, z)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            # Contrastive accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
                train_correct += (preds == labels).sum().item()
                train_total += labels.shape[0]

            n_batches += 1
            if n_batches % 100 == 0:
                avg_loss = np.mean(train_losses[-100:])
                acc = train_correct / max(1, train_total)
                print(f"  Epoch {epoch+1} | batch {n_batches} | "
                      f"loss={avg_loss:.4f} | acc={acc:.3f}")

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = train_correct / max(1, train_total)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits, z, mask = model(batch)
                    loss = model.compute_loss(logits, z)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = val_correct / max(1, val_total)

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
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s"
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
        "encoder_h", "context_layers", "context_heads", "mask_rate",
        "mask_span", "temp", "num_negatives", "epochs", "batch_size",
        "learning_rate", "weight_decay", "num_workers", "val_fraction",
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
        help="EEG channel indices to use (default: channel 0)",
    )
    parser.add_argument(
        "--segment-sec", type=float, default=60.0,
        help="Training segment length in seconds",
    )
    parser.add_argument("--target-fs", type=float, default=256.0,
                        help="Target sampling rate")
    parser.add_argument("--encoder-h", type=int, default=512,
                        help="Encoder hidden dimension")
    parser.add_argument("--context-layers", type=int, default=8,
                        help="Transformer layers")
    parser.add_argument("--context-heads", type=int, default=8,
                        help="Attention heads")
    parser.add_argument("--mask-rate", type=float, default=0.1,
                        help="Masking probability")
    parser.add_argument("--mask-span", type=int, default=6,
                        help="Contiguous mask span length")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="Contrastive loss temperature")
    parser.add_argument("--num-negatives", type=int, default=100,
                        help="Negative samples per position")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
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
        mask_rate=args.mask_rate,
        mask_span=args.mask_span,
        temp=args.temp,
        num_negatives=args.num_negatives,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        segments_per_file=args.segments_per_file,
        val_fraction=args.val_fraction,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
