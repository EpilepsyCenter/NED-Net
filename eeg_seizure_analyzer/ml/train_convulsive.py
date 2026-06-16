"""Training for the convulsive classifier (Stage 2 of the cascade).

Stage 1 (the U-Net) detects seizures; this trains a small 1D CNN that decides,
for each *seizure crop*, whether it is convulsive.  Because convulsive ⊂
seizure, the task is balanced binary classification (≈1:1.5) rather than the
heavily imbalanced per-sample segmentation the detector's ch1 has to solve — so
it comfortably beats the multi-task model's convulsive F1.

The module is deliberately self-contained: it reuses ``MODELS_DIR`` /
``TrainConfig`` and the AMP recipe from ``train.py`` but keeps the proven
seizure ``train_model`` untouched.  Models are saved with the same metadata
conventions (``architecture="convulsive_classifier"``) so ``list_models`` /
``load_trained_model`` consume them with no special-casing beyond one branch.

Example
-------
    python -m eeg_seizure_analyzer.ml.train_convulsive \
        --data-dir /path/to/edfs --model-name conv_v1 \
        --epochs 20 --exclude-animals 355676
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_seizure_analyzer.io.dataset_store import scan_annotation_files
from eeg_seizure_analyzer.ml.convulsive_model import build_convulsive_classifier
from eeg_seizure_analyzer.ml.dataset import (
    DatasetConfig,
    build_convulsive_datasets,
)
from eeg_seizure_analyzer.ml.train import MODELS_DIR, TrainConfig


# ---------------------------------------------------------------------------
# Scalar classification metrics
# ---------------------------------------------------------------------------


def _binary_metrics(probs: np.ndarray, targets: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Precision / recall / F1 for scalar binary predictions at a threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _best_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple[float, dict]:
    """Sweep thresholds 0.05..0.95 and return the (threshold, metrics) maxing F1."""
    best_t, best = 0.5, None
    for i in range(1, 20):
        t = i / 20.0
        m = _binary_metrics(probs, targets, threshold=t)
        if best is None or m["event_f1"] > best["event_f1"]:
            best, best_t = m, t
    return best_t, (best or {})


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_convulsive_model(
    dataset_def: dict,
    dataset_config: DatasetConfig | None = None,
    train_config: TrainConfig | None = None,
    model_name: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    stop_check_fn: Callable[[], bool] | None = None,
) -> dict:
    """Train a convulsive classifier from a dataset definition.

    Mirrors ``train.train_model``'s signature and progress-dict shape so the
    existing UI training machinery works unchanged.  ``val_metrics`` uses the
    same keys the seizure path emits (``event_f1`` / ``best_event_f1`` /
    ``best_threshold`` …) so ``_EPOCH_COLS`` and ``poll_training`` render it.

    Returns
    -------
    dict with model_path, best_metrics, history, n_params, stopped_by_user.
    """
    if dataset_config is None:
        dataset_config = DatasetConfig()
    if train_config is None:
        train_config = TrainConfig()
    train_config.architecture = "convulsive_classifier"
    if model_name is None:
        model_name = dataset_def.get("name", "unnamed")

    # ── Datasets ─────────────────────────────────────────────────
    train_ds, val_ds, dataset_config = build_convulsive_datasets(
        dataset_def, dataset_config
    )
    n_eeg_channels = 1

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Training requires at least 2 animals for a proper train/val "
            f"split. Got {len(train_ds)} train and {len(val_ds)} val samples."
        )

    # Class balance (for pos_weight) and animal coverage.
    n_conv = sum(1 for s in train_ds.specs if s.center_convulsive)
    n_nonconv = len(train_ds.specs) - n_conv
    val_conv = sum(1 for s in val_ds.specs if s.center_convulsive)
    train_animals = set(s.animal_id for s in train_ds.specs)
    val_animals = set(s.animal_id for s in val_ds.specs)
    print(f"Train: {n_conv} convulsive / {n_nonconv} non-convulsive")
    print(f"Val convulsive windows: {val_conv}")
    print(f"Train animals: {len(train_animals)} {sorted(train_animals)}")
    print(f"Val animals: {len(val_animals)} {sorted(val_animals)}")

    # ── Preload cache ────────────────────────────────────────────
    if dataset_config.cache_windows:
        n_w = max(1, train_config.num_workers or 8)
        print(f"Pre-loading {len(train_ds) + len(val_ds)} windows ({n_w} threads)...")
        t_cache = time.time()
        train_ds.preload(max_workers=n_w)
        val_ds.preload(max_workers=n_w)
        print(f"Cache ready in {time.time() - t_cache:.0f}s.")

    def collate_fn(batch):
        eeg = torch.stack([b[0] for b in batch])
        label = torch.stack([b[1] for b in batch])
        meta = [b[2] for b in batch]
        return eeg, label, meta

    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Device ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Mixed precision (CUDA only) — same recipe as train.py ────
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and amp_dtype == torch.float16
    )

    def autocast_ctx():
        return (torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp else nullcontext())

    if use_amp:
        print(f"Mixed precision: ON ({amp_dtype}".replace("torch.", "") + ")")

    # ── Model ────────────────────────────────────────────────────
    model = build_convulsive_classifier(
        n_eeg_channels=n_eeg_channels,
        base_filters=train_config.base_filters,
        depth=train_config.depth,
        dropout=train_config.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvulsiveClassifier — {n_params:,} trainable params")

    # ── Loss / optimizer / scheduler ─────────────────────────────
    # Upweight the minority (convulsive) class so recall isn't sacrificed.
    pos_weight = (n_nonconv / n_conv) if n_conv else 1.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    print(f"pos_weight (non-conv/conv): {pos_weight:.2f}")

    # ── Train ────────────────────────────────────────────────────
    history: list[dict] = []
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_metrics: dict = {}
    best_epoch = 0
    best_threshold = 0.5
    epochs_without_improvement = 0

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    stopped = False
    for epoch in range(1, train_config.epochs + 1):
        t0 = time.time()

        model.train()
        train_losses = []
        for eeg, label, meta in train_loader:
            if stop_check_fn is not None and stop_check_fn():
                stopped = True
                break
            eeg = eeg.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast_ctx():
                logits = model(eeg)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        if stopped:
            print(f"Training stopped by user during epoch {epoch}. "
                  f"Keeping best model so far (epoch {best_epoch}).")
            break

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        val_losses = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for eeg, label, meta in val_loader:
                eeg = eeg.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with autocast_ctx():
                    logits = model(eeg)
                    loss = criterion(logits, label)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                all_probs.append(probs)
                all_targets.append(label.cpu().numpy().reshape(-1))

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        probs_arr = np.concatenate(all_probs) if all_probs else np.array([])
        targs_arr = np.concatenate(all_targets) if all_targets else np.array([])

        if probs_arr.size:
            val_metrics = _binary_metrics(probs_arr, targs_arr, threshold=0.5)
            bt, bm = _best_threshold(probs_arr, targs_arr)
            val_metrics["best_threshold"] = round(bt, 2)
            val_metrics["best_event_f1"] = bm.get("event_f1", 0.0)
            val_metrics["best_event_precision"] = bm.get("event_precision", 0.0)
            val_metrics["best_event_recall"] = bm.get("event_recall", 0.0)
        else:
            val_metrics = {}
            bt = 0.5

        scheduler.step(avg_val_loss)

        # Select best epoch by (best-threshold) convulsive F1; tie-break on loss.
        val_f1 = float(val_metrics.get("best_event_f1",
                                       val_metrics.get("event_f1", 0.0)) or 0.0)
        is_better = (val_f1 > best_val_f1 + 1e-9) or (
            abs(val_f1 - best_val_f1) <= 1e-9 and avg_val_loss < best_val_loss
        )
        if is_better:
            best_val_f1 = val_f1
            best_val_loss = avg_val_loss
            best_metrics = val_metrics
            best_epoch = epoch
            best_threshold = bt
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - t0
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(elapsed, 1),
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch}/{train_config.epochs} — "
            f"train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, "
            f"F1@.5: {val_metrics.get('event_f1', 'N/A')}, "
            f"best_F1: {val_metrics.get('best_event_f1', 'N/A')}"
            f"@{val_metrics.get('best_threshold', 'N/A')}, {elapsed:.1f}s"
        )

        if progress_callback:
            progress_callback(epoch_info)

        if epochs_without_improvement >= train_config.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {train_config.patience} epochs)")
            break

    # ── Save artifacts ───────────────────────────────────────────
    if best_epoch == 0:
        torch.save(model.state_dict(), model_dir / "best_model.pt")

    metadata = {
        "model_name": model_name,
        "architecture": "convulsive_classifier",
        "dataset_name": dataset_def.get("name", ""),
        "dataset_folder": dataset_def.get("folder", ""),
        "created": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "n_params": n_params,
        "n_eeg_channels": n_eeg_channels,
        "n_activity_channels": 0,
        "n_classes": 1,
        "include_activity": False,
        "target_fs": dataset_config.target_fs,
        "window_sec": dataset_config.window_sec,
        "best_threshold": round(float(best_threshold), 2),
        "train_config": asdict(train_config),
        "dataset_config": asdict(dataset_config),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 4),
        "best_metrics": best_metrics,
        "n_epochs_trained": len(history),
        "stopped_by_user": stopped,
        "history": history,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    torch.save(model.state_dict(), model_dir / "final_model.pt")

    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"Best convulsive F1: {best_val_f1:.4f} "
          f"@ threshold {best_threshold:.2f}")
    print(f"Model saved to: {model_dir}")

    return {
        "model_path": str(model_dir),
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_metrics": best_metrics,
        "best_threshold": float(best_threshold),
        "history": history,
        "n_params": n_params,
        "stopped_by_user": stopped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_dataset_def(data_dir: str, name: str) -> dict:
    files = scan_annotation_files(data_dir, annotation_type="seizure")
    files = [f for f in files if f["n_confirmed"]]
    return {"name": name, "folder": data_dir, "files": files}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True,
                   help="folder of EDFs + *_ned_annotations.json (scanned recursively)")
    p.add_argument("--model-name", default="conv_v1")
    p.add_argument("--exclude-animals", nargs="*", default=[], metavar="ID",
                   help="animal IDs to drop entirely, e.g. --exclude-animals 355676")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args(argv)

    dataset_def = _build_dataset_def(args.data_dir, args.model_name)
    if not dataset_def["files"]:
        print(f"ERROR: no confirmed seizures found under {args.data_dir}",
              file=sys.stderr)
        return 1

    dataset_config = DatasetConfig(
        include_activity=False,
        exclude_animals=tuple(args.exclude_animals),
    )
    if args.exclude_animals:
        print(f"Excluding animals from dataset: {list(args.exclude_animals)}")
    train_config = TrainConfig(
        architecture="convulsive_classifier",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        base_filters=args.base_filters,
        depth=args.depth,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    print(f"Training convulsive classifier '{args.model_name}' on "
          f"{len(dataset_def['files'])} EDFs")
    result = train_convulsive_model(
        dataset_def,
        dataset_config=dataset_config,
        train_config=train_config,
        model_name=args.model_name,
    )
    print(f"\nBest convulsive F1: "
          f"{result['best_metrics'].get('best_event_f1', 'N/A')}")
    print(f"Model: {result['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
