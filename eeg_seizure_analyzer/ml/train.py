"""Training loop for seizure detection models (U-Net and BENDR).

Handles:
- Training with mixed-precision (if GPU available)
- Dice + BCE combined loss (handles class imbalance better than BCE alone)
- Validation with per-event metrics (not just per-sample)
- Model checkpointing and early stopping
- Progress reporting via callback (for UI integration)
- Architecture selection: U-Net (from scratch) or BENDR (fine-tuning
  from pre-trained self-supervised weights)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_seizure_analyzer.io.dataset_store import DATASETS_DIR
from eeg_seizure_analyzer.ml.dataset import (
    DatasetConfig,
    SeizureDataset,
    build_datasets,
    build_window_specs,
    split_by_animal,
)
from eeg_seizure_analyzer.ml.model import SeizureUNet, build_model
from eeg_seizure_analyzer.ml.bendr_model import BENDRSeizureModel, build_bendr_model

# ---------------------------------------------------------------------------
# Model storage
# ---------------------------------------------------------------------------

MODELS_DIR = Path.home() / ".eeg_seizure_analyzer" / "models"


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss.

    Dice loss handles class imbalance naturally (doesn't penalise the
    model for correctly predicting the dominant class). BCE provides
    stable gradients early in training.

    Parameters
    ----------
    dice_weight : weight for Dice component
    bce_weight : weight for BCE component
    smooth : smoothing factor for Dice (prevents division by zero)
    pos_weight : weight for positive class in BCE (helps with imbalance)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits, targets):
        """
        Parameters
        ----------
        logits : (batch, n_classes, n_samples) — raw model output
        targets : (batch, n_classes, n_samples) — multi-channel mask
        """
        # BCE
        bce_loss = self.bce(logits, targets)

        # Dice
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    all_preds: list[np.ndarray],
    all_targets: list[np.ndarray],
    threshold: float = 0.5,
    fs: int = 250,
    merge_gap_sec: float = 1.0,
) -> dict:
    """Compute per-sample and per-event metrics.

    Parameters
    ----------
    all_preds : list of probability arrays (n_samples,)
    all_targets : list of binary mask arrays (n_samples,)
    threshold : probability threshold for positive prediction
    fs : sampling rate of predictions
    merge_gap_sec : merge predicted segments closer than this

    Returns
    -------
    dict with sample_f1, event_precision, event_recall, event_f1, etc.
    """
    # Per-sample metrics
    preds_cat = np.concatenate(all_preds)
    targets_cat = np.concatenate(all_targets)
    pred_binary = (preds_cat > threshold).astype(int)
    target_binary = (targets_cat > 0.5).astype(int)

    tp = int(np.sum((pred_binary == 1) & (target_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (target_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (target_binary == 1)))
    tn = int(np.sum((pred_binary == 0) & (target_binary == 0)))

    sample_precision = tp / max(tp + fp, 1)
    sample_recall = tp / max(tp + fn, 1)
    sample_f1 = (
        2 * sample_precision * sample_recall
        / max(sample_precision + sample_recall, 1e-8)
    )

    # Per-event metrics (using overlap-based matching)
    merge_gap = int(merge_gap_sec * fs)

    def _segments(binary_arr):
        """Extract contiguous segments as (start, end) index pairs."""
        segs = []
        in_seg = False
        start = 0
        for i, v in enumerate(binary_arr):
            if v and not in_seg:
                start = i
                in_seg = True
            elif not v and in_seg:
                segs.append((start, i))
                in_seg = False
        if in_seg:
            segs.append((start, len(binary_arr)))

        # Merge close segments
        if merge_gap > 0 and len(segs) > 1:
            merged = [segs[0]]
            for s, e in segs[1:]:
                if s - merged[-1][1] <= merge_gap:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            segs = merged
        return segs

    # Compute event-level metrics across all windows
    total_true_events = 0
    total_pred_events = 0
    total_matched = 0

    for pred_prob, target in zip(all_preds, all_targets):
        pred_bin = (pred_prob > threshold).astype(int)
        target_bin = (target > 0.5).astype(int)

        true_segs = _segments(target_bin)
        pred_segs = _segments(pred_bin)

        total_true_events += len(true_segs)
        total_pred_events += len(pred_segs)

        # Match: a true event is "detected" if any predicted segment
        # overlaps it by at least 20%
        for ts, te in true_segs:
            t_len = te - ts
            for ps, pe in pred_segs:
                overlap = max(0, min(te, pe) - max(ts, ps))
                if overlap > 0.2 * t_len:
                    total_matched += 1
                    break

    event_recall = total_matched / max(total_true_events, 1)
    # Precision: fraction of predicted events that match a true event
    matched_pred = 0
    for pred_prob, target in zip(all_preds, all_targets):
        pred_bin = (pred_prob > threshold).astype(int)
        target_bin = (target > 0.5).astype(int)
        true_segs = _segments(target_bin)
        pred_segs = _segments(pred_bin)
        for ps, pe in pred_segs:
            p_len = pe - ps
            for ts, te in true_segs:
                overlap = max(0, min(te, pe) - max(ts, ps))
                if overlap > 0.2 * p_len:
                    matched_pred += 1
                    break

    event_precision = matched_pred / max(total_pred_events, 1)
    event_f1 = (
        2 * event_precision * event_recall
        / max(event_precision + event_recall, 1e-8)
    )

    return {
        "sample_precision": round(sample_precision, 4),
        "sample_recall": round(sample_recall, 4),
        "sample_f1": round(sample_f1, 4),
        "event_precision": round(event_precision, 4),
        "event_recall": round(event_recall, 4),
        "event_f1": round(event_f1, 4),
        "true_events": total_true_events,
        "pred_events": total_pred_events,
        "matched_events": total_matched,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _best_threshold_metrics(
    all_preds: list[np.ndarray],
    all_targets: list[np.ndarray],
    fs: int = 250,
) -> tuple[float, dict]:
    """Sweep decision thresholds and return (best_threshold, metrics) by event
    F1. Reveals whether the model has *ranking* signal even when its outputs are
    miscalibrated (e.g. saturated all-positive at 0.5)."""
    best_t, best = 0.5, None
    for i in range(1, 20):  # 0.05 .. 0.95
        t = i / 20.0
        m = _compute_metrics(all_preds, all_targets, threshold=t, fs=fs)
        if best is None or m["event_f1"] > best["event_f1"]:
            best, best_t = m, t
    return best_t, (best or {})


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 5.0  # upweight seizure samples in loss
    patience: int = 10  # early stopping patience (epochs)
    base_filters: int = 32
    depth: int = 4
    dropout: float = 0.2
    num_workers: int = 0  # DataLoader workers (0 = main process)

    # Architecture selection: "unet" or "bendr"
    architecture: str = "unet"

    # BENDR-specific settings (ignored for U-Net)
    pretrained_path: str = ""  # path to pre-trained BENDR weights
    encoder_h: int = 512  # BENDR encoder hidden dimension
    context_layers: int = 8  # BENDR transformer layers
    context_heads: int = 8  # BENDR attention heads
    encoder_lr: float = 1e-5  # lower LR for pre-trained encoder
    freeze_encoder_epochs: int = 5  # freeze encoder for first N epochs
    freeze_backbone: bool = False  # freeze encoder AND transformer for the
    #   whole run, training only the decoder head (linear-probe style). Best for
    #   small datasets — slashes trainable params (~155M -> ~5M) to curb
    #   overfitting. Overrides freeze_encoder_epochs / unfreezing.


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_model(
    dataset_def: dict,
    dataset_config: DatasetConfig | None = None,
    train_config: TrainConfig | None = None,
    model_name: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    stop_check_fn: Callable[[], bool] | None = None,
) -> dict:
    """Train a seizure detection model from a dataset definition.

    Parameters
    ----------
    dataset_def : dict
        From dataset_store.load_dataset().
    dataset_config : DatasetConfig, optional
    train_config : TrainConfig, optional
    model_name : str, optional
        Name for saving. Defaults to dataset name.
    progress_callback : callable, optional
        Called with a dict after each epoch:
        {"epoch", "train_loss", "val_loss", "val_metrics", "best_epoch"}.

    Returns
    -------
    dict with training results: model_path, best_metrics, history.
    """
    if dataset_config is None:
        dataset_config = DatasetConfig()
    if train_config is None:
        train_config = TrainConfig()
    if model_name is None:
        model_name = dataset_def.get("name", "unnamed")

    # ── Build datasets ───────────────────────────────────────────
    # 250 Hz end-to-end: the BENDR encoder is pre-trained at 250 Hz
    # (bendr_pretrain.py --target-fs 250) and inference runs at 250
    # (predict.py / dataset.py), so fine-tuning must stay at 250 too. An
    # earlier override bumped BENDR fine-tuning to 256 Hz, which fed the
    # 250 Hz-pre-trained encoder off-distribution input — removed.

    train_ds, val_ds, dataset_config = build_datasets(
        dataset_def, dataset_config
    )

    # Single-channel model: 1 EEG + optionally 1 activity
    n_eeg_channels = 1
    n_act_channels = 1 if dataset_config.include_activity else 0

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Input channels: {n_eeg_channels + n_act_channels} "
          f"(1 EEG{' + 1 activity' if n_act_channels else ''})")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Training requires at least 2 animals for a proper train/val "
            f"split. Got {len(train_ds)} train and {len(val_ds)} val samples. "
            "Add recordings from more animals to the dataset."
        )

    # Count unique animals
    train_animals = set(s.animal_id for s in train_ds.specs)
    val_animals = set(s.animal_id for s in val_ds.specs)
    print(f"Train animals: {len(train_animals)} {sorted(train_animals)}")
    print(f"Val animals: {len(val_animals)} {sorted(val_animals)}")

    # ── DataLoaders ──────────────────────────────────────────────
    # Custom collate to handle the meta dict
    def collate_fn(batch):
        eeg = torch.stack([b[0] for b in batch])
        mask = torch.stack([b[1] for b in batch])
        meta = [b[2] for b in batch]
        return eeg, mask, meta

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
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

    # ── Model ────────────────────────────────────────────────────
    is_bendr = train_config.architecture == "bendr"

    if is_bendr:
        pretrained = train_config.pretrained_path or None
        in_channels = n_eeg_channels + n_act_channels
        freeze_backbone = train_config.freeze_backbone
        model = build_bendr_model(
            n_eeg_channels=in_channels,
            encoder_h=train_config.encoder_h,
            n_classes=2,
            context_layers=train_config.context_layers,
            context_heads=train_config.context_heads,
            pretrained_path=pretrained,
            freeze_encoder=freeze_backbone or train_config.freeze_encoder_epochs > 0,
            freeze_contextualizer=freeze_backbone,
            finetuning=True,
            decoder_dropout=train_config.dropout,
        )
        if freeze_backbone:
            # Ensure frozen even when no pretrained weights were loaded
            # (freezing otherwise happens inside load_pretrained_combined).
            model.encoder.freeze(True)
            model.contextualizer.freeze(True)
        print(f"Architecture: BENDR (encoder_h={train_config.encoder_h})")
        if freeze_backbone:
            print("Backbone FROZEN (encoder + transformer) — training decoder "
                  "head only (linear-probe).")
        if pretrained:
            print(f"Pre-trained weights: {pretrained}")
    else:
        model = build_model(
            n_eeg_channels=n_eeg_channels,
            include_activity=dataset_config.include_activity,
            n_activity_channels=n_act_channels,
            base_filters=train_config.base_filters,
            depth=train_config.depth,
            dropout=train_config.dropout,
            n_classes=2,
        )
        print(f"Architecture: U-Net (base_filters={train_config.base_filters})")

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} trainable / {n_params_total:,} total")

    # ── Loss, optimizer, scheduler ───────────────────────────────
    criterion = DiceBCELoss(pos_weight=train_config.pos_weight)
    criterion = criterion.to(device)

    if is_bendr and train_config.freeze_backbone:
        # Backbone frozen: optimise the decoder head only.
        optimizer = torch.optim.AdamW(
            [p for p in model.decoder.parameters() if p.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        print(f"Head-only LR: {train_config.learning_rate:.1e}")
    elif is_bendr and train_config.pretrained_path:
        # Differential learning rate: lower LR for pre-trained encoder,
        # higher LR for new decoder head
        encoder_params = list(model.encoder.parameters()) + list(model.contextualizer.parameters())
        decoder_params = list(model.decoder.parameters())
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": train_config.encoder_lr},
            {"params": decoder_params, "lr": train_config.learning_rate},
        ], weight_decay=train_config.weight_decay)
        print(f"Differential LR: encoder={train_config.encoder_lr:.1e}, "
              f"decoder={train_config.learning_rate:.1e}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Training loop ────────────────────────────────────────────
    history = []
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0

    # Create model save directory
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    stopped = False
    for epoch in range(1, train_config.epochs + 1):
        t0 = time.time()

        # Unfreeze encoder after initial frozen epochs (BENDR only). Skipped
        # entirely when the backbone is frozen for the whole run.
        if (
            is_bendr
            and not train_config.freeze_backbone
            and train_config.freeze_encoder_epochs > 0
            and epoch == train_config.freeze_encoder_epochs + 1
        ):
            model.unfreeze_all()
            # Rebuild optimizer with all params trainable
            if train_config.pretrained_path:
                encoder_params = (
                    list(model.encoder.parameters())
                    + list(model.contextualizer.parameters())
                )
                decoder_params = list(model.decoder.parameters())
                optimizer = torch.optim.AdamW([
                    {"params": encoder_params, "lr": train_config.encoder_lr},
                    {"params": decoder_params, "lr": train_config.learning_rate},
                ], weight_decay=train_config.weight_decay)
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=train_config.learning_rate,
                    weight_decay=train_config.weight_decay,
                )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  → Encoder unfrozen at epoch {epoch}. "
                  f"Trainable params: {n_params:,}")

        # --- Train ---
        model.train()
        train_losses = []
        for eeg, mask, meta in train_loader:
            # Cooperative stop: break promptly (within a batch) on user request.
            if stop_check_fn is not None and stop_check_fn():
                stopped = True
                break
            eeg = eeg.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(eeg)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        if stopped:
            print(f"Training stopped by user during epoch {epoch}. "
                  f"Keeping best model so far (epoch {best_epoch}).")
            break

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        all_preds = []      # channel 0: all seizures
        all_targets = []
        all_preds_c = []    # channel 1: convulsive subset
        all_targets_c = []

        with torch.no_grad():
            for eeg, mask, meta in val_loader:
                eeg = eeg.to(device)
                mask = mask.to(device)

                logits = model(eeg)
                loss = criterion(logits, mask)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = mask.cpu().numpy()

                # ch 0 = all seizures, ch 1 = convulsive — score both.
                for i in range(probs.shape[0]):
                    all_preds.append(probs[i, 0])
                    all_targets.append(targets[i, 0])
                    if probs.shape[1] > 1:
                        all_preds_c.append(probs[i, 1])
                        all_targets_c.append(targets[i, 1])

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        val_metrics = _compute_metrics(
            all_preds, all_targets, fs=dataset_config.target_fs
        ) if all_preds else {}
        # Convulsive-channel readout (only meaningful where convulsive events
        # exist in the val set).
        if all_preds_c and any(t.max() > 0.5 for t in all_targets_c):
            cm = _compute_metrics(
                all_preds_c, all_targets_c, fs=dataset_config.target_fs)
            cbt, cbm = _best_threshold_metrics(
                all_preds_c, all_targets_c, fs=dataset_config.target_fs)
            val_metrics["conv_event_f1"] = cm.get("event_f1", 0.0)
            val_metrics["conv_event_precision"] = cm.get("event_precision", 0.0)
            val_metrics["conv_event_recall"] = cm.get("event_recall", 0.0)
            val_metrics["conv_best_event_f1"] = cbm.get("event_f1", 0.0)
            val_metrics["conv_best_threshold"] = round(cbt, 2)
        if all_preds:
            best_t, best_m = _best_threshold_metrics(
                all_preds, all_targets, fs=dataset_config.target_fs)
            val_metrics["best_threshold"] = round(best_t, 2)
            val_metrics["best_event_f1"] = best_m.get("event_f1", 0.0)
            val_metrics["best_event_precision"] = best_m.get("event_precision", 0.0)
            val_metrics["best_event_recall"] = best_m.get("event_recall", 0.0)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Check for improvement. Select the best model by validation event F1,
        # NOT val_loss: with sparse seizure masks an all-negative prediction can
        # have a lower loss than a useful detector, so loss-based selection
        # would happily pick a model that detects nothing. Tie-break on lower
        # val_loss (and that also covers the early all-zero-F1 epochs).
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
            epochs_without_improvement = 0

            # Save best model
            torch.save(model.state_dict(), model_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - t0

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(float(avg_train_loss), 4),
            "val_loss": round(float(avg_val_loss), 4),
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(elapsed, 1),
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch}/{train_config.epochs} — "
            f"train_loss: {avg_train_loss:.4f}, "
            f"val_loss: {avg_val_loss:.4f}, "
            f"event_f1: {val_metrics.get('event_f1', 'N/A')}, "
            f"lr: {optimizer.param_groups[0]['lr']:.1e}, "
            f"{elapsed:.1f}s"
        )

        if progress_callback:
            progress_callback(epoch_info)

        # Early stopping
        if epochs_without_improvement >= train_config.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {train_config.patience} epochs)")
            break

    # ── Save final artifacts ─────────────────────────────────────

    # If stopped before any epoch completed validation, there is no best
    # checkpoint yet — save the current weights so the model is usable.
    if best_epoch == 0:
        torch.save(model.state_dict(), model_dir / "best_model.pt")

    # Save training metadata
    metadata = {
        "model_name": model_name,
        "architecture": train_config.architecture,  # "unet" or "bendr"
        "dataset_name": dataset_def.get("name", ""),
        "dataset_folder": dataset_def.get("folder", ""),
        "created": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "n_params": n_params,
        "n_eeg_channels": n_eeg_channels,
        "n_activity_channels": n_act_channels,
        "n_classes": 2,
        "include_activity": dataset_config.include_activity,
        "target_fs": dataset_config.target_fs,
        "window_sec": dataset_config.window_sec,
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

    # Save final model too (in case best != last)
    torch.save(model.state_dict(), model_dir / "final_model.pt")

    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best metrics: {best_metrics}")
    print(f"Model saved to: {model_dir}")

    return {
        "model_path": str(model_dir),
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_metrics": best_metrics,
        "history": history,
        "n_params": n_params,
        "stopped_by_user": stopped,
    }


# ---------------------------------------------------------------------------
# Model listing / loading
# ---------------------------------------------------------------------------


def list_models() -> list[dict]:
    """List available trained models with their metadata summaries."""
    if not MODELS_DIR.exists():
        return []
    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            models.append({
                "name": meta.get("model_name", d.name),
                "architecture": meta.get("architecture", "unet"),
                "dataset": meta.get("dataset_name", ""),
                "created": meta.get("created", ""),
                "best_event_f1": meta.get("best_metrics", {}).get("event_f1", 0),
                "n_params": meta.get("n_params", 0),
                "n_eeg_channels": meta.get("n_eeg_channels", 0),
                "include_activity": meta.get("include_activity", False),
                "path": str(d),
            })
        except Exception:
            continue
    return models


def load_trained_model(
    model_name: str,
) -> tuple[SeizureUNet | BENDRSeizureModel, dict]:
    """Load a trained model and its metadata.

    Supports both U-Net and BENDR architectures.  The architecture is
    determined from the saved metadata.

    Parameters
    ----------
    model_name : name of the model directory

    Returns
    -------
    (model, metadata)
    """
    model_dir = MODELS_DIR / model_name
    meta_path = model_dir / "metadata.json"

    with open(meta_path) as f:
        metadata = json.load(f)

    architecture = metadata.get("architecture", "unet")
    tc = metadata.get("train_config", {})

    if architecture == "bendr":
        n_in = metadata.get("n_eeg_channels", 1)
        if metadata.get("include_activity", False):
            n_in += metadata.get("n_activity_channels", 0)
        model = build_bendr_model(
            n_eeg_channels=n_in,
            encoder_h=tc.get("encoder_h", 512),
            n_classes=metadata.get("n_classes", 2),
            context_layers=tc.get("context_layers", 8),
            context_heads=tc.get("context_heads", 8),
            finetuning=False,  # no masking at inference
            decoder_dropout=0.0,  # no dropout at inference
        )
    else:
        model = build_model(
            n_eeg_channels=metadata["n_eeg_channels"],
            include_activity=metadata.get("include_activity", False),
            n_activity_channels=metadata.get("n_activity_channels", 0),
            base_filters=tc.get("base_filters", 32),
            depth=tc.get("depth", 4),
            dropout=0.0,  # no dropout at inference
            n_classes=metadata.get("n_classes", 1),
        )

    weights_path = model_dir / "best_model.pt"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata
