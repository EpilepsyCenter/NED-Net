"""Training pipeline for interictal spike (IS) detection models.

Mirrors the seizure training pipeline (train.py) but uses:
- SpikeDatasetConfig / SpikeDataset (4s windows, spike masks)
- n_classes=1 (spike / no-spike)
- Spike-appropriate event metrics (shorter min_duration)

Models are saved to the same MODELS_DIR and listed alongside seizure models.
The metadata includes ``"model_type": "spike"`` to distinguish them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_seizure_analyzer.ml.model import build_model
from eeg_seizure_analyzer.ml.train import (
    DiceBCELoss,
    TrainConfig,
    MODELS_DIR,
    _compute_metrics,
    load_trained_model,
)
from eeg_seizure_analyzer.ml.spike_dataset import (
    SpikeDatasetConfig,
    build_spike_datasets,
)


def train_spike_model(
    dataset_def: dict,
    dataset_config: SpikeDatasetConfig | None = None,
    train_config: TrainConfig | None = None,
    model_name: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Train an interictal spike detection model.

    Parameters
    ----------
    dataset_def : dict
        From dataset_store.load_dataset() — must contain spike annotations.
    dataset_config : SpikeDatasetConfig, optional
    train_config : TrainConfig, optional
    model_name : str, optional
    progress_callback : callable, optional
        Called with epoch info dict after each epoch.

    Returns
    -------
    dict with training results.
    """
    if dataset_config is None:
        dataset_config = SpikeDatasetConfig()
    if train_config is None:
        train_config = TrainConfig()
    if model_name is None:
        model_name = dataset_def.get("name", "unnamed") + "_spikes"

    # ── Build datasets ───────────────────────────────────────────
    train_ds, val_ds, dataset_config = build_spike_datasets(
        dataset_def, dataset_config
    )

    n_eeg_channels = 1
    n_act_channels = 1 if dataset_config.include_activity else 0

    print(f"[IS] Training samples: {len(train_ds)}, "
          f"Validation samples: {len(val_ds)}")
    print(f"[IS] Input channels: {n_eeg_channels + n_act_channels}")

    train_animals = set(s.animal_id for s in train_ds.specs)
    val_animals = set(s.animal_id for s in val_ds.specs)
    print(f"[IS] Train animals: {len(train_animals)} {sorted(train_animals)}")
    print(f"[IS] Val animals: {len(val_animals)} {sorted(val_animals)}")

    # ── DataLoaders ──────────────────────────────────────────────
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
    print(f"[IS] Using device: {device}")

    # ── Model — n_classes=1 for spike detection ──────────────────
    model = build_model(
        n_eeg_channels=n_eeg_channels,
        include_activity=dataset_config.include_activity,
        n_activity_channels=n_act_channels,
        base_filters=train_config.base_filters,
        depth=train_config.depth,
        dropout=train_config.dropout,
        n_classes=1,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[IS] Model parameters: {n_params:,}")

    # ── Loss, optimizer, scheduler ───────────────────────────────
    criterion = DiceBCELoss(pos_weight=train_config.pos_weight)
    criterion = criterion.to(device)

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
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        for eeg, mask, meta in train_loader:
            eeg = eeg.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(eeg)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for eeg, mask, meta in val_loader:
                eeg = eeg.to(device)
                mask = mask.to(device)

                logits = model(eeg)
                loss = criterion(logits, mask)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = mask.cpu().numpy()

                for i in range(probs.shape[0]):
                    all_preds.append(probs[i, 0])
                    all_targets.append(targets[i, 0])

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")

        # Compute spike-level metrics with shorter merge gap
        val_metrics = _compute_metrics(
            all_preds, all_targets, fs=dataset_config.target_fs,
            merge_gap_sec=0.05,  # 50ms merge gap for spikes
        ) if all_preds else {}

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
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
            f"[IS] Epoch {epoch}/{train_config.epochs} — "
            f"train: {avg_train_loss:.4f}, "
            f"val: {avg_val_loss:.4f}, "
            f"event_f1: {val_metrics.get('event_f1', 'N/A')}, "
            f"{elapsed:.1f}s"
        )

        if progress_callback:
            progress_callback(epoch_info)

        if epochs_without_improvement >= train_config.patience:
            print(f"[IS] Early stopping at epoch {epoch}")
            break

    # ── Save metadata ────────────────────────────────────────────
    metadata = {
        "model_name": model_name,
        "model_type": "spike",  # distinguishes from seizure models
        "dataset_name": dataset_def.get("name", ""),
        "dataset_folder": dataset_def.get("folder", ""),
        "created": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "n_params": n_params,
        "n_eeg_channels": n_eeg_channels,
        "n_activity_channels": n_act_channels,
        "n_classes": 1,
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
        "history": history,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    torch.save(model.state_dict(), model_dir / "final_model.pt")

    print(f"\n[IS] Training complete. Best epoch: {best_epoch}")
    print(f"[IS] Best val loss: {best_val_loss:.4f}")
    print(f"[IS] Best metrics: {best_metrics}")
    print(f"[IS] Model saved to: {model_dir}")

    return {
        "model_path": str(model_dir),
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_metrics": best_metrics,
        "history": history,
        "n_params": n_params,
    }


def list_spike_models() -> list[dict]:
    """List trained IS models (model_type == 'spike')."""
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
            if meta.get("model_type") != "spike":
                continue
            models.append({
                "name": d.name,
                "dataset": meta.get("dataset_name", ""),
                "best_event_f1": meta.get("best_metrics", {}).get("event_f1", 0),
                "n_params": meta.get("n_params", 0),
                "created": meta.get("created", ""),
                "window_sec": meta.get("window_sec", 4),
                "target_fs": meta.get("target_fs", 250),
            })
        except Exception:
            continue
    return models
