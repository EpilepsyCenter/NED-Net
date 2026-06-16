#!/usr/bin/env python
"""Evaluate a trained seizure model on its held-out validation animals.

Reproduces the exact (deterministic, animal-wise) train/val split the model was
trained with, runs the saved best checkpoint over the validation windows, and
reports event-level metrics for BOTH output channels — channel 0 (all seizures)
and channel 1 (convulsive subset) — at the 0.5 threshold and at the best
threshold (swept). Also prints a per-animal recall breakdown, because the val
set is usually only a few animals and the aggregate number hides that variance.

Usage
-----
    python scripts/evaluate_model.py                 # list available models
    python scripts/evaluate_model.py U-Netv1_20260611
    python scripts/evaluate_model.py /path/to/model_dir
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from eeg_seizure_analyzer.io.dataset_store import scan_annotation_files
from eeg_seizure_analyzer.ml.dataset import (
    DatasetConfig, build_datasets, build_convulsive_datasets,
)
from eeg_seizure_analyzer.ml.model import build_model
from eeg_seizure_analyzer.ml.train import (
    MODELS_DIR, _compute_metrics, _best_threshold_metrics,
)


def _resolve_model_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.is_dir():
            return p
        cand = MODELS_DIR / arg
        if cand.is_dir():
            return cand
        sys.exit(f"Model not found: {arg}")
    # No arg: list models with a one-line summary.
    if not MODELS_DIR.exists():
        sys.exit(f"No models directory at {MODELS_DIR}")
    rows = sorted(MODELS_DIR.glob("*/metadata.json"),
                  key=lambda f: f.stat().st_mtime, reverse=True)
    if not rows:
        sys.exit(f"No models found in {MODELS_DIR}")
    print("Available models (most recent first):\n")
    for f in rows:
        d = json.load(open(f))
        print(f"  {f.parent.name:32s} arch={d.get('architecture','?'):6s} "
              f"best_epoch={d.get('best_epoch','?')} "
              f"best_val_loss={d.get('best_val_loss','?')}")
    print("\nRun again with a model name to evaluate it.")
    sys.exit(0)


def _build_net(meta: dict, device):
    """Recreate the architecture from metadata and load best_model.pt."""
    tc = meta.get("train_config", {})
    arch = meta.get("architecture", "unet")
    if arch == "bendr":
        from eeg_seizure_analyzer.ml.bendr_model import build_bendr_model
        in_ch = meta["n_eeg_channels"] + meta["n_activity_channels"]
        model = build_bendr_model(
            n_eeg_channels=in_ch,
            encoder_h=tc.get("encoder_h", 512), n_classes=2,
            context_layers=tc.get("context_layers", 8),
            context_heads=tc.get("context_heads", 8),
            pretrained_path=None,
            decoder_dropout=tc.get("dropout", 0.2),
        )
    else:
        model = build_model(
            n_eeg_channels=meta["n_eeg_channels"],
            include_activity=meta["include_activity"],
            n_activity_channels=meta["n_activity_channels"],
            base_filters=tc.get("base_filters", 32),
            depth=tc.get("depth", 4), dropout=tc.get("dropout", 0.2),
            n_classes=2,
        )
    state = torch.load(Path(meta["_model_dir"]) / "best_model.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _report(name: str, preds, targets, fs: int) -> None:
    n_windows_with_events = sum(int((t > 0.5).any()) for t in targets)
    print(f"\n=== {name} ===")
    print(f"val windows with events: {n_windows_with_events}")
    if n_windows_with_events == 0:
        print("  (no positive events in val — nothing to score)")
        return
    m = _compute_metrics(preds, targets, threshold=0.5, fs=fs)
    bt, bm = _best_threshold_metrics(preds, targets, fs=fs)
    print(f"  @0.5   F1={m['event_f1']:.3f}  P={m['event_precision']:.3f}  "
          f"R={m['event_recall']:.3f}   (true_events={m['true_events']})")
    print(f"  best   F1={bm['event_f1']:.3f}  P={bm['event_precision']:.3f}  "
          f"R={bm['event_recall']:.3f}   @thr={bt}")


def _evaluate_convulsive(meta: dict, model_dir: Path) -> None:
    """Evaluate a Stage-2 convulsive classifier on its held-out val animals."""
    from eeg_seizure_analyzer.ml.convulsive_model import build_convulsive_classifier
    from eeg_seizure_analyzer.ml.train_convulsive import (
        _binary_metrics, _best_threshold,
    )

    dc = meta["dataset_config"]
    folder = meta.get("dataset_folder") or meta.get("dataset", {}).get("folder")
    if not folder or not Path(folder).is_dir():
        sys.exit(f"Dataset folder from metadata not found: {folder}")

    print(f"Model:   {model_dir.name}  (arch=convulsive_classifier, "
          f"best_epoch={meta.get('best_epoch')})")
    print(f"Folder:  {folder}")

    valid = {f.name for f in fields(DatasetConfig)}
    cfg = DatasetConfig(**{k: v for k, v in dc.items() if k in valid})
    scan = scan_annotation_files(folder, "seizure")
    dataset_def = {"name": meta.get("dataset_name", "eval"), "folder": folder,
                   "type": "seizure",
                   "files": [{"edf_path": r["edf_path"]} for r in scan]}
    _, val_ds, cfg = build_convulsive_datasets(dataset_def, cfg)
    animals = sorted({s.animal_id for s in val_ds.specs})
    print(f"Val:     {len(val_ds)} windows from {len(animals)} animals: {animals}")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    tc = meta.get("train_config", {})
    model = build_convulsive_classifier(
        n_eeg_channels=meta.get("n_eeg_channels", 1),
        base_filters=tc.get("base_filters", 32),
        depth=tc.get("depth", 4), dropout=0.0,
    )
    state = torch.load(model_dir / "best_model.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    probs, targs, win_animal = [], [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            eeg, label, _ = val_ds[i]
            out = torch.sigmoid(model(eeg.unsqueeze(0).to(device)))[0].cpu().numpy()
            probs.append(float(out.reshape(-1)[0]))
            targs.append(float(label.reshape(-1)[0]))
            win_animal.append(val_ds.specs[i].animal_id)

    probs = np.array(probs)
    targs = np.array(targs)
    n_conv = int(targs.sum())
    print(f"\n=== CONVULSIVE CLASSIFIER ===")
    print(f"val windows: {len(targs)}  ({n_conv} convulsive / "
          f"{len(targs) - n_conv} non-convulsive)")
    if n_conv == 0:
        print("  (no convulsive events in val — nothing to score)")
        return
    m = _binary_metrics(probs, targs, threshold=0.5)
    bt, bm = _best_threshold(probs, targs)
    print(f"  @0.5   F1={m['event_f1']:.3f}  P={m['event_precision']:.3f}  "
          f"R={m['event_recall']:.3f}")
    print(f"  best   F1={bm['event_f1']:.3f}  P={bm['event_precision']:.3f}  "
          f"R={bm['event_recall']:.3f}   @thr={bt}")

    print(f"\n=== Per-animal convulsive recall (@thr={bt}) ===")
    for a in animals:
        idx = [i for i, w in enumerate(win_animal) if w == a]
        ap = probs[idx]
        at = targs[idx]
        if at.sum() == 0:
            print(f"  {a:10s} (no convulsive events in val windows)")
            continue
        am = _binary_metrics(ap, at, threshold=bt)
        print(f"  {a:10s} R={am['event_recall']:.3f}  P={am['event_precision']:.3f}  "
              f"convulsive={int(at.sum())}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    model_dir = _resolve_model_dir(arg)
    meta = json.load(open(model_dir / "metadata.json"))
    meta["_model_dir"] = str(model_dir)

    if meta.get("architecture") == "convulsive_classifier":
        _evaluate_convulsive(meta, model_dir)
        return

    dc = meta["dataset_config"]
    folder = meta.get("dataset_folder") or meta.get("dataset", {}).get("folder")
    if not folder or not Path(folder).is_dir():
        sys.exit(f"Dataset folder from metadata not found: {folder}")

    print(f"Model:   {model_dir.name}  (arch={meta.get('architecture')}, "
          f"best_epoch={meta.get('best_epoch')})")
    print(f"Folder:  {folder}")

    # Reproduce the deterministic, animal-wise val split.
    valid = {f.name for f in fields(DatasetConfig)}
    cfg = DatasetConfig(**{k: v for k, v in dc.items() if k in valid})
    scan = scan_annotation_files(folder, meta.get("dataset_type", "seizure"))
    dataset_def = {"name": meta.get("dataset_name", "eval"), "folder": folder,
                   "type": "seizure",
                   "files": [{"edf_path": r["edf_path"]} for r in scan]}
    _, val_ds, cfg = build_datasets(dataset_def, cfg)
    fs = cfg.target_fs
    animals = sorted({s.animal_id for s in val_ds.specs})
    print(f"Val:     {len(val_ds)} windows from {len(animals)} animals: {animals}")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model = _build_net(meta, device)

    p0, t0, p1, t1, win_animal = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            eeg, mask, _ = val_ds[i]
            out = torch.sigmoid(model(eeg.unsqueeze(0).to(device)))[0].cpu().numpy()
            m = mask.numpy()
            p0.append(out[0]); t0.append(m[0])
            p1.append(out[1]); t1.append(m[1])
            win_animal.append(val_ds.specs[i].animal_id)

    _report("SEIZURE (channel 0)", p0, t0, fs)
    _report("CONVULSIVE (channel 1)", p1, t1, fs)

    # Per-animal seizure recall at the best global threshold — exposes how much
    # the aggregate depends on individual animals (val sets are small).
    thr, _ = _best_threshold_metrics(p0, t0, fs=fs)
    print(f"\n=== Per-animal seizure recall (ch0 @thr={thr}) ===")
    for a in animals:
        idx = [i for i, w in enumerate(win_animal) if w == a]
        ap = [p0[i] for i in idx]
        at = [t0[i] for i in idx]
        if not any((t > 0.5).any() for t in at):
            print(f"  {a:10s} (no seizures in val windows)")
            continue
        am = _compute_metrics(ap, at, threshold=thr, fs=fs)
        print(f"  {a:10s} R={am['event_recall']:.3f}  P={am['event_precision']:.3f}  "
              f"events={am['true_events']}")


if __name__ == "__main__":
    main()
