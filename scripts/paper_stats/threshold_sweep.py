#!/usr/bin/env python
"""Dump the event-level precision/recall/F1 curve over detection thresholds.

Reproduces the deterministic, animal-wise validation split the model was
trained with (exactly as ``scripts/evaluate_model.py`` does), runs the saved
best checkpoint over the validation windows once, then scores the cached
predictions at every threshold from 0.05 to 0.95.

    python scripts/paper_stats/threshold_sweep.py

Writes ``threshold_sweep.json`` next to this script, consumed by
``supp_fig3.py``.

NOTE: this re-evaluation is not bit-identical to the metrics stored in the
model's ``metadata.json``. The annotation corpus grew slightly after the model
was trained, so the reproduced validation set holds 1,505 windows against the
1,561 recorded at training time. The numbers are close but not equal; whichever
set is quoted in the manuscript, quote it consistently.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from eeg_seizure_analyzer.io.dataset_store import scan_annotation_files
from eeg_seizure_analyzer.ml.dataset import DatasetConfig, build_datasets
from eeg_seizure_analyzer.ml.model import build_model
from eeg_seizure_analyzer.ml.train import MODELS_DIR, _compute_metrics

HERE = Path(__file__).parent
MODEL = "UNetv2_20260615"


def main() -> None:
    model_dir = MODELS_DIR / MODEL
    meta = json.loads((model_dir / "metadata.json").read_text())
    folder = meta["dataset_folder"]

    valid = {f.name for f in fields(DatasetConfig)}
    cfg = DatasetConfig(**{k: v for k, v in meta["dataset_config"].items() if k in valid})
    scan = scan_annotation_files(folder, "seizure")
    dataset_def = {"name": "sweep", "folder": folder, "type": "seizure",
                   "files": [{"edf_path": r["edf_path"]} for r in scan]}
    _, val_ds, cfg = build_datasets(dataset_def, cfg)
    fs = cfg.target_fs
    animals = sorted({s.animal_id for s in val_ds.specs})
    print(f"val: {len(val_ds)} windows from {len(animals)} animals: {animals}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tc = meta["train_config"]
    model = build_model(
        n_eeg_channels=meta["n_eeg_channels"],
        include_activity=meta["include_activity"],
        n_activity_channels=meta["n_activity_channels"],
        base_filters=tc["base_filters"], depth=tc["depth"],
        dropout=tc["dropout"], n_classes=2)
    model.load_state_dict(torch.load(model_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model = model.to(device).eval()

    preds, targets = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            eeg, mask, _ = val_ds[i]
            out = torch.sigmoid(model(eeg.unsqueeze(0).to(device)))[0].cpu().numpy()
            preds.append(out[0])
            targets.append(mask.numpy()[0])
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(val_ds)}")

    thresholds, precision, recall, f1 = [], [], [], []
    s_precision, s_recall, s_f1 = [], [], []
    for i in range(1, 20):
        t = i / 20.0
        m = _compute_metrics(preds, targets, threshold=t, fs=fs)
        thresholds.append(t)
        precision.append(m["event_precision"])
        recall.append(m["event_recall"])
        f1.append(m["event_f1"])
        s_precision.append(m["sample_precision"])
        s_recall.append(m["sample_recall"])
        s_f1.append(m["sample_f1"])
        print(f"  thr={t:.2f}  event P={m['event_precision']:.3f} "
              f"R={m['event_recall']:.3f} F1={m['event_f1']:.3f}   |   "
              f"sample P={m['sample_precision']:.3f} R={m['sample_recall']:.3f}")

    best = int(np.argmax(f1))
    i50 = thresholds.index(0.5)
    out = {"model": MODEL, "n_val_windows": len(val_ds), "val_animals": animals,
           "thresholds": thresholds, "precision": precision,
           "recall": recall, "f1": f1,
           "sample_precision": s_precision, "sample_recall": s_recall,
           "sample_f1": s_f1,
           "best_threshold": thresholds[best], "best_f1": f1[best],
           "best_precision": precision[best], "best_recall": recall[best],
           "f1_at_0.5": f1[i50],
           "precision_at_0.5": precision[i50],
           "recall_at_0.5": recall[i50],
           "sample_precision_at_0.5": s_precision[i50],
           "sample_recall_at_0.5": s_recall[i50]}
    (HERE / "threshold_sweep.json").write_text(json.dumps(out, indent=1))
    print(f"\nbest F1 {out['best_f1']:.3f} @ {out['best_threshold']}; "
          f"@0.5 F1 {out['f1_at_0.5']:.3f}")


if __name__ == "__main__":
    main()
