#!/usr/bin/env python
"""Dump per-crop Stage-2 convulsive probabilities on the held-out validation set.

Reproduces the convulsive classifier's animal-wise validation split, runs the
saved checkpoint over every validation crop, and records each crop's predicted
probability alongside the expert's convulsive/non-convulsive label.

    python scripts/paper_stats/convulsive_probs.py

Writes ``convulsive_probs.json`` next to this script, consumed by
``supp_fig3.py`` to draw panel g.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from eeg_seizure_analyzer.io.dataset_store import scan_annotation_files
from eeg_seizure_analyzer.ml.convulsive_model import build_convulsive_classifier
from eeg_seizure_analyzer.ml.dataset import DatasetConfig, build_convulsive_datasets
from eeg_seizure_analyzer.ml.train import MODELS_DIR

HERE = Path(__file__).parent
MODEL = "Convulsive_v4LUNARC_20260616"


def main() -> None:
    model_dir = MODELS_DIR / MODEL
    meta = json.loads((model_dir / "metadata.json").read_text())
    folder = meta["dataset_folder"]
    if not Path(folder).is_dir():
        # metadata records the LUNARC path; fall back to the local corpus.
        folder = "/Users/marcoledri/Software/edf/SV2A_2024"
        print(f"(dataset folder from metadata not present; using {folder})")

    valid = {f.name for f in fields(DatasetConfig)}
    cfg = DatasetConfig(**{k: v for k, v in meta["dataset_config"].items() if k in valid})
    scan = scan_annotation_files(folder, "seizure")
    dataset_def = {"name": "convprobs", "folder": folder, "type": "seizure",
                   "files": [{"edf_path": r["edf_path"]} for r in scan]}
    _, val_ds, cfg = build_convulsive_datasets(dataset_def, cfg)
    animals = sorted({s.animal_id for s in val_ds.specs})
    print(f"val: {len(val_ds)} crops from {len(animals)} animals: {animals}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tc = meta["train_config"]
    model = build_convulsive_classifier(
        n_eeg_channels=meta.get("n_eeg_channels", 1),
        base_filters=tc["base_filters"], depth=tc["depth"], dropout=0.0)
    model.load_state_dict(torch.load(model_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model = model.to(device).eval()

    probs, labels, who = [], [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            eeg, label, _ = val_ds[i]
            out = torch.sigmoid(model(eeg.unsqueeze(0).to(device)))[0].cpu().numpy()
            probs.append(float(out.reshape(-1)[0]))
            labels.append(int(label.reshape(-1)[0]))
            who.append(val_ds.specs[i].animal_id)

    thr = float(meta.get("best_threshold", 0.45))
    p = np.array(probs); y = np.array(labels)
    tp = int(((p > thr) & (y == 1)).sum()); fn = int(((p <= thr) & (y == 1)).sum())
    fp = int(((p > thr) & (y == 0)).sum()); tn = int(((p <= thr) & (y == 0)).sum())
    print(f"threshold {thr}:  TP={tp} FN={fn} FP={fp} TN={tn}   "
          f"n_conv={int(y.sum())} n_nonconv={int((y == 0).sum())}")

    out = {"model": MODEL, "threshold": thr, "animals": animals,
           "prob": probs, "label": labels, "animal": who,
           "tp": tp, "fn": fn, "fp": fp, "tn": tn}
    (HERE / "convulsive_probs.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {HERE / 'convulsive_probs.json'}")


if __name__ == "__main__":
    main()
