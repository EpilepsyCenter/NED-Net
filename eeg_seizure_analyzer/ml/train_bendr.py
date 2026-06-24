"""CLI entrypoint for fine-tuning a pre-trained BENDR seizure detector.

Mirror of ``train_unet`` (same dataset scan, same per-animal split, same
LUNARC job pattern) but selects ``architecture="bendr"`` and loads a
self-supervised pre-trained encoder/contextualizer
(``bendr_pretrain.py`` -> ``best_model.pt``) for fine-tuning.

This exists because BENDR fine-tuning was previously GUI-only; the one BENDR
detector trained that way ran on Apple MPS (a backend known to silently corrupt
training, see the convulsive-classifier divergence) and scored event_f1 ~0.06.
This CLI lets the same fine-tune run headless on a clean CUDA backend (LUNARC
A100) so the MPS confound can be ruled out before BENDR is shelved.

Examples
--------
Analyse the class balance / split (no training)::

    python -m eeg_seizure_analyzer.ml.train_bendr \
        --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data --analyze

Fine-tune::

    python -m eeg_seizure_analyzer.ml.train_bendr \
        --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data \
        --model-name bendr_cuda_v1 \
        --pretrained ~/.eeg_seizure_analyzer/pretrained/run1_best.pt \
        --neg-pos-ratio 4 --pos-weight 5 --epochs 40
"""
from __future__ import annotations

import argparse
import os
import sys

from eeg_seizure_analyzer.ml.dataset import DatasetConfig
from eeg_seizure_analyzer.ml.train import TrainConfig, train_model
# Reuse the U-Net CLI's dataset scan + analysis helpers verbatim: the dataset
# is identical (EDFs + *_ned_annotations.json), only the architecture differs.
from eeg_seizure_analyzer.ml.train_unet import _build_dataset_def, analyze


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True,
                   help="folder of EDFs + *_ned_annotations.json (scanned recursively)")
    p.add_argument("--model-name", default="bendr_cuda")
    p.add_argument("--analyze", action="store_true",
                   help="report class balance + split and a recommended ratio, then exit")

    # Dataset / balance (mirrors train_unet so a BENDR run can match the U-Net's)
    p.add_argument("--neg-pos-ratio", type=float, default=4.0,
                   help="negative windows per positive (<=0 keeps all hard negatives)")
    p.add_argument("--neg-source", choices=["hard", "random"], default="hard")
    p.add_argument("--include-activity", action="store_true",
                   help="add the paired activity channel as a 2nd input channel")
    p.add_argument("--exclude-animals", nargs="*", default=[], metavar="ID",
                   help="animal IDs to drop from the dataset entirely")

    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="head / decoder learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="BCE positive-class weight; default: match neg-pos-ratio")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)

    # BENDR-specific
    p.add_argument("--pretrained", default="",
                   help="path to pre-trained BENDR weights (encoder+contextualizer); "
                        "blank = train BENDR architecture from scratch")
    p.add_argument("--encoder-lr", type=float, default=1e-5,
                   help="lower LR for the pre-trained encoder (differential LR)")
    p.add_argument("--freeze-encoder-epochs", type=int, default=5,
                   help="freeze the encoder for the first N epochs (head warm-up)")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="freeze encoder AND contextualizer for the whole run "
                        "(train only the head; strongest regulariser for tiny label sets)")
    p.add_argument("--encoder-h", type=int, default=512)
    p.add_argument("--context-layers", type=int, default=8)
    p.add_argument("--context-heads", type=int, default=8)
    args = p.parse_args(argv)

    dataset_def = _build_dataset_def(args.data_dir, args.model_name)
    if not dataset_def["files"]:
        print(f"ERROR: no *_ned_annotations.json found under {args.data_dir}",
              file=sys.stderr)
        return 1

    if args.analyze:
        analyze(dataset_def)
        return 0

    pretrained = os.path.expanduser(args.pretrained) if args.pretrained else ""
    if pretrained and not os.path.isfile(pretrained):
        print(f"ERROR: pretrained weights not found: {pretrained}", file=sys.stderr)
        return 1
    if not pretrained:
        print("WARNING: no --pretrained given -> BENDR architecture trained from "
              "scratch (this is NOT the pre-training transfer test).", file=sys.stderr)

    use_hard = args.neg_source == "hard"
    neg_pos_ratio = args.neg_pos_ratio
    if not use_hard and neg_pos_ratio <= 0:
        print("neg-source=random needs a positive --neg-pos-ratio; using 2.0",
              file=sys.stderr)
        neg_pos_ratio = 2.0

    if args.pos_weight is not None:
        pos_weight = args.pos_weight
    elif neg_pos_ratio > 0:
        pos_weight = max(neg_pos_ratio, 1.0)
    else:
        n_conf = sum(f["n_confirmed"] for f in dataset_def["files"]) or 1
        n_rej = sum(f["n_rejected"] for f in dataset_def["files"])
        pos_weight = max(n_rej / n_conf, 1.0)

    dataset_config = DatasetConfig(
        neg_pos_ratio=neg_pos_ratio,
        use_hard_negatives=use_hard,
        include_activity=args.include_activity,
        exclude_animals=tuple(args.exclude_animals),
    )
    if args.exclude_animals:
        print(f"Excluding animals from dataset: {list(args.exclude_animals)}")

    train_config = TrainConfig(
        architecture="bendr",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        pretrained_path=pretrained,
        encoder_h=args.encoder_h,
        context_layers=args.context_layers,
        context_heads=args.context_heads,
        encoder_lr=args.encoder_lr,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        freeze_backbone=args.freeze_backbone,
    )

    print(f"Fine-tuning BENDR '{args.model_name}' on {len(dataset_def['files'])} EDFs "
          f"(neg-source={args.neg_source}, neg/pos={neg_pos_ratio}, "
          f"pos_weight={pos_weight:.1f})")
    print(f"  pretrained={pretrained or '(none/from-scratch)'} "
          f"encoder_lr={args.encoder_lr:.1e} "
          f"freeze_encoder_epochs={args.freeze_encoder_epochs} "
          f"freeze_backbone={args.freeze_backbone}")
    result = train_model(
        dataset_def,
        dataset_config=dataset_config,
        train_config=train_config,
        model_name=args.model_name,
    )
    print(f"\nBest event_f1: {result['best_metrics'].get('event_f1', 'N/A')}")
    print(f"Model: {result['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
