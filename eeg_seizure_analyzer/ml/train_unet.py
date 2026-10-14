"""CLI entrypoint for supervised U-Net seizure-detection training.

Builds a dataset definition by scanning a folder of EDFs + their
``*_ned_annotations.json`` sidecars, then trains the from-scratch U-Net
(``eeg_seizure_analyzer.ml.train.train_model`` with ``architecture="unet"``).

Mirrors the BENDR pre-training CLI (``bendr_pretrain``) so it slots into the
same cluster job pattern (see ``scripts/slurm/``).

Examples
--------
Analyse the class balance and per-animal split (no training)::

    python -m eeg_seizure_analyzer.ml.train_unet --data-dir /path/to/edfs --analyze

Train::

    python -m eeg_seizure_analyzer.ml.train_unet \
        --data-dir /path/to/edfs \
        --model-name unet_v1 --neg-pos-ratio 8 --epochs 50
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from eeg_seizure_analyzer.io.dataset_store import scan_annotation_files
from eeg_seizure_analyzer.ml.dataset import (
    DatasetConfig,
    build_window_specs,
    split_specs,
)
from eeg_seizure_analyzer.ml.train import TrainConfig, train_model


def _build_dataset_def(data_dir: str, name: str) -> dict:
    files = scan_annotation_files(data_dir, annotation_type="seizure")
    files = [f for f in files if f["n_confirmed"] or f["n_rejected"]]
    return {"name": name, "folder": data_dir, "files": files}


def analyze(dataset_def: dict, exclude_animals: tuple = (),
            stable_val_split: bool = False, max_positive_sec: float = 0.0,
            bg_avoid_rejected: bool = False,
            hard_neg_exclude_methods: tuple = (),
            val_mode: str = "animal",
            dump_val_split: str | None = None,
            target: str = "seizure",
            val_pairs_file: str = "",
            nonconv_negatives: bool = True) -> dict:
    """Report class balance + the per-animal train/val split, and recommend a
    neg/pos ratio and pos_weight.  Returns the recommendation dict.

    ``exclude_animals`` must mirror what training will use, or --analyze
    reports a split that the real run will not produce.
    """
    files = dataset_def["files"]
    n_conf = sum(f["n_confirmed"] for f in files)
    n_rej = sum(f["n_rejected"] for f in files)
    print("=" * 64)
    print("DATASET ANALYSIS")
    print("=" * 64)
    print(f"Annotated EDFs:        {len(files)}")
    print(f"Confirmed seizures:    {n_conf}")
    print(f"Rejected (hard neg):   {n_rej}")
    print(f"Raw hard-neg : pos  =  {n_rej / max(n_conf, 1):.1f} : 1")

    # Plan windows with ALL hard negatives so we see the full pool, then look at
    # how build_window_specs actually lays them out per animal-group.
    cfg = DatasetConfig(neg_pos_ratio=0.0, augment=False,  # 0 => keep all
                        exclude_animals=tuple(exclude_animals),
                        max_positive_sec=max_positive_sec,
                        bg_avoid_rejected=bg_avoid_rejected,
                        hard_neg_exclude_methods=tuple(hard_neg_exclude_methods),
                        val_mode=val_mode, stable_convulsive_val=stable_val_split,
                        target=target, val_pairs_file=val_pairs_file,
                        nonconv_negatives=nonconv_negatives)
    specs = build_window_specs(dataset_def, cfg)
    pos = [s for s in specs if s.is_positive]
    neg = [s for s in specs if not s.is_positive]
    print(f"\nPositive windows:      {len(pos)}")
    print(f"Hard-negative windows: {len(neg)}")

    # Per animal-group (animal_id falls back to 'ch<idx>' when no channel-ID
    # files exist, so the whole corpus collapses into ~8 groups).
    by_animal = collections.Counter(s.animal_id for s in specs)
    pos_by_animal = collections.Counter(s.animal_id for s in pos)
    print(f"\nAnimal groups: {len(by_animal)}  (positives per group)")
    for aid in sorted(by_animal):
        print(f"  {aid:10} windows={by_animal[aid]:5}  positives={pos_by_animal.get(aid, 0)}")
    if len(by_animal) <= 8:
        print("  [!] Only ~8 groups: animal_id is the channel index, so every\n"
              "     recording's ch0 is pooled together. Consider writing batch-\n"
              "     aware channel IDs (*_ned_channels.json) for a finer split.")

    # Simulate the split the real run will use -- including the stable-stratum
    # option, or --analyze reports a split training will not produce.
    train_specs, val_specs = split_specs(specs, cfg, val_fraction=0.2)
    # Record the held-out side so detection can be scored on it alone. The split is
    # deterministic in (specs, seed, val_fraction, val_mode), so re-running --analyze
    # with the same flags reproduces exactly what training used.
    if dump_val_split:
        val_keys = sorted({(s.animal_id, s.edf_path) for s in val_specs})
        trn_keys = {(s.animal_id, s.edf_path) for s in train_specs}
        payload = {
            "val_mode": val_mode,
            "seed": 42,
            "val_fraction": 0.2,
            "exclude_animals": list(exclude_animals),
            "max_positive_sec": max_positive_sec,
            "n_val_positives": sum(s.is_positive for s in val_specs),
            # (animal, recording) pairs: the unit the split guarantees is unbroken
            "val_pairs": [{"animal_id": a, "edf": p} for a, p in val_keys],
            "overlap_with_train": sorted(
                f"{a}|{p}" for a, p in set(val_keys) & trn_keys),
        }
        Path(dump_val_split).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote held-out split to {dump_val_split}: "
              f"{len(val_keys)} (animal, recording) pairs, "
              f"{payload['n_val_positives']} positives, "
              f"{len(payload['overlap_with_train'])} overlaps with train (must be 0)")

    tp = sum(s.is_positive for s in train_specs)
    vp = sum(s.is_positive for s in val_specs)
    if val_mode in ("recording", "temporal"):
        print(f"\nSplit (val_fraction=0.2, by {val_mode} — every animal in BOTH sides; "
              f"measures DEPLOYMENT, not generalisation to a new animal):")
    else:
        print(f"\nSplit (val_fraction=0.2, by animal — whole animals held out; "
              f"stable_convulsive_val={stable_val_split}):")
    print(f"  train: {len(train_specs):5} windows, {tp} positive")
    print(f"  val:   {len(val_specs):5} windows, {vp} positive")
    if vp == 0:
        print("  [!] VALIDATION HAS ZERO POSITIVES -- event_f1 will be meaningless.\n"
              "     Re-seed the split or assign animal IDs so seizures land in both.")
    elif tp == 0:
        print("  [!] TRAIN HAS ZERO POSITIVES -- the model can't learn seizures.")

    # ── Recommendation ──────────────────────────────────────────────
    # The negatives here are reviewed-and-rejected spike candidates -- i.e. all
    # *hard* negatives that teach the model to suppress false positives.  So
    # prefer keeping many of them; a moderate cap keeps the imbalance trainable
    # for a first run.  pos_weight is matched to the chosen ratio so the BCE
    # term stays balanced (the Dice term is already imbalance-robust).
    max_ratio = len(neg) / max(len(pos), 1)
    rec_ratio = round(min(max_ratio, 10.0))
    rec_pos_weight = float(max(rec_ratio, 1))
    print("\n" + "=" * 64)
    print("RECOMMENDATION")
    print("=" * 64)
    print(f"  --neg-pos-ratio {rec_ratio}   (max available = {max_ratio:.0f}; "
          f"these are all HARD negatives, so keep plenty)")
    print(f"  --pos-weight    {rec_pos_weight:.0f}   (~= the ratio, to balance BCE)")
    print(f"  Alternative: --neg-pos-ratio 0 keeps ALL {len(neg)} hard negatives;\n"
          f"  pair with --pos-weight {max_ratio:.0f}. Best precision, slowest epochs.")
    print("=" * 64)
    return {"neg_pos_ratio": rec_ratio, "pos_weight": rec_pos_weight,
            "n_pos": len(pos), "n_neg": len(neg), "val_pos": vp}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True,
                   help="folder of EDFs + *_ned_annotations.json (scanned recursively)")
    p.add_argument("--model-name", default="unet_v1")
    p.add_argument("--analyze", action="store_true",
                   help="report class balance + split and a recommended ratio, then exit")

    # Dataset / balance
    p.add_argument("--neg-pos-ratio", type=float, default=2.0,
                   help="negative windows per positive (<=0 keeps all hard negatives)")
    p.add_argument("--neg-source", choices=["hard", "random"], default="hard",
                   help="'hard': rejected events become negatives (hard-negative "
                        "mining). 'random': ignore rejected labels, sample negatives "
                        "as random background from the recordings (the old behaviour).")
    p.add_argument("--include-activity", action="store_true",
                   help="add the paired activity channel as a 2nd input channel")
    p.add_argument("--stable-val-split", action="store_true",
                   help="fill the convulsive validation stratum with the smallest "
                        "animals first, keeping the dominant ones in train. Positives "
                        "are heavily concentrated, so a random draw can put half of "
                        "them in val. Use for any run compared against another.")
    p.add_argument("--dump-val-split", metavar="PATH",
                   help="with --analyze, write the held-out (animal, recording) pairs to "
                        "PATH as JSON. The split is deterministic, so this reproduces "
                        "exactly what a training run with the same flags used — needed to "
                        "score detection on the held-out side only.")
    p.add_argument("--val-mode", choices=["animal", "recording", "temporal"],
                   default="animal",
                   help="how the train/val split is drawn. 'animal' (default) holds out "
                        "whole animals = the TRANSFER question. 'recording' holds out "
                        "whole recordings with every animal on both sides = the DEPLOYMENT "
                        "question, which is what a lab faces since every animal gets some "
                        "annotation. 'temporal' is the same but holds out each animal's "
                        "LATEST recordings, matching annotate-early-deploy-on-the-rest. "
                        "Recording is the finest safe unit: splitting individual events "
                        "leaks, because events in one 90-min file share electrode state, "
                        "noise and often one seizure cluster. 'recording'/'temporal' "
                        "numbers are optimistic vs a new animal and must not be quoted as "
                        "generalisation, nor compared with 'animal'-split runs.")
    p.add_argument("--max-positive-sec", type=float, default=0.0,
                   help="drop confirmed events LONGER than this from the positive "
                        "windows (0 = no cap). Long imported 'convulsive' rows can be "
                        "chained seizures annotated as one block, which a 60 s window "
                        "cannot learn from. Capped events are still "
                        "labelled where they fall inside other windows and still "
                        "block background sampling. Suggested: 100.")
    p.add_argument("--hard-neg-exclude-method", nargs="*", default=[], metavar="M",
                   help="detection_method values whose `rejected` rows are not used "
                        "as hard negatives, while their regions still block random "
                        "background (with --bg-avoid-rejected). For imported reference "
                        "annotations whose rejections can contain real seizures.")
    p.add_argument("--bg-avoid-rejected", action="store_true",
                   help="random background windows avoid `rejected` regions as well "
                        "as `confirmed` ones. A rejection meaning 'not a convulsive "
                        "seizure' can contain real non-convulsive activity, so background drawn there trains "
                        "the detector to suppress real events.")
    p.add_argument("--target", choices=["seizure", "convulsive"], default="seizure",
                   help="what the detector learns: all confirmed seizures (default) "
                        "or convulsive seizures only. With 'convulsive', non-convulsive "
                        "seizures are hard negatives (see --nonconv-left-out), and every "
                        "detection of the trained model is convulsive.")
    p.add_argument("--nonconv-left-out", action="store_true",
                   help="with --target convulsive: leave non-convulsive seizures out of "
                        "training instead of using them as hard negatives (no window "
                        "centred on them; windows overlapping one are dropped).")
    p.add_argument("--val-pairs-file", default="", metavar="JSON",
                   help="use exactly the (animal, recording) validation pairs in this "
                        "file (from --analyze --dump-val-split) instead of computing a "
                        "split; to share another run's split.")
    p.add_argument("--exclude-animals", nargs="*", default=[], metavar="ID",
                   help="animal IDs to drop from the dataset entirely (no "
                        "train/val windows), e.g. noisy recordings: "
                        "--exclude-animals A12 A17")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="BCE positive-class weight; default: match neg-pos-ratio")
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--export-windows", metavar="DIR",
                   help="instead of training, write the exact train/val windows this "
                        "configuration produces (un-augmented model inputs + masks + a "
                        "per-window table) to DIR, then exit. Same code path as training.")
    p.add_argument("--fp32", action="store_true",
                   help="disable bf16 mixed precision (and TF32) on CUDA (full fp32, the "
                        "same numerics as training on Apple MPS)")
    args = p.parse_args(argv)

    dataset_def = _build_dataset_def(args.data_dir, args.model_name)
    if not dataset_def["files"]:
        print(f"ERROR: no *_ned_annotations.json found under {args.data_dir}",
              file=sys.stderr)
        return 1

    if args.analyze:
        analyze(dataset_def, tuple(args.exclude_animals), args.stable_val_split,
                args.max_positive_sec, args.bg_avoid_rejected,
                tuple(args.hard_neg_exclude_method), args.val_mode,
                args.dump_val_split, args.target, args.val_pairs_file,
                not args.nonconv_left_out)
        return 0

    use_hard = args.neg_source == "hard"
    neg_pos_ratio = args.neg_pos_ratio
    # "keep all" (<=0) only makes sense for hard negatives; random mode needs a
    # finite multiplier to know how many background windows to draw.
    if not use_hard and neg_pos_ratio <= 0:
        print("neg-source=random needs a positive --neg-pos-ratio; using 2.0",
              file=sys.stderr)
        neg_pos_ratio = 2.0

    # Default pos_weight tracks the ratio (>=1); for "keep all" use the raw imbalance.
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
        stable_convulsive_val=args.stable_val_split,
        max_positive_sec=args.max_positive_sec,
        bg_avoid_rejected=args.bg_avoid_rejected,
        hard_neg_exclude_methods=tuple(args.hard_neg_exclude_method),
        val_mode=args.val_mode,
        target=args.target,
        val_pairs_file=args.val_pairs_file,
        nonconv_negatives=not args.nonconv_left_out,
    )
    if args.target != "seizure" or args.val_pairs_file:
        print(f"Target: {args.target}   val pairs file: {args.val_pairs_file or 'none'}"
              + (f"   non-convulsive: {'LEFT OUT' if args.nonconv_left_out else 'hard negatives'}"
                 if args.target == "convulsive" else ""))
    if args.val_mode != "animal":
        print(f"Split mode: {args.val_mode} — every animal appears in BOTH train and val. "
              f"This measures deployment, NOT generalisation to a new animal.")
    if args.hard_neg_exclude_method:
        print(f"Not using as hard negatives: rejected rows from "
              f"{list(args.hard_neg_exclude_method)}")
    if args.max_positive_sec > 0:
        print(f"Positive-window cap: dropping confirmed events over "
              f"{args.max_positive_sec:.0f} s")
    if args.bg_avoid_rejected:
        print("Random background will avoid rejected regions as well as confirmed")
    if args.exclude_animals:
        print(f"Excluding animals from dataset: {list(args.exclude_animals)}")
    train_config = TrainConfig(
        architecture="unet",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        base_filters=args.base_filters,
        depth=args.depth,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        mixed_precision=not args.fp32,
    )

    if args.export_windows:
        return _export_windows(dataset_def, dataset_config, Path(args.export_windows))

    print(f"Training U-Net '{args.model_name}' on {len(dataset_def['files'])} EDFs "
          f"(neg-source={args.neg_source}, neg/pos={neg_pos_ratio}, "
          f"pos_weight={pos_weight:.1f})")
    result = train_model(
        dataset_def,
        dataset_config=dataset_config,
        train_config=train_config,
        model_name=args.model_name,
    )
    print(f"\nBest event_f1: {result['best_metrics'].get('event_f1', 'N/A')}")
    print(f"Model: {result['model_path']}")
    return 0


def _export_windows(dataset_def: dict, dataset_config: DatasetConfig, out: Path) -> int:
    """Write the windows a training run with these flags would see, without training.

    Uses ``build_datasets``, the call ``train_model`` makes, so the specs, split and
    background draws are identical (seed in DatasetConfig). Augmentation is switched
    off, because that is applied per epoch at random and is not part of the data.
    Per split it writes X_<split>.npy (float32, n x channels x samples: the
    resampled, per-window z-scored input exactly as the model receives it),
    Y_<split>.npy (uint8, n x 2 x samples: row 0 seizure, row 1 convulsive) and
    windows_<split>.csv (one row per window).
    """
    import numpy as np
    import pandas as pd
    from eeg_seizure_analyzer.ml.dataset import build_datasets

    dataset_config.augment = False
    train_ds, val_ds, cfg = build_datasets(dataset_def, dataset_config)
    out.mkdir(parents=True, exist_ok=True)
    for split, ds in (("train", train_ds), ("val", val_ds)):
        n = len(ds)
        eeg0, mask0, _ = ds[0]
        X = np.lib.format.open_memmap(out / f"X_{split}.npy", mode="w+",
                                      dtype=np.float32, shape=(n, *eeg0.shape))
        Y = np.lib.format.open_memmap(out / f"Y_{split}.npy", mode="w+",
                                      dtype=np.uint8, shape=(n, *mask0.shape))
        rows = []
        for i in range(n):
            eeg, mask, _ = ds[i]
            X[i] = eeg.numpy()
            Y[i] = (mask.numpy() > 0.5).astype(np.uint8)
            sp = ds.specs[i]
            rows.append({
                "window": i,
                "recording": Path(sp.edf_path).stem,
                "animal_id": sp.animal_id,
                "eeg_channel": sp.eeg_channel,
                "start_sec": round(float(sp.start_sec), 3),
                "duration_sec": float(sp.duration_sec),
                "contains_seizure": bool(sp.is_positive),
                "centre_is_convulsive": bool(getattr(sp, "center_convulsive", False)),
                "seizure_seconds": round(float(Y[i][0].sum()) / cfg.target_fs, 2),
            })
            if (i + 1) % 1000 == 0:
                print(f"  {split}: {i + 1}/{n}", flush=True)
        X.flush(); Y.flush()
        pd.DataFrame(rows).to_csv(out / f"windows_{split}.csv", index=False)
        print(f"{split}: {n} windows -> X {X.shape}, Y {Y.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
