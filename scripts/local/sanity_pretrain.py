#!/usr/bin/env python
"""Local CPU/MPS learnability gate for BENDR data2vec pre-training.

This is the gate described in ``eeg_seizure_analyzer/ml/BENDR_PRETRAIN_FIX_TODO.md``:
prove the self-supervised objective actually *learns and generalises* before
spending cluster time. It runs the **real** model (``BENDRData2VecPretrainModel``)
and the **real** EDF streaming + 250 Hz decimation pipeline, shrunk so it
finishes in minutes on a laptop.

Methodology (matters — earlier debugging was misled by a weaker metric):

* Validation is on **whole held-out files** (``--val-files`` count). A held-out
  *channel* would share its file's recording session/day/rig and leak; a whole
  held-out file does not. Animals differ between cohorts/batches, so holding out
  whole files measures generalisation to unseen recordings.
* The metric is the **prediction–target cosine** at masked positions on the
  fixed held-out batch, tracked over training. A healthy objective makes it rise
  and *hold* (the old contrastive objective made it spike then decay to chance).

PASS requires: held-out cosine clearly above zero, still high at the end (no
decay), and held-out loss decreasing. Exit code 0 on PASS, 1 on FAIL.

Usage
-----
    python scripts/local/sanity_pretrain.py --data /path/to/edf_dir
    python scripts/local/sanity_pretrain.py --data /path/to/edf_dir --steps 1200
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eeg_seizure_analyzer.ml.bendr_model import build_data2vec_pretrain_model
from eeg_seizure_analyzer.ml.bendr_pretrain import EdfStreamDataset, find_edf_files


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Local data2vec learnability gate for BENDR pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True,
                   help="Directory searched recursively for *.edf (needs >= 3 files)")
    p.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7],
                   help="Channel indices to stream (each is an independent example)")
    p.add_argument("--val-files", type=int, default=2,
                   help="Number of whole files held out for validation")
    p.add_argument("--steps", type=int, default=1200, help="Optimizer steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--segment-sec", type=float, default=20.0)
    p.add_argument("--target-fs", type=float, default=250.0)
    # Shrunk model — full pre-training is encoder_h=512, 8 layers.
    p.add_argument("--encoder-h", type=int, default=128)
    p.add_argument("--context-layers", type=int, default=4)
    p.add_argument("--context-heads", type=int, default=4)
    p.add_argument("--top-k-layers", type=int, default=3)
    p.add_argument("--mask-rate", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cos-threshold", type=float, default=0.05,
                   help="Final held-out cosine must exceed this")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    edf_paths = find_edf_files(args.data) if Path(args.data).is_dir() else [args.data]
    if len(edf_paths) < args.val_files + 1:
        print(f"FAIL: need at least {args.val_files + 1} EDF files, found "
              f"{len(edf_paths)} at {args.data}", file=sys.stderr)
        return 1
    val_files = edf_paths[-args.val_files:]
    train_files = edf_paths[:-args.val_files]

    device = pick_device(args.device)
    print("=" * 70)
    print("BENDR data2vec learnability gate (local, whole-file holdout)")
    print(f"  train files : {len(train_files)}")
    print(f"  val files   : {[Path(f).name for f in val_files]}")
    print(f"  device      : {device.type}")
    print(f"  model       : encoder_h={args.encoder_h}, layers={args.context_layers}, "
          f"top-k={args.top_k_layers}")
    print(f"  segment     : {args.segment_sec}s @ {args.target_fs} Hz")
    print(f"  steps       : {args.steps}, batch={args.batch_size}, peak lr={args.lr}")
    print("=" * 70)

    def loader(files, shuffle):
        ds = EdfStreamDataset(files, args.channels, segment_sec=args.segment_sec,
                              target_fs=args.target_fs, shuffle=shuffle)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                           num_workers=0, drop_last=True)

    # Fixed held-out batch (never trained on)
    val_batch = next(iter(loader(val_files, shuffle=False))).to(device)
    train_loader = loader(train_files, shuffle=True)

    model = build_data2vec_pretrain_model(
        n_eeg_channels=1, encoder_h=args.encoder_h,
        context_layers=args.context_layers, context_heads=args.context_heads,
        mask_rate=args.mask_rate, top_k_layers=args.top_k_layers,
        ema_anneal_steps=max(1, args.steps // 5),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * prog))

    def validate():
        model.eval()
        with torch.no_grad():
            pred, target, mask = model(val_batch)
            loss = model.compute_loss(pred, target, mask).item()
            pn = F.normalize(pred[mask], dim=1)
            tn = F.normalize(target[mask], dim=1)
            cos = (pn * tn).sum(1).mean().item()
        model.train()
        return cos, loss

    model.train()
    cos_hist, loss_hist = [], []
    step = 0
    while step < args.steps:
        produced = False
        for batch in train_loader:
            produced = True
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = batch.to(device)
            opt.zero_grad()
            pred, target, mask = model(batch)
            loss = model.compute_loss(pred, target, mask)
            if not torch.isfinite(loss):
                print(f"  step {step:4d} | non-finite loss — skipping")
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            model.update_ema()
            if step % 100 == 0 or step == args.steps - 1:
                vc, vl = validate()
                cos_hist.append(vc)
                loss_hist.append(vl)
                print(f"  step {step:4d} | lr={lr_at(step):.2e} | "
                      f"VAL_cos={vc:.4f} | VAL_loss={vl:.4f}")
            step += 1
            if step >= args.steps:
                break
        if not produced:
            print("FAIL: dataset yielded no usable segments", file=sys.stderr)
            return 1

    # ── Verdict ──────────────────────────────────────────────────
    peak_cos = max(cos_hist)
    final_cos = float(np.mean(cos_hist[-2:]))
    loss_dropped = loss_hist[-1] < loss_hist[0]
    learns = final_cos > args.cos_threshold
    no_decay = final_cos >= 0.6 * peak_cos

    print("-" * 70)
    print(f"  held-out cosine : peak={peak_cos:.4f}  final={final_cos:.4f}  "
          f"(need final > {args.cos_threshold} and >= 60% of peak)")
    print(f"  held-out loss   : {loss_hist[0]:.4f} → {loss_hist[-1]:.4f}  "
          f"({'decreased' if loss_dropped else 'NOT decreased'})")
    print("=" * 70)

    if learns and no_decay and loss_dropped:
        print("PASS ✓  data2vec generalises to unseen files and does NOT decay. "
              "Safe to run pretrain_short.sh on the cluster.")
        return 0
    print("FAIL ✗  Objective is not learning cleanly — do NOT spend cluster time.")
    if not learns:
        print(f"        final held-out cosine {final_cos:.4f} <= {args.cos_threshold}")
    if not no_decay:
        print(f"        held-out cosine decayed (final {final_cos:.4f} < 60% of "
              f"peak {peak_cos:.4f})")
    if not loss_dropped:
        print("        held-out loss did not decrease")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
