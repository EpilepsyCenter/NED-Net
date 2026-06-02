"""Pinpoint the BENDR pre-training NaN on GPU.

A single forward pass is finite on the A100, so the NaN develops during
training (backward + optimizer steps). This runs real training steps on
real data and reports, per step, whether the loss / gradients / weights
go non-finite first and which parameter — under both the default
(fused flash/mem-efficient) and math-only SDPA backends.

Run on a GPU node:
    interactive -A lu2026-2-60 -p gpua100 --gres=gpu:1 -t 00:15:00
    module purge && module load Anaconda3/2024.06-1 && source config_conda.sh
    conda activate bendr
    cd ~/NED-Net && python scripts/lunarc/diagnose_nan.py
"""
import argparse
import glob
import os

import torch
import torch.nn as nn

from eeg_seizure_analyzer.ml.bendr_model import build_pretrain_model
from eeg_seizure_analyzer.ml.bendr_pretrain import EdfStreamDataset

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir",
                default="/lunarc/nobackup/projects/lu2026-2-60/edf_data")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  torch={torch.__version__}  lr={args.lr}")

files = sorted(glob.glob(os.path.join(args.data_dir, "*.edf")))
ds = EdfStreamDataset(files, channels=list(range(8)),
                      segments_per_file=5, shuffle=True)
pool = []
for s in ds:
    pool.append(s)
    if len(pool) >= args.steps * args.batch:
        break
print(f"pooled {len(pool)} segments\n")


def run(tag):
    torch.manual_seed(0)
    model = build_pretrain_model(n_eeg_channels=1).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    i = 0
    print(f"===== {tag} =====")
    for step in range(args.steps):
        chunk = pool[i:i + args.batch]
        i += args.batch
        if len(chunk) < args.batch:
            print("  (ran out of pooled segments)")
            break
        batch = torch.stack(chunk).to(dev)
        opt.zero_grad()
        logits, z, _ = model(batch)
        loss = model.compute_loss(logits, z)
        loss_fin = bool(torch.isfinite(loss))
        loss.backward()
        gbad = [n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        wbad = next((n for n, p in model.named_parameters()
                     if not torch.isfinite(p).all()), None)
        if not loss_fin or gbad or wbad:
            print(f"  step {step:2d} loss={float(loss):.4f} "
                  f"loss_fin={loss_fin} gnorm={float(gnorm):.3e} "
                  f"n_grad_nan={len(gbad)} first_weight_nan={wbad}")
            if gbad:
                print(f"    FIRST non-finite GRADIENTS: {gbad[:6]}")
            print(f"  --> first failure at step {step}")
            return
        if step < 5 or step % 5 == 0:
            print(f"  step {step:2d} loss={float(loss):.4f} "
                  f"gnorm={float(gnorm):.3e}")
    print("  completed all steps with finite loss/grads/weights")


run("default SDPA backends")
print()
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    run("math-only SDPA backend")
except Exception as e:
    print(f"could not force math SDPA backend: {e}")
