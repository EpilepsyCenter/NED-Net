"""Pinpoint the BENDR pre-training NaN on GPU.

Runs a real-data forward pass with finiteness hooks on every submodule and
prints the FIRST module whose output goes non-finite while its inputs were
finite — i.e. the exact op introducing the NaN. Then retries with the
math-only SDPA backend (the CPU-equivalent attention kernel) to test
whether fused flash/mem-efficient attention is the cause.

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

from eeg_seizure_analyzer.ml.bendr_model import build_pretrain_model
from eeg_seizure_analyzer.ml.bendr_pretrain import EdfStreamDataset

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir",
                default="/lunarc/nobackup/projects/lu2026-2-60/edf_data")
ap.add_argument("--n", type=int, default=16)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  torch={torch.__version__}")

files = sorted(glob.glob(os.path.join(args.data_dir, "*.edf")))
ds = EdfStreamDataset(files, channels=list(range(8)),
                      segments_per_file=3, shuffle=False)
segs = []
for s in ds:
    segs.append(s)
    if len(segs) >= args.n:
        break
batch = torch.stack(segs).to(dev)
print(f"batch={tuple(batch.shape)} finite={bool(torch.isfinite(batch).all())}")

torch.manual_seed(0)
model = build_pretrain_model(n_eeg_channels=1).to(dev)
model.train()

records = []


def mk_hook(name):
    def hook(mod, inp, out):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if torch.is_tensor(o) and not bool(torch.isfinite(o).all()):
                in_fin = all(bool(torch.isfinite(i).all())
                             for i in inp if torch.is_tensor(i))
                records.append((name, type(mod).__name__, in_fin))
                return
    return hook


for n, m in model.named_modules():
    if n:
        m.register_forward_hook(mk_hook(n))


def run(tag):
    records.clear()
    logits, z, _ = model(batch)
    loss = model.compute_loss(logits, z)
    print(f"\n[{tag}] loss_finite={bool(torch.isfinite(loss))} "
          f"logits_finite={bool(torch.isfinite(logits).all())}")
    if records:
        print(f"[{tag}] first non-finite outputs (innermost first):")
        for name, typ, infin in records[:10]:
            print(f"    {name:45s} {typ:24s} inputs_finite={infin}")
    else:
        print(f"[{tag}] no non-finite module outputs detected")


run("default SDPA backends")

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    run("math-only SDPA backend")
except Exception as e:
    print(f"could not force math SDPA backend: {e}")
