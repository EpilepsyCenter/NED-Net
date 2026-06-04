#!/bin/bash
# ============================================================
# Arrhenius — Tiny GPU smoke test (2–9 EDF files, ≤ 30 min)
# ============================================================
# Same as test_gpu.sh but tuned for a *very* small dataset.
# Defaults like 8 workers and a 5% validation hold-out break or
# no-op when you only have a handful of files, so this variant
# overrides them.
#
# Use this BEFORE shipping the full dataset to confirm:
#   - SSH + sbatch + queue submission works for you
#   - The NGC PyTorch container starts on a Grace Hopper node
#   - torch sees the H200 GPU through the container
#   - bendr_pretrain ingests YOUR EDFs (not just generic ones)
#
# Output goes to a separate folder so it doesn't pollute real runs:
#   $PROJECT_STORAGE/bendr_output/tiny_run/
#
# Usage:
#   sbatch scripts/arrhenius/test_gpu_tiny.sh
# ============================================================

#SBATCH -J bendr_tiny
#SBATCH -o logs/bendr_tiny_%j.out
#SBATCH -e logs/bendr_tiny_%j.err
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue
#SBATCH -A naiss2026-3-358-gpu
#SBATCH -p gpu

set -euo pipefail

# Under sbatch, $0 is the spooled script copy (…/slurmd/jobNNNN/), so
# dirname-based cd lands in the Slurm spool tree, not the repo. Use
# SLURM_SUBMIT_DIR (where `sbatch` was invoked — run it from the repo
# root); fall back to the $0 trick only when run directly with `bash`.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
# sbatch exports the submitting shell's env (--export=ALL default), so a
# stale PROJECT_STORAGE/NAISS_PROJECT exported there would override
# _common.sh's ${VAR:-default} and break paths (e.g. CONTAINER_PATH would
# miss the personal/<user> segment). Clear them so _common.sh is the single
# source of truth; to override intentionally, edit _common.sh, not the env.
unset PROJECT_STORAGE NAISS_PROJECT EDF_DIR CODE_DIR OUTPUT_DIR \
      CONTAINER_PATH EXTRAS_VENV NVME_SCRATCH GPU_PARTITION GPU_QOS
source scripts/arrhenius/_common.sh

mkdir -p logs

echo "========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Project:     ${NAISS_PROJECT}"
echo "Storage:     ${PROJECT_STORAGE}"
echo "EDF dir:     ${EDF_DIR}"
echo "========================================="
cat "$0"
echo "========================================="

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Apptainer 2>/dev/null || true
fi

arrhenius_run "nvidia-smi"
arrhenius_run "python -c 'import torch, platform; \
print(f\"Arch:      {platform.machine()}\"); \
print(f\"PyTorch:   {torch.__version__}\"); \
print(f\"CUDA:      {torch.version.cuda}\"); \
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"; \
print(f\"Device:    {dev}\"); \
assert torch.cuda.is_available(), \"CUDA not visible inside container\"'"

echo "EDF files found:"
ls "${EDF_DIR}" || echo "(none — check the path / upload completed)"

# Workers dropped to 2 so a handful of files don't leave 6+ workers
# idle (or crash on an empty shard). Validation disabled because 5%
# of <20 files rounds to 0.
arrhenius_run "python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir '${EDF_DIR}' \
    --output-dir '${PROJECT_STORAGE}/bendr_output/tiny_run' \
    --channels 0 1 2 3 4 5 6 7 \
    --epochs 2 \
    --batch-size 16 \
    --num-workers 2 \
    --segments-per-file 3 \
    --val-fraction 0 \
    --checkpoint-every 1"

echo "========================================="
echo "Tiny smoke test finished at $(date)"
echo "Output: ${PROJECT_STORAGE}/bendr_output/tiny_run/"
echo "========================================="
