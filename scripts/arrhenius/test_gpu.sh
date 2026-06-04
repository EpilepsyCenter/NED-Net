#!/bin/bash
# ============================================================
# Arrhenius — GH200 GPU smoke test (≤ 30 min)
# ============================================================
# Confirms:
#   - container starts on a GH200 node
#   - torch sees the H100 GPU on the Grace ARM CPU
#   - EDF directory is readable
#   - 2-epoch tiny pretrain checkpoints successfully
#
# Usage:
#   sbatch scripts/arrhenius/test_gpu.sh
# ============================================================

#SBATCH -J bendr_test
#SBATCH -o logs/bendr_test_%j.out
#SBATCH -e logs/bendr_test_%j.err
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
# Project comes from your SUPR allocation (find it via `storagequota`
# or at https://supr.naiss.se/account/). Update the line below once
# the account is approved.
#SBATCH -A naiss2026-3-358-gpu
#SBATCH -p gpu

set -euo pipefail

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

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Apptainer 2>/dev/null || true
fi

# Verify GPU visible to the container
arrhenius_run "nvidia-smi"
arrhenius_run "python -c 'import torch, platform; \
print(f\"Arch:      {platform.machine()}\"); \
print(f\"PyTorch:   {torch.__version__}\"); \
print(f\"CUDA:      {torch.version.cuda}\"); \
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"; \
print(f\"Device:    {dev}\"); \
assert torch.cuda.is_available(), \"CUDA not visible inside container\"'"

# Tiny pretrain on a few segments per file
arrhenius_run "python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir '${EDF_DIR}' \
    --output-dir '${PROJECT_STORAGE}/bendr_output/test_run' \
    --channels 0 1 2 3 4 5 6 7 \
    --epochs 2 \
    --batch-size 32 \
    --num-workers 8 \
    --segments-per-file 5 \
    --checkpoint-every 1"

echo "========================================="
echo "Test finished at $(date)"
echo "Check: ${PROJECT_STORAGE}/bendr_output/test_run/"
echo "========================================="
