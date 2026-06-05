#!/bin/bash
# ============================================================
# Arrhenius — Short BENDR pre-training (5 epochs, GH200)
# ============================================================
# Full dataset, 5 epochs only. Lets you confirm the loss is
# decreasing on GH200 and produces an early model to test in
# NED-Net before committing to the long run. `resume.sh` will
# continue from the latest checkpoint to 30 epochs with no
# wasted work.
#
# Usage:
#   sbatch scripts/arrhenius/pretrain_short.sh
# ============================================================

#SBATCH -J bendr_5ep
#SBATCH -o logs/bendr_5ep_%j.out
#SBATCH -e logs/bendr_5ep_%j.err
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue
#SBATCH -A naiss2026-3-358-gpu
#SBATCH -p gpu

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
# sbatch exports the submitting shell's env (--export=ALL default), so a
# stale PROJECT_STORAGE/NAISS_PROJECT exported there would override
# _common.sh's ${VAR:-default} and break paths (e.g. CONTAINER_PATH would
# miss the personal/<user> segment). Clear them so _common.sh is the single
# source of truth; to override intentionally, edit _common.sh, not the env.
unset PROJECT_STORAGE NAISS_PROJECT EDF_DIR CODE_DIR OUTPUT_DIR BAD_CHANNELS \
      CONTAINER_PATH EXTRAS_VENV NVME_SCRATCH GPU_PARTITION GPU_QOS
source scripts/arrhenius/_common.sh

mkdir -p logs

echo "========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "========================================="
cat "$0"
echo "========================================="

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Apptainer 2>/dev/null || true
fi

arrhenius_run "nvidia-smi"

# GH200 has 96 GB HBM per GPU vs A100's 80 GB. Keeping batch-size
# at 64 mirrors LUNARC for apples-to-apples loss curves; raise
# only after a deliberate hyper-parameter sweep.
arrhenius_run "python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir '${EDF_DIR}' \
    --output-dir '${OUTPUT_DIR}' \
    --channels 0 1 2 3 4 5 6 7 \
    --bad-channels '${BAD_CHANNELS}' \
    --epochs 5 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --num-workers 16 \
    --segment-sec 60 \
    --target-fs 250 \
    --encoder-h 512 \
    --context-layers 8 \
    --context-heads 8 \
    --checkpoint-every 1 \
    --val-fraction 0.05"

echo "========================================="
echo "Short run finished at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Next: sbatch scripts/arrhenius/resume.sh    (continues to 30 epochs)"
echo "========================================="
