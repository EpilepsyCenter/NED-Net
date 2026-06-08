#!/bin/bash
# ============================================================
# Arrhenius — Resume BENDR pre-training
# ============================================================
# Resumes from the latest checkpoint in OUTPUT_DIR. Use this
# after `pretrain_short.sh` (continue to 30 epochs) or after
# `pretrain.sh` if it hit the 3-day (72 h) `gpu` partition walltime.
# At 4–8 h/epoch, 30 epochs (~120–240 h) typically needs 2–4 of
# these resubmits on Arrhenius — just run it again each time.
#
# Usage:
#   sbatch scripts/arrhenius/resume.sh
#
# Or chain automatically:
#   JOBID=$(sbatch --parsable scripts/arrhenius/pretrain.sh)
#   sbatch -d afterok:$JOBID scripts/arrhenius/resume.sh
# ============================================================

#SBATCH -J bendr_resume
#SBATCH -o logs/bendr_resume_%j.out
#SBATCH -e logs/bendr_resume_%j.err
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

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Apptainer 2>/dev/null || true
fi

arrhenius_run "nvidia-smi"

LATEST_CKPT=$(ls -t "${OUTPUT_DIR}"/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -z "${LATEST_CKPT}" ]; then
    echo "ERROR: No checkpoint found in ${OUTPUT_DIR}"
    echo "Run scripts/arrhenius/pretrain_short.sh or pretrain.sh first."
    exit 1
fi

echo "Resuming from: ${LATEST_CKPT}"

arrhenius_run "python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir '${EDF_DIR}' \
    --output-dir '${OUTPUT_DIR}' \
    --channels 0 1 2 3 4 5 6 7 \
    --bad-channels '${BAD_CHANNELS}' \
    --epochs 30 \
    --batch-size 64 \
    --lr 5e-4 \
    --method data2vec \
    --warmup-steps 500 \
    --weight-decay 1e-4 \
    --num-workers 16 \
    --segment-sec 60 \
    --target-fs 250 \
    --encoder-h 512 \
    --context-layers 8 \
    --context-heads 8 \
    --checkpoint-every 5 \
    --val-fraction 0.05 \
    --resume '${LATEST_CKPT}'"

echo "========================================="
echo "Resume finished at $(date)"
echo "========================================="
