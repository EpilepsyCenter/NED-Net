#!/bin/bash
# ============================================================
# LUNARC COSMOS — Convulsive classifier training (Stage 2)
# ============================================================
# Trains the convulsive-vs-not classifier on the converted
# annotations (*_ned_annotations.json next to each EDF). This is
# Stage 2 of the seizure→convulsive cascade; Stage 1 is the U-Net
# (scripts/lunarc/train_unet.sh).
#
# Run it on a GPU node (CUDA) — NOT the Mac. The classifier diverges
# intermittently on Apple MPS (a backend bug); CUDA is numerically
# clean and fast, and the model is tiny so this finishes in minutes.
#
# SELF-SUBMITTING + INTERACTIVE:
#   Run it directly on the login/desktop node and it asks for the
#   hyperparameters (Enter accepts the default), then submits the
#   SLURM job for you:
#       bash scripts/lunarc/train_convulsive.sh
#   It can't prompt from inside the batch job (compute nodes have no
#   terminal), so it prompts first, then sbatch's itself with your
#   answers passed as environment variables.
#
#   Non-interactive / scripted use still works — preset any of the
#   vars and they're used as-is:
#       EPOCHS=40 LR=1e-4 sbatch scripts/lunarc/train_convulsive.sh
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J conv_train
#SBATCH -o logs/conv_train_%j.out
#SBATCH -e logs/conv_train_%j.err
#SBATCH -A lu2026-2-60                    # LUNARC compute allocation (SUPR: LU 2026/2-60)
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- Defaults (used if the var isn't already set / left blank) ----
: "${MODEL_NAME:=conv_kaha_v1}"
: "${EPOCHS:=30}"
: "${BATCH_SIZE:=16}"
: "${LR:=3e-4}"
: "${PATIENCE:=10}"
: "${EXCLUDE_ANIMALS:=355676}"   # space-separated IDs to drop; 355676 is noisy

# ============================================================
# Phase 1: not under SLURM -> prompt, then submit this script.
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    ask() {  # ask VAR "prompt"
        local cur; eval "cur=\${$1}"
        read -r -p "$2 [$cur]: " ans
        [ -n "$ans" ] && eval "$1=\"\$ans\""
    }
    echo "=== Convulsive classifier — set hyperparameters (Enter = keep default) ==="
    ask MODEL_NAME "Model name"
    ask EPOCHS     "Epochs"
    ask BATCH_SIZE "Batch size"
    ask LR         "Learning rate"
    ask PATIENCE   "Patience"
    read -r -p "Exclude animal IDs (space-separated, blank = none) [$EXCLUDE_ANIMALS]: " ans
    [ -n "$ans" ] && EXCLUDE_ANIMALS="$ans"
    echo "-------------------------------------------------------------"
    echo "Submitting: model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
    echo "            patience=$PATIENCE exclude=${EXCLUDE_ANIMALS:-none}"
    export MODEL_NAME EPOCHS BATCH_SIZE LR PATIENCE EXCLUDE_ANIMALS
    sbatch --export=ALL "$0"
    exit $?
fi

# ============================================================
# Phase 2: running under SLURM on a GPU node -> train.
# ============================================================
echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Settings:    model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
echo "             patience=$PATIENCE exclude=${EXCLUDE_ANIMALS:-none}"
echo "========================================="

# Activate environment (same conda env as BENDR / the U-Net job)
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

nvidia-smi

cd $HOME/NED-Net
mkdir -p logs

# EDF data + their *_ned_annotations.json sidecars live in project storage.
EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/edf_data"

# Space-separated IDs -> multiple --exclude-animals values (intentionally unquoted).
EXCLUDE_ARG=()
[ -n "$EXCLUDE_ANIMALS" ] && EXCLUDE_ARG=(--exclude-animals $EXCLUDE_ANIMALS)

python -m eeg_seizure_analyzer.ml.train_convulsive \
    --data-dir "$EDF_DIR" \
    --model-name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    "${EXCLUDE_ARG[@]}" \
    --num-workers 4

echo "========================================="
echo "Training finished at $(date)"
echo "Model saved under ~/.eeg_seizure_analyzer/models/$MODEL_NAME"
echo "========================================="
