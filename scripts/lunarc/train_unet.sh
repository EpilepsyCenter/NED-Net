#!/bin/bash
# ============================================================
# LUNARC COSMOS — Supervised U-Net seizure-detection training
# ============================================================
# Trains the from-scratch U-Net on the converted annotations
# (*_ned_annotations.json sit next to each EDF).
#
# SELF-SUBMITTING + INTERACTIVE:
#   Run it directly on the login/desktop node and it asks for the
#   hyperparameters (Enter accepts the default), then submits the
#   SLURM job for you:
#       bash scripts/lunarc/train_unet.sh
#   It can't prompt from inside the batch job (compute nodes have no
#   terminal), so it prompts first, then sbatch's itself with your
#   answers passed as environment variables.
#
#   Non-interactive / scripted use still works — preset any of the
#   vars and they're used as-is (no prompt for those):
#       EPOCHS=80 LR=1e-4 sbatch scripts/lunarc/train_unet.sh
#
# Pick the neg/pos ratio first if unsure:
#   srun -p gpua100 -A lu2026-2-60 -t 00:15:00 --gres=gpu:1 --pty \
#     python -m eeg_seizure_analyzer.ml.train_unet \
#       --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data --analyze
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J unet_train
#SBATCH -o logs/unet_train_%j.out
#SBATCH -e logs/unet_train_%j.err
#SBATCH -A lu2026-2-60                    # LUNARC compute allocation (SUPR: LU 2026/2-60)
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- Defaults (used if the var isn't already set / left blank) ----
: "${MODEL_NAME:=unet_kaha_v1}"
: "${EPOCHS:=50}"
: "${BATCH_SIZE:=16}"
: "${LR:=3e-4}"
: "${PATIENCE:=10}"
: "${NEG_POS_RATIO:=8}"
# POS_WEIGHT left unset/blank => auto (train_unet sets it to NEG_POS_RATIO).

# ============================================================
# Phase 1: not under SLURM -> prompt, then submit this script.
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    ask() {  # ask VAR "prompt" "default"
        local cur; eval "cur=\${$1}"
        read -r -p "$2 [$cur]: " ans
        [ -n "$ans" ] && eval "$1=\"\$ans\""
    }
    echo "=== U-Net training — set hyperparameters (Enter = keep default) ==="
    ask MODEL_NAME    "Model name"        # default unet_kaha_v1
    ask EPOCHS        "Epochs"
    ask BATCH_SIZE    "Batch size"
    ask LR            "Learning rate"
    ask PATIENCE      "Patience"
    ask NEG_POS_RATIO "Neg/pos ratio"
    read -r -p "Pos weight (blank = auto = neg/pos ratio): " POS_WEIGHT
    echo "-------------------------------------------------------------"
    echo "Submitting: model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
    echo "            patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=${POS_WEIGHT:-auto}"
    sbatch --export=ALL,MODEL_NAME="$MODEL_NAME",EPOCHS="$EPOCHS",\
BATCH_SIZE="$BATCH_SIZE",LR="$LR",PATIENCE="$PATIENCE",\
NEG_POS_RATIO="$NEG_POS_RATIO",POS_WEIGHT="$POS_WEIGHT" "$0"
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
echo "             patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=${POS_WEIGHT:-auto}"
echo "========================================="

# Activate environment (same conda env as BENDR)
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

nvidia-smi

cd $HOME/NED-Net
mkdir -p logs

# EDF data + their *_ned_annotations.json sidecars live in project storage.
EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/edf_data"

# Optional pos-weight: only pass the flag if the user set it (else train_unet
# auto-picks pos_weight = neg/pos ratio).
POS_WEIGHT_ARG=()
[ -n "$POS_WEIGHT" ] && POS_WEIGHT_ARG=(--pos-weight "$POS_WEIGHT")

python -m eeg_seizure_analyzer.ml.train_unet \
    --data-dir "$EDF_DIR" \
    --model-name "$MODEL_NAME" \
    --neg-source hard \
    --neg-pos-ratio "$NEG_POS_RATIO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    "${POS_WEIGHT_ARG[@]}" \
    --weight-decay 1e-4 \
    --base-filters 32 \
    --depth 4 \
    --dropout 0.2 \
    --num-workers 8

echo "========================================="
echo "Training finished at $(date)"
echo "Model saved under ~/.eeg_seizure_analyzer/models/$MODEL_NAME"
echo "========================================="
