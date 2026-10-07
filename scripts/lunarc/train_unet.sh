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
#SBATCH -t 06:00:00
# 06:00 not 02:00 — at ~226 s/epoch (observed, job 3795251 on the RAM_GDNF-only
# arm) 50 epochs is >3 h, and the SV2A+RAM_GDNF arm has more data so its epochs
# are slower. Patience normally stops well short of this; the ceiling only
# matters in the good case where val_loss keeps improving.
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
: "${BATCH_SIZE:=32}"   # A100 + fp16 has plenty of headroom; bigger batch keeps
#                         the tensor cores fed. Bump to 64 if memory allows.
: "${LR:=3e-4}"
: "${PATIENCE:=10}"
: "${NEG_POS_RATIO:=8}"
: "${EXCLUDE_ANIMALS:=}"   # space-separated animal IDs to drop, e.g. "355676"
: "${NEG_SOURCE:=hard}"   # hard | random.
#   `hard` uses `rejected` annotations as negatives. That is right for SV2A, whose
#   rejections were adjudicated for seizure-ness -- but WRONG for Mir's RAM_GDNF
#   `False` rows, which mean "not a convulsive/behavioural seizure" and were
#   confirmed by visual review (2026-10-07) to contain real non-convulsive activity,
#   including 180 s blocks of clear bursting. Used as hard negatives they train the
#   detector to suppress the events it should find.
#   `random` ignores every rejected label and samples background negatives instead:
#   no sidecar edits, so completed human reviews on disk are untouched.
: "${STABLE_VAL_SPLIT:=1}"  # 1 = keep the dominant convulsive animals in train.
#   Positives are heavily concentrated (355675 alone carries ~1/3 of all convulsive
#   events), so a random split can land half of them in validation -- which wastes
#   training data and makes the result a lottery on the seed. Set to 0 only to
#   reproduce a run from before this flag existed.
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
    # Use ask() here too: a bare `read` sets the var to EMPTY on Enter, which
    # silently destroys a preset passed in the environment. That is how job
    # 3795249 was submitted with exclude=none despite EXCLUDE_ANIMALS being set,
    # which would have trained a leave-one-batch-out fold on its own test set.
    ask POS_WEIGHT      "Pos weight (Enter keeps current; unset = auto = neg/pos ratio)"
    ask EXCLUDE_ANIMALS "Exclude animal IDs (space-separated)"
    echo "-------------------------------------------------------------"
    echo "Submitting: model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
    echo "            patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=${POS_WEIGHT:-auto}"
    echo "            exclude=${EXCLUDE_ANIMALS:-none}"
    # Pass settings via the (exported) environment + --export=ALL — robust for
    # values that contain spaces (e.g. multiple excluded IDs).
    export MODEL_NAME EPOCHS BATCH_SIZE LR PATIENCE NEG_POS_RATIO POS_WEIGHT \
           EXCLUDE_ANIMALS EDF_DIR STABLE_VAL_SPLIT NEG_SOURCE
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
echo "             patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=${POS_WEIGHT:-auto}"
echo "             exclude=${EXCLUDE_ANIMALS:-none}"
echo "Data dir:    $EDF_DIR"
echo "Stable val split: $STABLE_VAL_SPLIT   neg-source: $NEG_SOURCE"
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
# Overridable so one script serves both retraining arms:
#   arm A (SV2A + RAM_GDNF): EDF_DIR=/lunarc/nobackup/projects/lu2026-2-60
#   arm B (RAM_GDNF only):   EDF_DIR=/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025
# The scan is recursive and keyed on *_ned_annotations.json sidecars, so the
# data-dir choice IS the dataset definition -- check what lives under it first.
: "${EDF_DIR:=/lunarc/nobackup/projects/lu2026-2-60/edf_data}"

# Optional pos-weight: only pass the flag if the user set it (else train_unet
# auto-picks pos_weight = neg/pos ratio).
POS_WEIGHT_ARG=()
[ -n "$POS_WEIGHT" ] && POS_WEIGHT_ARG=(--pos-weight "$POS_WEIGHT")

# Space-separated IDs -> multiple --exclude-animals values (intentionally unquoted).
EXCLUDE_ARG=()
[ -n "$EXCLUDE_ANIMALS" ] && EXCLUDE_ARG=(--exclude-animals $EXCLUDE_ANIMALS)

STABLE_ARG=()
[ "$STABLE_VAL_SPLIT" = "1" ] && STABLE_ARG=(--stable-val-split)

python -m eeg_seizure_analyzer.ml.train_unet \
    --data-dir "$EDF_DIR" \
    --model-name "$MODEL_NAME" \
    --neg-source "$NEG_SOURCE" \
    --neg-pos-ratio "$NEG_POS_RATIO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    "${POS_WEIGHT_ARG[@]}" \
    "${EXCLUDE_ARG[@]}" \
    "${STABLE_ARG[@]}" \
    --weight-decay 1e-4 \
    --base-filters 32 \
    --depth 4 \
    --dropout 0.2 \
    --num-workers 8

echo "========================================="
echo "Training finished at $(date)"
echo "Model saved under ~/.eeg_seizure_analyzer/models/$MODEL_NAME"
echo "========================================="
