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
#SBATCH -t 02:00:00
# 2 h, not 30 min: the model is small but with SV2A + RAM_GDNF the dataset scan
# and window extraction dominate, and 1,216 annotated EDFs is 7x the original.
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
: "${CONV_NEG_FROM_REJECTED:=0}"  # 1 = train on `rejected` events as NON-convulsive.
#   Stage 2 is trained on confirmed seizures but DEPLOYED on every detection, most of
#   which are false positives -- so by default it has never seen a non-seizure. With the
#   frozen classifier that failed safely (RAM_GDNF noise was unfamiliar, scored low,
#   precision 33% -> 61%); after retraining on RAM_GDNF convulsive events the same noise
#   resembles them and precision is FLAT AT 11% at every threshold. Negatives fix that,
#   tuning cannot. Mir's rejected rows are the right source: video-adjudicated "not a
#   convulsive/behavioural seizure" -- wrong for the detector, exactly right here
#   (`build_merged_sidecars.py:76`).
: "${CONV_NEG_METHODS:=}"      # restrict those negatives to these detection_method
#   values (space-separated); empty = every rejected event. e.g. "mir_candidate"
: "${CONV_NEG_POS_RATIO:=5}"   # cap on TOTAL negatives per convulsive positive.
#   Non-convulsive SEIZURES are kept in full; rejected events fill the rest, so the
#   convulsive-vs-non-convulsive-seizure boundary is not swamped by easy noise.
: "${MAX_POSITIVE_SEC:=0}"     # 0 = no cap; 100 drops Mir's chained blocks.

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
    ask EXCLUDE_ANIMALS         "Exclude animal IDs (space-separated, blank = none)"
    ask CONV_NEG_FROM_REJECTED  "Train rejected events as non-convulsive (1/0)"
    ask CONV_NEG_METHODS        "Restrict those to detection_methods (blank = all)"
    ask CONV_NEG_POS_RATIO      "Total negatives per convulsive positive"
    ask MAX_POSITIVE_SEC        "Max positive duration in s (0 = no cap)"
    echo "-------------------------------------------------------------"
    echo "Submitting: model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
    echo "            patience=$PATIENCE exclude=${EXCLUDE_ANIMALS:-none}"
    echo "            neg_from_rejected=$CONV_NEG_FROM_REJECTED methods=${CONV_NEG_METHODS:-all}"
    echo "            neg/pos cap=$CONV_NEG_POS_RATIO max_positive_sec=$MAX_POSITIVE_SEC"
    export MODEL_NAME EPOCHS BATCH_SIZE LR PATIENCE EXCLUDE_ANIMALS EDF_DIR \
           CONV_NEG_FROM_REJECTED CONV_NEG_METHODS CONV_NEG_POS_RATIO MAX_POSITIVE_SEC
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
echo "             neg_from_rejected=$CONV_NEG_FROM_REJECTED methods=${CONV_NEG_METHODS:-all}"
echo "             neg/pos cap=$CONV_NEG_POS_RATIO max_positive_sec=$MAX_POSITIVE_SEC"
echo "Data dir:    $EDF_DIR"
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
# Overridable, same as train_unet.sh:
#   SV2A only              : .../lu2026-2-60/edf_data
#   SV2A + RAM_GDNF (arm A) : .../lu2026-2-60          <- the parent
#   RAM_GDNF only           : .../lu2026-2-60/RAM_GDNF_2025
# The scan is recursive on *_ned_annotations.json, so the data-dir choice IS the
# dataset. Mir's RAM_GDNF labels carry features.convulsive from his
# candidate_type (368 convulsive, 51 behaviour), which is exactly what Stage 2
# needs -- so the parent is the right choice for a convulsive retrain.
: "${EDF_DIR:=/lunarc/nobackup/projects/lu2026-2-60/edf_data}"

# Space-separated IDs -> multiple --exclude-animals values (intentionally unquoted).
EXCLUDE_ARG=()
[ -n "$EXCLUDE_ANIMALS" ] && EXCLUDE_ARG=(--exclude-animals $EXCLUDE_ANIMALS)

NEG_ARG=()
[ "$CONV_NEG_FROM_REJECTED" = "1" ] && NEG_ARG=(--conv-neg-from-rejected)
# Intentionally unquoted: space-separated values -> multiple argparse values.
[ -n "$CONV_NEG_METHODS" ] && NEG_ARG+=(--conv-neg-method $CONV_NEG_METHODS)

python -m eeg_seizure_analyzer.ml.train_convulsive \
    --data-dir "$EDF_DIR" \
    --model-name "$MODEL_NAME" \
    "${NEG_ARG[@]}" \
    --conv-neg-pos-ratio "$CONV_NEG_POS_RATIO" \
    --max-positive-sec "$MAX_POSITIVE_SEC" \
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
