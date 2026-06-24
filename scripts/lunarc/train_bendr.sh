#!/bin/bash
# ============================================================
# LUNARC COSMOS — BENDR fine-tuning (pre-trained encoder)
# ============================================================
# Fine-tunes a self-supervised pre-trained BENDR encoder on the converted
# annotations (*_ned_annotations.json sit next to each EDF).
#
# WHY THIS EXISTS: the only BENDR detector trained so far ran on Apple MPS
# (a backend known to silently corrupt training) and scored event_f1 ~0.06,
# while the from-scratch U-Net scored ~0.78 on the SAME data. This runs the
# identical fine-tune on a clean CUDA A100 to decide whether BENDR is actually
# broken or just an MPS artefact. Dataset settings below MATCH the working
# U-Net (neg/pos 4, pos_weight 5, no exclusions) so architecture + pretraining
# are the only differences.
#
# PREREQUISITE — upload the pre-trained checkpoint to LUNARC first:
#   rsync -avP ~/.eeg_seizure_analyzer/pretrained/run1_best.pt \
#     <user>@cosmos.lunarc.lu.se:~/.eeg_seizure_analyzer/pretrained/
#
# SELF-SUBMITTING + INTERACTIVE: run it directly on the login node; it asks
# for hyperparameters then sbatch's itself:
#       bash scripts/lunarc/train_bendr.sh
# Non-interactive: preset any var and it is used as-is:
#       EPOCHS=40 FREEZE_BACKBONE=1 sbatch scripts/lunarc/train_bendr.sh
#
# Pick the neg/pos ratio first if unsure:
#   srun -p gpua100 -A lu2026-2-60 -t 00:15:00 --gres=gpu:1 --pty \
#     python -m eeg_seizure_analyzer.ml.train_bendr \
#       --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data --analyze
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J bendr_ft
#SBATCH -o logs/bendr_ft_%j.out
#SBATCH -e logs/bendr_ft_%j.err
#SBATCH -A lu2026-2-60                    # LUNARC compute allocation (SUPR: LU 2026/2-60)
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- Defaults (used if the var isn't already set / left blank) ----
: "${MODEL_NAME:=bendr_cuda_v1}"
: "${PRETRAINED:=$HOME/.eeg_seizure_analyzer/pretrained/run1_best.pt}"
: "${EPOCHS:=40}"
: "${BATCH_SIZE:=8}"
: "${LR:=3e-4}"                # head / decoder LR
: "${ENCODER_LR:=1e-5}"        # differential LR for the pre-trained encoder
: "${FREEZE_ENCODER_EPOCHS:=5}"
: "${FREEZE_BACKBONE:=}"       # set to 1 to train head only (regularise tiny label sets)
: "${PATIENCE:=12}"
: "${NEG_POS_RATIO:=4}"        # match the working U-Net
: "${POS_WEIGHT:=5}"           # match the working U-Net
: "${EXCLUDE_ANIMALS:=}"       # space-separated animal IDs to drop, e.g. "355676"

# ============================================================
# Phase 1: not under SLURM -> prompt, then submit this script.
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    ask() {  # ask VAR "prompt"
        local cur; eval "cur=\${$1}"
        read -r -p "$2 [$cur]: " ans
        [ -n "$ans" ] && eval "$1=\"\$ans\""
    }
    echo "=== BENDR fine-tune — set hyperparameters (Enter = keep default) ==="
    ask MODEL_NAME            "Model name"
    ask PRETRAINED            "Pre-trained .pt path (blank = from scratch)"
    ask EPOCHS                "Epochs"
    ask BATCH_SIZE            "Batch size"
    ask LR                    "Head LR"
    ask ENCODER_LR            "Encoder LR"
    ask FREEZE_ENCODER_EPOCHS "Freeze-encoder epochs"
    read -r -p "Freeze whole backbone? 1=yes, blank=no [${FREEZE_BACKBONE:-no}]: " FREEZE_BACKBONE
    ask PATIENCE              "Patience"
    ask NEG_POS_RATIO         "Neg/pos ratio"
    ask POS_WEIGHT            "Pos weight"
    read -r -p "Exclude animal IDs (space-separated, blank = none): " EXCLUDE_ANIMALS
    echo "-------------------------------------------------------------"
    echo "Submitting: model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR enc_lr=$ENCODER_LR"
    echo "            freeze_enc_epochs=$FREEZE_ENCODER_EPOCHS freeze_backbone=${FREEZE_BACKBONE:-no}"
    echo "            patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=$POS_WEIGHT"
    echo "            pretrained=$PRETRAINED"
    echo "            exclude=${EXCLUDE_ANIMALS:-none}"
    export MODEL_NAME PRETRAINED EPOCHS BATCH_SIZE LR ENCODER_LR FREEZE_ENCODER_EPOCHS \
           FREEZE_BACKBONE PATIENCE NEG_POS_RATIO POS_WEIGHT EXCLUDE_ANIMALS
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
echo "Settings:    model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR enc_lr=$ENCODER_LR"
echo "             freeze_enc_epochs=$FREEZE_ENCODER_EPOCHS freeze_backbone=${FREEZE_BACKBONE:-no}"
echo "             patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=$POS_WEIGHT"
echo "             pretrained=$PRETRAINED"
echo "             exclude=${EXCLUDE_ANIMALS:-none}"
echo "========================================="

# Activate environment (same conda env as BENDR pre-training)
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

nvidia-smi

cd $HOME/NED-Net
mkdir -p logs

# EDF data + their *_ned_annotations.json sidecars live in project storage.
EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/edf_data"

if [ -n "$PRETRAINED" ] && [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED" >&2
    echo "Upload it first (see header), or set PRETRAINED= to train from scratch." >&2
    exit 1
fi

# Optional pre-trained path: omit the flag entirely if blank (-> from scratch).
PRETRAINED_ARG=()
[ -n "$PRETRAINED" ] && PRETRAINED_ARG=(--pretrained "$PRETRAINED")

# Freeze the whole backbone only if requested.
FREEZE_ARG=()
[ -n "$FREEZE_BACKBONE" ] && FREEZE_ARG=(--freeze-backbone)

# Space-separated IDs -> multiple --exclude-animals values (intentionally unquoted).
EXCLUDE_ARG=()
[ -n "$EXCLUDE_ANIMALS" ] && EXCLUDE_ARG=(--exclude-animals $EXCLUDE_ANIMALS)

python -m eeg_seizure_analyzer.ml.train_bendr \
    --data-dir "$EDF_DIR" \
    --model-name "$MODEL_NAME" \
    --neg-source hard \
    --neg-pos-ratio "$NEG_POS_RATIO" \
    --pos-weight "$POS_WEIGHT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --encoder-lr "$ENCODER_LR" \
    --freeze-encoder-epochs "$FREEZE_ENCODER_EPOCHS" \
    --patience "$PATIENCE" \
    "${PRETRAINED_ARG[@]}" \
    "${FREEZE_ARG[@]}" \
    "${EXCLUDE_ARG[@]}" \
    --weight-decay 1e-4 \
    --dropout 0.2 \
    --num-workers 8

echo "========================================="
echo "Training finished at $(date)"
echo "Model saved under ~/.eeg_seizure_analyzer/models/$MODEL_NAME"
echo "========================================="
