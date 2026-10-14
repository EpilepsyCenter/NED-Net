#!/bin/bash
# Train the Stage-1 U-Net seizure detector on a GPU node.
#
# Data: a folder of EDFs with their *_ned_annotations.json sidecars (reviewed in the
# app's Training tab), scanned recursively. Check the class balance and split first:
#   python -m eeg_seizure_analyzer.ml.train_unet --data-dir /path/to/edfs --analyze
#
# Submit (any setting below can be overridden from the environment):
#   EDF_DIR=/path/to/edfs MODEL_NAME=my_unet sbatch scripts/slurm/train_unet.sbatch
#
# The model is saved to ~/.eeg_seizure_analyzer/models/$MODEL_NAME and appears in the app.

#SBATCH -J unet_train
#SBATCH -p gpu                  # your GPU partition
#SBATCH -A my-allocation        # your compute allocation
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00             # ~ epochs x windows / ~13 windows/s on an A100, plus margin
#SBATCH -o logs/unet_train_%j.out
#SBATCH -e logs/unet_train_%j.err

: "${EDF_DIR:?set EDF_DIR to the folder of EDFs + annotation sidecars}"
: "${MODEL_NAME:=my_unet}"
: "${EPOCHS:=50}"
: "${BATCH_SIZE:=8}"
: "${LR:=1e-3}"
: "${POS_WEIGHT:=5}"
: "${PATIENCE:=10}"
: "${NEG_POS_RATIO:=8}"         # negative windows per positive
: "${NEG_SOURCE:=hard}"         # hard: reviewed rejections are negatives; random: background
: "${VAL_MODE:=animal}"         # animal | recording | temporal (see --help)
: "${TARGET:=seizure}"          # seizure | convulsive
: "${EXCLUDE_ANIMALS:=}"        # space-separated animal IDs to leave out
: "${EXTRA_ARGS:=}"             # any other train_unet options, e.g. "--max-positive-sec 100"

module purge
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nednet
nvidia-smi

cd "$HOME/NED-Net"
mkdir -p logs

EXCL=(); [ -n "$EXCLUDE_ANIMALS" ] && EXCL=(--exclude-animals $EXCLUDE_ANIMALS)
# shellcheck disable=SC2086
python -m eeg_seizure_analyzer.ml.train_unet \
    --data-dir "$EDF_DIR" --model-name "$MODEL_NAME" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
    --pos-weight "$POS_WEIGHT" --patience "$PATIENCE" \
    --neg-pos-ratio "$NEG_POS_RATIO" --neg-source "$NEG_SOURCE" \
    --val-mode "$VAL_MODE" --target "$TARGET" --stable-val-split --fp32 \
    "${EXCL[@]}" $EXTRA_ARGS --num-workers 8
