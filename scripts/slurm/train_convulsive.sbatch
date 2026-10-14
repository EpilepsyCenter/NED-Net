#!/bin/bash
# Train the Stage-2 convulsive classifier on a GPU node.
# Use the same split mode as the Stage-1 run it will be paired with.
#
#   EDF_DIR=/path/to/edfs MODEL_NAME=my_conv sbatch scripts/slurm/train_convulsive.sbatch

#SBATCH -J conv_train
#SBATCH -p gpu
#SBATCH -A my-allocation
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o logs/conv_train_%j.out
#SBATCH -e logs/conv_train_%j.err

: "${EDF_DIR:?set EDF_DIR to the folder of EDFs + annotation sidecars}"
: "${MODEL_NAME:=my_conv}"
: "${EPOCHS:=30}"
: "${VAL_MODE:=animal}"
: "${EXTRA_ARGS:=}"             # e.g. "--max-positive-sec 100 --conv-label-method manual"

module purge
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nednet

cd "$HOME/NED-Net"
mkdir -p logs

# --conv-neg-from-rejected: rejected detections become non-convulsive examples, so the
# classifier has seen the false positives it will be applied to.
# shellcheck disable=SC2086
python -m eeg_seizure_analyzer.ml.train_convulsive \
    --data-dir "$EDF_DIR" --model-name "$MODEL_NAME" --epochs "$EPOCHS" \
    --val-mode "$VAL_MODE" --conv-neg-from-rejected $EXTRA_ARGS
