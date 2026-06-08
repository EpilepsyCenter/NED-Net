#!/bin/bash
# ============================================================
# LUNARC COSMOS — Full BENDR pre-training job
# ============================================================
# Self-supervised pre-training on ~25,000 hours of rodent EEG.
# All 8 EDF channels treated independently (8x data multiplier).
#
# Estimated time: ~4-8 hours per epoch on A100 80GB.
# 30 epochs = ~5-10 days. May need 1-2 job submissions.
#
# Usage:
#   sbatch scripts/lunarc/pretrain.sh
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1          # 1 of the node's 2 A100s — don't reserve the whole node
#SBATCH -J bendr_pretrain
#SBATCH -o logs/bendr_pretrain_%j.out
#SBATCH -e logs/bendr_pretrain_%j.err
#SBATCH -A lu2026-2-60                    # LUNARC compute allocation (SUPR: LU 2026/2-60)
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# Print job info
echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "========================================="
cat $0
echo "========================================="

# Activate environment
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

nvidia-smi

cd $HOME/NED-Net
mkdir -p logs

# EDF data lives in the project storage (same SUPR ID as compute: LU 2026/2-60)
EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/edf_data"
OUTPUT_DIR="$HOME/bendr_output/run1"
# Per-batch noisy-channel exclusions. Generate once with
#   python scripts/make_bad_channels.py --data-dir "$EDF_DIR"
# (writes to <edf_data>/../bad_channels.json). The trainer hard-fails if
# this file is missing — that's intentional: better than silently training
# on the noisy channels you meant to drop.
BAD_CHANNELS="/lunarc/nobackup/projects/lu2026-2-60/bad_channels.json"

python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir "$EDF_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --channels 0 1 2 3 4 5 6 7 \
    --bad-channels "$BAD_CHANNELS" \
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
    --val-fraction 0.05

echo "========================================="
echo "Training finished at $(date)"
echo "Output: $OUTPUT_DIR"
echo "========================================="
