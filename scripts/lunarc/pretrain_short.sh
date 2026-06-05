#!/bin/bash
# ============================================================
# LUNARC COSMOS — Short BENDR pre-training (5 epochs)
# ============================================================
# Quick training run on the FULL dataset but only 5 epochs.
# Enough to verify the loss is decreasing and produce an early
# model you can fine-tune in NED-Net to check it's useful.
#
# The full run (pretrain.sh) can RESUME from this checkpoint,
# so no work is wasted.
#
# Usage:
#   sbatch scripts/lunarc/pretrain_short.sh
#
# After this finishes and you're happy with the result:
#   sbatch scripts/lunarc/resume.sh    (continues to 30 epochs)
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1          # 1 of the node's 2 A100s — don't reserve the whole node
#SBATCH -J bendr_5ep
#SBATCH -o logs/bendr_5ep_%j.out
#SBATCH -e logs/bendr_5ep_%j.err
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
# Per-batch noisy-channel exclusions — generate once with
#   python scripts/make_bad_channels.py --data-dir "$EDF_DIR"
# The trainer hard-fails if this file is missing (intentional).
BAD_CHANNELS="/lunarc/nobackup/projects/lu2026-2-60/bad_channels.json"

python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir "$EDF_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --channels 0 1 2 3 4 5 6 7 \
    --bad-channels "$BAD_CHANNELS" \
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
    --val-fraction 0.05

echo "========================================="
echo "Short run finished at $(date)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Download best_model.pt and try fine-tuning in NED-Net"
echo "  2. If it looks good, continue to 30 epochs:"
echo "     sbatch scripts/lunarc/resume.sh"
echo "========================================="
