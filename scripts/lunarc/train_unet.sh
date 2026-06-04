#!/bin/bash
# ============================================================
# LUNARC COSMOS — Supervised U-Net seizure-detection training
# ============================================================
# Trains the from-scratch U-Net on the colleague's converted
# annotations (*_ned_annotations.json sit next to each EDF).
#
# The U-Net is small (~23M params) and the labelled set is tiny
# (tens of seizures + a few thousand hard negatives), so this is
# a SHORT job — minutes per epoch, well under an hour total.
# Single GPU is plenty; no A100-80GB needed.
#
# Run the analysis first to pick the ratio (see RECOMMENDATION):
#   srun -p gpua100 -A lu2026-2-60 -t 00:15:00 --gres=gpu:1 --pty \
#     python -m eeg_seizure_analyzer.ml.train_unet --data-dir "$EDF_DIR" --analyze
#
# Then submit:
#   sbatch scripts/lunarc/train_unet.sh
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

echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "========================================="
cat $0
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

# Class balance: --neg-source hard uses the reviewed-and-rejected spike
# candidates as HARD negatives (default); --neg-source random ignores them and
# samples background from the recordings (the old behaviour). Worth running both
# as an A/B — just change --neg-source and --model-name.
# neg-pos-ratio 8 + matching pos-weight is a sensible first run;
# --neg-source hard --neg-pos-ratio 0 keeps every hard negative.
python -m eeg_seizure_analyzer.ml.train_unet \
    --data-dir "$EDF_DIR" \
    --model-name unet_kaha_v1 \
    --neg-source hard \
    --neg-pos-ratio 8 \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --base-filters 32 \
    --depth 4 \
    --dropout 0.2 \
    --patience 10 \
    --num-workers 8

echo "========================================="
echo "Training finished at $(date)"
echo "Model saved under ~/.eeg_seizure_analyzer/models/unet_kaha_v1"
echo "========================================="
