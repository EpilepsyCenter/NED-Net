#!/bin/bash
# ============================================================
# LUNARC COSMOS — Quick GPU test (test queue, max 1 hour)
# ============================================================
# Runs a 2-epoch pre-training on a tiny sample to verify:
#   - GPU is detected and working
#   - EDF files are found and readable
#   - Model trains and checkpoints save correctly
#
# Usage:
#   sbatch scripts/lunarc/test_gpu.sh
# ============================================================

#SBATCH -p gpua100
#SBATCH --qos=test
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1          # 1 of the node's 2 A100s — don't reserve the whole node
#SBATCH -J bendr_test
#SBATCH -o logs/bendr_test_%j.out
#SBATCH -e logs/bendr_test_%j.err
#SBATCH -A lu2026-2-60                    # LUNARC compute allocation (SUPR: LU 2026/2-60)
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# Print job info for debugging
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

# Verify GPU
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd $HOME/NED-Net
mkdir -p logs

# EDF data lives in the project storage (same SUPR ID as compute: LU 2026/2-60)
EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/edf_data"

python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir "$EDF_DIR" \
    --output-dir $HOME/bendr_output/test_run \
    --channels 0 1 2 3 4 5 6 7 \
    --epochs 2 \
    --batch-size 32 \
    --num-workers 8 \
    --segments-per-file 5 \
    --checkpoint-every 1

echo "========================================="
echo "Test finished at $(date)"
echo "Check output in: $HOME/bendr_output/test_run/"
echo "========================================="
