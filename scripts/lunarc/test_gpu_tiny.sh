#!/bin/bash
# ============================================================
# LUNARC COSMOS — Tiny GPU smoke test (2-3 EDF files)
# ============================================================
# Same as test_gpu.sh but tuned for a *very* small dataset
# (literally 2-3 converted EDF files). Defaults like 8 workers
# and a 5% validation hold-out break or no-op when you only
# have a handful of files, so this variant overrides them.
#
# Use this BEFORE shipping the full dataset to confirm:
#   - SSH + sbatch + queue submission works for you
#   - The conda env + GPU module path on COSMOS resolves
#   - bendr_pretrain ingests YOUR EDFs (not just generic ones)
#
# Output is sent to a separate folder so it doesn't pollute
# real runs:  $HOME/bendr_output/tiny_run/
#
# Usage:
#   sbatch scripts/lunarc/test_gpu_tiny.sh
# ============================================================

#SBATCH -p gpua100
#SBATCH --qos=test
#SBATCH --gres=gpu:1          # 1 of the node's 2 A100s — don't reserve the whole node
#SBATCH -n 1                  # single task
#SBATCH -c 8                  # 8 cores: 2 dataloader workers + main loop + headroom
#SBATCH --mem=16G             # last tiny run used 253 MB; 16G is generous slack
#SBATCH -t 00:15:00           # last run was 49 s — 15 min is plenty
#SBATCH -J bendr_tiny
#SBATCH -o logs/bendr_tiny_%j.out
#SBATCH -e logs/bendr_tiny_%j.err
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

module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd $HOME/NED-Net
mkdir -p logs

EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/edf_data"

echo "EDF files found:"
ls "$EDF_DIR" || echo "(none — check the path / upload completed)"

# Workers dropped to 2 so 3 files don't leave 6 workers idle (or
# crash on an empty shard). Validation disabled because 5% of 3
# files rounds to 0.
python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir "$EDF_DIR" \
    --output-dir "$HOME/bendr_output/tiny_run" \
    --channels 0 1 2 3 4 5 6 7 \
    --epochs 2 \
    --batch-size 16 \
    --num-workers 2 \
    --segments-per-file 3 \
    --val-fraction 0 \
    --checkpoint-every 1

echo "========================================="
echo "Tiny smoke test finished at $(date)"
echo "Output: $HOME/bendr_output/tiny_run/"
echo "========================================="
