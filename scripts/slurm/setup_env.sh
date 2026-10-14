#!/bin/bash
# One-time setup of a conda environment for NED-Net on a SLURM cluster.
# Adjust the module name to your cluster (`module avail` lists them).
#
#   cd ~/NED-Net && bash scripts/slurm/setup_env.sh
set -euo pipefail

module purge
module load Anaconda3            # or Miniforge3 / your site's conda module
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n nednet python=3.11
conda activate nednet

# PyTorch with CUDA: pick the CUDA version your GPU nodes support.
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo "Activate in job scripts with: conda activate nednet"
