#!/bin/bash
# =====================================================================
# Per-channel signal-quality sweep over the RAM_GDNF cohort.
# Model-independent basis for excluding unusable recordings -- see
# docs/methods_paper/EXCLUSION_CRITERIA.md.
#   Submit from ~/NED-Net:  sbatch scripts/lunarc/signal_quality.sbatch
# Reads EDFs only; writes ONE csv to $HOME. No project-storage inodes.
# =====================================================================
#SBATCH -p lu48
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -J sigqual
#SBATCH -o logs/sigqual_%j.out
#SBATCH -e logs/sigqual_%j.err
#SBATCH -A lu2026-2-60
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

EDF_DIR="/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025"
OUT="$HOME/signal_quality_ramgdnf.csv"
PATH_INCLUDE="Batch_[1-4]_Recordings"

module purge
module load Anaconda3/2024.06-1
cd "$HOME/NED-Net"
source config_conda.sh 2>/dev/null || true
conda activate bendr
mkdir -p logs

N=$(find "$EDF_DIR" -type f -iname '*.edf' | grep -Ec "$PATH_INCLUDE")
echo "cohort EDFs matched: $N (expected 1377)"
if [ "$N" -eq 0 ]; then echo "ERROR: no EDFs under $EDF_DIR" >&2; exit 1; fi

python scripts/lunarc/signal_quality.py \
    --edf-dir "$EDF_DIR" \
    --path-include "$PATH_INCLUDE" \
    --out "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-24}"

echo "done $(date). Copy $OUT back and analyse the distribution before setting a threshold."
