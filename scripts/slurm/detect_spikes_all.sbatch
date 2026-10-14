#!/bin/bash
# Batch classical interictal-spike detection (no model) into its own project database,
# on one CPU node. Resumable.
#
#   EDF_DIR=/path/to/edfs DB_PATH=~/.eeg_seizure_analyzer/projects/my_spikes.db \
#   sbatch scripts/slurm/detect_spikes_all.sbatch
#
# EDFs stored in mV: give --min-amplitude-uv in the file's units (see --help).

#SBATCH -J ned_spikes
#SBATCH -p cpu
#SBATCH -A my-allocation
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -o logs/ned_spikes_%j.out
#SBATCH -e logs/ned_spikes_%j.err

: "${EDF_DIR:?set EDF_DIR}"
: "${DB_PATH:?set DB_PATH}"
: "${ZSCORE:=7}"
: "${METADATA_CSV:=}"
: "${EXTRA_ARGS:=}"             # other detector options, see detect_spikes_batch.py --help
: "${WORKERS:=${SLURM_CPUS_ON_NODE:-8}}"

module purge
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nednet

cd "$HOME/NED-Net"
mkdir -p logs

OPT=(); [ -n "$METADATA_CSV" ] && OPT+=(--metadata-csv "$METADATA_CSV")
# shellcheck disable=SC2086
python scripts/slurm/detect_spikes_batch.py \
    --edf-dir "$EDF_DIR" --db-path "$DB_PATH" --zscore "$ZSCORE" \
    --workers "$WORKERS" --tmpdir "${TMPDIR:-/tmp}" "${OPT[@]}" $EXTRA_ARGS
