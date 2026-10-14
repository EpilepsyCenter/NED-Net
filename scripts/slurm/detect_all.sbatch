#!/bin/bash
# Batch seizure detection over a folder of EDFs into a project database, on one CPU
# node (inference is CPU/IO-bound; a GPU is not needed). Resumable: files already in
# the database are skipped.
#
#   EDF_DIR=/path/to/edfs DB_PATH=~/.eeg_seizure_analyzer/projects/my_project.db \
#   MODEL=my_unet CONV_MODEL=my_conv sbatch scripts/slurm/detect_all.sbatch
#
# Copy the database back to ~/.eeg_seizure_analyzer/projects/ to open it in Results.

#SBATCH -J ned_detect
#SBATCH -p cpu                  # your CPU partition
#SBATCH -A my-allocation
#SBATCH -N 1
#SBATCH --exclusive             # a whole node; or --cpus-per-task=N with WORKERS=N
#SBATCH -t 1-00:00:00
#SBATCH -o logs/ned_detect_%j.out
#SBATCH -e logs/ned_detect_%j.err

: "${EDF_DIR:?set EDF_DIR}"
: "${DB_PATH:?set DB_PATH}"
: "${MODEL:?set MODEL to a trained U-Net name}"
: "${CONV_MODEL:=}"             # blank = the U-Net's own convulsive output
: "${THRESHOLD:=0.5}"
: "${BOUNDARY_THRESHOLD:=0.1}"  # hysteresis; >= THRESHOLD turns it off
: "${CONV_THRESHOLD:=0.5}"
: "${CONV_MIN_DURATION:=0}"     # shorter events are never convulsive; 0 = off
: "${METADATA_CSV:=}"           # optional: filename -> cohort / group / animal per channel
: "${WORKERS:=${SLURM_CPUS_ON_NODE:-8}}"

module purge
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nednet

cd "$HOME/NED-Net"
mkdir -p logs

OPT=()
[ -n "$CONV_MODEL" ] && OPT+=(--convulsive-model "$CONV_MODEL")
[ -n "$METADATA_CSV" ] && OPT+=(--metadata-csv "$METADATA_CSV")
python scripts/slurm/detect_batch.py \
    --edf-dir "$EDF_DIR" --db-path "$DB_PATH" --model "$MODEL" \
    --threshold "$THRESHOLD" --boundary-threshold "$BOUNDARY_THRESHOLD" \
    --conv-threshold "$CONV_THRESHOLD" --conv-min-duration "$CONV_MIN_DURATION" \
    --workers "$WORKERS" --tmpdir "${TMPDIR:-/tmp}" "${OPT[@]}"
