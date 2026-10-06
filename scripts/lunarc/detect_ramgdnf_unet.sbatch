#!/bin/bash
# =====================================================================
# RAM_GDNF_2025 — Phase-1 frozen-model validation detection (U-Net)
# Runs the production cascade (U-Net seizure detector + convulsive
# classifier) over the RAM_GDNF cohort EDFs -> project SQLite DB.
# This is the ANALYSIS DB (precision/recall/burden), NOT training sidecars.
#   Submit:  sbatch detect_ramgdnf_unet.sbatch
# =====================================================================
#SBATCH -p lu48
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J ramgdnf_unet
#SBATCH -o logs/ramgdnf_unet_%j.out
#SBATCH -e logs/ramgdnf_unet_%j.err
#SBATCH -A lu2026-2-60
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- paths (EDIT if your layout differs) ----
: "${EDF_DIR:=/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025}"
: "${DB_PATH:=$HOME/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db}"
: "${METADATA_CSV:=$HOME/NED-Net/scripts/lunarc/RAM_GDNF_2025_batch_metadata.csv}"
# cohort EDFs ONLY -> excludes the 8 non-cohort files (483580 baseline + Hamza 2023):
: "${PATH_INCLUDE:=Batch_[1-4]_Recordings}"

# ---- frozen production operating point (DO NOT change for the Phase-1 validation) ----
# Every setting below is an env override so ONE script serves both the frozen
# validation and each retrained fold's evaluation. Keep the operating point fixed
# when comparing models -- only MODEL, DB_PATH and PATH_INCLUDE should change.
#   e.g. MODEL=ramgdnf_armB_holdB3 PATH_INCLUDE='Batch_3_Recordings' \
#        DB_PATH=~/.eeg_seizure_analyzer/projects/armB_holdB3_on_B3.db \
#        sbatch scripts/lunarc/detect_ramgdnf_unet.sbatch
: "${MODEL:=UNetv2_20260615}"
: "${CONV_MODEL:=Convulsive_v4LUNARC_20260616}"
: "${THRESHOLD:=0.5}"
: "${BOUNDARY_THRESHOLD:=0.1}"
: "${CONV_THRESHOLD:=0.45}"
: "${MIN_DURATION:=5}"
: "${MERGE_GAP:=2}"

module purge
module load Anaconda3/2024.06-1
cd "$HOME/NED-Net"
source config_conda.sh 2>/dev/null || true
conda activate bendr
mkdir -p logs "$(dirname "$DB_PATH")"

# sanity: how many cohort EDFs will be processed (expect 1377)
# no 2>/dev/null here: a missing EDF_DIR must be loud, not a silent zero
N=$(find "$EDF_DIR" -type f -iname '*.edf' | grep -Ec "$PATH_INCLUDE")
echo "cohort EDFs matched by PATH_INCLUDE='$PATH_INCLUDE': $N  (expected 1377)"
if [ "$N" -eq 0 ]; then
    echo "ERROR: no EDFs matched under $EDF_DIR" >&2
    echo "       check EDF_DIR and PATH_INCLUDE='$PATH_INCLUDE'" >&2
    exit 1
fi

WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "node=$(hostname) workers=$WORKERS  model=$MODEL  conv=$CONV_MODEL"
echo "  db=$DB_PATH  include=$PATH_INCLUDE  start=$(date)"

python scripts/lunarc/detect_batch.py \
    --edf-dir "$EDF_DIR" \
    --db-path "$DB_PATH" \
    --model "$MODEL" \
    --convulsive-model "$CONV_MODEL" \
    --metadata-csv "$METADATA_CSV" \
    --path-include "$PATH_INCLUDE" \
    --threshold "$THRESHOLD" \
    --boundary-threshold "$BOUNDARY_THRESHOLD" \
    --conv-threshold "$CONV_THRESHOLD" \
    --min-duration "$MIN_DURATION" \
    --merge-gap "$MERGE_GAP" \
    --workers "$WORKERS" \
    --tmpdir "${SNIC_TMP:-/tmp}"

echo "done $(date).  Copy $DB_PATH back and open in NED-Net."
