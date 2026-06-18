#!/bin/bash
# ============================================================
# LUNARC COSMOS — Full-dataset CLASSICAL spike detection -> DB
# ============================================================
# Runs the rule-based interictal-spike detector (NO trained model) over
# every EDF in the project storage, IN PARALLEL across one CPU node's
# cores, writing spike events to a project SQLite database (not sidecars).
# Copy that DB back to your Mac and open it in NED-Net (Results -> Spikes).
#
# CPU partition (lu48), NOT GPU: classical detection is CPU/I-O-bound, so
# this spends 0 GPU-hours. One 48-core node chews through the whole
# dataset in parallel.
#
# SELF-SUBMITTING + INTERACTIVE: run it on the login node, answer the
# prompts (Enter = default), and it sbatch's itself.
#     bash scripts/lunarc/detect_spikes_all.sh
#
# Write to a DEDICATED spikes DB (one detector family per DB). The merge
# is destructive per file, so don't point this at a seizure DB unless you
# mean to overwrite those files.
#
# Subsample days with PATH_INCLUDE (a regex over the full path): e.g.
# 'Week[123]-' for the first 3 weeks, or a day pattern to thin the timeline.
# ============================================================

#SBATCH -p lu48
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J ned_spikes
#SBATCH -o logs/ned_spikes_%j.out
#SBATCH -e logs/ned_spikes_%j.err
#SBATCH -A lu2026-2-60
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- Defaults (used if the var isn't already set / left blank) ----
: "${EDF_DIR:=/lunarc/nobackup/projects/lu2026-2-60/edf_data}"
: "${DB_PATH:=$HOME/.eeg_seizure_analyzer/projects/sv2a_spikes.db}"
: "${METADATA_CSV:=$HOME/NED-Net/scripts/lunarc/batch_metadata.csv}"  # full SV2A run
: "${PATH_INCLUDE:=Week[123]-}"   # only first 3 weeks; blank = all files
: "${ZSCORE:=4.0}"                # amplitude threshold = mean + z*std (GUI z=4-5)
: "${BANDPASS_LOW:=10.0}"
: "${BANDPASS_HIGH:=70.0}"
: "${PROMINENCE:=1.5}"
: "${ISO_WINDOW:=2.0}"            # burst-rejection window (s)
: "${ISO_MAX:=6}"                 # max neighbours before a spike is rejected
: "${OVERWRITE:=0}"              # 1 = re-detect files already in the DB

# ============================================================
# Phase 1: not under SLURM -> prompt, then submit this script.
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    ask() {  # ask VAR "prompt"
        local cur; eval "cur=\${$1}"
        read -r -p "$2 [$cur]: " ans
        [ -n "$ans" ] && eval "$1=\"\$ans\""
    }
    echo "=== Classical spike detection — settings (Enter = keep default) ==="
    ask EDF_DIR      "EDF folder (scanned recursively)"
    ask DB_PATH      "Output project DB path (use a dedicated spikes DB)"
    ask METADATA_CSV "Batch metadata CSV (blank = none)"
    ask PATH_INCLUDE "Path-include regex (blank = all files)"
    ask ZSCORE       "Amplitude z-score threshold"
    ask PROMINENCE   "Prominence x baseline"
    ask ISO_MAX      "Isolation max neighbours (burst rejection)"
    read -r -p "Re-detect files already in the DB? (y/N): " ow
    [ "$ow" = "y" ] || [ "$ow" = "Y" ] && OVERWRITE=1
    echo "-------------------------------------------------------------"
    echo "Submitting: edf=$EDF_DIR"
    echo "            db=$DB_PATH"
    echo "            meta=${METADATA_CSV:-none}  path_include=${PATH_INCLUDE:-all}"
    echo "            z=$ZSCORE prom=$PROMINENCE iso_max=$ISO_MAX overwrite=$OVERWRITE"
    export EDF_DIR DB_PATH METADATA_CSV PATH_INCLUDE ZSCORE BANDPASS_LOW \
           BANDPASS_HIGH PROMINENCE ISO_WINDOW ISO_MAX OVERWRITE
    sbatch --export=ALL "$0"
    exit $?
fi

# ============================================================
# Phase 2: running under SLURM on a CPU node -> detect.
# ============================================================
echo "========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)   cores=$(nproc)"
echo "Start time: $(date)"
echo "EDF dir:    $EDF_DIR"
echo "DB:         $DB_PATH"
echo "Meta:       ${METADATA_CSV:-none}   path_include=${PATH_INCLUDE:-all}"
echo "Detector:   z=$ZSCORE bp=$BANDPASS_LOW-$BANDPASS_HIGH prom=$PROMINENCE iso=$ISO_MAX/$ISO_WINDOW"
echo "========================================="

module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

cd $HOME/NED-Net
mkdir -p logs "$(dirname "$DB_PATH")"

# Optional args only when set.
META_ARG=();      [ -n "$METADATA_CSV" ] && META_ARG=(--metadata-csv "$METADATA_CSV")
PATH_ARG=();      [ -n "$PATH_INCLUDE" ] && PATH_ARG=(--path-include "$PATH_INCLUDE")
OVERWRITE_ARG=(); [ "$OVERWRITE" = "1" ] && OVERWRITE_ARG=(--overwrite)

python scripts/lunarc/detect_spikes_batch.py \
    --edf-dir "$EDF_DIR" \
    --db-path "$DB_PATH" \
    --zscore "$ZSCORE" \
    --bandpass-low "$BANDPASS_LOW" \
    --bandpass-high "$BANDPASS_HIGH" \
    --prominence "$PROMINENCE" \
    --isolation-window-sec "$ISO_WINDOW" \
    --isolation-max-neighbours "$ISO_MAX" \
    --workers "$(nproc)" \
    --tmpdir "${SNIC_TMP:-/tmp}" \
    "${META_ARG[@]}" "${PATH_ARG[@]}" "${OVERWRITE_ARG[@]}"

echo "========================================="
echo "Detection finished at $(date)"
echo "Results DB: $DB_PATH   (copy it back to your Mac and open in NED-Net)"
echo "========================================="
