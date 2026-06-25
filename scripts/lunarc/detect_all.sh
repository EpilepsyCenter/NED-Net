#!/bin/bash
# ============================================================
# LUNARC COSMOS — Full-dataset batch seizure detection -> DB
# ============================================================
# Runs the cascade (U-Net seizure detector + convulsive classifier)
# over every EDF in the project storage, IN PARALLEL across one CPU
# node's cores, writing results to a project SQLite database (not
# sidecars). Copy that DB back to your Mac and open it in NED-Net.
#
# CPU partition (lu48), NOT GPU: detection is CPU/I-O-bound, so this
# spends 0 GPU-hours. One 48-core node chews through the whole dataset
# in parallel; bump to an array later only if you need it faster.
#
# SELF-SUBMITTING + INTERACTIVE: run it on the login node, answer the
# prompts (Enter = default), and it sbatch's itself.
#     bash scripts/lunarc/detect_all.sh
#
# Metadata CSV (optional): one row per EDF with columns filename, cohort,
# group_id, animal_ch0..7, cohort_ch0..7, group_ch0..7 — same format the
# Analysis tab's "metadata browse" reads (eeg_seizure_analyzer/io/
# batch_metadata.py). Sidecar *_ned_channels.json / *_ned_meta.json next
# to an EDF take precedence over the CSV.
# ============================================================

#SBATCH -p lu48
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH -J ned_detect
#SBATCH -o logs/ned_detect_%j.out
#SBATCH -e logs/ned_detect_%j.err
#SBATCH -A lu2026-2-60
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- Defaults (used if the var isn't already set / left blank) ----
: "${EDF_DIR:=/lunarc/nobackup/projects/lu2026-2-60/edf_data}"
# New-workflow run writes to a *_v2 DB so the original wk1-3 run is kept for
# side-by-side comparison.
: "${DB_PATH:=$HOME/.eeg_seizure_analyzer/projects/lunarc_detect_wk1-3_v2.db}"
: "${MODEL:=UNetv2_20260615}"
: "${CONV_MODEL:=Convulsive_v4LUNARC_20260616}"   # blank = use detector ch1 instead of cascade
: "${METADATA_CSV:=$HOME/NED-Net/scripts/lunarc/batch_metadata.csv}"   # full SV2A run, 1893 EDFs
: "${PATH_INCLUDE:=Week[123]-}"   # only first 3 weeks (WeekN-DayNN subfolders); blank = all
# New workflow: lower detection threshold for recall, then hysteresis grows the
# boundaries back out and the re-ranker filters the extra false positives.
# (Old full run used THRESHOLD=0.7, no boundary, no re-ranker.)
: "${THRESHOLD:=0.5}"             # detection core; was 0.7 (F1-opt). Lower = more sensitive
: "${BOUNDARY_THRESHOLD:=0.3}"    # hysteresis: grow onset/offset out to this. blank/>=THRESHOLD = off
: "${RERANKER_MODEL:=Re-rankerv2_20260625}"   # event re-ranker (AUC 0.97); blank = none. NEEDS scikit-learn+joblib in the env
: "${CONV_THRESHOLD:=0.45}"       # trained operating point (best F1 0.8848 @ 0.45, job 3286330)
: "${MIN_DURATION:=5}"
: "${MERGE_GAP:=2}"
: "${OVERWRITE:=0}"               # 1 = re-detect files already in the DB

# ============================================================
# Phase 1: not under SLURM -> prompt, then submit this script.
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    ask() {  # ask VAR "prompt"
        local cur; eval "cur=\${$1}"
        read -r -p "$2 [$cur]: " ans
        [ -n "$ans" ] && eval "$1=\"\$ans\""
    }
    echo "=== Batch detection — settings (Enter = keep default) ==="
    ask EDF_DIR        "EDF folder (scanned recursively)"
    ask DB_PATH        "Output project DB path"
    ask MODEL          "Seizure detector model"
    ask CONV_MODEL     "Convulsive classifier model (blank = none)"
    ask METADATA_CSV   "Batch metadata CSV (blank = none)"
    ask PATH_INCLUDE   "Path-include regex (blank = all files)"
    ask THRESHOLD      "Seizure detection threshold"
    ask BOUNDARY_THRESHOLD "Boundary (hysteresis) threshold (blank/>=det = off)"
    ask RERANKER_MODEL "Event re-ranker model (blank = none)"
    ask CONV_THRESHOLD "Convulsive threshold"
    read -r -p "Re-detect files already in the DB? (y/N): " ow
    [ "$ow" = "y" ] || [ "$ow" = "Y" ] && OVERWRITE=1
    echo "-------------------------------------------------------------"
    echo "Submitting: edf=$EDF_DIR"
    echo "            db=$DB_PATH"
    echo "            model=$MODEL conv=${CONV_MODEL:-none} meta=${METADATA_CSV:-none}"
    echo "            path_include=${PATH_INCLUDE:-all}"
    echo "            thr=$THRESHOLD boundary=${BOUNDARY_THRESHOLD:-off} reranker=${RERANKER_MODEL:-none}"
    echo "            conv_thr=$CONV_THRESHOLD overwrite=$OVERWRITE"
    export EDF_DIR DB_PATH MODEL CONV_MODEL METADATA_CSV PATH_INCLUDE THRESHOLD \
           BOUNDARY_THRESHOLD RERANKER_MODEL CONV_THRESHOLD MIN_DURATION MERGE_GAP OVERWRITE
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
echo "Model:      $MODEL   conv=${CONV_MODEL:-none}   meta=${METADATA_CSV:-none}"
echo "Filter:     path_include=${PATH_INCLUDE:-all}   thr=$THRESHOLD   conv_thr=$CONV_THRESHOLD"
echo "Workflow:   boundary=${BOUNDARY_THRESHOLD:-off}   reranker=${RERANKER_MODEL:-none}"
echo "========================================="

module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

cd $HOME/NED-Net
mkdir -p logs "$(dirname "$DB_PATH")"

# Optional args only when set.
CONV_ARG=();     [ -n "$CONV_MODEL" ]   && CONV_ARG=(--convulsive-model "$CONV_MODEL")
META_ARG=();     [ -n "$METADATA_CSV" ] && META_ARG=(--metadata-csv "$METADATA_CSV")
PATH_ARG=();     [ -n "$PATH_INCLUDE" ] && PATH_ARG=(--path-include "$PATH_INCLUDE")
OVERWRITE_ARG=(); [ "$OVERWRITE" = "1" ] && OVERWRITE_ARG=(--overwrite)
BND_ARG=();      [ -n "$BOUNDARY_THRESHOLD" ] && BND_ARG=(--boundary-threshold "$BOUNDARY_THRESHOLD")
RERANK_ARG=();   [ -n "$RERANKER_MODEL" ]     && RERANK_ARG=(--reranker-model "$RERANKER_MODEL")

python scripts/lunarc/detect_batch.py \
    --edf-dir "$EDF_DIR" \
    --db-path "$DB_PATH" \
    --model "$MODEL" \
    --threshold "$THRESHOLD" \
    --conv-threshold "$CONV_THRESHOLD" \
    --min-duration "$MIN_DURATION" \
    --merge-gap "$MERGE_GAP" \
    --workers "$(nproc)" \
    --tmpdir "${SNIC_TMP:-/tmp}" \
    "${CONV_ARG[@]}" "${META_ARG[@]}" "${PATH_ARG[@]}" "${OVERWRITE_ARG[@]}" \
    "${BND_ARG[@]}" "${RERANK_ARG[@]}"

echo "========================================="
echo "Detection finished at $(date)"
echo "Results DB: $DB_PATH   (copy it back to your Mac and open in NED-Net)"
echo "========================================="
