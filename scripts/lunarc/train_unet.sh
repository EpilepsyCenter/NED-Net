#!/bin/bash
# ============================================================
# LUNARC COSMOS — Supervised U-Net seizure-detection training
# ============================================================
# Trains the from-scratch U-Net on the converted annotations
# (*_ned_annotations.json sit next to each EDF).
#
# SELF-SUBMITTING + INTERACTIVE:
#   Run it directly on the login/desktop node and it asks for the
#   hyperparameters (Enter accepts the default), then submits the
#   SLURM job for you:
#       bash scripts/lunarc/train_unet.sh
#   It can't prompt from inside the batch job (compute nodes have no
#   terminal), so it prompts first, then sbatch's itself with your
#   answers passed as environment variables.
#
#   Non-interactive / scripted use still works — preset any of the
#   vars and they're used as-is (no prompt for those):
#       EPOCHS=80 LR=1e-4 sbatch scripts/lunarc/train_unet.sh
#
# Pick the neg/pos ratio first if unsure:
#   srun -p gpua100 -A lu2026-2-60 -t 00:15:00 --gres=gpu:1 --pty \
#     python -m eeg_seizure_analyzer.ml.train_unet \
#       --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data --analyze
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 14:00:00
# Size this from the window count, not from habit:
#   epochs x (train+val windows) / ~13 windows/s, then add ~35% margin.
# Overridable with WALL_TIME (the self-submit passes -t, which beats this directive).
# History: 02:00 was too short; 06:00 was too short (job 3825989 at NEG_POS_RATIO=10,
# 14,805 windows, ran 1,165 s/epoch and was killed at epoch 18 of 50 with event_f1
# still rising); 24:00 was over-insurance. At NEG_POS_RATIO=6 the dataset is ~9,400
# windows => ~740 s/epoch => ~10.3 h for 50 epochs, and patience normally stops sooner.
# NOTE on queueing: a longer request does NOT lower SLURM priority here. `sprio` on
# gpua100 (2026-10-09) showed QOS 60000 + FAIRSHARE 5534 dominating, with only AGE
# differing between jobs, and cutting 24 h -> 14 h on two pending jobs moved neither
# start estimate. The partition has 6 nodes; waiting is node contention, not wall clock.
# Size honestly anyway — it costs nothing and avoids holding a node longer than needed.
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J unet_train
#SBATCH -o logs/unet_train_%j.out
#SBATCH -e logs/unet_train_%j.err
#SBATCH -A lu2026-2-60                    # LUNARC compute allocation (SUPR: LU 2026/2-60)
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# ---- Defaults (used if the var isn't already set / left blank) ----
: "${MODEL_NAME:=unet_kaha_v1}"
: "${EPOCHS:=50}"
: "${BATCH_SIZE:=32}"   # A100 + fp16 has plenty of headroom; bigger batch keeps
#                         the tensor cores fed. Bump to 64 if memory allows.
: "${LR:=3e-4}"
: "${PATIENCE:=10}"
: "${NEG_POS_RATIO:=8}"
: "${EXCLUDE_ANIMALS:=}"   # space-separated animal IDs to drop, e.g. "355676"
: "${NEG_SOURCE:=hard}"   # hard | random.
#   `hard` uses `rejected` annotations as negatives. That is right for SV2A, whose
#   rejections were adjudicated for seizure-ness -- but WRONG for Mir's RAM_GDNF
#   `False` rows, which mean "not a convulsive/behavioural seizure" and were
#   confirmed by visual review (2026-10-07) to contain real non-convulsive activity,
#   including 180 s blocks of clear bursting. Used as hard negatives they train the
#   detector to suppress the events it should find.
#   `random` ignores every rejected label and samples background negatives instead:
#   no sidecar edits, so completed human reviews on disk are untouched.
: "${MAX_POSITIVE_SEC:=0}"   # 0 = no cap. Drops confirmed events LONGER than this
#   from the positive windows. Mir's long "convulsive" rows are chained seizures
#   annotated as one block (reviewer: "real convulsive seizures are never minutes
#   long"), and 0 of 13 events over 120 s were ever detected -- a 240 s block fills a
#   60 s window edge to edge, so per-channel z-scoring leaves no contrast to learn.
#   Capped events are still labelled inside other windows and still block background
#   sampling. Use 100 for RAM_GDNF.
: "${BG_AVOID_REJECTED:=0}"  # 1 = random background avoids `rejected` regions too.
#   Mir's `rejected` means "not a convulsive/behavioural seizure" and contains real
#   non-convulsive activity (2026-10-07), so background drawn from those regions
#   trains the detector to suppress real events -- the same reason those rows are
#   dropped as hard negatives by refresh_training_tree.py. Set 1 whenever the data
#   includes Mir's labels.
: "${HARD_NEG_EXCLUDE_METHODS:=}"  # space-separated detection_method values whose
#   `rejected` rows are NOT used as hard negatives but still block background (with
#   BG_AVOID_REJECTED=1). Use "mir_candidate" when training on Mir's labels: those
#   rows are wrong as negatives but right as regions to keep background out of.
#   This supersedes deleting them from the sidecars -- deletion destroyed the
#   region information, so the two settings could not both be honoured.
: "${VAL_MODE:=animal}"    # animal | recording | temporal — see --val-mode.
#   `animal` holds out whole animals (TRANSFER: a new, unannotated animal).
#   `recording`/`temporal` keep every animal on both sides (DEPLOYMENT: having annotated
#   some of this animal, does the model find its other seizures?) — the question a lab
#   actually faces, since in practice every animal gets some annotation. Those numbers are
#   optimistic relative to a new animal and are NOT comparable with `animal`-split runs.
: "${WALL_TIME:=14:00:00}"   # passed to sbatch as -t, overriding the directive above.
: "${STABLE_VAL_SPLIT:=1}"  # 1 = keep the dominant convulsive animals in train.
#   Positives are heavily concentrated (355675 alone carries ~1/3 of all convulsive
#   events), so a random split can land half of them in validation -- which wastes
#   training data and makes the result a lottery on the seed. Set to 0 only to
#   reproduce a run from before this flag existed.
# POS_WEIGHT left unset/blank => auto (train_unet sets it to NEG_POS_RATIO).

# ---- Guard: refuse to submit from a checkout that is behind origin ----
# Four runs were lost to one failure mode: an env var is set, the checked-out script
# predates the feature, the variable is silently ignored, and the job trains with the
# wrong config while looking fine (3825994 missed --conv-neg-from-rejected; 3831658/9
# missed VAL_MODE; 3795249 lost EXCLUDE_ANIMALS to a bare `read`). Checking for
# "unknown variables" cannot work -- a script cannot know a name it has never heard of.
# The invariant that actually covers every case is: the checkout must not be behind.
if [ -z "$SLURM_JOB_ID" ] && [ "${ALLOW_STALE:-0}" != "1" ]; then
    if git -C "$HOME/NED-Net" rev-parse --git-dir >/dev/null 2>&1; then
        git -C "$HOME/NED-Net" fetch -q origin 2>/dev/null || true
        _behind=$(git -C "$HOME/NED-Net" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
        if [ "${_behind:-0}" -gt 0 ]; then
            echo "ERROR: this checkout is $_behind commit(s) behind origin/main." >&2
            echo "       Any setting this copy does not know would be SILENTLY IGNORED" >&2
            echo "       and the job would train with the wrong configuration." >&2
            echo "       Fix:      cd \$HOME/NED-Net && git pull" >&2
            echo "       Override: ALLOW_STALE=1 (only to reproduce an older run)" >&2
            exit 1
        fi
    fi
fi

# ============================================================
# Phase 1: not under SLURM -> prompt, then submit this script.
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    ask() {  # ask VAR "prompt" "default"
        local cur; eval "cur=\${$1}"
        read -r -p "$2 [$cur]: " ans
        [ -n "$ans" ] && eval "$1=\"\$ans\""
    }
    echo "=== U-Net training — set hyperparameters (Enter = keep default) ==="
    ask MODEL_NAME    "Model name"        # default unet_kaha_v1
    ask EPOCHS        "Epochs"
    ask BATCH_SIZE    "Batch size"
    ask LR            "Learning rate"
    ask PATIENCE      "Patience"
    ask NEG_POS_RATIO "Neg/pos ratio"
    # Use ask() here too: a bare `read` sets the var to EMPTY on Enter, which
    # silently destroys a preset passed in the environment. That is how job
    # 3795249 was submitted with exclude=none despite EXCLUDE_ANIMALS being set,
    # which would have trained a leave-one-batch-out fold on its own test set.
    ask POS_WEIGHT      "Pos weight (Enter keeps current; unset = auto = neg/pos ratio)"
    ask EXCLUDE_ANIMALS "Exclude animal IDs (space-separated)"
    ask MAX_POSITIVE_SEC  "Max positive duration in s (0 = no cap)"
    ask BG_AVOID_REJECTED "Background avoids rejected regions (1/0)"
    ask HARD_NEG_EXCLUDE_METHODS "detection_methods NOT used as hard negatives"
    ask VAL_MODE                 "Split mode (animal | recording | temporal)"
    echo "-------------------------------------------------------------"
    echo "Submitting: model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
    echo "            patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=${POS_WEIGHT:-auto}"
    echo "            exclude=${EXCLUDE_ANIMALS:-none}"
    echo "            max_positive_sec=${MAX_POSITIVE_SEC} bg_avoid_rejected=${BG_AVOID_REJECTED}"
    echo "            hard_neg_exclude=${HARD_NEG_EXCLUDE_METHODS:-none}"
    echo "            wall time=$WALL_TIME val_mode=$VAL_MODE"
    # Pass settings via the (exported) environment + --export=ALL — robust for
    # values that contain spaces (e.g. multiple excluded IDs).
    export MODEL_NAME EPOCHS BATCH_SIZE LR PATIENCE NEG_POS_RATIO POS_WEIGHT \
           EXCLUDE_ANIMALS EDF_DIR STABLE_VAL_SPLIT NEG_SOURCE \
           MAX_POSITIVE_SEC BG_AVOID_REJECTED HARD_NEG_EXCLUDE_METHODS WALL_TIME VAL_MODE
    sbatch --export=ALL -t "$WALL_TIME" "$0"
    exit $?
fi

# ============================================================
# Phase 2: running under SLURM on a GPU node -> train.
# ============================================================
echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Settings:    model=$MODEL_NAME epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
echo "             patience=$PATIENCE neg/pos=$NEG_POS_RATIO pos_weight=${POS_WEIGHT:-auto}"
echo "             exclude=${EXCLUDE_ANIMALS:-none}"
echo "Data dir:    $EDF_DIR"
echo "Stable val split: $STABLE_VAL_SPLIT   neg-source: $NEG_SOURCE"
echo "max_positive_sec: $MAX_POSITIVE_SEC   bg_avoid_rejected: $BG_AVOID_REJECTED"
echo "hard_neg_exclude: ${HARD_NEG_EXCLUDE_METHODS:-none}   val_mode: $VAL_MODE"
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
# Overridable so one script serves both retraining arms:
#   arm A (SV2A + RAM_GDNF): EDF_DIR=/lunarc/nobackup/projects/lu2026-2-60
#   arm B (RAM_GDNF only):   EDF_DIR=/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025
# The scan is recursive and keyed on *_ned_annotations.json sidecars, so the
# data-dir choice IS the dataset definition -- check what lives under it first.
: "${EDF_DIR:=/lunarc/nobackup/projects/lu2026-2-60/edf_data}"

# Optional pos-weight: only pass the flag if the user set it (else train_unet
# auto-picks pos_weight = neg/pos ratio).
POS_WEIGHT_ARG=()
[ -n "$POS_WEIGHT" ] && POS_WEIGHT_ARG=(--pos-weight "$POS_WEIGHT")

# Space-separated IDs -> multiple --exclude-animals values (intentionally unquoted).
EXCLUDE_ARG=()
[ -n "$EXCLUDE_ANIMALS" ] && EXCLUDE_ARG=(--exclude-animals $EXCLUDE_ANIMALS)

STABLE_ARG=()
[ "$STABLE_VAL_SPLIT" = "1" ] && STABLE_ARG=(--stable-val-split)

BG_ARG=()
[ "$BG_AVOID_REJECTED" = "1" ] && BG_ARG=(--bg-avoid-rejected)

# Intentionally unquoted: space-separated values -> multiple argparse values.
HNX_ARG=()
[ -n "$HARD_NEG_EXCLUDE_METHODS" ] && HNX_ARG=(--hard-neg-exclude-method $HARD_NEG_EXCLUDE_METHODS)

python -m eeg_seizure_analyzer.ml.train_unet \
    --data-dir "$EDF_DIR" \
    --model-name "$MODEL_NAME" \
    --neg-source "$NEG_SOURCE" \
    --neg-pos-ratio "$NEG_POS_RATIO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    "${POS_WEIGHT_ARG[@]}" \
    "${EXCLUDE_ARG[@]}" \
    "${STABLE_ARG[@]}" \
    "${BG_ARG[@]}" \
    "${HNX_ARG[@]}" \
    --max-positive-sec "$MAX_POSITIVE_SEC" \
    --val-mode "$VAL_MODE" \
    --weight-decay 1e-4 \
    --base-filters 32 \
    --depth 4 \
    --dropout 0.2 \
    --num-workers 8

echo "========================================="
echo "Training finished at $(date)"
echo "Model saved under ~/.eeg_seizure_analyzer/models/$MODEL_NAME"
echo "========================================="
