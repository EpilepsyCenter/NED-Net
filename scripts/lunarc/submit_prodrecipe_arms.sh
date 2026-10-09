#!/bin/bash
# ============================================================
# Retrain every U-Net arm at the PRODUCTION RECIPE (2026-10-09)
# ============================================================
# UNetv2_20260615 was trained at lr 1e-3 / batch 8 / pos_weight 5 / fp32. Every
# retraining arm before 2026-10-09 used lr 3e-4 / batch 32 / bf16 (see RUN_LOG
# 2026-10-09 "DECISION"). This queues one rerun per arm. Each keeps its original
# data flags (copied from its job log header) and changes only the recipe.
# Model names get the suffix _pr.
#
# None of these is an exact repeat: the label trees have taken in review work
# since the originals ran. Only ramgdnf_temporal (3831662) vs ramgdnf_temporal_pr
# differs in the recipe alone.
#
# Run on the login node from ~/NED-Net:
#     bash scripts/lunarc/submit_prodrecipe_arms.sh            # checks, asks, submits all
#     bash scripts/lunarc/submit_prodrecipe_arms.sh --dry-run  # checks and prints only
#     ONLY="temporal A3" bash scripts/lunarc/submit_prodrecipe_arms.sh
# ============================================================
set -euo pipefail
cd "$HOME/NED-Net"

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
PROJ=/lunarc/nobackup/projects/lu2026-2-60
TREE=$HOME/train_nomirneg
HOLD_B3="459657 459658 459659 459660 459661 459662 459663 459664"
RETAINED_EXCL="449382 450093 450094 483550 483551 483552 483553 483554 483555 483557 483559"

# ---- 1. checkout must match origin (sbatch directly skips train_unet.sh's guard) ----
git fetch -q origin
_behind=$(git rev-list --count HEAD..origin/main)
if [ "$_behind" -gt 0 ]; then
    echo "ERROR: checkout is $_behind commit(s) behind origin/main. git pull first." >&2
    exit 1
fi
grep -q -- '--fp32' eeg_seizure_analyzer/ml/train_unet.py \
    || { echo "ERROR: train_unet.py has no --fp32; this checkout predates 22e54b2." >&2; exit 1; }
echo "checkout: $(git log --oneline -1)"

# ---- 2. what each data dir will actually scan ----
# Arms A/A2 scan the project ROOT, so anything with sidecars under it joins the
# dataset. ad_edf_data sits under the root too, so it must contain no sidecars.
echo
echo "== sidecars per top-level folder under $PROJ (arms A, A2 scan all of these) =="
find "$PROJ" -name '*_ned_annotations.json' 2>/dev/null \
    | sed "s|^$PROJ/||" | cut -d/ -f1 | sort | uniq -c
n_ad=$(find "$PROJ/ad_edf_data" -name '*_ned_annotations.json' 2>/dev/null | wc -l)
if [ "$n_ad" -gt 0 ]; then
    echo "ERROR: $n_ad sidecars under ad_edf_data would join arms A/A2. Resolve first." >&2
    exit 1
fi
echo "== $TREE: $(find -L "$TREE" -maxdepth 1 -name '*_ned_annotations.json' | wc -l) sidecars =="
echo
echo "== refresh_training_tree --dry-run (record the manual/adjudicated counts in RUN_LOG) =="
python scripts/lunarc/refresh_training_tree.py --dry-run --keep-mir-rejections 2>&1 | tail -15 \
    || echo "(dry-run failed; check the tree by hand before submitting)"

# ---- 3. the arms ----
# name | EDF_DIR | NEG_POS_RATIO | EXCLUDE | NEG_SOURCE | MAX_POS | BG_AVOID | HNX | VAL_MODE | WALL
# Sources: log headers of 3831662, 3831584 (= 3825989 at ratio 6), 3820377, 3800378,
# 3809460, 3801062. A3's original tree had Mir's rejections DELETED. The tree now keeps
# them, so A3 needs HNX=mir_candidate to stay "no Mir negatives". BG_AVOID stays 0, as
# in the original, where background could fall in those regions.
# Wall time: fp32/batch 8 is untimed on the A100. The originals ran 0.05-0.09 s/window
# per epoch at bf16. Sized for ~2x that over the full 50 epochs. Patience usually stops sooner,
# and metadata.json is now written every epoch, so a timeout still leaves a usable model.
ARMS=(
"temporal|ramgdnf_temporal_pr|$TREE|6|$RETAINED_EXCL|hard|100|1|mir_candidate|temporal|24:00:00"
"all_prod|ramgdnf_all_prod_pr|$TREE|6|$RETAINED_EXCL|hard|100|1|mir_candidate|animal|36:00:00"
"A3|ramgdnf_armA3_holdB3_pr|$TREE|10|$HOLD_B3|hard|0|0|mir_candidate|animal|48:00:00"
"A|ramgdnf_armA_holdB3_pr|$PROJ|10|$HOLD_B3|hard|0|0||animal|48:00:00"
"A2|ramgdnf_armA2_holdB3_randneg_pr|$PROJ|10|$HOLD_B3|random|0|0||animal|48:00:00"
"B|ramgdnf_armB_holdB3_stable_pr|$PROJ/RAM_GDNF_2025|10|$HOLD_B3|hard|0|0||animal|14:00:00"
)

echo
printf "%-9s %-34s %-5s %-6s %-4s %-3s %-13s %-9s %s\n" key model ratio neg cap bg hnx split wall
for a in "${ARMS[@]}"; do
    IFS='|' read -r key name dir ratio excl neg cap bg hnx split wall <<<"$a"
    if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $key "* ]]; then continue; fi
    printf "%-9s %-34s %-5s %-6s %-4s %-3s %-13s %-9s %s\n" \
        "$key" "$name" "$ratio" "$neg" "$cap" "$bg" "${hnx:--}" "$split" "$wall"
    printf "          dir=%s\n          exclude=%s\n" "$dir" "$excl"
done
echo "All: epochs 50, batch 8, lr 1e-3, pos_weight 5, FP32=1, patience 10, stable split 1"

[ "$DRY" = 1 ] && { echo; echo "--dry-run: nothing submitted."; exit 0; }
echo
read -r -p "Submit these jobs? [y/N] " ok
[ "$ok" = "y" ] || [ "$ok" = "Y" ] || { echo "Nothing submitted."; exit 0; }

mkdir -p logs
for a in "${ARMS[@]}"; do
    IFS='|' read -r key name dir ratio excl neg cap bg hnx split wall <<<"$a"
    if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $key "* ]]; then continue; fi
    # Subshell, every variable set explicitly: nothing leaks from one arm to the next.
    (
        export MODEL_NAME="$name" EDF_DIR="$dir" NEG_POS_RATIO="$ratio" \
               EXCLUDE_ANIMALS="$excl" NEG_SOURCE="$neg" MAX_POSITIVE_SEC="$cap" \
               BG_AVOID_REJECTED="$bg" HARD_NEG_EXCLUDE_METHODS="$hnx" VAL_MODE="$split" \
               EPOCHS=50 BATCH_SIZE=8 LR=1e-3 POS_WEIGHT=5 FP32=1 PATIENCE=10 \
               STABLE_VAL_SPLIT=1 WALL_TIME="$wall"
        jid=$(sbatch --parsable --export=ALL -t "$wall" scripts/lunarc/train_unet.sh)
        echo "$key -> $name : job $jid (wall $wall)"
    )
done
echo
echo "Check each header once it starts:  sed -n '/^Settings/,/^fp32/p' logs/unet_train_<jobid>.out"
