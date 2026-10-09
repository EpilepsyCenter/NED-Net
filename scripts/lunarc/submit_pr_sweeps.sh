#!/bin/bash
# ============================================================
# Operating-point sweeps for the production-recipe (_pr) U-Nets
# ============================================================
# Six-point Stage-1 grid (thr/bnd) per model, each a full lu48 detection over
# B1-B3: 0.5/0.1 0.7/0.3 0.8/0.5 0.9/0.3 0.9/0.5 0.95/0.5. It adds 0.95, because the
# old A3 winner sat on the top edge, and 0.9/0.3, so the boundary is no longer tied
# to the threshold. Stage 2 runs too, but its threshold is chosen afterwards from
# the stored convulsive_confidence (select_operating_point.py). CONV_THRESHOLD
# therefore stays at the script default and only sets the stored `type` label.
#
# All three models sweep B1-B3:
#   temporal, all_prod -- trained on B1-B3, so the winning sweep DB IS the
#                         in-sample detection run.
#   A3                 -- select on held-in B1+B2, then read held-out B3 from the SAME DB
#                         at the winning point (select_operating_point.py
#                         --select-batches 12 --report-batches 3). No second round-trip.
#
# DB names are new on purpose: detect_batch.py skips files already in a DB, so a
# reused DB_PATH (or the default, the frozen ram_gdnf_unet_v0.db) silently does nothing.
#
#   bash scripts/lunarc/submit_pr_sweeps.sh --dry-run
#   bash scripts/lunarc/submit_pr_sweeps.sh            # only models whose training finished
#   DEPEND=1 bash scripts/lunarc/submit_pr_sweeps.sh   # queue now, start after training (afterany)
#   ONLY="temporal" bash scripts/lunarc/submit_pr_sweeps.sh
# ============================================================
set -euo pipefail
cd "$HOME/NED-Net"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
PROJDB="$HOME/.eeg_seizure_analyzer/projects"
MODELS="$HOME/.eeg_seizure_analyzer/models"
GRID="0.5/0.1 0.7/0.3 0.8/0.5 0.9/0.3 0.9/0.5 0.95/0.5"

git fetch -q origin
_behind=$(git rev-list --count HEAD..origin/main)
[ "$_behind" -eq 0 ] || { echo "ERROR: $_behind commit(s) behind origin/main; git pull" >&2; exit 1; }
echo "checkout: $(git log --oneline -1)"

# key | model | conv model | training job
ARMS=(
"temporal|ramgdnf_temporal_pr|conv_temporal|3832977"
"all_prod|ramgdnf_all_prod_pr|conv_temporal|3832978"
"A3|ramgdnf_armA3_holdB3_pr|conv_armA_holdB3|3832979"
)

n=0
for a in "${ARMS[@]}"; do
    IFS='|' read -r key model conv tjob <<<"$a"
    if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $key "* ]]; then continue; fi
    [ -d "$MODELS/$conv" ] || { echo "ERROR: Stage-2 model $conv missing" >&2; exit 1; }
    dep=()
    if [ "${DEPEND:-0}" = "1" ]; then
        # afterany, not afterok: a wall-clock kill still leaves a usable model, since
        # metadata.json is written every epoch.
        dep=(--dependency=afterany:$tjob)
        state="queued after training job $tjob"
    else
        done_=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('training_complete',''))" \
                "$MODELS/$model/metadata.json" 2>/dev/null || echo missing)
        if [ "$done_" != "True" ]; then
            echo "skip $key: $model training_complete=$done_ (job $tjob). Rerun later, or DEPEND=1."
            continue
        fi
        state="model ready"
    fi
    for p in $GRID; do
        thr=${p%/*}; bnd=${p#*/}
        db="$PROJDB/${key}_pr_sweep_t${thr}_b${bnd}.db"
        if [ -e "$db" ]; then echo "exists, skipped: $db"; continue; fi
        echo "$key  $model + $conv  thr $thr / bnd $bnd  -> $(basename "$db")  [$state]"
        n=$((n+1))
        [ "$DRY" = 1 ] && continue
        (
            export MODEL="$model" CONV_MODEL="$conv" THRESHOLD="$thr" \
                   BOUNDARY_THRESHOLD="$bnd" PATH_INCLUDE='Batch_[123]_Recordings' \
                   DB_PATH="$db"
            jid=$(sbatch --parsable --export=ALL ${dep[@]+"${dep[@]}"} -J "sw_${key}_${thr}_${bnd}" \
                  scripts/lunarc/detect_ramgdnf_unet.sbatch)
            echo "    job $jid"
        )
    done
done
echo
[ "$DRY" = 1 ] && echo "--dry-run: $n job(s) would be submitted." || echo "$n job(s) submitted."
echo "Record the job IDs in RUN_LOG. When done, copy the DBs to the Mac and run"
echo "scripts/local/select_operating_point.py (see NEXT_STEPS)."
