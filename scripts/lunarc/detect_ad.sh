#!/bin/bash
# ============================================================
# AD animals (KAHA recordings) — seizure detection (U-Net + convulsive)
# ============================================================
# Thin preset over scripts/lunarc/detect_all.sh for the AD cohort, using the
# SAME final production workflow as the SV2A run (decided 2026-06-26):
#   U-Net UNetv2_20260615 @ 0.5 + hysteresis boundary 0.1
#   + cascade convulsive classifier Convulsive_v4LUNARC_20260616 @ 0.45
#   NO re-ranker (shelved — didn't generalise out-of-sample)
# Those all come from detect_all.sh's defaults; nothing about the model or the
# thresholds is overridden here, so the two cohorts stay comparable.
#
# Only the cohort-specific bits are set:
#   EDF_DIR       -> ad_edf_data (NOT edf_data, which is SV2A)
#   DB_PATH       -> a fresh, dedicated ad_seizures.db
#   METADATA_CSV  -> ad_metadata.csv (animals 1-4 on ch0/ch3/ch5/ch6, 5xFAD)
#   PATH_INCLUDE  -> blank (all files); the SV2A default 'Week[123456]-' would
#                    match ZERO AD paths, which are 'Week_1/W1_D1/...'
#
# Expected outcome: few or no events. These are 5xFAD control animals and no
# seizures are anticipated — this run is the check, not a measurement.
#
# NOTE: unlike spike detection, the seizure path detects on ALL EEG channels,
# not just animal-mapped ones. The four unused Biopot channels (Marco's
# Ch2/Ch3/Ch5/Ch8) can therefore contribute events with a BLANK animal_id.
# Filter on animal 1-4 in Results; a pile of blank-animal events means one of
# the unused channels is floating rather than flat.
#
# Run on the LUNARC login node; it prompts (Enter = keep the value shown) and
# then sbatch's itself onto the lu48 CPU partition:
#     cd ~/NED-Net && bash scripts/lunarc/detect_ad.sh
#
# With lu48 full (`sinfo -p lu48` showing 'mix' but no 'idle'), answer the Cores
# prompt with a number (e.g. 24) instead of leaving it blank — an exclusive
# whole-node request waits for a node to drain completely.
#
# When it finishes:
#     scp cosmos:~/.eeg_seizure_analyzer/projects/ad_seizures.db \
#         ~/.eeg_seizure_analyzer/projects/
# ============================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EDF_DIR="${EDF_DIR:-/lunarc/nobackup/projects/lu2026-2-60/ad_edf_data}"
export DB_PATH="${DB_PATH:-$HOME/.eeg_seizure_analyzer/projects/ad_seizures.db}"
export METADATA_CSV="${METADATA_CSV:-$HERE/ad_metadata.csv}"   # NOT the SV2A one
export PATH_INCLUDE=""     # all files

if [ ! -f "$METADATA_CSV" ]; then
  echo "!! Missing $METADATA_CSV — events would land with no animal attribution."
  echo "   Generate it with:"
  echo "     python scripts/lunarc/make_ad_metadata_csv.py --edf-dir \"$EDF_DIR\" \\"
  echo "         --out \"$METADATA_CSV\""
  exit 1
fi

if [ -e "$DB_PATH" ]; then
  echo "!! $DB_PATH already exists."
  echo "   The batch merge is destructive per file. Move it aside, or set"
  echo "   DB_PATH=<other path> (and answer y to re-detect to redo the files in it)."
  read -r -p "   Continue anyway? (y/N): " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
fi

exec bash "$HERE/detect_all.sh"
