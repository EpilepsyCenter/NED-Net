#!/bin/bash
# ============================================================
# AD animals (KAHA recordings) — classical interictal-spike detection
# ============================================================
# Thin preset over scripts/lunarc/detect_spikes_all.sh for the AD cohort that
# scripts/local/sync_ad_edf_to_lunarc.sh pushes to ad_edf_data/.
#
# It only overrides the three things that are cohort-specific:
#   EDF_DIR       -> ad_edf_data (NOT edf_data, which is SV2A)
#   DB_PATH       -> a fresh, dedicated ad_spikes.db
#   METADATA_CSV  -> ad_metadata.csv, NOT the SV2A batch_metadata.csv. This is
#                    required, not cosmetic: the detector only runs on channels
#                    carrying an animal ID, and the AD headers are generic
#                    ('Ch1 Biopot'...), so without it every file errors with
#                    "No animal-ID-mapped EEG channels". Regenerate the CSV with
#                    make_ad_metadata_csv.py if the IDs or genotypes change.
#   PATH_INCLUDE  -> blank (all files); the SV2A default 'Week[123]-' would
#                    match ZERO AD paths, which are 'Week_1/W1_D1/...'
#
# Detector parameters are left at the detect_spikes_all.sh defaults, i.e. the
# same operating point used for the SV2A spike run:
#   z=4.0, bandpass 3-50 Hz, prominence 6x baseline, refractory 750 ms,
#   isolation 1 neighbour / 2 s, then conf>=0.7, SNR>=10, amp/baseline>=15.
#
# Run on the LUNARC login node; it prompts (Enter = keep the value shown) and
# then sbatch's itself onto the lu48 CPU partition:
#     cd ~/NED-Net && bash scripts/lunarc/detect_spikes_ad.sh
#
# When it finishes, copy the DB back to your Mac and open it in NED-Net:
#     scp cosmos:~/.eeg_seizure_analyzer/projects/ad_spikes.db \
#         ~/.eeg_seizure_analyzer/projects/
# ============================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EDF_DIR="${EDF_DIR:-/lunarc/nobackup/projects/lu2026-2-60/ad_edf_data}"
export DB_PATH="${DB_PATH:-$HOME/.eeg_seizure_analyzer/projects/ad_spikes.db}"
export METADATA_CSV="${METADATA_CSV:-$HERE/ad_metadata.csv}"   # NOT the SV2A one
export PATH_INCLUDE=""     # all files

if [ ! -f "$METADATA_CSV" ]; then
  echo "!! Missing $METADATA_CSV — every file would fail with"
  echo "   'No animal-ID-mapped EEG channels'. Generate it with:"
  echo "     python scripts/lunarc/make_ad_metadata_csv.py --edf-dir \"$EDF_DIR\" \\"
  echo "         --out \"$METADATA_CSV\""
  exit 1
fi

if [ -e "$DB_PATH" ]; then
  echo "!! $DB_PATH already exists."
  echo "   The batch merge is destructive per file. Move it aside, or set"
  echo "   DB_PATH=<other path> (and OVERWRITE=1 to re-detect files already in it)."
  read -r -p "   Continue anyway? (y/N): " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
fi

exec bash "$HERE/detect_spikes_all.sh"
