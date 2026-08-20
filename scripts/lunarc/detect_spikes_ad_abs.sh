#!/bin/bash
# ============================================================
# AD animals — spike detection with an ABSOLUTE amplitude criterion
# ============================================================
# CONTROL RUN for the Week1->Week3 spike-rate increase. It is not a better
# detector and its counts are NOT comparable to ad_spikes.db — the only
# question it answers is whether the WEEK-OVER-WEEK TREND survives when the
# detection criterion cannot drift with the background.
#
# Why: background amplitude falls in all four animals over the three weeks
# (17.5->15.1, 36.1->16.1, 30.9->18.6, 23.8->21.3 uV; Spearman -0.70 to -0.99,
# all p<0.001). The production detector is baseline-relative in FOUR places:
#     threshold      = baseline_mean + 4*baseline_std      (spike.py:72)
#     min_prominence = prominence_x_baseline * baseline    (spike.py:189)
#     min_local_snr, min_amplitude_x_baseline              (post-filters)
# A falling background lowers all four, so counts can rise with no change in
# activity. This run neutralises every one of them:
#
#   MIN_AMPLITUDE=0.15   absolute floor; threshold = max(relative, this), and
#                        0.15 mV = 150 uV exceeds the highest per-animal
#                        relative threshold observed (148.5 uV, animal 2 W1),
#                        so the absolute value binds in every week.
#   PROMINENCE=1.0       still baseline-relative, but 1.0 * baseline is at most
#                        ~36 uV here — far below the 150 uV height floor, so it
#                        no longer binds. (Production uses 6.0.)
#   MIN_CONF/SNR/XBL=0   all three are relative; off.
#
# THE UNITS TRAP: --min-amplitude-uv is applied to the data in its own physical
# units, and these EDFs are in mV. So 150 uV is written 0.15, NOT 150. Passing
# 150 would mean 150 mV and detect precisely nothing.
#
# Reading the result: compare the WEEK RATIOS against ad_spikes.db via
#     python scripts/local/ad_spike_trend.py --db ~/.eeg_seizure_analyzer/projects/ad_spikes_abs.db
# Trend survives  -> the increase is real activity, not a moving threshold.
# Trend vanishes  -> it was largely the falling background; do not report the
#                    rise without correcting for it.
#
#     cd ~/NED-Net && bash scripts/lunarc/detect_spikes_ad_abs.sh
# ============================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EDF_DIR="${EDF_DIR:-/lunarc/nobackup/projects/lu2026-2-60/ad_edf_data}"
export DB_PATH="${DB_PATH:-$HOME/.eeg_seizure_analyzer/projects/ad_spikes_abs.db}"
export METADATA_CSV="${METADATA_CSV:-$HERE/ad_metadata.csv}"
export PATH_INCLUDE=""

# The absolute criterion (see above). In mV, because that is what these EDFs are.
export MIN_AMPLITUDE="${MIN_AMPLITUDE:-0.15}"
export PROMINENCE="${PROMINENCE:-1.0}"
# Every post-filter is baseline-relative — off, or the confound comes back in.
export MIN_CONF="${MIN_CONF:-0}"
export MIN_SNR="${MIN_SNR:-0}"
export MIN_XBL="${MIN_XBL:-0}"

if [ ! -f "$METADATA_CSV" ]; then
  echo "!! Missing $METADATA_CSV — every file would fail with"
  echo "   'No animal-ID-mapped EEG channels'."
  exit 1
fi

case "$MIN_AMPLITUDE" in
  [1-9]*) echo "!! MIN_AMPLITUDE=$MIN_AMPLITUDE looks like microvolts."
          echo "   These EDFs are in mV: 150 uV must be written 0.15."
          read -r -p "   Continue anyway? (y/N): " a
          [ "$a" = "y" ] || [ "$a" = "Y" ] || exit 1 ;;
esac

if [ -e "$DB_PATH" ]; then
  echo "!! $DB_PATH already exists — move it aside or set DB_PATH=<other>."
  read -r -p "   Continue anyway? (y/N): " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
fi

echo "=== CONTROL RUN: absolute amplitude floor ${MIN_AMPLITUDE} mV "
echo "=== (counts are NOT comparable to ad_spikes.db — only the trend is)"
exec bash "$HERE/detect_spikes_all.sh"
