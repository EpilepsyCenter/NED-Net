#!/usr/bin/env bash
# ============================================================
# Push EDF data from Arrhenius -> LUNARC, fixing the layout mismatch
# ============================================================
# On Arrhenius the cohorts live under subfolders, e.g.
#     edf_data/SV2A_2024/<batch>/*.edf
#     edf_data/RAM_GDNF_2025/<batch>/*.edf
# On LUNARC the SV2A batches apparently live directly under the root
#     edf_data/<batch>/*.edf            (no SV2A_2024 wrapper)
# A naive `rsync edf_data/ -> edf_data/` would create a *duplicate*
# edf_data/SV2A_2024/ tree on LUNARC next to the existing root batches,
# and the recursive trainer would then count every SV2A file twice.
#
# This script rsyncs **per cohort** with an explicit destination so files
# merge into LUNARC's layout. The mapping is in COHORT_MAP below — VERIFY it
# against `--inspect` before transferring; the defaults are a best guess.
#
# Run from the ARRHENIUS login node, inside tmux (it's a long transfer):
#     tmux new -s edfsync
#
# Usage:
#   LUNARC_USER=xyz scripts/arrhenius/sync_edf_to_lunarc.sh --inspect  # show both trees, move nothing
#   LUNARC_USER=xyz scripts/arrhenius/sync_edf_to_lunarc.sh            # DRY-RUN (default): preview
#   LUNARC_USER=xyz scripts/arrhenius/sync_edf_to_lunarc.sh --go       # real transfer
#
# Overridable env: LUNARC_HOST, LUNARC_EDF, ARRHENIUS_EDF
# ============================================================

set -euo pipefail

: "${LUNARC_USER:?Set LUNARC_USER=<your lunarc username>}"
LUNARC_HOST="${LUNARC_HOST:-cosmos.lunarc.lu.se}"
LUNARC_EDF="${LUNARC_EDF:-/lunarc/nobackup/projects/lu2026-2-60/edf_data}"

# Resolve the Arrhenius EDF root from _common.sh (sets EDF_DIR=$PROJECT_STORAGE/edf_data).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/_common.sh"
ARRHENIUS_EDF="${ARRHENIUS_EDF:-$EDF_DIR}"

# ── Cohort layout mapping ────────────────────────────────────────────
# "<arrhenius subdir under edf_data>|<dest subdir under LUNARC edf_data>"
# An EMPTY destination means "land directly in LUNARC's edf_data root",
# i.e. strip the Arrhenius cohort wrapper. EDIT to match what --inspect shows.
COHORT_MAP=(
  "SV2A_2024|"                    # Arrhenius edf_data/SV2A_2024/* -> LUNARC edf_data/*  (strip wrapper)
  "RAM_GDNF_2025|RAM_GDNF_2025"   # keep its own folder — change to "RAM_GDNF_2025|" to also land in root
)

echo "Arrhenius source : $ARRHENIUS_EDF"
echo "LUNARC dest      : $LUNARC_USER@$LUNARC_HOST:$LUNARC_EDF"
echo "------------------------------------------------------------"

# ── Inspect mode: print both directory trees, transfer nothing ───────
if [[ "${1:-}" == "--inspect" ]]; then
  echo "ARRHENIUS edf_data (depth 2):"
  find "$ARRHENIUS_EDF" -maxdepth 2 -type d 2>/dev/null | sort | sed "s|$ARRHENIUS_EDF|  edf_data|"
  echo
  echo "LUNARC edf_data (depth 2):"
  ssh "$LUNARC_USER@$LUNARC_HOST" "find '$LUNARC_EDF' -maxdepth 2 -type d 2>/dev/null | sort | sed 's|$LUNARC_EDF|  edf_data|'"
  echo
  echo "Per-cohort EDF counts (Arrhenius):"
  for m in "${COHORT_MAP[@]}"; do
    src="${m%%|*}"
    n=$(find "$ARRHENIUS_EDF/$src" -type f -iname '*.edf' 2>/dev/null | wc -l)
    echo "  $src : $n .edf"
  done
  exit 0
fi

# ── Transfer (dry-run unless --go) ───────────────────────────────────
DRY="--dry-run"
if [[ "${1:-}" == "--go" ]]; then DRY=""; echo "MODE: REAL TRANSFER"; else echo "MODE: DRY-RUN (pass --go to transfer)"; fi
echo "------------------------------------------------------------"

for m in "${COHORT_MAP[@]}"; do
  src="${m%%|*}"
  dest="${m#*|}"
  src_path="$ARRHENIUS_EDF/$src/"
  dest_path="$LUNARC_EDF/${dest:+$dest/}"   # collapse to root when dest is empty

  if [[ ! -d "$ARRHENIUS_EDF/$src" ]]; then
    echo "!! skip '$src' — not found under $ARRHENIUS_EDF (check COHORT_MAP)"
    continue
  fi

  echo ">>> $src/  ->  $dest_path"
  # Ensure the destination subdir exists (rsync needs the parent present).
  [[ -z "$DRY" && -n "$dest" ]] && ssh "$LUNARC_USER@$LUNARC_HOST" "mkdir -p '$LUNARC_EDF/$dest'"

  # -a archive, -z compress, --partial resume interrupted EDFs, --progress live.
  # Default size+mtime compare only sends missing/changed files.
  rsync -az $DRY --partial --progress \
    "$src_path" \
    "$LUNARC_USER@$LUNARC_HOST:$dest_path"
done

echo "------------------------------------------------------------"
[[ -n "$DRY" ]] && echo "Dry-run only. Re-run with --go once the file lists look right." \
               || echo "Done. Regenerate LUNARC bad_channels.json if cohort paths changed."
