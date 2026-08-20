#!/usr/bin/env bash
# ============================================================
# Push the AD-animal EDFs (KAHA recordings) from the research share -> LUNARC
# ============================================================
# Source layout on the mounted research volume:
#     AD_Animals_Recordings/Week_<N>/EDF and dat files/W<N>_D<M>/*.edf
# plus huge .dat files and "* movies" folders we do NOT want (1.2 TB total;
# the EDFs alone are ~50 GB / 319 files).
#
# Destination layout on LUNARC (space-free, wrapper folder stripped):
#     ad_edf_data/Week_<N>/W<N>_D<M>/*.edf
#
# NOTE the destination is a SEPARATE root next to edf_data/, not inside it.
# edf_data/ holds the SV2A cohort and is scanned recursively by the trainers /
# pretraining scripts — dropping a different study in there would silently pull
# AD files into every future training set.
#
# Run from your Mac with the research share mounted, inside tmux/screen
# (~50 GB, so it takes a while):
#     tmux new -s adsync
#
# Usage:
#   scripts/local/sync_ad_edf_to_lunarc.sh --inspect   # show src/dst trees + free space, move nothing
#   scripts/local/sync_ad_edf_to_lunarc.sh             # DRY-RUN (default): preview the file list
#   scripts/local/sync_ad_edf_to_lunarc.sh --go        # real transfer
#   scripts/local/sync_ad_edf_to_lunarc.sh --verify    # compare per-day EDF counts src vs LUNARC
#
# Overridable env: SRC_ROOT, LUNARC_HOST, LUNARC_DEST, COMPRESS=1
# ============================================================

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/Volumes/research/LU26D1055-epicenter/Data/KAHA recordings/AD_Animals_Recordings}"
LUNARC_HOST="${LUNARC_HOST:-cosmos}"   # ~/.ssh/config alias -> cosmos.lunarc.lu.se
LUNARC_DEST="${LUNARC_DEST:-/lunarc/nobackup/projects/lu2026-2-60/ad_edf_data}"
WEEKS=(Week_1 Week_2 Week_3)
WRAPPER="EDF and dat files"            # the space-y folder we strip on the way out

# EDF is int16 — compression buys little and costs CPU. Set COMPRESS=1 to enable.
ZFLAG=(); [ "${COMPRESS:-0}" = "1" ] && ZFLAG=(-z)

# macOS 15+ ships Apple's `openrsync` as /usr/bin/rsync, which wins on PATH and
# rejects --chmod (and other rsync 3 flags). Prefer a real rsync 3 if one is
# installed (Homebrew), and degrade the flag set if only openrsync exists.
pick_rsync() {
  local c
  for c in "${RSYNC:-}" /opt/homebrew/bin/rsync /usr/local/bin/rsync "$(command -v rsync || true)"; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    "$c" --version 2>&1 | head -1 | grep -qi openrsync && continue
    echo "$c"; return 0
  done
  command -v rsync
}
RSYNC="$(pick_rsync)"
# Drop SMB's meaningless 0700 mode bits — only real rsync understands --chmod.
PERM_FLAGS=(--no-perms --chmod=D755,F644)
if "$RSYNC" --version 2>&1 | head -1 | grep -qi openrsync; then
  PERM_FLAGS=()
  echo "!! Only Apple's openrsync found — no --partial resume of half-sent files."
  echo "   'brew install rsync' is strongly recommended for a 50 GB transfer."
fi

# List the day folders of one week that actually contain EDFs (skips "Results").
day_dirs() {  # day_dirs <week>
  local wk="$1" d
  for d in "$SRC_ROOT/$wk/$WRAPPER/"*/; do
    [ -d "$d" ] || continue
    if compgen -G "$d*.edf" > /dev/null || compgen -G "$d*.EDF" > /dev/null; then
      basename "$d"
    fi
  done
}

count_src() {  # count_src <week> <day>
  find "$SRC_ROOT/$1/$WRAPPER/$2" -maxdepth 1 -type f -iname '*.edf' | wc -l | tr -d ' '
}

[ -d "$SRC_ROOT" ] || { echo "!! Source not found (is the research share mounted?): $SRC_ROOT" >&2; exit 1; }

echo "Source : $SRC_ROOT"
echo "Dest   : $LUNARC_HOST:$LUNARC_DEST"
echo "------------------------------------------------------------"

MODE="${1:-}"

# ── Inspect: show what's here, what's there, and whether it fits ─────
if [ "$MODE" = "--inspect" ]; then
  total=0
  for wk in "${WEEKS[@]}"; do
    for d in $(day_dirs "$wk"); do
      n=$(count_src "$wk" "$d")
      total=$((total + n))
      printf '  %s/%s : %s .edf\n' "$wk" "$d" "$n"
    done
  done
  echo "  TOTAL: $total .edf"
  find "$SRC_ROOT" -type f -iname '*.edf' -print0 \
    | xargs -0 stat -f '%z' \
    | awk '{s+=$1} END {printf "  SIZE : %.1f GB\n", s/1073741824}'
  echo
  echo "LUNARC side:"
  ssh "$LUNARC_HOST" "
    echo '  dest tree (depth 2):'
    find '$LUNARC_DEST' -maxdepth 2 -type d 2>/dev/null | sort | sed 's|$LUNARC_DEST|    ad_edf_data|' || echo '    (does not exist yet)'
    echo '  existing .edf there:' \$(find '$LUNARC_DEST' -type f -iname '*.edf' 2>/dev/null | wc -l)
    echo '  free space:'
    df -h '$(dirname "$LUNARC_DEST")' 2>/dev/null | tail -1 | sed 's/^/    /'
  "
  exit 0
fi

# ── Verify: per-day counts on both sides ────────────────────────────
if [ "$MODE" = "--verify" ]; then
  remote_counts=$(ssh "$LUNARC_HOST" "
    for f in \$(find '$LUNARC_DEST' -type f -iname '*.edf' 2>/dev/null); do
      echo \"\$(basename \$(dirname \$(dirname \$f)))/\$(basename \$(dirname \$f))\"
    done | sort | uniq -c | awk '{print \$2, \$1}'")
  bad=0
  for wk in "${WEEKS[@]}"; do
    for d in $(day_dirs "$wk"); do
      src=$(count_src "$wk" "$d")
      dst=$(echo "$remote_counts" | awk -v k="$wk/$d" '$1==k {print $2}')
      dst="${dst:-0}"
      if [ "$src" = "$dst" ]; then
        printf '  OK   %s/%s : %s\n' "$wk" "$d" "$src"
      else
        printf '  MISM %s/%s : src=%s lunarc=%s\n' "$wk" "$d" "$src" "$dst"
        bad=1
      fi
    done
  done
  [ "$bad" = 0 ] && echo "All day folders match." || echo "Mismatches above — re-run --go (rsync resumes)."
  exit $bad
fi

# ── Transfer (dry-run unless --go) ──────────────────────────────────
DRY="--dry-run"
if [ "$MODE" = "--go" ]; then
  DRY=""; echo "MODE: REAL TRANSFER"
else
  echo "MODE: DRY-RUN (pass --go to transfer)"
fi
echo "------------------------------------------------------------"

# Create the whole destination tree up front — one ssh instead of one per day.
if [ -z "$DRY" ]; then
  mkdirs=""
  for wk in "${WEEKS[@]}"; do
    for d in $(day_dirs "$wk"); do mkdirs="$mkdirs '$LUNARC_DEST/$wk/$d'"; done
  done
  eval ssh "$LUNARC_HOST" "mkdir -p $mkdirs"
fi

for wk in "${WEEKS[@]}"; do
  for d in $(day_dirs "$wk"); do
    echo ">>> $wk/$WRAPPER/$d/  ->  $LUNARC_DEST/$wk/$d/"
    # EDFs only: --include the pattern, --exclude the rest (drops .dat, movies).
    # --partial resumes half-sent 170 MB files after a dropped connection.
    # ${ARR[@]+...} guard: macOS bash 3.2 treats an empty array as unset under `set -u`.
    "$RSYNC" -rt $DRY ${ZFLAG[@]+"${ZFLAG[@]}"} --partial --progress \
      ${PERM_FLAGS[@]+"${PERM_FLAGS[@]}"} \
      --include='*.edf' --include='*.EDF' --exclude='*' \
      "$SRC_ROOT/$wk/$WRAPPER/$d/" \
      "$LUNARC_HOST:$LUNARC_DEST/$wk/$d/"
  done
done

echo "------------------------------------------------------------"
if [ -n "$DRY" ]; then
  echo "Dry-run only. Re-run with --go once the file list looks right."
else
  echo "Transfer done. Check it with:  $0 --verify"
  echo
  echo "Then, on LUNARC (login node):"
  echo "    cd ~/NED-Net && bash scripts/lunarc/detect_spikes_ad.sh"
fi
