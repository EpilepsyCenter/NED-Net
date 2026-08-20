#!/usr/bin/env bash
# ============================================================
# Fetch ONLY the AD EDFs needed to review flagged seizure events
# ============================================================
# The AD recordings live on LUNARC (50 GB, 319 files). Reviewing the flagged
# detections needs 14 of them (~2.4 GB), so this pulls just those, keeping the
# Week_N/W N_D M layout so the files stay identifiable.
#
# The file list is derived from a review CSV (default: the one
# ad_seizures.db's flagged events were exported to), so re-running it after
# widening the review criteria fetches only what is newly needed — rsync skips
# whatever is already local.
#
# Usage:
#   scripts/local/fetch_ad_review_edfs.sh              # DRY-RUN: show the list
#   scripts/local/fetch_ad_review_edfs.sh --go         # transfer
#   scripts/local/fetch_ad_review_edfs.sh --go --rewrite-db
#
# --rewrite-db additionally writes ad_seizures_local.db, a COPY of the project
# DB whose chunks.path values point at the local files instead of the LUNARC
# ones, so anything reading the stored paths resolves. The original DB is never
# modified.
#
# Overridable env: LIST, DEST, LUNARC_HOST, LUNARC_EDF, DB_SRC
# ============================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST="${LIST:-$HERE/ad_figure/seizure_review_list.csv}"
DEST="${DEST:-$HOME/Software/edf/AD_2026}"
LUNARC_HOST="${LUNARC_HOST:-cosmos}"
LUNARC_EDF="${LUNARC_EDF:-/lunarc/nobackup/projects/lu2026-2-60/ad_edf_data}"
DB_SRC="${DB_SRC:-$HOME/.eeg_seizure_analyzer/projects/ad_seizures.db}"

[ -f "$LIST" ] || { echo "!! Review list not found: $LIST" >&2; exit 1; }

# macOS 15 ships Apple's openrsync as /usr/bin/rsync; it lacks --files-from.
pick_rsync() {
  local c
  for c in "${RSYNC:-}" /opt/homebrew/bin/rsync /usr/local/bin/rsync "$(command -v rsync || true)"; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    "$c" --version 2>&1 | head -1 | grep -qi openrsync && continue
    echo "$c"; return 0
  done
  return 1
}
RSYNC="$(pick_rsync)" || { echo "!! Need real rsync 3 (brew install rsync)" >&2; exit 1; }

# Column 1 of the review CSV is the path relative to the EDF root.
TMPLIST="$(mktemp)"
trap 'rm -f "$TMPLIST"' EXIT
python3 - "$LIST" > "$TMPLIST" <<'PY'
import csv, sys
seen = []
for row in csv.DictReader(open(sys.argv[1])):
    f = (row.get("file") or "").strip()
    if f and f not in seen:
        seen.append(f)
print("\n".join(sorted(seen)))
PY

N=$(wc -l < "$TMPLIST" | tr -d ' ')
echo "Review list : $LIST"
echo "Files needed: $N   (~$(python3 -c "print(f'{$N*173/1024:.1f}')") GB)"
echo "From        : $LUNARC_HOST:$LUNARC_EDF"
echo "To          : $DEST"
echo "------------------------------------------------------------"
sed 's/^/  /' "$TMPLIST"
echo "------------------------------------------------------------"

if [ "${1:-}" != "--go" ] && [ "${2:-}" != "--go" ]; then
  echo "DRY-RUN. Re-run with --go to transfer."
  exit 0
fi

mkdir -p "$DEST"
# --files-from paths are relative to the REMOTE source root; -R is implied, so
# the Week_N/W N_D M structure is recreated locally.
"$RSYNC" -rt --partial --progress --no-perms --chmod=D755,F644 \
  --files-from="$TMPLIST" \
  "$LUNARC_HOST:$LUNARC_EDF/" "$DEST/"

echo "------------------------------------------------------------"
echo "Transferred to $DEST"

if [ "${1:-}" = "--rewrite-db" ] || [ "${2:-}" = "--rewrite-db" ]; then
  OUT="${DB_SRC%.db}_local.db"
  cp "$DB_SRC" "$OUT"
  python3 - "$OUT" "$LUNARC_EDF" "$DEST" <<'PY'
import sqlite3, sys, os
db, remote_root, local_root = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/")
conn = sqlite3.connect(db)
n = 0
for cid, path in conn.execute("SELECT id, path FROM chunks").fetchall():
    if not path.startswith(remote_root):
        continue
    new = os.path.join(local_root, path[len(remote_root):].lstrip("/"))
    # Only repoint files that actually arrived; the rest keep their LUNARC path
    # so it stays obvious they were not fetched.
    if os.path.exists(new):
        conn.execute("UPDATE chunks SET path=? WHERE id=?", (new, cid))
        n += 1
conn.commit(); conn.close()
print(f"Rewrote {n} chunk paths -> {db}")
PY
  echo "Open that DB in NED-Net (Results) to review with the signal available."
fi
