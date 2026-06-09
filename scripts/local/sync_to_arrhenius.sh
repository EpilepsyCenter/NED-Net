#!/usr/bin/env bash
# ============================================================
# Sync NED-Net CODE (only) to Arrhenius (NAISS)
# ============================================================
# Ships the repo to ~/NED-Net on the Arrhenius login node, which is the
# CODE_DIR the sbatch scripts expect (see scripts/arrhenius/_common.sh).
# Uses the repo's .gitignore as the exclude list, so data, .venv,
# checkpoints (*.pt), bendr_output/, EDFs, etc. are never uploaded — only
# code/scripts. This mirrors scripts/arrhenius/GUIDE.md Step 2.
#
# After syncing, on the login node:
#   sbatch scripts/arrhenius/pretrain_short.sh
#
# Usage:
#   scripts/local/sync_to_arrhenius.sh            # real transfer
#   scripts/local/sync_to_arrhenius.sh -n         # dry-run (preview file list)
#   scripts/local/sync_to_arrhenius.sh --delete   # also remove stale code on
#                                                 # the cluster (excluded data
#                                                 # is protected); preview first
#
# Override host/dest if your account or layout differs:
#   ARRHENIUS_HOST=ledri@arrhenius1.hpc.arrhenius.naiss.se \
#     scripts/local/sync_to_arrhenius.sh
# ============================================================

set -euo pipefail

# Repo root (this script lives at scripts/local/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARRHENIUS_HOST="${ARRHENIUS_HOST:-ledri@login.hpc.arrhenius.naiss.se}"
DEST="${DEST:-~/NED-Net/}"

echo "Source : ${REPO_ROOT}/"
echo "Dest   : ${ARRHENIUS_HOST}:${DEST}"
echo "Extra  : $*"
echo "-----------------------------------------------------------"

# -a archive, -v verbose, -z compress; .gitignore drives the excludes.
# -c (checksum) compares file *contents*, not size+mtime: the cluster copy was
# first synced from a different machine (Windows), so mtimes never match and a
# plain sync would re-list every file. With -c only genuinely-changed files
# transfer. Any extra args ("$@") pass straight through, e.g. -n for a dry-run.
rsync -avzc "$@" \
  --filter=':- .gitignore' --exclude '.git' \
  "${REPO_ROOT}/" \
  "${ARRHENIUS_HOST}:${DEST}"

echo "-----------------------------------------------------------"
echo "Done. On the login node: sbatch scripts/arrhenius/pretrain_short.sh"
