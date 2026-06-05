# ============================================================
# Arrhenius — shared variables sourced by all job scripts
# ============================================================
# Edit this single file when project IDs, paths, or container
# tags change instead of touching each sbatch script.
# ============================================================

# SUPR allocation. IMPORTANT: on Arrhenius the GPU Slurm *account* and
# the *storage project* are DIFFERENT strings:
#   - storage / PROJECT_STORAGE path → `naiss2026-3-358`   (this var)
#   - GPU Slurm account (#SBATCH -A) → `naiss2026-3-358-gpu` (the `-gpu`
#     suffix; confirmed via `sacctmgr -n show assoc user=$USER`)
# This var drives ONLY the storage path below. The account is a static
# literal in each script's `#SBATCH -A` line (Slurm reads it before this
# file is sourced, so it can't reference this var). If your allocation
# changes, update BOTH: this var AND every `#SBATCH -A …-gpu` line.
# Verify at https://supr.naiss.se/account/, `storagequota`, and
# `sacctmgr -n show assoc user=$USER format=Account%30,Partition,QOS`.
export NAISS_PROJECT="${NAISS_PROJECT:-naiss2026-3-358}"

# Project storage on Arrhenius. NAISS publishes two tiers:
#   /nobackup/proj/disk/<PROJECT>   – default bulk storage (slower, larger)
#   /nobackup/proj/flash/<PROJECT>  – fast scratch (use if granted)
# The tier root itself (…/disk/<PROJECT>) is root-owned and NOT writable
# by users; each member gets an auto-created private dir at
# `personal/<username>` (see the README in the tier root), with `shared`
# and `apps` for group-wide data. We point at the per-user `personal`
# dir since BENDR runs are single-user. Run `storagequota` to see tiers
# and quota; switch the prefix to `/nobackup/proj/flash/…` if you want
# the faster tier for hot data.
export PROJECT_STORAGE="${PROJECT_STORAGE:-/nobackup/proj/disk/${NAISS_PROJECT}/personal/${USER}}"

# Data, code, output, container.
export EDF_DIR="${EDF_DIR:-${PROJECT_STORAGE}/edf_data}"
# Per-batch noisy-channel exclusion map for the real pretrain/resume runs.
# Generate once after the data is staged:
#   python scripts/make_bad_channels.py --data-dir "$EDF_DIR"
# (writes to ${PROJECT_STORAGE}/bad_channels.json). The trainer hard-fails
# if it's missing — intentional, so a forgotten generate doesn't silently
# train on the channels you meant to drop. Smoke tests don't use it.
export BAD_CHANNELS="${BAD_CHANNELS:-${PROJECT_STORAGE}/bad_channels.json}"
export CODE_DIR="${CODE_DIR:-${HOME}/NED-Net}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_STORAGE}/bendr_output/run1}"
export CONTAINER_PATH="${CONTAINER_PATH:-${PROJECT_STORAGE}/containers/pytorch-ngc-arm64.sif}"
export EXTRAS_VENV="${EXTRAS_VENV:-${PROJECT_STORAGE}/venvs/bendr-extras}"

# Local NVMe per node (Arrhenius advertises ~1.8 TB). Use it to
# stage the EDF index / scratch caches so we don't hammer Lustre
# from every worker process.
export NVME_SCRATCH="${NVME_SCRATCH:-/scratch/local/${SLURM_JOB_ID:-tmp}}"

# Partition / QoS. The Arrhenius GPU partition (Grace Hopper / H200)
# is `gpu` (confirmed via `sinfo -s`: 382 nodes, 3-day max walltime;
# the default partition is `cpu`). No dedicated short-job QoS is
# documented, so we leave QOS unset and rely on walltime to gate jobs.
export GPU_PARTITION="${GPU_PARTITION:-gpu}"
export GPU_QOS="${GPU_QOS:-}"

# Helper: run a command inside the container with GPU and our
# extras venv activated, with the project bound in.
arrhenius_run() {
    apptainer exec --nv \
        --bind "${PROJECT_STORAGE}:${PROJECT_STORAGE}" \
        --bind "${CODE_DIR}:${CODE_DIR}" \
        "${CONTAINER_PATH}" \
        bash -lc "source '${EXTRAS_VENV}/bin/activate'; cd '${CODE_DIR}'; $*"
}
