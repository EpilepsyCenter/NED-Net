# BENDR Pre-training on Arrhenius — Step-by-Step Guide

Arrhenius (NAISS) is the planned replacement for the current
infrastructure. Its GPU partition uses **NVIDIA GH200 superchips**:
a 72-core ARM (aarch64) CPU fused with an H100 GPU and 96 GB of
HBM, four per node, on a single Lustre filesystem.

This guide mirrors the LUNARC one but uses an **NVIDIA NGC PyTorch
container** instead of conda. The container ships a PyTorch build
that is properly compiled for aarch64 + CUDA + the GH200's
NVLink-C2C path, so we don't have to fight pip wheels or wait for
conda-forge to catch up on every release.

---

## Settings baked into the scripts

All operational details are wired in. If your SUPR project ever changes,
update `NAISS_PROJECT` in `_common.sh` *and* the `#SBATCH -A` line of
every sbatch script (Slurm reads `-A` before `_common.sh` runs).

| Item | Value | Where it lives |
|------|-------|----------------|
| Login | `login.hpc.arrhenius.naiss.se` | Step 1 |
| Storage project | `naiss2026-3-358` (storage path) | `_common.sh` (`NAISS_PROJECT`) |
| GPU Slurm account | `naiss2026-3-358-gpu` (note the `-gpu` suffix — differs from storage ID) | `#SBATCH -A` in every script |
| Project storage | `/nobackup/proj/disk/naiss2026-3-358/personal/$USER` (tier root is root-owned; write under `personal/<user>`) | `_common.sh` |
| GPU partition | `gpu` (382 nodes, 3-day max walltime; default partition is `cpu`) | `_common.sh` + `#SBATCH -p` |
| Account status | https://supr.naiss.se/account/ | this guide |
| Apptainer module | `Apptainer` | `_common.sh` — fall back to `singularity` if `module spider` shows that instead |

---

## Before you start

- [ ] SUPR allocation approved for Arrhenius
- [ ] LUNARC compute allocation (`LU 2026/2-60`) still active — useful
      as a fallback while Arrhenius onboarding is in progress
- [ ] EDF files staged somewhere transferable (currently on LUNARC at
      `/lunarc/nobackup/projects/lu2026-2-60/edf_data` — copying
      cluster-to-cluster is much faster than re-uploading from your Mac)

---

## Step 1: Log in

Before the first login, confirm your account is "enabled / Active" at
[supr.naiss.se/account/](https://supr.naiss.se/account/). If the
status is still "missing → enabled, transferring", wait — SSH will
fail with `Permission denied (publickey)` until the transfer
completes.

```bash
ssh ledri@login.hpc.arrhenius.naiss.se
```

Fallback login nodes (load-balanced): `arrhenius1.hpc.arrhenius.naiss.se`,
`arrhenius3.hpc.arrhenius.naiss.se`.

---

## Step 2: Upload code to Arrhenius

From your **local machine** (macOS Terminal, Linux, or WSL on Windows):

```bash
rsync -avz --filter=':- .gitignore' --exclude '.git' \
  ~/Software/NED-Net/ \
  ledri@login.hpc.arrhenius.naiss.se:~/NED-Net/
```

From Windows (PowerShell → WSL Ubuntu):

```bash
rsync -avz --filter=':- .gitignore' --exclude '.git' \
  /mnt/c/Users/Marco/Software/NED-Net/ \
  ledri@login.hpc.arrhenius.naiss.se:~/NED-Net/
```

The same SSH ControlMaster setup from the LUNARC guide works here —
just add an entry for `Host arrhenius login.hpc.arrhenius.naiss.se`.

---

## Step 3: Get the EDF data onto Arrhenius

The fastest path is **cluster-to-cluster** rsync over SSH. From
Arrhenius's login node:

```bash
# Replace LUNARC_USER with your LUNARC username
rsync -avz --progress \
  LUNARC_USER@cosmos.lunarc.lu.se:/lunarc/nobackup/projects/lu2026-2-60/edf_data/ \
  $PROJECT_STORAGE/edf_data/
```

`$PROJECT_STORAGE` is set by `_common.sh` (default
`/nobackup/proj/disk/$NAISS_PROJECT`). Resolve it before transfer:

```bash
cd ~/NED-Net
source scripts/arrhenius/_common.sh
echo "Will write to: $PROJECT_STORAGE/edf_data"
mkdir -p "$PROJECT_STORAGE/edf_data"
```

If `mkdir` fails, run `storagequota` on the login node — it lists
the project directories you actually have access to. Update
`PROJECT_STORAGE` in `_common.sh` if the doc-published convention
doesn't match your allocation (e.g. your project only granted the
`flash` tier).

> **No backups.** `/nobackup/proj/...` is, as the name says, not
> backed up. Keep the originals safe elsewhere.

---

## Step 4: Project ID + paths

Already set for `naiss2026-3-358` with storage at
`/nobackup/proj/disk/naiss2026-3-358` (both `_common.sh` and every
`#SBATCH -A` line). Nothing to edit unless your allocation changes —
if it does, update `NAISS_PROJECT` in `_common.sh` *and* the inline
`#SBATCH -A` lines in `test_gpu.sh`, `test_gpu_tiny.sh`,
`pretrain_short.sh`, `pretrain.sh`, `resume.sh` (Slurm reads `-A`
before `_common.sh` runs).

Override `PROJECT_STORAGE` only if `storagequota` reports something
other than `/nobackup/proj/disk/...` (e.g. you were granted only the
flash tier — then point it at `/nobackup/proj/flash/<id>`).

---

## Step 5: Set up the container environment (one-time)

```bash
cd ~/NED-Net
bash scripts/arrhenius/setup_env.sh
```

What this does:

1. Loads the `Apptainer` module
2. Pulls `nvcr.io/nvidia/pytorch:24.10-py3` (aarch64) into
   `$PROJECT_STORAGE/containers/`
3. Creates a small extras venv at `$PROJECT_STORAGE/venvs/bendr-extras`
   that lives outside the SIF — so updating `pyedflib` or your own
   package later doesn't require re-pulling the container
4. Installs `pyedflib`, `scipy`, `tqdm`, and `eeg_seizure_analyzer`
   into that venv

Expect 10–30 minutes for the pull (the SIF is ~10 GB).

---

## Step 6: Create logs directory

```bash
mkdir -p ~/NED-Net/logs
```

`OUTPUT_DIR` and the container directory are created automatically.

---

## Step 7: Quick GPU smoke tests

Run the **tiny** smoke first — it's tuned for 2–9 EDFs and finishes
in well under 30 minutes. If it passes, run the full `test_gpu.sh`
to validate at scale.

### 7a — Tiny test (2–9 EDFs)

Upload 2–9 EDFs to `$PROJECT_STORAGE/edf_data/` (Step 3 commands work
for any subset — point them at a small folder), then:

```bash
cd ~/NED-Net
sbatch scripts/arrhenius/test_gpu_tiny.sh
```

Workers dropped to 2, validation disabled, 3 segments/file.

### 7b — Full-scale test

After the full dataset is uploaded:

```bash
sbatch scripts/arrhenius/test_gpu.sh
```

### Watch either job

```bash
squeue -u $USER                            # job state
tail -f logs/bendr_tiny_<JOBID>.out        # tiny
tail -f logs/bendr_test_<JOBID>.out        # full
```

### What to look for

1. `Arch: aarch64` — confirms ARM kernel/userspace (Grace CPU)
2. `PyTorch 2.x.x, CUDA 12.x` — container layer healthy
3. `Device: NVIDIA H200` (or `GH200`) — GPU detected
4. `EDF files found:` listing — Lustre path correct
5. `Epoch 1/2` … `Epoch 2/2` — training actually runs
6. Checkpoint saved under `$PROJECT_STORAGE/bendr_output/{tiny,test}_run/`

---

## Step 7.5: Generate the bad-channels map

The real training runs (Steps 8–9) exclude known-noisy channels on a
**per (cohort, batch)** basis. The data is staged one folder per cohort,
each with its own `batch {1,2,3}` — and the bad channels differ per batch
*and* per cohort:

```
$EDF_DIR/
  SV2A_2024/     batch {1,2,3}/Week*-Day*/*.edf
  RAM_GDNF_2025/ batch {1,2,3}/W*-D*/*.edf
```

The trainer reads the exclusions from a `bad_channels.json` file, which
`scripts/make_bad_channels.py` generates from the cohort + `batch N` folders
in your uploaded data. The lab's plain-text source of truth is
`edf_data_for_bendr/bad channels.txt`; the script encodes the same rules as
0-based indices.

Run it once, on the **login node**. It's pure standard-library Python and
only reads directory paths (not EDF contents), so it needs **no container
and no extras venv** — the login node's `python3` is enough:

```bash
cd ~/NED-Net
source scripts/arrhenius/_common.sh        # resolves $EDF_DIR / $BAD_CHANNELS
python scripts/make_bad_channels.py --data-dir "$EDF_DIR"
```

Expected output (counts will match your data):

```
Scanned 2976 EDF(s) under .../personal/<user>/edf_data
  RAM_GDNF_2025 / batch 1:   338 file(s)  exclude indices [1, 4]
  RAM_GDNF_2025 / batch 2:   339 file(s)  exclude indices [2, 3, 6]
  RAM_GDNF_2025 / batch 3:   341 file(s)  exclude indices none
  SV2A_2024 / batch 1:   660 file(s)  exclude indices [0, 2, 7]
  SV2A_2024 / batch 2:   670 file(s)  exclude indices [4, 5]
  SV2A_2024 / batch 3:   628 file(s)  exclude indices [0, 2, 5]

Wrote 2635 file entr(ies) ... to:
  .../personal/<user>/bad_channels.json
```

By default it writes `bad_channels.json` to `<edf_data>/../`, which is
exactly `$BAD_CHANNELS` from `_common.sh` — the path the
`pretrain_short.sh` / `pretrain.sh` / `resume.sh` scripts pass to
`--bad-channels`. Nothing else to wire up.

Sanity-check it landed:

```bash
head -20 "$BAD_CHANNELS"
```

> **Why this isn't optional.** The real-run scripts pass `--bad-channels`
> unconditionally, so they **hard-fail at startup** if this file is
> missing — that's deliberate: better a fast, loud failure than silently
> training on the noisy channels you meant to drop. The smoke tests
> (Step 7) do *not* use the map, so they run fine before you generate it.

The generator itself also hard-fails (non-zero exit, nothing written) if:

- the **same EDF filename appears under groups with different exclusions**
  (the map keys on filename, so this would be ambiguous — this is exactly
  how a misplaced/duplicated recording gets caught), or
- an **EDF sits outside any known cohort folder** (`SV2A_2024`,
  `RAM_GDNF_2025`, …), or
- an **EDF sits outside any `batch N` folder** (no rule to apply).

If you hit any of these, the data layout needs a look before training — the
cohort and `batch N` folders must be preserved from the transfer (don't
flatten).

### Adjusting for future cohorts/batches or different bad channels

All the exclusion logic lives in **one dict** at the top of
`scripts/make_bad_channels.py`, keyed on **cohort → batch → 0-based indices**:

```python
COHORT_EXCLUDE = {
    "SV2A_2024": {
        1: [0, 2, 7],   # electrodes 1, 3, 8
        2: [4, 5],      # electrodes 5, 6
        3: [0, 2, 5],   # electrodes 1, 3, 6
    },
    "RAM_GDNF_2025": {
        1: [1, 4],      # electrodes 2, 5
        2: [2, 3, 6],   # electrodes 3, 4, 7
        3: [],          # all ok
    },
}
```

> **Electrode label vs. array index.** The lab numbers electrodes
> **1-based** (1–8); the model indexes channels **0-based** (0–7). So
> electrode *N* → index *N−1*. Electrode 1 → `0`, electrode 6 → `5`,
> electrode 8 → `7`. Edit the dict in **index** space. The human-readable
> source is `edf_data_for_bendr/bad channels.txt`.

To adjust:

1. **New cohort arrives** (e.g. another study): add a top-level key with its
   own per-batch sub-dict. EDFs under a folder name not in `COHORT_EXCLUDE`
   **hard-fail** the run (so a new cohort can't be silently mis-handled).
2. **New batch within a cohort**: add a line to that cohort's sub-dict, e.g.
   `4: [2],` to "drop electrode 3", or `4: [],` if clean. A batch folder
   with no entry is left **untouched (all channels)** with a warning to
   stderr — *not* treated as "exclude none" — so a forgotten batch is
   visible in the log.
3. **Bad channels differ from what's wired** (your channel check found
   something else): change the index list for that cohort/batch.
4. **Re-run the same command** above. It overwrites `bad_channels.json`
   in place; the training scripts already point at it, so no other edits
   are needed. Re-run it any time the data set or the rules change.

---

## Step 8: Short pre-training (5 epochs)

Same logic as LUNARC: run a short pass on the full dataset before
committing to the long job. The resume script picks up from the
last checkpoint, so this is not wasted compute.

```bash
sbatch scripts/arrhenius/pretrain_short.sh
```

Monitor:

```bash
squeue -u $USER
tail -f logs/bendr_5ep_<JOBID>.out
```

Loss should drop noticeably across the 5 epochs. If it doesn't,
stop and debug before continuing.

---

## Step 9: Continue to 30 epochs

```bash
sbatch scripts/arrhenius/resume.sh
```

Resumes from the latest `checkpoint_epoch_*.pt` in `OUTPUT_DIR`.
If the 3-day walltime (the `gpu` partition max) hits and 30 epochs
aren't done, just submit `resume.sh` again — it keeps picking up from
the latest checkpoint.

Auto-chain after `pretrain_short.sh`:

```bash
JOBID=$(sbatch --parsable scripts/arrhenius/pretrain_short.sh)
sbatch -d afterok:$JOBID scripts/arrhenius/resume.sh
```

---

## Step 10: Download the trained model

```bash
# Best model location
ls $PROJECT_STORAGE/bendr_output/run1/best_model.pt
```

From your **local Mac**:

```bash
mkdir -p ~/.eeg_seizure_analyzer/pretrained
scp <YOUR_USERNAME>@<arrhenius-login>:$PROJECT_STORAGE/bendr_output/run1/best_model.pt \
  ~/.eeg_seizure_analyzer/pretrained/bendr_rodent_25k.pt
```

(You can leave it on Lustre and re-use from there if you fine-tune
on Arrhenius too.)

---

## Step 11: Use the pre-trained model in NED-Net

Identical to LUNARC — see **Step 11** in `scripts/lunarc/GUIDE.md` for the
full tab-by-tab flow. In short: the cluster produces a self-supervised
BENDR *backbone* (`bendr_rodent_25k.pt` in
`~/.eeg_seizure_analyzer/pretrained/`), not a detector. Inside NED-Net you
generate candidates in **Detection → Seizure**, refine them (confirm /
reject) in **Training → Seizure**, then fine-tune a detector in
**Dataset / Model** (Architecture = BENDR, Pre-trained weights =
`bendr_rodent_25k`, **Start Training**), and finally run batch/live
detection in the **Analysis** tab. These are all separate top-level tabs.

To **keep improving** the model you re-run that loop — detect with the
trained model (Detection offers both **BENDR** and **U-Net** methods),
correct the candidates, and **retrain on the larger annotation set**. You
do *not* load your trained model as **Pre-trained weights**: that dropdown
only lists the self-supervised backbones in `~/.eeg_seizure_analyzer/
pretrained/`, never your trained detectors in `…/models/`. BENDR
re-fine-tunes from the backbone; U-Net retrains from scratch.

---

## Why a container, not conda?

- aarch64 PyTorch wheels exist on PyPI (since 2.3) but the
  GH200-specific NCCL tuning and CUDA-12.x driver path is not
  guaranteed to match an arbitrary pip install.
- NVIDIA's NGC PyTorch image is built and tested by NVIDIA for
  exactly this hardware. It's the same recipe their reference
  benchmarks use, so reproducing results later is easier.
- Lustre + many-file workloads benefit from the container's
  consistent libc/NCCL versions across nodes — useful if we ever
  scale to multi-node DDP.

If a future PyTorch release ships a known-good aarch64+GH200 wheel
through conda-forge, we can switch — but the container path is the
zero-surprise option for the first runs.

---

## Useful commands on Arrhenius

| What | Command |
|------|---------|
| List your jobs | `squeue -u $USER` |
| Cancel a job | `scancel <JOBID>` |
| Storage / project info | `storagequota` |
| Account status | https://supr.naiss.se/account/ |
| Watch a log | `tail -f logs/bendr_*_<JOBID>.out` |
| Estimated start | `squeue --start -j <JOBID>` |
| GPU partition status | `sinfo -p gpu` |
| Inspect container | `apptainer inspect $CONTAINER_PATH` |
| Shell into container | `apptainer shell --nv $CONTAINER_PATH` |
| Interactive GPU shell | `interactive -p gpu --gpus 1 -t 30` |

---

## Troubleshooting

**`apptainer: command not found`**
The module is named differently at this site. Try
`module spider apptainer` and `module spider singularity`. Edit
the `module load Apptainer` line in `_common.sh` and the job
scripts.

**`ERROR: CUDA not visible inside container`**
Check the script used `apptainer exec --nv`. The `--nv` flag is
what mounts the host NVIDIA driver into the container.

**`No checkpoint found`**
`OUTPUT_DIR` doesn't exist or is on a different filesystem than
where the previous job wrote to. Confirm
`echo $PROJECT_STORAGE/bendr_output/run1` resolves to the same path
the previous job logged.

**Job stuck pending**
`squeue --start -j <JOBID>` shows the estimated start. Until NAISS
publishes how Arrhenius prioritises jobs, treat long waits as
normal during the rollout period.

**EDF reader complains about samples < 1**
This was the LUNARC-era bug — fixed in `adicht_reader.py` to handle
records where individual channels are empty. Make sure your code
checkout includes that fix before re-converting on Arrhenius.
