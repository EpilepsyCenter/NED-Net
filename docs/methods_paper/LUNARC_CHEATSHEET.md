# LUNARC COSMOS — commands cheatsheet

Account `lu2026-2-60` for **both** compute (`-A`) and storage. `lu2026-12-29` is the
SLURM allocation ID; a convenience symlink `~/lu2026-12-29` in the LUNARC home dir points
at the same storage, but **the canonical path is `lu2026-2-60`**. Using the other one
silently matched 0 EDFs once (job 3769954).

## Environment

```bash
cd ~/NED-Net                      # REQUIRED: config_conda.sh is a relative path
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr
```

Activate **before** any `srun`, not inside it: `srun --pty` starts a fresh shell that
inherits nothing, and you get `ModuleNotFoundError: No module named 'numpy'`. Once
activated in the login shell, `srun` propagates `PATH` to the compute node.

## Interactive vs batch

| use | why |
|---|---|
| `srun --pty` | short checks whose output you want now (`--analyze`, a quick query). Dies with your terminal. |
| `sbatch` | anything producing an artefact worth logging (a model, a DB). Survives logout. |

```bash
# short CPU check (sidecar + EDF-header work only -- no GPU needed)
srun -p lu48 -A lu2026-2-60 -t 00:20:00 -c 8 --pty <command>

# short GPU check
srun -p gpua100 -A lu2026-2-60 -t 00:15:00 --gres=gpu:1 --pty <command>
```

Partitions: **`lu48`** = CPU (detection is CPU/IO-bound, use this), **`gpua100`** = GPU
(training only).

## Paths

```
/lunarc/nobackup/projects/lu2026-2-60/edf_data/         SV2A cohort + sidecars
/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025/    RAM_GDNF cohort
    Batch_{1,2,3,4}_Recordings/<subdirs>/*.edf          sidecars live in SUBDIRS,
                                                        so always glob recursively
~/train_nomirneg/                                       flat training tree:
                                                        SV2A symlinked, RAM_GDNF copied
~/.eeg_seizure_analyzer/models/<name>/                   trained models
~/.eeg_seizure_analyzer/projects/<name>.db               detection output
```

`/lunarc/nobackup` is **not backed up**, and the sidecars are the only copy of the human
labels — see [[annotations-source-of-truth-lunarc]].

## Queue

```bash
squeue -u $USER                   # mine
squeue --start -j <jobid>         # estimated start (it drifts)
scancel <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f logs/unet_train_<jobid>.out
```

## The training loop

Four steps, and **step 2 is the one that fails silently**:

```bash
# 1. detect           -> DB + pending sidecar events
sbatch scripts/lunarc/detect_ramgdnf_unet.sbatch

# 2. review in the UI, then FOLD THE REVIEW INTO THE TRAINING TREE
python scripts/lunarc/refresh_training_tree.py --keep-mir-rejections --dry-run
python scripts/lunarc/refresh_training_tree.py --keep-mir-rejections

# 3. check the dataset is what you think BEFORE spending GPU time
srun -p lu48 -A lu2026-2-60 -t 00:20:00 -c 8 --pty \
  python -m eeg_seizure_analyzer.ml.train_unet --data-dir $HOME/train_nomirneg --analyze \
    --stable-val-split --max-positive-sec 100 \
    --hard-neg-exclude-method mir_candidate --bg-avoid-rejected \
    --exclude-animals <ids>

# 4. train
EDF_DIR=$HOME/train_nomirneg MODEL_NAME=<name> \
NEG_SOURCE=hard NEG_POS_RATIO=10 \
MAX_POSITIVE_SEC=100 BG_AVOID_REJECTED=1 HARD_NEG_EXCLUDE_METHODS=mir_candidate \
EXCLUDE_ANIMALS="<ids>" \
bash scripts/lunarc/train_unet.sh
```

**Two failures to check for every time**, both of which have already happened:

1. **Read the submission summary line.** A bare `read` used to wipe env presets, so job
   3795249 was submitted with `exclude=none` and would have trained a hold-out fold on
   its own test set. Confirm `exclude=`, `max_positive_sec=`, `bg_avoid_rejected=` and
   `hard_neg_exclude=` all read as you set them.
2. **Refresh the tree before every submission.** It was last run before the 2026-10-07/08
   review sessions, so arms A/A2/A3 trained with none of that work — 52 manual additions
   and 344 adjudicated detections were missing. `train_unet.py` takes only `--data-dir`
   and re-scans the folder, so it cannot be pinned to a frozen dataset definition and will
   quietly train on whatever is there.
