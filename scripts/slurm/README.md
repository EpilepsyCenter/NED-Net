# SLURM examples

Example job scripts for running NED-Net's trainers and batch detectors on a cluster. They are
starting points: set the partition (`-p`), allocation (`-A`), module names and wall times for
your site.

| Script | Node | What it does |
|---|---|---|
| `setup_env.sh` | login | one-time conda environment (`nednet`) with PyTorch + CUDA |
| `train_unet.sbatch` | GPU | Stage-1 U-Net seizure detector |
| `train_convulsive.sbatch` | GPU | Stage-2 convulsive classifier |
| `detect_all.sbatch` → `detect_batch.py` | CPU | batch seizure detection into a project database |
| `detect_spikes_all.sbatch` → `detect_spikes_batch.py` | CPU | batch interictal-spike detection |

## Workflow

1. Copy your EDFs to the cluster, with the `*_ned_annotations.json` sidecars written by the
   app's Training tab (and `*_ned_channels.json`, so animals are known per channel).
2. `bash scripts/slurm/setup_env.sh` once.
3. Check the dataset: `python -m eeg_seizure_analyzer.ml.train_unet --data-dir /path/to/edfs --analyze`.
4. Train: `EDF_DIR=/path/to/edfs MODEL_NAME=my_unet sbatch scripts/slurm/train_unet.sbatch`,
   then the classifier the same way. Models land in `~/.eeg_seizure_analyzer/models/`;
   copy them to the same folder on your computer to use them in the app.
5. Detect: `sbatch scripts/slurm/detect_all.sbatch` with `EDF_DIR`, `DB_PATH` and `MODEL` set,
   and copy the database to `~/.eeg_seizure_analyzer/projects/` to review it in Results.

Every setting in the job scripts is an environment variable with a default; override any of
them on the `sbatch` line. Run the Python modules with `--help` for all options.

**Same split for both stages.** The detector and classifier should use the same `VAL_MODE`
(`animal`, `recording` or `temporal`), or the cascade's validation numbers are not coherent.

**Detection is CPU work.** Inference is bound by reading and resampling EDFs, so a whole CPU
node with many workers is faster and cheaper than a GPU.
