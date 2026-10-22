# NED-Net

**Neural Event Detection Network**: a desktop application for detecting, reviewing and
quantifying seizures and interictal spikes in long-term rodent EEG.

NED-Net offers two routes to detection:
- **rule-based detectors** that need no training: Spike-Train, Spectral Band, Autocorrelation
  and Ensemble;
- **a trained two-stage model**: a 1-D U-Net finds seizures, and a convulsive classifier
  labels each one convulsive or not.

Detections are reviewed in the app, and the reviewed events become training data for the
next model. Retraining on each new cohort is part of the intended workflow. Results go to
per-project databases for group statistics.

It runs locally as a Dash app, in its own desktop window or in a browser.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Licence](https://img.shields.io/badge/licence-GPL--3.0-green)

---

## Features

| Area | What it does |
|---|---|
| **Loading** | EDF/EDF+ with multi-rate channels and automatic EEG–activity pairing. Animal ID, cohort and group per channel, from the table or a batch metadata template. Load one file or a folder. The start screen lists recent files and folders. ADICHT (LabChart) via the built-in converter. |
| **Viewer** | Scrollable multi-channel EEG with band-pass/notch filters, min/max downsampling, activity overlay, event overlays, synchronised video, and vector SVG export of the current view. |
| **Seizure detection** | Four rule-based methods plus the trained U-Net. The U-Net runs with an optional convulsive classifier and hysteresis boundaries. Result filters, an event inspector, CSV export, and "Add to project database". |
| **Interictal spikes** | Rule-based or U-Net spike detection with morphology features, spike-rate metrics and filters. |
| **Event inspector** | Pan through the recording, change the context (±10–120 s), and click an amplitude overview strip (±5 min, ±15 min or the whole recording) to jump. Step through events with Prev/Next or the keyboard. |
| **Training (review)** | Confirm or reject candidates, edit boundaries, flag convulsive seizures, add or remove events by hand, jump to any event from the event list, autoscale with `A`. Every candidate shows where it came from. Confirmed and rejected events become training data. Seizure-free background stretches are reviewed the same way. |
| **Dataset & Model** | Build datasets from reviewed recordings and train the U-Net or the convulsive classifier, with live progress. **Blind Review** checks a model's detections on held-out recordings without showing its scores. |
| **Batch Analysis** | Run a trained model on one file, a folder (batch) or a watched folder (live monitoring). Choosing a model fills in its recommended operating point. Output goes to a project database, the training review queue, or both. HVSW/HPD subtype classification. |
| **Results** | Per-project statistics: group comparison (raw or per animal-hour), per-animal summary with include/valid-until controls, longitudinal, distributions, daily burden and circadian plots, an interictal-spike panel, and CSV/XLSX export. |
| **Tools** | WMV→MP4 video conversion, and ADICHT→EDF conversion (Windows). |

---

## Installation

### Prerequisites

- **Python 3.9+** (3.11 or later recommended)
- **ffmpeg** (optional, for video conversion)
- **Linux only:** `zenity` or `kdialog` for the Browse dialogs. Most desktops ship one.

### Install

```bash
git clone https://github.com/EpilepsyCenter/NED-Net.git
cd NED-Net
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[desktop,ml]"
```

Extras:

| Extra | Adds | Needed for |
|---|---|---|
| (core) | numpy, scipy, pandas, pyEDFlib, plotly, matplotlib, dash, dash-bootstrap-components, dash-ag-grid, openpyxl, watchdog | everything else, incl. rule-based detection and live monitoring |
| `ml` | PyTorch (plus scikit-learn/joblib for retained research code) | training models and running the U-Net / convulsive classifier |
| `desktop` | pywebview, waitress | the native desktop window |
| `windows` | adi-reader | opening ADICHT (LabChart) files on Windows |

PyTorch uses the Apple-silicon GPU (MPS) automatically. For an NVIDIA GPU, install the CUDA
build of PyTorch first, following [pytorch.org](https://pytorch.org/get-started/locally/).

`pip install -r requirements.txt` installs the core and desktop dependencies without the
editable package.

---

## Quick start

**Desktop app.** Double-click the launcher for your platform, or run `nednet`
(also `python -m eeg_seizure_analyzer.dash_app.desktop`):

| Platform | Launcher |
|---|---|
| macOS | `NED-Net.app` |
| Windows | `Start NED-Net (Native).bat` |
| Linux | `start-nednet-native.sh` (first time: `chmod +x start-nednet-native.sh`) |

The desktop app serves on `127.0.0.1:8051`, or a free port if that one is taken.

**Packaged app.** Self-contained apps (no Python needed) for macOS, Windows and Linux are on
the **Releases** page: pushing a version tag builds them on GitHub. See
[docs/PACKAGING.md](docs/PACKAGING.md).

**Browser.**

```bash
python -m eeg_seizure_analyzer.dash_app.main      # then open http://127.0.0.1:8050
```

### Typical workflows

**Rule-based, one recording**
1. **Load** a recording and set the animal IDs.
2. **Seizures** (Detection in the sidebar): pick a method, tune it, detect.
3. Inspect events, filter them, export CSV, or add them to the project database.

**Trained model, a whole study**
1. **Batch Analysis:** choose a project and a model (its recommended thresholds are filled
   in), then run on a folder.
2. **Results:** group and per-animal statistics.
3. On a new cohort or implant, send output to the **training review queue** as well. Review a
   sample in **Seizure Annotation**, then retrain in **Dataset & Model**. Models transfer poorly to
   animals they have not seen, so plan to retrain on each new cohort.

---

## App layout

A sidebar lists the pages; the toolbar shows the open recording and a status bar what is saved.
The Viewer, detection and annotation pages keep their settings in a panel on the right that can
be resized and hidden. The desktop app has native menus (File, Go, Tools, Help) and keyboard
shortcuts (`⌘`/`Ctrl` + `1`–`9` for the pages).

| Sidebar group | Pages | Purpose |
|---|---|---|
| Recording | **Load**, **Viewer** | Start screen, loading, channel selection, IDs and tags; multi-channel EEG browser, filters, overlays, SVG export, video |
| Detection | **Seizures**, **Interictal Spikes** | Rule-based and U-Net detection, filters, inspector, export |
| Training | **Seizure Annotation**, **Spike Annotation**, **Background** | Review: label candidates, edit boundaries, add or remove events; seizure-free background stretches |
| Model | **Dataset & Model**, **Blind Review**, **Batch Analysis**, **Results** | Build datasets and train models; blind check of a model's detections; model inference (single file, batch, live); per-project statistics, charts and exports |
| Tools | **Video Converter**, **ADICHT → EDF** | Conversion utilities |

---

## Detection pipeline

### Rule-based seizure detectors
- **Spike-Train:** spike detection with a baseline z-score, grouping into trains, and RMS
  boundary refinement.
- **Spectral Band:** power in a 17–25 Hz band relative to 1–50 Hz (spectral band index).
- **Autocorrelation:** range autocorrelation (rhythmicity) and spike frequency; both must exceed
  their thresholds.
- **Ensemble:** a vote across the methods, with union or intersection boundaries.

Each event gets quality metrics (local baseline ratio, amplitude, spike frequency and others),
which the result filters use.

### Trained two-stage model (seizures)
1. **Stage 1: U-Net detection.** The U-Net outputs a seizure probability for every sample (EEG
   at 250 Hz, 60 s windows, 15 s overlap). Events use **hysteresis**:
   - a *core* must exceed the **detection threshold**;
   - the event then grows outward until the probability falls below the lower **boundary
     threshold**;
   - then come a minimum duration and a merge gap.
2. **Stage 2: convulsive classifier (optional).** A 1-D CNN scores a 60 s crop around each
   detected event and sets the convulsive flag. It **labels events and never removes them**.
   Without it, the U-Net's own convulsive output decides.

**Operating points.** Choosing a model in Batch Analysis, or in the Seizures page's U-Net settings,
fills in its recommended operating point (`ml/operating_points.py`). In order of preference:
- the model's own `operating_point` entry in its `metadata.json`, for a point you validated
  yourself (detection and boundary thresholds, convulsive classifier and its threshold);
- the threshold that maximised event F1 during training.

NED-Net ships no trained models: you train them on your own annotated recordings.

**HVSW / HPD.** Non-convulsive events can be sub-classified as HVSW (slow, high-amplitude
waves: low dominant frequency, high slow-wave index) or HPD (fast periodic discharges: high
dominant frequency, high high-frequency index). The thresholds are set in Analysis →
Advanced.

### Models and training
- **U-Net (Stage 1):**
  - 1-D encoder–decoder with skip connections: 4 levels, 32 base filters, kernel 7;
  - one EEG channel in, optionally the activity channel too;
  - two outputs per sample: seizure and convulsive.
- **Convulsive classifier (Stage 2):** the U-Net's encoder blocks, global average pooling and
  a linear head, on 60 s crops.
- **Training:**
  - Dice + BCE loss with a positive-class weight, AdamW, ReduceLROnPlateau, early stopping on
    patience;
  - the checkpoint kept is the epoch with the best **event F1 over swept thresholds**, and
    that threshold is saved as the model's training optimum;
  - negatives are reviewed rejections (hard negatives) or random background;
  - validation splits by animal (default), by recording, or temporally (each animal's later
    recordings).
- **Recommended recipe:** learning rate 1e-3, batch 8, positive weight 5, fp32. On CUDA, mixed
  precision (bf16) is the default; pass `--fp32` for full fp32.

---

## Projects and databases

Analysis and **Add to project database** write to the **active project**. Results reads from
it.
- Each project is a SQLite file in `~/.eeg_seizure_analyzer/projects/<name>.db`. The project
  "Default" is `~/.eeg_seizure_analyzer/analysis.db`.
- Create, switch and delete projects in Analysis. Use one project per experiment.
- The database stores events, the files and animals analysed, and recorded time per animal,
  so rates can be expressed per animal-hour.

---

## Files

### Next to each recording

| File | Contents | Written by |
|---|---|---|
| `<name>_ned_annotations.json` | **Seizure review state, and the only source of training data.** Every candidate with its label (pending / confirmed / rejected), edits, notes and origin. | Training; a Seizures-page run (as pending); Analysis with the review-queue output |
| `<name>_ned_detections.json` | The last seizure run on the Seizures page: events, parameters and filter settings, so the page can restore itself. Training edits never change it. | Seizures |
| `<name>_ned_spikes.json` | The last interictal-spike run | Interictal Spikes |
| `<name>_ned_spike_annotations.json` | Spike review state | Spike Annotation |
| `<name>_ned_background_annotations.json` | Background stretches and their review state | Seizures (Propose background); Background |
| `<name>_ned_channels.json` | Animal ID per channel | Load |
| `<name>_ned_meta.json` | Cohort and group per channel | Load |

Model training uses only **confirmed** events (positives) and **rejected** events (hard
negatives). Pending candidates never reach a model until someone reviews them.

### `~/.eeg_seizure_analyzer/`

| Path | Contents |
|---|---|
| `models/<name>/` | Trained models: `best_model.pt` and `metadata.json` (configuration, metrics, threshold) |
| `datasets/` | Saved dataset definitions |
| `projects/`, `analysis.db`, `active_project.txt` | Project databases and the active project |
| `defaults.json`, `spike_defaults.json` | Your saved detection parameters (Parameters menu) |
| `recent.json`, `window.json`, `results_state.json` | Recent files, window size, Results view state |

---

## Headless use and HPC

The trainers and batch detectors run without the UI, for clusters (example SLURM job scripts
are in `scripts/slurm/`):

```bash
# Stage 1: U-Net (--analyze reports the class balance and split, then exits)
python -m eeg_seizure_analyzer.ml.train_unet --data-dir /path/to/edfs \
    --model-name my_unet --neg-source hard --val-mode animal --fp32

# Stage 2: convulsive classifier
python -m eeg_seizure_analyzer.ml.train_convulsive --data-dir /path/to/edfs \
    --model-name my_conv --conv-neg-from-rejected

# Batch detection into a project database, in parallel across cores
python scripts/slurm/detect_batch.py --edf-dir /path/to/edfs \
    --db-path ~/.eeg_seizure_analyzer/projects/my_project.db \
    --model my_unet --convulsive-model my_conv --workers 48
```

Run any of them with `--help` for all options. `scripts/slurm/detect_spikes_batch.py` does
the same for interictal spikes. ADICHT files convert from the command line with
`eeg-convert input.adicht output.edf` (Windows).

---

## Project structure

```
eeg_seizure_analyzer/
├── dash_app/
│   ├── main.py, desktop.py        # app, sidebar navigation; native window, menus, splash
│   ├── pages/                     # one module per page (upload, viewer, seizures, spikes,
│   │                              #   training, training_spikes, training_background,
│   │                              #   ml_datasets, blind_review, analysis, results, tools,
│   │                              #   adicht_converter)
│   ├── components.py              # shared UI pieces (two-pane pages, parameter controls,
│   │                              #   filter panel, menus)
│   ├── inspector_view.py          # pan / context / overview strip / grid navigation
│   ├── file_dialogs.py            # native file and folder dialogs per platform
│   ├── recent.py                  # recent files on the start screen
│   └── assets/                    # CSS, user manual, screenshots
├── detection/                     # rule-based detectors, boundaries, quality metrics, burden
├── ml/
│   ├── model.py, convulsive_model.py           # U-Net and Stage-2 classifier
│   ├── dataset.py, train.py, predict.py        # windows, training loop, inference
│   ├── train_unet.py, train_convulsive.py      # headless trainers
│   ├── operating_points.py                     # recommended thresholds per model
│   └── spike_*.py                              # spike model
├── io/                            # EDF/ADICHT reading, sidecars, channel IDs, datasets
├── processing/                    # filtering, features, spectra, activity
├── analysis.py                    # single / batch / live analysis (shared by UI and scripts)
├── db.py                          # project databases
└── config.py                      # default parameters
scripts/slurm/                     # example SLURM job scripts (setup, training, batch detection)
packaging/                         # PyInstaller recipe, app entry point, Windows installer (docs/PACKAGING.md)
```

Earlier research code is kept in `ml/` but not used by the app: the BENDR foundation-model
experiments (`bendr_*`, `train_bendr.py`, CLI `eeg-bendr-pretrain`) and an event re-ranker
(`train_reranker.py`). Both were evaluated and set aside.

---

## User manual

The full manual is [`eeg_seizure_analyzer/dash_app/assets/USER_MANUAL.md`](eeg_seizure_analyzer/dash_app/assets/USER_MANUAL.md).
The app shows the same file at **Help** in the sidebar (and **Help › NED-Net Manual** in the desktop app), and it also works offline.

---

## Licence

NED-Net is released under the **GNU General Public License v3.0**. See [LICENSE](LICENSE).

---

## Citation

If you use NED-Net, please cite the software:

> Ledri, M. (2026). NED-Net: Neural Event Detection Network for automated seizure detection in
> rodent EEG (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.23372245

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23372245.svg)](https://doi.org/10.5281/zenodo.23372245)

Each release is archived on Zenodo with its own DOI. To cite all versions, use the concept DOI
[10.5281/zenodo.23372245](https://doi.org/10.5281/zenodo.23372245), which resolves to the
latest release. GitHub's "Cite this repository" button (from [CITATION.cff](CITATION.cff))
gives the current reference.
