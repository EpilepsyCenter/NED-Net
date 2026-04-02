# NED-Net

**Neural Event Detection Network** — a desktop application for detecting and quantifying seizures and interictal spikes in mouse EEG recordings.

NED-Net combines classical signal-processing algorithms (spike-train detection, line-length/energy thresholding) with a trainable 1D U-Net deep-learning model. It ships as a multi-tab Dash web app that runs locally in your browser.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)

---

## Features

| Area | What it does |
|---|---|
| **File loading** | EDF and ADICHT (LabChart) files with multi-rate channel support, auto-pairing of EEG + activity channels |
| **Viewer** | Scrollable multi-channel EEG with bandpass/notch filtering, min/max downsampling, activity overlay, and synchronized video playback |
| **Seizure detection** | Spike-train method with HVSW / HPD / convulsive subtype classification, boundary refinement, quality scoring, and seizure burden metrics |
| **Spike detection** | Interictal spike detection with z-score thresholding, prominence/width filtering, and spike-rate metrics |
| **ML training** | Annotate events in-app, build datasets, and train a 1D U-Net (SeizureUNet) with live epoch-by-epoch progress |
| **ML detection** | Run trained models on new recordings with post-hoc spectral feature enrichment |
| **ML results** | Events table with filters (duration, confidence, frequency), click-to-inspect with EEG/spectrogram/power plots, per-animal statistics, spike-train comparison, CSV export |
| **Video sync** | Convert LabChart WMV recordings to MP4, synchronized playhead overlay on EEG traces |
| **Export** | CSV and JSON export of seizure events, spike events, quality metrics, and validation reports |

---

## Installation

### Prerequisites

- **Python 3.9+** (3.10 or later recommended)
- **ffmpeg** (optional, for video conversion)
- **PyTorch** (optional, for ML training and detection — see [ML extras](#ml-extras) below)

### 1. Clone the repository

```bash
git clone https://github.com/marcoledri/NED-Net.git
cd NED-Net
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install the package

```bash
pip install -e .
```

This installs the core dependencies:

| Package | Purpose |
|---|---|
| numpy | Numerical computing |
| scipy | Signal processing, filtering |
| pyedflib | EDF file reading/writing |
| pandas | Tabular data handling |
| plotly | Interactive graphs |
| matplotlib | Static plots |
| streamlit | Legacy UI (Streamlit version) |

### 4. Install the Dash app dependencies

The primary UI is a Dash application. Install its dependencies:

```bash
pip install dash dash-bootstrap-components dash-ag-grid
```

### ML extras

To use the machine-learning pipeline (training, detection, results), install PyTorch and scikit-learn:

```bash
pip install torch scikit-learn
```

On **Apple Silicon** Macs, PyTorch will automatically use the MPS (Metal) GPU backend for accelerated training.

For GPU support on other platforms, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate CUDA version.

### Windows extras (LabChart files)

To read ADICHT (LabChart) files on Windows:

```bash
pip install "eeg-seizure-analyzer[windows]"
```

### Development extras

```bash
pip install "eeg-seizure-analyzer[dev]"
```

---

## Quick start

### Launch the Dash app

```bash
python -m eeg_seizure_analyzer.dash_app.main
```

Open http://127.0.0.1:8050 in your browser.

### Launch the Streamlit app (legacy)

```bash
streamlit run eeg_seizure_analyzer/app/main.py
```

---

## Usage guide

### Loading a recording

1. Go to the **Upload** tab
2. Enter the path to your EDF file (or browse)
3. Select the EEG and activity channels to load
4. Click **Load**

NED-Net auto-detects channel pairings (e.g. `LH` + `LH_EMG`) and handles files with mixed sampling rates.

### Seizure detection (spike-train method)

1. Go to **Detection > Seizures**
2. Adjust detection parameters or use defaults:
   - **Spike front-end**: amplitude threshold, min/max width, prominence, refractory period
   - **Train grouping**: max inter-spike interval, min spikes per train, min duration
   - **Subtype criteria**: HVSW, HPD, convulsive thresholds
   - **Boundary refinement**: signal-based (RMS envelope), spike-density, or none
3. Click **Run Detection**
4. Review results: event list, seizure burden metrics, click any event to inspect
5. Save detected events and parameter presets

### Interictal spike detection

1. Go to **Detection > Spikes**
2. Configure bandpass filter (default 10-70 Hz), z-score threshold, and width limits
3. Click **Run Detection**
4. Review spike rate metrics and event list

### ML workflow

#### 1. Annotate training data

- Go to **Training > Seizure**
- Load a recording and its spike-train detections
- Scroll through events and **Confirm**, **Reject**, or **Skip** each one
- Annotations are saved to `{filename}_ned_annotations.json`

#### 2. Build a dataset and train

- Go to **Machine Learning > Dataset**
- Scan a folder containing annotated EDF files
- Select files to include, then **Save Dataset**
- Configure hyperparameters:
  - **Epochs** (default 50)
  - **Batch size** (default 8)
  - **Learning rate** (default 1e-3)
  - **Patience** for early stopping (default 8)
  - **Positive weight** for class imbalance (default 5.0)
  - **Negative ratio** — ratio of background-to-seizure chunks (default 3.0)
- Click **Start Training** — progress bar shows epoch metrics (loss, F1, precision, recall) in real time
- Trained models are saved to `~/.eeg_seizure_analyzer/models/`

#### 3. Run detection on new recordings

- Go to **Machine Learning > Detection**
- Select a trained model from the dropdown (shows F1 score and dataset info)
- Adjust inference settings:
  - **Threshold** (default 0.5)
  - **Min duration** (default 3 s)
  - **Merge gap** (default 1 s)
- Click **Run ML Detection**
- Results are saved as `{filename}_ned_ml_detections.json` and auto-loaded when the file is reopened

#### 4. Inspect results

- Go to **Machine Learning > Results**
- View the events table with quality metrics (confidence, spectral entropy, peak frequency, signal-to-baseline, theta/delta ratio)
- Filter by channel, duration, confidence, or frequency
- Click any row to inspect the event with EEG trace, spectrogram, and power-over-time plots
- Compare ML detections with spike-train results (matched / ML-only / spike-train-only)
- View per-animal statistics
- Export to CSV

### Video synchronization

1. Go to **Tools > Video Converter**
2. Point to the folder containing LabChart WMV recordings
3. Convert to a single MP4 file
4. The video player appears in the Viewer and inspector panels, synchronized with EEG traces

---

## ML model architecture

NED-Net uses a **1D U-Net** (SeizureUNet) for per-sample seizure segmentation:

```
Input:  (batch, n_channels, n_samples) at 250 Hz
Output: (batch, 1, n_samples) — seizure probability per sample
```

- **Encoder**: progressive 2x downsampling with skip connections (configurable depth, default 4)
- **Decoder**: transposed convolutions with skip-connection concatenation
- **Loss**: combined Dice + BCE with adjustable positive weight for class imbalance
- **Validation**: train/val split by animal ID to prevent data leakage
- **Early stopping** with ReduceLROnPlateau scheduler

---

## File formats

### Input
| Format | Extension | Platform | Notes |
|---|---|---|---|
| EDF / EDF+ | `.edf` | All | Primary format, includes annotations |
| ADICHT | `.adicht` | Windows | Requires `adi-reader` + `cffi` |

### Output (sidecar JSON files)
| File | Contents |
|---|---|
| `{name}_ned_spikes.json` | Detected interictal spikes |
| `{name}_ned_detections.json` | Spike-train seizure detections |
| `{name}_ned_annotations.json` | Manual confirm/reject labels |
| `{name}_ned_ml_detections.json` | ML model predictions with quality metrics |

---

## Project structure

```
eeg_seizure_analyzer/
├── dash_app/                # Dash web application
│   ├── main.py              # App entrypoint, routing, layout
│   ├── state.py             # Server-side session state
│   ├── pages/               # Tab modules
│   │   ├── upload.py        # File loading and channel selection
│   │   ├── viewer.py        # Multi-channel EEG viewer
│   │   ├── seizures.py      # Spike-train seizure detection
│   │   ├── spikes.py        # Interictal spike detection
│   │   ├── training.py      # Annotation / labeling UI
│   │   ├── ml_datasets.py   # Dataset builder + model training
│   │   ├── ml_detection.py  # ML inference on recordings
│   │   ├── ml_results.py    # Results table, filters, inspector
│   │   └── tools.py         # Video converter
│   └── assets/              # JS (video sync) and CSS
├── detection/               # Detection algorithms
│   ├── seizure_detector.py  # Line-length / energy method
│   ├── spike_train.py       # Spike-train seizure detection
│   ├── spike_detector.py    # Interictal spike detection
│   ├── confidence.py        # Quality metrics and scoring
│   └── activity.py          # Movement artifact flagging
├── ml/                      # Machine learning pipeline
│   ├── model.py             # SeizureUNet architecture
│   ├── dataset.py           # Dataset loading and augmentation
│   ├── train.py             # Training loop with validation
│   └── predict.py           # Inference on new recordings
├── processing/              # Signal processing
│   ├── preprocess.py        # Bandpass, notch, artifact rejection
│   ├── features.py          # Line-length, energy features
│   └── spectral.py          # PSD, band power, spectrograms
├── io/                      # File I/O
│   ├── edf_reader.py        # EDF/EDF+ reader
│   ├── channel_ids.py       # Channel ID management
│   └── dataset_store.py     # Training dataset persistence
├── config.py                # Default parameters and frequency bands
└── export/                  # CSV/JSON export utilities
```

---

## Configuration

Detection parameters are defined as dataclasses in `eeg_seizure_analyzer/config.py` and can be tuned in the UI:

- **SeizureDetectionParams** — line-length/energy thresholds, min duration, merge gap, baseline method
- **SpikeDetectionParams** — bandpass range, z-score threshold, prominence, width, refractory period
- **SpikeTrainSeizureParams** — full spike-train pipeline: spike front-end, train grouping, subtype criteria, boundary refinement
- **PreprocessParams** — notch filter (50/60 Hz), artifact threshold, filter order

Frequency bands are configured for mouse EEG: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma-low (30-50 Hz), gamma-high (50-100 Hz).

---

## CLI tools

### Convert ADICHT to EDF (Windows)

```bash
eeg-convert input.adicht output.edf
```

---

## License

This project is currently under development. License information will be added in a future release.

---

## Citation

If you use NED-Net in your research, please cite:

> Ledri, M. (2026). NED-Net: Neural Event Detection Network for automated seizure detection in mouse EEG. GitHub. https://github.com/marcoledri/NED-Net
