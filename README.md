# NED-Net

**Neural Event Detection Network** — a desktop application for detecting and quantifying seizures and interictal spikes in rodent EEG recordings.

NED-Net combines multiple rule-based detection algorithms (Spike-Train, Spectral Band, Autocorrelation, Ensemble) with a trainable 1D U-Net deep-learning model. It ships as a multi-tab Dash web app that runs locally in your browser.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)

---

## Features

| Area | What it does |
|---|---|
| **File loading** | EDF files (primary) with multi-rate channel support, auto-pairing of EEG + activity channels. ADICHT (LabChart) files can be converted to EDF via the built-in converter |
| **Viewer** | Scrollable multi-channel EEG with bandpass/notch filtering, min/max downsampling, activity overlay, and synchronized video playback |
| **Seizure detection** | Four methods: Spike-Train, Spectral Band (17–25 Hz), Autocorrelation, and Ensemble. Boundary refinement, quality scoring, and seizure burden metrics |
| **Spike detection** | Interictal spike detection with z-score thresholding, prominence/width filtering, morphology analysis, and spike-rate metrics |
| **Training** | In-app annotation interface for reviewing and labeling detected events to build training datasets |
| **Dataset / Model** | Build datasets from annotated files and train a 1D U-Net (2-class output for seizures, binary for spikes) with live progress |
| **Analysis** | Run trained CNN models on single files, batch folders, or live-monitored directories. HVSW/HPD subtype classification |
| **Results** | Aggregated results across all analysis runs with daily seizure burden, circadian charts, and CSV export |
| **Video sync** | Convert LabChart WMV recordings to MP4, synchronized playhead overlay on EEG traces |
| **Export** | CSV export of seizure events and interictal spikes with configurable field groups (core, morphology, context, spectral) |

---

## Installation

### Prerequisites

- **Python 3.9+** (3.10 or later recommended)
- **ffmpeg** (optional, for video conversion)

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

Core dependencies:

| Package | Purpose |
|---|---|
| numpy | Numerical computing |
| scipy | Signal processing, filtering |
| pyedflib | EDF file reading/writing |
| pandas | Tabular data handling |
| plotly | Interactive graphs |
| dash | Web application framework |
| dash-bootstrap-components | UI components |
| dash-ag-grid | Interactive data tables |

### ML extras

To use the machine-learning pipeline (model training and CNN-based analysis), install the ML dependencies:

```bash
pip install -e ".[ml]"
```

This installs **PyTorch**. On Apple Silicon Macs (M1/M2/M3/M4), PyTorch automatically uses the Metal GPU backend for accelerated training. For NVIDIA GPU support on Linux/Windows, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Windows extras (LabChart files)

To directly open and convert ADICHT (LabChart) files on Windows:

```bash
pip install -e ".[windows]"
```

---

## Quick start

```bash
python -m eeg_seizure_analyzer.dash_app.main
```

Open http://127.0.0.1:8050 in your browser.

### Basic workflow

1. **Load** → Enter the path to your EDF file, select channels, click Load
2. **View** → Browse multi-channel EEG traces in the Viewer tab
3. **Detect** → Go to Detection, choose a method, adjust parameters, click Detect Seizures
4. **Review** → Inspect results in the table, click events to view EEG + spectrogram
5. **Export** → Download CSV with selected field groups
6. **Train** → Annotate events in the Training tab, build datasets, train a U-Net model
7. **Analyse** → Run trained models on new recordings in the Analysis tab

---

## Tab structure

| Tab | Subtabs | Description |
|---|---|---|
| **Load** | — | File loading and channel selection |
| **Viewer** | — | Multi-channel EEG browser with filtering |
| **Detection** | Seizure, Interictal Spikes | Rule-based detection with configurable parameters |
| **Training** | Seizure, Interictal Spikes | Annotation interface for labeling events |
| **Dataset / Model** | Dataset | Build datasets and train U-Net models |
| **Analysis** | — | CNN inference (single file, batch, live monitoring) with HVSW/HPD classification |
| **Results** | — | Aggregated analysis results with charts and filters |
| **Tools** | Video Converter, ADICHT → EDF | File conversion utilities |

---

## ML model architecture

NED-Net uses a **1D U-Net** for per-sample event segmentation:

- **Seizure model**: 2-class output (seizure probability + convulsive probability)
- **Spike model**: binary output (spike probability)
- **Encoder**: 4-level progressive 2× downsampling with skip connections
- **Decoder**: transposed convolutions with skip-connection concatenation
- **Loss**: combined Dice + BCE with adjustable positive weight
- **Validation**: train/val split by animal ID to prevent data leakage
- **Training**: mixed precision (AMP), ReduceLROnPlateau scheduler, early stopping

---

## File formats

### Input

| Format | Extension | Platform | Notes |
|---|---|---|---|
| EDF / EDF+ | `.edf` | All | Primary format for all features |
| ADICHT | `.adicht` | Windows | Convert to EDF first via Tools tab or `eeg-convert` CLI |

### Output

| File | Contents |
|---|---|
| `{name}_ned_detections.json` | Rule-based seizure detections |
| `{name}_ned_spikes.json` | Detected interictal spikes |
| `{name}_ned_spike_annotations.json` | Spike annotation labels |
| `{name}_ned_annotations.json` | Seizure annotation labels |
| `results.db` | SQLite database with CNN analysis results |

---

## Project structure

```
eeg_seizure_analyzer/
├── dash_app/                    # Dash web application
│   ├── main.py                  # App entrypoint, routing, layout
│   ├── server_state.py          # Server-side session state
│   ├── components.py            # Shared UI components and helpers
│   ├── pages/                   # Tab modules
│   │   ├── upload.py            # File loading and channel selection
│   │   ├── viewer.py            # Multi-channel EEG viewer
│   │   ├── seizures.py          # Seizure detection (4 methods)
│   │   ├── spikes.py            # Interictal spike detection
│   │   ├── training.py          # Seizure annotation UI
│   │   ├── training_spikes.py   # Spike annotation UI
│   │   ├── ml_datasets.py       # Dataset builder + model training
│   │   ├── analysis.py          # CNN inference + HVSW/HPD classification
│   │   ├── results.py           # Aggregated results and charts
│   │   ├── tools.py             # Video converter + ADICHT converter
│   │   └── adicht_converter.py  # ADICHT → EDF conversion logic
│   └── assets/                  # CSS, JS, manual, screenshots
├── detection/                   # Detection algorithms
│   ├── base.py                  # DetectedEvent dataclass
│   ├── spike_train_seizure.py   # Spike-train seizure detection
│   ├── spectral_band_seizure.py # Spectral band (17–25 Hz) method
│   ├── autocorrelation_seizure.py # Autocorrelation method
│   ├── ensemble_seizure.py      # Ensemble (vote across methods)
│   ├── seizure.py               # Legacy seizure detector
│   ├── spike.py                 # Interictal spike detection
│   ├── spike_utils.py           # Spike feature extraction
│   ├── boundary_utils.py        # Event boundary refinement
│   ├── confidence.py            # Quality metrics and scoring
│   └── burden.py                # Seizure burden calculation
├── ml/                          # Machine learning pipeline
│   ├── model.py                 # SeizureUNet architecture
│   ├── dataset.py               # Seizure dataset loading
│   ├── spike_dataset.py         # Spike dataset loading
│   ├── train.py                 # Training loop with validation
│   ├── predict.py               # Seizure inference
│   ├── spike_predict.py         # Spike inference
│   └── spike_train.py           # Spike model training
├── processing/                  # Signal processing
│   ├── preprocess.py            # Bandpass, notch, artifact rejection
│   ├── features.py              # RMS, line-length, energy features
│   ├── spectral.py              # PSD, band power, spectrograms
│   └── activity.py              # Activity channel processing
├── io/                          # File I/O
│   ├── base.py                  # EEGRecording dataclass
│   ├── edf_reader.py            # EDF/EDF+ reader
│   ├── adicht_reader.py         # ADICHT reader (Windows)
│   ├── adicht_to_edf.py         # ADICHT → EDF converter
│   ├── channel_ids.py           # Channel / animal ID management
│   ├── persistence.py           # Detection result save/load
│   ├── annotations.py           # Annotation file handling
│   ├── annotation_store.py      # Annotation persistence
│   ├── dataset_store.py         # Training dataset persistence
│   └── batch_metadata.py        # Batch Excel metadata parsing
├── validation/                  # Validation utilities
│   └── metrics.py               # Detection quality metrics
├── analysis.py                  # CNN analysis orchestration
├── db.py                        # SQLite results database
├── config.py                    # Default parameters and frequency bands
└── export/                      # CSV/JSON export utilities
```

---

## User manual

A comprehensive user manual is available at `eeg_seizure_analyzer/dash_app/assets/USER_MANUAL.md` and is also served in-app at http://127.0.0.1:8050/assets/manual.html when the app is running.

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

> Ledri, M. (2026). NED-Net: Neural Event Detection Network for automated seizure detection in rodent EEG. GitHub. https://github.com/marcoledri/NED-Net
