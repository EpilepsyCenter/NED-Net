# NED-Net User Manual

**Neural Event Detection Network** — v0.1

A desktop application for automated detection, annotation, and machine-learning-based analysis of seizures and interictal spikes in long-term EEG recordings.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Getting Started](#3-getting-started)
4. [Loading Data](#4-loading-data)
5. [EEG Viewer](#5-eeg-viewer)
6. [Seizure Detection](#6-seizure-detection)
7. [Result Filters & Inspector](#7-result-filters--inspector)
8. [Exporting Detected Events](#8-exporting-detected-events)
9. [Training / Annotation](#9-training--annotation)
10. [Interictal Spike Detection & Annotation](#10-interictal-spike-detection--annotation)
11. [Dataset & Model Training](#11-dataset--model-training)
12. [Analysis (CNN Detection)](#12-analysis-cnn-detection)
13. [Results](#13-results)
14. [Batch Processing](#14-batch-processing)
15. [Saving & Recalling Parameters](#15-saving--recalling-parameters)
16. [Tools](#16-tools)
17. [File Reference](#17-file-reference)

---

## 1. Introduction

NED-Net is a browser-based desktop application for neuroscience researchers who work with long-term EEG recordings in rodent models of epilepsy. It provides:

- **Multiple rule-based seizure detection algorithms** (Spike-Train, Spectral Band, Autocorrelation) that can be run individually or combined via an Ensemble method.
- **Interictal spike detection** with configurable thresholds.
- **A visual annotation interface** for reviewing, confirming, or rejecting detected events to build training datasets.
- **A machine-learning pipeline** for training a 1D U-Net deep-learning model on your annotated data, with separate models for seizures (2-class: seizure + convulsive) and interictal spikes.
- **CNN-based analysis** with single-file, batch, and live-monitoring modes, including HVSW/HPD subtype classification.
- **Results aggregation** with daily seizure burden, circadian analysis, and CSV export.
- **Batch processing** across multiple recordings.

NED-Net runs locally on your machine — no data leaves your computer. It opens in your web browser but does not require an internet connection (except for the initial font loading, which is cached).

### Supported file formats

- **EDF** (European Data Format) — the primary format for all detection, training, and ML features.
- **ADICHT** (LabChart) — can be viewed and converted to EDF using the built-in converter tool (Windows only for direct reading; cross-platform for pre-exported files).

---

## 2. Installation

This section guides you through installing NED-Net step by step. No advanced Python knowledge is required.

### 2.1 Prerequisites

You need **Python 3.9 or later** installed on your computer. To check if you already have Python:

1. Open a terminal (macOS/Linux) or Command Prompt (Windows).
2. Type:
   ```
   python --version
   ```
   or, on some systems:
   ```
   python3 --version
   ```
3. You should see something like `Python 3.10.12`. If you see `Python 3.9` or higher, you are ready. If not, install Python from [python.org](https://www.python.org/downloads/) or via [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

> **Tip for beginners:** If you are new to Python, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html). It is a lightweight Python distribution that makes managing packages easy and avoids conflicts with other software on your system.

### 2.2 Download NED-Net

Download or clone the NED-Net repository to your computer:

**Option A — Git clone (recommended):**
```bash
git clone https://github.com/marcoledri/NED-Net.git
cd NED-Net
```

**Option B — Download ZIP:**
1. Go to the NED-Net GitHub page.
2. Click the green "Code" button, then "Download ZIP".
3. Unzip the file and open a terminal in the resulting folder.

### 2.3 Create a virtual environment

A virtual environment keeps NED-Net's packages separate from other Python projects on your computer. This step is optional but strongly recommended.

**Using venv (standard Python):**
```bash
python -m venv .venv
```

Activate the virtual environment:
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

You should see `(.venv)` at the beginning of your terminal prompt, confirming the environment is active.

**Using Conda (alternative):**
```bash
conda create -n nednet python=3.10
conda activate nednet
```

### 2.4 Install NED-Net

With the virtual environment active, run:

```bash
pip install -e .
```

This installs NED-Net and all required packages:

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing |
| scipy | Signal processing and filtering |
| pyedflib | Reading/writing EDF files |
| pandas | Tabular data handling |
| plotly | Interactive plots |
| matplotlib | Static plots (export) |
| dash | Web application framework |
| dash-bootstrap-components | UI components |
| dash-ag-grid | Interactive data tables |

### 2.5 Install ML support (optional)

If you want to use the machine-learning features (model training and CNN-based detection), also install:

```bash
pip install -e ".[ml]"
```

This adds **PyTorch** and **scikit-learn**. On Apple Silicon Macs (M1/M2/M3/M4), PyTorch automatically uses the Metal GPU backend for accelerated training. For NVIDIA GPU support on Linux/Windows, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct CUDA version.

### 2.6 Install LabChart support (Windows only, optional)

To directly open and convert ADICHT (LabChart) files:

```bash
pip install -e ".[windows]"
```

### 2.7 Verify the installation

Run the app to verify everything is working:

```bash
python -m eeg_seizure_analyzer.dash_app.main
```

Open your web browser and go to **http://127.0.0.1:8050**. You should see the NED-Net landing page.

> **Troubleshooting:** If you see `ModuleNotFoundError`, make sure your virtual environment is activated (you should see `(.venv)` or `(nednet)` in your terminal prompt). If the port is in use, NED-Net will tell you — close any other instance first.

---

## 3. Getting Started

### 3.1 Launching the app

From your terminal (with the virtual environment active):

```bash
python -m eeg_seizure_analyzer.dash_app.main
```

Then open **http://127.0.0.1:8050** in your browser (Chrome or Firefox recommended).

### 3.2 The interface

NED-Net has three main areas:

![Screenshot: Full app interface showing sidebar, tab bar, and main content area](screenshots/interface_overview.png)

- **Sidebar (left):** Shows the loaded file name, recording info (channels, sampling rate, duration), detection status with per-method breakdown, annotation progress, and a light/dark theme toggle. A **Help** link at the bottom opens this manual.
- **Tab bar (top):** Navigate between sections. Some tabs have subtabs that appear in a second row when selected.
- **Main content area:** The active tab's content.

#### Tab structure

| Tab | Subtabs | Purpose |
|-----|---------|---------|
| **Load** | — | Load EDF files, assign animal IDs, pair channels |
| **Viewer** | — | Scrollable multi-channel EEG viewer |
| **Detection** | Seizure, Interictal Spikes | Rule-based detection with configurable parameters |
| **Training** | Seizure, Interictal Spikes | Review and annotate detected events |
| **Dataset / Model** | Dataset | Create training datasets and train CNN models |
| **Analysis** | — | Run trained CNN models (single file, batch, or live monitoring) |
| **Results** | — | Aggregated results across all analysis runs |
| **Tools** | Video Converter, ADICHT → EDF | File conversion utilities |

### 3.3 Theme toggle

NED-Net defaults to a light theme. To switch to dark mode, use the **Dark mode** toggle at the bottom of the sidebar. The theme applies instantly to all UI elements. Plotly figures update on the next tab switch or interaction.

![Screenshot: Theme toggle in sidebar footer](screenshots/theme_toggle.png)

---

## 4. Loading Data

### 4.1 Loading a single EDF file

1. On the landing page (or the **Load** tab), click **Load File**.
2. Enter the full path to your `.edf` file in the text field and click **Load**.

![Screenshot: Load tab with file path input](screenshots/load_single.png)

NED-Net reads the EDF header and displays:
- Number of channels, sampling rate, and recording duration.
- A channel table showing each channel's name, sampling rate, and physical unit.

### 4.2 Animal IDs

Each channel must be assigned an **Animal ID** before running detection. This links detected events to specific animals, which is essential for multi-animal recordings and for ML training.

- Fill in the **Animal ID** column in the channel table on the Load tab.
- All channels for the same animal should share the same ID (e.g., `Mouse_01`).

![Screenshot: Channel table with Animal ID column](screenshots/animal_ids.png)

### 4.3 Loading multiple files (batch/project)

Click **Load Multiple...** to load a folder of EDF files for batch processing. NED-Net scans the folder and lists all EDF files found. You can then run detection across all of them from the Detection tab (see [Batch Processing](#14-batch-processing)).

### 4.4 Pairing EEG with activity channels

If your recording includes activity/EMG channels alongside EEG channels, NED-Net can pair them automatically based on channel naming conventions, or you can set pairings manually on the Load tab. Paired activity channels appear as overlays in the Viewer and Training tabs.

### 4.5 Video association

If you have a synchronized video file (MP4) for the recording, you can associate it on the Load tab. The video playhead will sync with the EEG time position in the Viewer and Training tabs. If your video files are in WMV format, use the **Video Converter** tool (under Tools) to convert them to MP4 first.

---

## 5. EEG Viewer

The **Viewer** tab provides a scrollable, multi-channel EEG viewer.

![Screenshot: Viewer tab with EEG traces and time navigation](screenshots/viewer.png)

### 5.1 Time navigation

- Use the **time slider** to scroll through the recording.
- Set the **window size** (in seconds) to control how much data is visible at once.
- Click-drag on the plot to zoom into a specific time range.

### 5.2 Channel controls

- **Channel selection:** Choose which channels to display using the channel checkboxes.
- **Y-range:** Adjust the vertical scale of the EEG traces. A smaller value zooms in on the signal; a larger value shows more dynamic range.

### 5.3 Activity channel overlay

When activity channels are paired with EEG channels, they appear as a second trace overlaid on the EEG. The activity Y-range has its own min/max controls.

### 5.4 Detection shadows

After running seizure or spike detection, detected events appear as colored shadow regions behind the EEG traces, giving you a quick visual overview of where events were found.

---

## 6. Seizure Detection

The **Detection → Seizure** subtab is where you configure and run seizure detection. NED-Net offers four detection methods, each with its own set of parameters.

### 6.1 Choosing a detection method

Select a method from the **Method** dropdown at the top of the detection panel:

![Screenshot: Method selector dropdown](screenshots/method_selector.png)

| Method | Best for | How it works |
|--------|----------|-------------|
| **Spike-Train** | Seizures with clear spike-and-wave patterns | Detects individual spikes, then groups them into trains based on timing and density |
| **Spectral Band** | Seizures with sustained power increases in a frequency band | Monitors power in a target frequency band relative to a reference band |
| **Autocorrelation** | Seizures with rhythmic, repetitive patterns | Detects periodic structure in the EEG using autocorrelation scoring |
| **Ensemble** | Maximum sensitivity, combining multiple methods | Runs 2–3 methods and merges results by voting |

You can run multiple methods sequentially — results **accumulate** across methods. For example, running Spike-Train and then Spectral Band keeps both sets of detections. Re-running the same method replaces only that method's previous results.

### 6.2 Channel selection

Before detecting, select which channels to analyze using the **Channels** checklist. Only selected channels will be processed.

### 6.3 Spike-Train method

The Spike-Train detector works in three stages: (1) detect individual spikes, (2) group spikes into trains, (3) refine event boundaries.

#### Spike detection parameters

| Parameter | Default | Effect of increasing |
|-----------|---------|---------------------|
| **Bandpass low (Hz)** | 1.0 | Filters out more low-frequency drift. Increase if baseline is noisy |
| **Bandpass high (Hz)** | 50.0 | Filters out more high-frequency noise. Decrease if recording has high-frequency artifacts |
| **Spike amplitude (z-score)** | 3.0 | Z-score threshold: spike must exceed mean + z × std of the baseline. **Increase to reduce false positives; decrease for more sensitivity** |
| **Min amplitude (uV)** | 0.0 | Sets an absolute minimum spike height in microvolts. Useful if baseline is very low |
| **Spike prominence (x BL)** | 2.5 | Requires spikes to stand out more from surrounding signal. Increase to be more selective |
| **Max spike width (ms)** | 200 | Rejects wider deflections (e.g., slow waves). Decrease to require sharper spikes |
| **Min spike width (ms)** | 10 | Rejects very narrow spikes (likely artifacts). Increase if seeing noise spikes |
| **Refractory period (ms)** | 75 | Minimum time between successive spikes. Increase if double-counting spikes |

#### Train formation parameters

| Parameter | Default | Effect of increasing |
|-----------|---------|---------------------|
| **Max ISI (ms)** | 500 | Maximum inter-spike interval to keep spikes in the same train. **Increase to merge loosely spaced spikes; decrease for tighter trains** |
| **Min spikes** | 10 | Minimum number of spikes required to form a seizure. **Increase to reject short bursts** |
| **Min duration (s)** | 5.0 | Minimum seizure duration. Events shorter than this are discarded |
| **Min IEI (s)** | 3.0 | Minimum inter-event interval. Events closer than this are merged into one |

#### Baseline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Baseline method** | Percentile | How the baseline amplitude is estimated. "Percentile" uses a quiet portion of the signal; "Rolling" recomputes the baseline in sliding windows across the recording; "First N min" uses only the initial segment |
| **Baseline percentile** | 25 | Which percentile of signal amplitude to use as baseline (lower = quieter baseline) |
| **Baseline RMS window (s)** | 30 | Window size for RMS baseline calculation |

#### Boundary refinement

After a seizure is detected, its start and end times are refined to better match the actual onset and offset of ictal activity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Boundary method** | Signal | "Signal" uses RMS envelope; "Spike density" uses spike rate |
| **RMS window (ms)** | 100 | Window for RMS envelope smoothing |
| **RMS threshold (x BL)** | 3.0 | How much the RMS must exceed baseline to be considered ictal |
| **Max trim (s)** | 2.0 | Maximum amount the boundary can be trimmed inward |

When using **Spike density** boundary refinement:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Window (s)** | 2.0 | Sliding window for spike density calculation |
| **Rate threshold** | 2.0 | Minimum spike rate (spikes/s) to be considered part of the seizure |
| **Amp threshold (x BL)** | 2.0 | Minimum spike amplitude for counting toward density |

#### Pre-ictal local baseline

These parameters define a window *before* each detected event used to compute the local baseline ratio — a quality metric comparing ictal activity to the immediately preceding signal.

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Window start (s)** | 15 | How far before seizure onset to start the baseline window |
| **Window end (s)** | 5 | How far before seizure onset to end the baseline window (gap avoids pre-ictal buildup) |
| **Trim percentile (%)** | 30 | Percentage of extreme values trimmed from the baseline window to remove artifacts |

### 6.4 Spectral Band method

The Spectral Band detector monitors power in a target frequency band and flags periods where it significantly exceeds a reference band.

![Screenshot: Spectral Band parameter panel](screenshots/spectral_band_params.png)

| Parameter | Default | Effect of increasing |
|-----------|---------|---------------------|
| **Band low (Hz)** | 17 | Lower edge of the target frequency band |
| **Band high (Hz)** | 25 | Upper edge of the target frequency band |
| **Ref low (Hz)** | 1 | Lower edge of the reference (broadband) range |
| **Ref high (Hz)** | 50 | Upper edge of the reference range |
| **Window (s)** | 2.0 | Spectral analysis window. Longer windows give smoother estimates but less time precision |
| **Step (s)** | 1.0 | How far the window advances each step. Smaller steps = finer resolution |
| **Z-score threshold** | 3.0 | How many standard deviations above baseline the band power must be. **Decrease for more sensitivity; increase to reduce false positives** |
| **Baseline percentile** | 15 | Percentile of power values used to estimate baseline. Lower = more conservative baseline |
| **Min duration (s)** | 5.0 | Minimum event duration |
| **Merge gap (s)** | 3.0 | Events separated by less than this are merged |

The Spectral Band method also has its own **boundary refinement** and **pre-ictal local baseline** parameters, which work identically to those described for the Spike-Train method.

### 6.5 Autocorrelation method

The Autocorrelation detector identifies seizures by detecting sustained rhythmic (periodic) patterns in the EEG.

![Screenshot: Autocorrelation parameter panel](screenshots/autocorrelation_params.png)

| Parameter | Default | Effect of increasing |
|-----------|---------|---------------------|
| **Bandpass low (Hz)** | 1.0 | Bandpass filter low cutoff |
| **Bandpass high (Hz)** | 100 | Bandpass filter high cutoff |
| **Spike amplitude (x BL)** | 3.0 | Minimum spike amplitude for periodicity analysis |
| **Refractory (ms)** | 50 | Minimum time between detected spikes |
| **Sub-window (samples)** | 30 | Autocorrelation sub-window size |
| **Lookahead (samples)** | 60 | Maximum lag for autocorrelation |
| **Window (s)** | 12.0 | Analysis window for scoring. Larger windows detect longer-duration patterns |
| **Step (s)** | 4.0 | Step size between analysis windows |
| **Min frequency (Hz)** | 2.0 | Minimum detected periodicity frequency |
| **Z-score threshold** | 3.0 | **Decrease for more sensitivity; increase to reduce false positives** |
| **Min duration (s)** | 5.0 | Minimum event duration |
| **Merge gap (s)** | 3.0 | Events separated by less than this are merged |

The Autocorrelation method has its own **baseline**, **boundary refinement**, and **pre-ictal local baseline** parameter groups.

### 6.6 Ensemble method

The Ensemble method runs multiple detectors and combines their results through a voting/merging strategy.

![Screenshot: Ensemble parameter panel](screenshots/ensemble_params.png)

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Methods** | Spike-Train + Spectral Band | Which methods to include in the ensemble |
| **Vote threshold** | 2 | Minimum number of methods that must agree for an event to be kept. Set to 1 for union (keep all), set to N for intersection (require all methods to agree) |
| **Merge strategy** | Union | How overlapping events from different methods are combined |
| **Confidence merge** | Mean | How confidence scores are combined (mean or max) |

### 6.7 Running detection

1. Select your channels.
2. Choose a method and adjust parameters.
3. Click **Detect**.

The status bar shows progress. When complete, a summary banner shows the number of detected seizures. The results table appears below with all detected events.

![Screenshot: Detection results with summary banner and table](screenshots/detection_results.png)

### 6.8 Event features and quality metrics

Each detected seizure event carries a set of automatically computed features and quality metrics. These are visible in the inspector, used for filtering, and included in CSV exports.

#### Event features

| Feature | Description |
|---------|-------------|
| **n_spikes** | Total number of individual spikes detected within the seizure |
| **mean_spike_frequency_hz** | Average spike rate during the event (spikes per second) |
| **max_amplitude_x_baseline** | Peak spike amplitude expressed as a multiple of baseline |
| **mean_isi_ms** | Mean inter-spike interval in milliseconds |
| **spike_regularity** | Coefficient of variation of ISI — lower values indicate more regular (rhythmic) spiking |
| **mean_amplitude_uv** | Mean spike amplitude in microvolts |
| **max_amplitude_uv** | Maximum spike amplitude in microvolts |

#### Quality metrics

| Metric | Description |
|--------|-------------|
| **local_baseline_ratio** | Ratio of ictal to pre-ictal RMS amplitude — higher values indicate the event stands out more from background |
| **top_spike_amplitude_x** | Mean amplitude of the top 10% of spikes, expressed as baseline multiples |
| **peak_ll_zscore** | Peak line-length z-score — measures signal complexity relative to baseline |
| **peak_energy_zscore** | Peak energy z-score — measures signal power relative to baseline |
| **signal_to_baseline_ratio** | Overall signal-to-baseline ratio across the event |
| **theta_delta_ratio** | Ratio of theta (4–8 Hz) to delta (0.5–4 Hz) power — useful for characterising seizure type |
| **activity_zscore** | Z-score from paired activity channel (if available) — flags movement artifacts |

---

## 7. Result Filters & Inspector

### 7.1 Filter controls

After detection, the **Result Filters** panel lets you narrow down the displayed events. Toggle filters on/off with the switch.

![Screenshot: Filter controls row](screenshots/filter_controls.png)

| Filter | What it does |
|--------|-------------|
| **Confidence** | Min/max detector confidence score (0–1) |
| **Duration (s)** | Min/max event duration |
| **Spikes** | Min/max number of spikes in the event |
| **Amp (xBL)** | Min/max peak amplitude relative to baseline |
| **Local BL** | Min/max local baseline ratio (ictal vs pre-ictal activity) |
| **Top Amp** | Min/max top spike amplitude |
| **Freq (Hz)** | Min/max mean spike frequency |
| **Channel** | Show only events from a specific channel |
| **Method** | Show only events from a specific detection method |

### 7.2 The results table

The results table lists all detected (and filtered) events with sortable columns:

| Column | Description |
|--------|-------------|
| # | Event ID |
| Ch | Channel name |
| Onset (s) | Event start time in seconds |
| Duration | Event duration |
| Confidence | Detector confidence score |
| Spikes | Number of spikes detected |
| Method | Which detection method found this event |

Click any row to open the **Event Inspector**.

### 7.3 Event inspector

The inspector shows detailed information for the selected event:

![Screenshot: Event inspector with EEG trace, spectrogram, and metrics](screenshots/event_inspector.png)

- **EEG trace:** The raw signal around the event, with the seizure region highlighted. Optionally shows spike markers, baseline level, and detection threshold.
- **Spectrogram:** Time-frequency power distribution.
- **Band power:** Power in standard EEG bands (delta, theta, alpha, beta, gamma) over time.
- **Metric cards:** Key measurements including duration, confidence, spike count, amplitude, frequency, local baseline ratio, and more.

#### Inspector options

| Option | Description |
|--------|-------------|
| **Show spikes** | Overlay spike markers on the EEG trace |
| **Show baseline** | Show the computed baseline level |
| **Show threshold** | Show the spike detection threshold |
| **Bandpass** | Apply bandpass filter to the displayed trace |
| **Y-range** | Vertical scale of the EEG trace |

---

## 8. Exporting Detected Events

Both the Seizure and Interictal Spike detection tabs provide CSV export. The export panel appears below the inspector on each tab.

### 8.1 Seizure export

![Screenshot: Export controls panel](screenshots/export_csv.png)

#### Controls

| Control | Description |
|---------|-------------|
| **Methods** | Select which detection methods to include. All methods are checked by default. |
| **Channel** | Optionally limit export to a single channel, or leave on "All channels". |
| **Field groups** | Choose which columns to include in the CSV (see below). |
| **Filename** | The downloaded file name. Pre-filled with the recording name. |
| **Export CSV** | Click to download. |

#### Field groups

| Group | Columns included |
|-------|-----------------|
| **Core** | event_id, onset_sec, offset_sec, duration_sec, channel, channel_name, animal_id, detection_method, confidence, severity, movement_flag |
| **Features** | n_spikes, mean_spike_frequency_hz, max_amplitude_x_baseline, mean_isi_ms, spike_regularity, mean_amplitude_uv, max_amplitude_uv |
| **Quality** | local_baseline_ratio, top_spike_amplitude_x, peak_ll_zscore, peak_energy_zscore, signal_to_baseline_ratio, theta_delta_ratio, activity_zscore |
| **Spectral** | delta/theta/alpha/beta/gamma power (absolute and relative), total_power, dominant_freq_hz, spectral_entropy — computed on-the-fly from the raw EEG signal |

> **Note:** Spectral fields are computed at export time using Welch's method on each event's raw EEG segment. This adds a small delay for large numbers of events but does not require the spectral data to have been pre-computed during detection.

### 8.2 Interictal spike export

The spike export panel (on the **Detection → Interictal Spikes** tab) works the same way but with spike-appropriate field groups:

| Group | Columns included |
|-------|-----------------|
| **Core** | event_id, peak_time_sec, channel, channel_name, animal_id, amplitude, amplitude_x_baseline, confidence |
| **Morphology** | duration_ms, sharpness, phase_ratio, local_snr, rise_time_ms, fall_time_ms |
| **Context** | baseline_mean, threshold, neighbours, after_slow_wave, isolation_score |
| **Spectral** | Band powers, dominant frequency, spectral entropy (computed on-the-fly from a 1-second window around each spike) |

### 8.3 Multi-file export

If you have loaded multiple files, the export always applies to the **currently active file** (selected in the file selector at the top of the sidebar). Switch files and export again to get events from a different recording.

---

## 9. Training / Annotation

The **Training → Seizure** subtab provides a structured workflow for reviewing detected events and building a labeled training dataset for the CNN model.

### 9.1 Overview

Every detected event starts as **pending**. Your job is to review each one and label it as:
- **Confirmed** — this is a real seizure.
- **Rejected** — this is a false positive (artifact, noise, etc.).

You can also **skip** events to come back to them later.

![Screenshot: Training tab in review mode](screenshots/training_review.png)

### 9.2 Review mode vs Browse mode

- **Review mode:** Step through events one by one. The interface shows the EEG trace centered on the current event, with confirm/reject/skip buttons.
- **Browse mode:** Scroll freely through the entire recording. Useful for finding events the detector may have missed, which you can manually annotate.

### 9.3 Navigation

| Action | Button | Keyboard shortcut |
|--------|--------|-------------------|
| Previous event | ◀ Prev | `←` or `,` |
| Next event | Next ▶ | `→` or `.` |
| Confirm | Confirm | `C` |
| Reject | Reject | `R` |
| Skip | Skip | `S` |
| Toggle convulsive | Convulsive toggle | `V` |
| Jump to event # | Enter number in jump field | — |

### 9.4 Adjusting seizure boundaries

If the detector got the onset or offset slightly wrong, you can manually adjust:
1. Edit the **onset** or **offset** time fields below the EEG trace.
2. The trace updates immediately to show the new boundaries.

The original boundaries are preserved internally so you can always revert.

### 9.5 Filtering events

The Training tab has the same filter controls as the Detection tab:
- **Channel** filter — show only events from a specific channel.
- **Method** filter — show only events from a specific detection method.
- **Confidence, Duration, Local BL, Spikes, Amp, Top Amp, Freq** — all with min/max range filters.
- **Filter toggle** — enable/disable all filters at once.

### 9.6 Annotation counts

The annotation progress panel shows:
- Total events, how many are confirmed/rejected/pending.
- Progress percentage.
- Per-channel breakdown.

### 9.7 Notes and metadata

- **Annotator:** Enter your name so annotations are attributed.
- **Notes:** Add free-text notes to any event.
- **Convulsive toggle:** Mark events as convulsive or non-convulsive. This label is used by the 2-class CNN model (see [Dataset & Model Training](#11-dataset--model-training)).

### 9.8 Auto-save

Annotations are auto-saved to a JSON file alongside the EDF file every time you confirm, reject, skip, or modify an event. No manual save is needed.

---

## 10. Interictal Spike Detection & Annotation

The **Detection → Interictal Spikes** subtab detects interictal spikes (isolated sharp transients between seizures).

### 10.1 Detection parameters

| Parameter | Default | Effect of increasing |
|-----------|---------|---------------------|
| **Bandpass low (Hz)** | 3.0 | Filters out slow drift |
| **Bandpass high (Hz)** | 50.0 | Filters out high-frequency noise |
| **Amplitude threshold (z-score)** | 7.0 | Z-score multiplier: spike must exceed mean + z × std of baseline. **Increase to detect only large spikes; decrease for more sensitivity** |
| **Min amplitude (uV)** | 0.0 | Absolute minimum spike height |
| **Prominence (x BL)** | 6.0 | How much the spike must stand out from surrounding signal |
| **Max width (ms)** | 300 | Maximum spike duration |
| **Min width (ms)** | 10 | Minimum spike duration |
| **Refractory (ms)** | 750 | Minimum time between successive spikes |
| **Baseline percentile** | 25 | Percentile for baseline estimation |
| **Baseline RMS (s)** | 30 | RMS window for baseline |
| **Isolation window (s)** | 2.0 | Time window around each spike that must be "quiet" |
| **Max neighbours** | 1 | Maximum number of other spikes allowed within the isolation window. Spikes in dense bursts (e.g., seizures) are rejected when they exceed this count |

### 10.2 Spike features

Each detected spike carries the following computed features:

| Feature | Description |
|---------|-------------|
| **amplitude** | Peak-to-peak amplitude in the signal's native units (usually uV) |
| **amplitude_x_baseline** | Amplitude as a multiple of the computed baseline |
| **duration_ms** | Duration of the spike waveform in milliseconds |
| **sharpness** | Ratio of rise slope to fall slope — sharp spikes have higher values |
| **phase_ratio** | Ratio of the negative phase to the positive phase duration |
| **local_snr** | Local signal-to-noise ratio around the spike |
| **after_slow_wave** | Whether a slow wave follows the spike (typical of epileptiform discharges) |
| **neighbours** | Number of other spikes within the isolation window |

### 10.3 Spike annotation

The **Training → Interictal Spikes** subtab works identically to the seizure Training tab:
- Review mode with confirm/reject/skip.
- Same keyboard shortcuts (`C`, `R`, `S`, `←`, `→`).
- Boundary adjustment.
- Filter controls.

![Screenshot: Spike annotation in review mode](screenshots/spike_annotation.png)

---

## 11. Dataset & Model Training

The **Dataset / Model** tab provides tools for training CNN models on your annotated data.

### 11.1 Creating a training dataset

On the **Dataset** subtab:
1. Select annotation files from one or more recordings.
2. NED-Net extracts the labeled EEG segments (confirmed = positive, rejected = negative).
3. The dataset is saved for use in training.

![Screenshot: Dataset creation page](screenshots/ml_datasets.png)

### 11.2 Training parameters

| Parameter | Description |
|-----------|-------------|
| **Model name** | A unique name for the trained model |
| **Epochs** | Number of training passes through the dataset. More epochs = longer training but potentially better performance. Typical range: 20–100 |
| **Learning rate** | Step size for weight updates. Default is usually fine; decrease if training is unstable |
| **Batch size** | Number of samples per training step. Larger = faster but uses more memory |
| **Window size (s)** | Length of EEG segments fed to the model |
| **Validation split** | Fraction of data held out for validation (typically 0.2 = 20%). Splitting is done by animal — all events from a given animal go entirely into training or validation, never both |

### 11.3 Understanding training metrics

During training, NED-Net shows epoch-by-epoch progress:

![Screenshot: Training progress with loss and metric curves](screenshots/ml_training_progress.png)

| Metric | What it means |
|--------|--------------|
| **Loss** | How wrong the model's predictions are. Should decrease over epochs |
| **F1 score** | Balance between precision and recall. Higher is better (0–1). This is computed as an *event-level* metric, not per-sample |
| **Precision** | Of events the model predicts, how many are real. High precision = few false positives |
| **Recall** | Of actual events, how many the model finds. High recall = few missed events |

The **best model** (highest validation F1) is automatically saved.

### 11.4 The 1D U-Net architecture

The model is a **1D U-Net** — a neural network architecture commonly used for segmentation tasks. It takes a window of multi-channel raw EEG signal (resampled to 250 Hz) as input and outputs a per-sample probability map.

**Architecture details:**

- **Encoder path:** A series of convolutional blocks, each consisting of two 1D convolutions (kernel size 7) with batch normalisation and ReLU activation, followed by 2× max-pooling. The number of filters doubles at each level (32 → 64 → 128 → 256 → 512 by default with depth=4). This progressively captures patterns at multiple timescales — from individual spike morphology (milliseconds) to rhythmic evolution (seconds).
- **Bottleneck:** A convolutional block at the deepest level, operating on the most compressed representation.
- **Decoder path:** A series of upsampling blocks (transposed convolution for 2× upsampling), each concatenated with the corresponding encoder output via skip connections, then processed by a convolutional block. This recovers spatial (temporal) resolution for precise onset/offset prediction.
- **Output layer:** A 1×1 convolution producing per-sample logits, followed by sigmoid activation for probabilities.

**Two-class output:**

- **Seizure models** output 2 classes: channel 0 = seizure probability, channel 1 = convulsive probability. The convulsive channel is trained from the "convulsive" labels you assign during annotation. Legacy models with 1 output class (seizure only) are also supported.
- **Spike models** output 1 class: spike probability.

**Loss function:**

Training uses a combined **Dice + BCE (Binary Cross-Entropy)** loss. Dice loss naturally handles the class imbalance inherent in seizure detection (seizures are rare relative to background), while BCE provides stable gradients early in training. Both components are weighted equally by default (0.5 each).

**Mixed precision:** When a GPU is available, training uses mixed-precision (float16) for faster computation.

### 11.5 Retraining with new annotations

When you annotate more data, simply retrain from scratch by selecting all annotation files (old and new) and training a new model. With typical EEG dataset sizes (hundreds to low thousands of events), training takes minutes, and retraining from scratch avoids any forgetting issues compared to fine-tuning.

### 11.6 Where models are saved

Trained models are saved to:
```
~/.eeg_seizure_analyzer/models/<model_name>/
```

Each model folder contains:
- `best_model.pt` — the best weights (by validation F1), typically 2–8 MB.
- `final_model.pt` — weights at the end of training.
- `metadata.json` — training configuration, dataset name, best metrics, model architecture details.

To share a model with a collaborator, simply zip the model folder and send it. They can place it in their own `~/.eeg_seizure_analyzer/models/` directory and it will appear in the model selector.

### 11.7 Seizure vs Spike models

NED-Net supports two separate model pipelines:

| | Seizure model | Spike model |
|---|---|---|
| **Input** | Multi-channel EEG windows (default 30s at 250 Hz) | Multi-channel EEG windows (default 4s at 250 Hz) |
| **Output** | 2 classes (seizure + convulsive) | 1 class (spike) |
| **Annotations from** | Seizure Training tab | Interictal Spike Training tab |
| **Min event duration** | Configurable (default 5s) | Configurable (default 0.5s) |
| **Used in** | Analysis tab → Seizures mode | Analysis tab → Interictal Spikes mode |

Both model types are stored in the same models directory and are listed with their type and F1 score in the Analysis tab's model selector.

---

## 12. Analysis (CNN Detection)

The **Analysis** tab runs trained CNN models on EDF recordings. It replaces the need for manual parameter tuning — the model applies patterns it learned from your annotations.

### 12.1 Detection type

At the top of the tab, select whether to detect **Seizures** or **Interictal Spikes**. This determines which models appear in the model selector.

### 12.2 Model and parameters

| Control | Description |
|---------|-------------|
| **Trained model** | Select from your trained models. The dropdown shows each model's name, F1 score, and training dataset. |
| **Confidence threshold** | Minimum probability for a prediction to be kept (0.1–1.0, default 0.5). Lower = more sensitive but more false positives. |
| **Min event duration (s)** | Predictions shorter than this are discarded (default 5.0s for seizures). |
| **Merge gap (s)** | Predictions separated by less than this are merged into a single event (default 2.0s). |

### 12.3 HVSW / HPD classification

After CNN detection, each seizure event is automatically classified into electroclinical subtypes based on its spectral characteristics. This classification is configurable via the collapsible **HVSW / HPD classification** panel:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **HVSW — max dominant freq (Hz)** | 4.0 | Events whose dominant frequency is below this are candidates for HVSW |
| **HVSW — min slow-wave index** | 0.5 | Minimum fraction of power in the delta band (0.5–4 Hz) required. Higher values require more prominent slow waves |
| **HPD — min dominant freq (Hz)** | 15.0 | Events whose dominant frequency exceeds this are candidates for HPD |
| **HPD — min HF index** | 0.3 | Minimum fraction of power above 15 Hz required. Higher values require more prominent fast activity |

**Subtype definitions:**

- **HVSW (High-Voltage Spike-and-Wave):** Characterised by high-amplitude, low-frequency (<4 Hz) rhythmic discharges dominated by delta-band power. Common in genetic absence epilepsy models.
- **HPD (Hypersynchronous Paroxysmal Discharge):** Characterised by fast (>15 Hz) repetitive discharges. The spectral profile is dominated by beta/gamma frequencies.
- **Convulsive / Non-convulsive:** Determined by the model's second output channel (if the model was trained with convulsive labels). If the model only outputs 1 class, a spectral heuristic is used as fallback.

Events that do not meet HVSW or HPD criteria are classified as "unclassified" for subtype but retain their convulsive/non-convulsive label.

### 12.4 Modes

#### Single file

Run the model on a single EDF file. The file path defaults to the currently loaded recording but can be changed. Results appear in the summary panel and are stored in the SQLite database.

#### Batch

Process an entire folder of EDF files:
1. Select or browse for a folder.
2. Optionally enable **Include subfolders** to scan recursively.
3. Optionally provide a **batch metadata** Excel file mapping files to cohort, group, and animal IDs. Click **Generate template** to create a pre-filled template.
4. Click **Scan folder** to preview the file list.
5. Click **Run batch analysis**. Progress shows per-file status. You can **Pause** or **Cancel** at any time.

#### Live monitoring

Monitor a shared folder for new EDF files and process them automatically:
1. Select a **Watch folder**.
2. Set the **Wait before processing** delay (default 30s) — this ensures the acquisition software has finished writing the file before NED-Net reads it.
3. Optionally enable **Process backlog on startup** to analyse any existing files that haven't been processed yet.
4. Click **Start monitoring**. NED-Net polls the folder and processes new files as they appear.

### 12.5 Results summary

Below the mode panels, a summary shows the last analysis run's statistics: files processed, animals, total events, convulsive/non-convulsive counts, HVSW/HPD counts, and flagged events.

### 12.6 CNN vs rule-based detection

| Aspect | Rule-based (Detection tab) | CNN (Analysis tab) |
|--------|---------------------------|-------------------|
| **How it works** | Explicit parameter thresholds (spike height, frequency, etc.) | Learned patterns from your annotated examples |
| **Tuning** | Many parameters to adjust per recording | Train once, apply to many recordings |
| **Subtype classification** | Not available | HVSW/HPD classification with configurable spectral thresholds |
| **Batch/Live** | Detect-all across loaded files | Full batch and live-monitoring modes |
| **Best for** | Initial exploration, understanding what the detector is doing | Production use once you have sufficient annotated examples |
| **Output** | Events stored in JSON alongside EDF | Events stored in SQLite database, visible on Results tab |

---

## 13. Results

The **Results** tab aggregates detection results from all Analysis runs (single, batch, and live modes), stored in a local SQLite database.

### 13.1 Event category

At the top, select whether to view **Seizures** or **Interictal Spikes** results.

### 13.2 Summary statistics

The Results tab shows:
- **Total events**, **total duration**, **mean duration**, and **event rate** as metric cards.
- **Daily seizure burden:** A stacked bar chart showing event count per day.
- **Circadian distribution:** Event count by hour of day.

![Screenshot: Results summary with daily burden chart](screenshots/results_summary.png)

### 13.3 Filtering stored results

Filter by:
- **Source file** — select one or more specific files.
- **Date range** — start and end dates.
- **Mode** — single, batch, or live analysis.
- **Animals** — filter by animal ID.
- **Event type** — convulsive vs non-convulsive.
- **Min confidence** — minimum confidence threshold.

### 13.4 Event table

A detailed table of all stored events with sortable columns. Click any row to open the inspector (if the source EDF file is available on disk).

### 13.5 CSV export

Export all displayed results to CSV using the export button. The CSV includes all event parameters, quality metrics, and features.

---

## 14. Batch Processing

NED-Net supports batch processing in two ways:

### 14.1 Rule-based batch detection

When you have loaded a project folder (multiple EDF files via **Load Multiple...**), you can run rule-based detection across all of them:

1. Go to the **Detection → Seizure** tab.
2. Configure your detection parameters.
3. Click **Detect All Files**.

NED-Net processes each file sequentially, showing a progress bar with the current file name and count.

![Screenshot: Batch processing progress bar](screenshots/batch_progress.png)

### 14.2 CNN batch analysis

For CNN-based batch processing, use the **Analysis** tab in **Batch** mode (see [Analysis — Batch](#batch)). This supports folder scanning, subfolder inclusion, batch metadata Excel files, and pause/cancel controls.

### 14.3 Results persistence

Each file's rule-based detections are automatically saved to a JSON file alongside the EDF. When you re-open a file, its saved detections are loaded automatically — no need to re-run detection. CNN analysis results are stored in the SQLite database and appear on the Results tab.

---

## 15. Saving & Recalling Parameters

### 15.1 Saving detection parameters

After tuning your detection parameters, save them for future use:
- Click **Save User Params** in the detection panel.
- Parameters are saved to `~/.eeg_seizure_analyzer/user_defaults.json`.

### 15.2 Recalling parameters

- **Recall User Params:** Restores your saved parameter values while keeping the currently selected method.
- **Recall Detection Params:** Loads the parameters that were used for the saved detections on the current file. Useful when you want to see exactly what settings produced a particular set of results.
- **Restore Defaults:** Restores all parameters to factory defaults.

### 15.3 Theme

Switch between light and dark themes using the **Dark mode** toggle in the sidebar footer. The light theme is the default.

---

## 16. Tools

The **Tools** tab provides file conversion utilities.

### 16.1 Video Converter

Converts WMV video recordings (e.g., from LabChart Video Capture) into a single MP4 file. This is useful when you have fragmented WMV files from a recording session and need a single video to associate with your EDF file.

1. Set the **Input folder** containing WMV files.
2. Set the **Output path** for the MP4 file.
3. Click **Convert**. Progress shows the current file being processed.

> **Requirement:** FFmpeg must be installed on your system for video conversion. On macOS: `brew install ffmpeg`. On Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html).

### 16.2 ADICHT → EDF Converter

Converts LabChart ADICHT files to EDF format for use with NED-Net's detection and analysis features. This tool is available on all platforms for pre-exported files; direct ADICHT reading requires Windows with LabChart SDK (install via `pip install -e ".[windows]"`).

---

## 17. File Reference

NED-Net creates several files alongside your EDF recordings and in its configuration directory.

### 17.1 Files alongside the EDF

| File | Description |
|------|-------------|
| `<name>_ned_detections.json` | Seizure detection results: events, parameters used, filter settings, detection info |
| `<name>_ned_annotations.json` | Annotation labels (confirmed/rejected/pending), boundary adjustments, annotator info |
| `<name>_ned_spikes.json` | Interictal spike detection results |
| `<name>_ned_spike_annotations.json` | Interictal spike annotation labels |

These JSON files are human-readable and can be loaded by scripts for further analysis.

### 17.2 Configuration directory

Located at `~/.eeg_seizure_analyzer/`:

| Path | Description |
|------|-------------|
| `user_defaults.json` | Saved detection parameters (seizure and spike) |
| `models/` | Trained CNN model weights and metadata |
| `cache/` | Temporary files for batch processing progress |
| `results.db` | SQLite database with aggregated results from all Analysis runs |

### 17.3 Detection JSON structure

The `_ned_detections.json` file contains:

```json
{
  "edf_path": "/path/to/recording.edf",
  "detector_name": "SpikeTrainSeizureDetector",
  "channels": [0, 1],
  "params": { ... },
  "filter_settings": {
    "filter_enabled": true,
    "filter_values": { ... }
  },
  "events": [
    {
      "onset_sec": 1234.5,
      "offset_sec": 1267.8,
      "duration_sec": 33.3,
      "channel": 0,
      "event_type": "seizure",
      "confidence": 0.92,
      "features": {
        "detection_method": "spike_train",
        "n_spikes": 47,
        "mean_spike_frequency_hz": 8.2,
        "max_amplitude_x_baseline": 6.1
      },
      "quality_metrics": {
        "local_baseline_ratio": 3.4,
        "top_spike_amplitude_x": 8.7
      }
    }
  ]
}
```

### 17.4 Annotation JSON structure

The `_ned_annotations.json` file contains:

```json
{
  "annotations": [
    {
      "onset_sec": 1234.5,
      "offset_sec": 1267.8,
      "channel": 0,
      "label": "confirmed",
      "source": "detector",
      "annotator": "John",
      "animal_id": "Mouse_01",
      "detector_confidence": 0.92,
      "notes": "",
      "event_id": 1,
      "features": { ... },
      "quality_metrics": { ... }
    }
  ]
}
```

---

*NED-Net — Neural Event Detection Network*
*For questions or issues, contact the development team.*
