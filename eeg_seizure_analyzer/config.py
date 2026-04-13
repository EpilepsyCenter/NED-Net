"""Global configuration: frequency bands, detection thresholds, and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

# Frequency bands for mouse EEG (Hz)
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma_low": (30, 50),
    "gamma_high": (50, 100),
}


@dataclass
class SeizureDetectionParams:
    """User-tunable seizure detection parameters."""

    bandpass_low: float = 1.0
    bandpass_high: float = 50.0
    line_length_window_sec: float = 2.0
    line_length_threshold_zscore: float = 5.0
    energy_threshold_zscore: float = 4.0
    onset_offset_zscore: float = 2.0
    min_duration_sec: float = 5.0
    merge_gap_sec: float = 1.0
    baseline_method: str = "percentile"  # "percentile", "rolling", "first_n", "manual", "robust" (deprecated)
    baseline_percentile: int = 15         # percentile of RMS windows (1-50)
    baseline_rms_window_sec: float = 10.0 # RMS computation window size
    rolling_lookback_sec: float = 1800.0  # 30 min lookback for rolling baseline
    rolling_step_sec: float = 300.0       # 5 min step for rolling baseline
    baseline_duration_min: float = 5.0    # used only for "first_n"
    baseline_start_sec: float | None = None  # used only for "manual"
    baseline_end_sec: float | None = None    # used only for "manual"

    # Severity thresholds
    mild_max_duration_sec: float = 15.0
    moderate_max_duration_sec: float = 45.0
    severe_energy_zscore: float = 10.0


@dataclass
class SpikeDetectionParams:
    """User-tunable interictal spike detection parameters."""

    bandpass_low: float = 10.0
    bandpass_high: float = 70.0
    amplitude_threshold_zscore: float = 4.0   # z-score multiplier: mean + z × std
    spike_min_amplitude_uv: float = 0.0       # absolute floor (µV); 0 = disabled
    spike_prominence_x_baseline: float = 1.5  # prominence relative to baseline
    max_duration_ms: float = 70.0             # max spike half-width (ms)
    min_duration_ms: float = 2.0              # min spike half-width (ms)
    refractory_ms: float = 200.0
    # Baseline
    baseline_method: str = "percentile"       # "percentile", "rolling", "first_n"
    baseline_percentile: int = 15
    baseline_rms_window_sec: float = 10.0
    rolling_lookback_sec: float = 1800.0
    rolling_step_sec: float = 300.0
    # Isolation — reject spikes inside high-rate bursts (seizures)
    isolation_window_sec: float = 2.0         # window around spike to count neighbours
    isolation_max_neighbours: int = 6         # max spikes allowed in window (above → reject)
    # Confidence scoring weights (0–1, used in composite)
    w_amplitude: float = 0.20
    w_sharpness: float = 0.20
    w_local_snr: float = 0.25
    w_after_slow_wave: float = 0.20
    w_phase_ratio: float = 0.15


@dataclass
class SpikeTrainSeizureParams:
    """Parameters for spike-train-based seizure detection.

    Based on Twele et al. (2017) criteria for the intrahippocampal kainate
    mouse model.  Seizures are detected by first finding individual spikes,
    then grouping them into trains and classifying each train as HVSW, HPD,
    or electroclinical/convulsive.
    """

    # ── Spike detection front-end ────────────────────────────────────
    bandpass_low: float = 1.0
    bandpass_high: float = 100.0
    spike_amplitude_x_baseline: float = 3.0   # z-score multiplier: threshold = mean + z×std
    spike_min_amplitude_uv: float = 0.0       # absolute floor (µV); 0 = disabled
    spike_refractory_ms: float = 50.0         # min time between spikes (ms)
    spike_min_prominence_uv: float = 0.0      # min prominence (µV); 0 = use baseline-relative
    spike_prominence_x_baseline: float = 1.5  # min prominence as × baseline; spike must stand out
    spike_max_width_ms: float = 70.0          # max half-width (ms); rejects slow waves
    spike_min_width_ms: float = 2.0           # min half-width (ms); rejects single-sample noise

    # ── Spike-train grouping ─────────────────────────────────────────
    max_interspike_interval_ms: float = 500.0  # spikes further apart → separate trains
    min_train_spikes: int = 5                  # minimum spikes to form a train
    min_train_duration_sec: float = 5.0        # shortest accepted train
    min_interevent_interval_sec: float = 3.0   # gap to separate distinct events

    # ── HVSW classification ──────────────────────────────────────────
    hvsw_min_amplitude_x: float = 3.0          # ≥3× baseline
    hvsw_min_frequency_hz: float = 2.0         # spikes ≥2 Hz within train
    hvsw_min_duration_sec: float = 5.0
    hvsw_max_evolution: float = 0.4            # max CV(ISI) to still be "monomorphic"

    # ── HPD classification ───────────────────────────────────────────
    hpd_min_amplitude_x: float = 2.0           # ≥2× baseline (evolved phase)
    hpd_min_frequency_hz: float = 5.0          # faster evolved phase ≥5 Hz
    hpd_min_duration_sec: float = 10.0         # typically >20 s, use ≥10 as floor

    # ── Electroclinical / convulsive ─────────────────────────────────
    convulsive_min_duration_sec: float = 20.0
    convulsive_min_amplitude_x: float = 5.0    # very high amplitude
    convulsive_postictal_suppression_sec: float = 5.0  # quiet period after event

    # ── Boundary refinement ──────────────────────────────────────────
    boundary_method: str = "signal"            # "spike_density", "signal", "none"

    # Spike-density method params
    boundary_window_sec: float = 2.0           # sliding window for local spike rate
    boundary_min_rate_hz: float = 2.0          # min local spike rate at edges
    boundary_min_amplitude_x: float = 2.0      # edge spikes must exceed this

    # Signal-based method params (method C)
    boundary_rms_window_ms: float = 100.0      # short window for RMS envelope (ms)
    boundary_rms_threshold_x: float = 2.0      # onset/offset = RMS drops below N× baseline
    boundary_max_trim_sec: float = 5.0         # max seconds to trim from each edge

    # ── Classification ─────────────────────────────────────────────────
    classify_subtypes: bool = True  # False = skip HVSW/HPD/convulsive, just report "seizure"

    # ── Local baseline (pre-ictal comparison) ───────────────────────
    local_baseline_start_sec: float = -20.0  # seconds before event onset
    local_baseline_end_sec: float = -5.0     # seconds before event onset

    # ── Baseline ─────────────────────────────────────────────────────
    baseline_method: str = "percentile"    # "percentile", "rolling", "first_n"
    baseline_percentile: int = 15
    baseline_rms_window_sec: float = 10.0
    rolling_lookback_sec: float = 1800.0  # 30 min lookback for rolling baseline
    rolling_step_sec: float = 300.0       # 5 min step for rolling baseline


@dataclass
class SpectralBandParams:
    """Parameters for spectral-band seizure detection.

    Based on Casillas-Espinosa et al. (2019): seizures across rodent models
    share a characteristic spectral peak in the 17–25 Hz band that is absent
    in normal interictal EEG.  The detector computes a Spectral Band Index
    (SBI) per sliding window and thresholds it against the baseline
    distribution.
    """

    # Frequency band of interest
    band_low: float = 17.0
    band_high: float = 25.0

    # Reference band for SBI ratio (total power denominator)
    ref_band_low: float = 1.0
    ref_band_high: float = 50.0

    # Sliding window
    window_sec: float = 2.0
    step_sec: float = 1.0

    # Threshold (z-score above baseline SBI)
    threshold_z: float = 3.0

    # Baseline
    baseline_method: str = "percentile"   # "percentile" or "first_n"
    baseline_percentile: int = 15

    # Event grouping
    min_duration_sec: float = 5.0
    merge_gap_sec: float = 3.0

    # Boundary refinement (signal-based only — no spikes in this method)
    boundary_method: str = "none"          # "signal" or "none"
    boundary_rms_window_ms: float = 100.0
    boundary_rms_threshold_x: float = 2.0
    boundary_max_trim_sec: float = 5.0


@dataclass
class AutocorrelationParams:
    """Parameters for autocorrelation-based seizure detection.

    Based on White et al. (2006): computes range overlap between consecutive
    sub-windows to detect rhythmic, correlated activity characteristic of
    seizures.  Combined with spike frequency counting for high specificity.
    """

    # ── Spike front-end (shared with spike-train) ──────────────────
    bandpass_low: float = 1.0
    bandpass_high: float = 100.0
    spike_amplitude_x_baseline: float = 3.0
    spike_min_amplitude_uv: float = 0.0
    spike_refractory_ms: float = 50.0
    spike_prominence_x_baseline: float = 1.5
    spike_max_width_ms: float = 70.0
    spike_min_width_ms: float = 2.0

    # ── Autocorrelation-specific ───────────────────────────────────
    # Sub-window for range computation (in data points at recording fs)
    subwindow_points: int = 30       # ~120 ms at 250 Hz
    lookahead_points: int = 60       # next 2 sub-windows

    # Analysis window
    acorr_window_sec: float = 12.0   # 3000 points at 250 Hz = 12s
    acorr_step_sec: float = 4.0

    # Thresholds
    min_spike_freq_hz: float = 2.0   # min spikes/sec within window
    acorr_threshold_z: float = 3.0   # z-score above baseline range-overlap

    # Event grouping
    min_duration_sec: float = 5.0
    merge_gap_sec: float = 3.0

    # Boundary refinement (signal or spike_density — has spikes)
    boundary_method: str = "signal"        # "signal", "spike_density", or "none"
    boundary_rms_window_ms: float = 100.0
    boundary_rms_threshold_x: float = 2.0
    boundary_max_trim_sec: float = 5.0
    # spike_density boundary params
    boundary_window_sec: float = 2.0
    boundary_min_rate_hz: float = 2.0
    boundary_min_amplitude_x: float = 2.0

    # Baseline
    baseline_method: str = "percentile"
    baseline_percentile: int = 15
    baseline_rms_window_sec: float = 10.0


@dataclass
class EnsembleParams:
    """Parameters for ensemble seizure detection.

    Runs multiple sub-detectors and combines their results via temporal
    overlap voting.
    """

    # Which methods to include (subset of: spike_train, spectral_band, autocorrelation)
    methods: list = field(default_factory=lambda: ["spike_train", "spectral_band"])

    # Voting: event survives if >= voting_threshold methods detect it
    voting_threshold: int = 2

    # How to merge overlapping event boundaries
    merge_strategy: str = "union"      # "union" (widest) or "intersection" (tightest)

    # How to merge confidence scores
    confidence_merge: str = "mean"     # "mean" or "max"


@dataclass
class BENDRParams:
    """Parameters for BENDR-based seizure detection.

    Used when running a fine-tuned BENDR model in the Detection tab.
    """

    model_name: str = ""            # trained model to use for detection
    threshold: float = 0.5          # probability threshold for seizure
    min_duration_sec: float = 3.0   # discard events shorter than this
    merge_gap_sec: float = 2.0      # merge events closer than this
    overlap_sec: float = 15.0       # sliding window overlap


@dataclass
class PreprocessParams:
    """Preprocessing parameters."""

    notch_freq: float | None = 50.0  # None to disable; 50 Hz (EU) or 60 Hz (US)
    artifact_threshold_uv: float = 3000.0
    filter_order: int = 4


# MAD-to-std scaling factor (for normal distribution)
MAD_SCALE: float = 1.4826
