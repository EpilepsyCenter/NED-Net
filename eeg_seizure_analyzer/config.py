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
class PreprocessParams:
    """Preprocessing parameters."""

    notch_freq: float | None = 50.0  # None to disable; 50 Hz (EU) or 60 Hz (US)
    artifact_threshold_uv: float = 3000.0
    filter_order: int = 4


# MAD-to-std scaling factor (for normal distribution)
MAD_SCALE: float = 1.4826
