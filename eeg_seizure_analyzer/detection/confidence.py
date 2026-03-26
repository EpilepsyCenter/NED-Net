"""Post-detection confidence scoring using LL/energy and spectral features.

Computes quality metrics for each detected event and derives a composite
confidence score.  This helps distinguish true seizures from artifacts
without discarding events — the metrics are stored and the user/downstream
code can filter by confidence.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from eeg_seizure_analyzer.config import BANDS
from eeg_seizure_analyzer.detection.base import DetectedEvent
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.features import line_length, signal_energy
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter


def compute_event_quality(
    recording: EEGRecording,
    event: DetectedEvent,
    baseline_rms: float | None = None,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
) -> dict:
    """Compute quality metrics for a single detected event.

    Parameters
    ----------
    recording : EEGRecording
        The recording containing the event.
    event : DetectedEvent
        The event to score.
    baseline_rms : float | None
        Precomputed baseline RMS. If None, uses recording-wide estimate.
    bandpass_low, bandpass_high : float
        Bandpass filter range.

    Returns
    -------
    dict
        Quality metrics including:
        - peak_ll_zscore, peak_energy_zscore
        - spectral_entropy
        - dominant_freq_hz
        - theta_delta_ratio
        - signal_to_baseline_ratio
    """
    fs = recording.fs
    ch = event.channel

    # Extract event data with small padding
    pad_sec = 1.0
    start_idx = max(0, int((event.onset_sec - pad_sec) * fs))
    end_idx = min(recording.n_samples, int((event.offset_sec + pad_sec) * fs))
    raw_segment = recording.data[ch, start_idx:end_idx]

    if len(raw_segment) < int(fs * 0.5):
        return _empty_metrics()

    # Bandpass filter
    filtered = bandpass_filter(raw_segment, fs, bandpass_low, bandpass_high)

    # ── LL and energy z-scores ───────────────────────────────────────
    window_samples = max(1, int(2.0 * fs))
    step_samples = window_samples // 2

    ll = line_length(filtered, window_samples, step_samples)
    en = signal_energy(filtered, window_samples, step_samples)

    if baseline_rms is None:
        from eeg_seizure_analyzer.processing.features import compute_zscore_baseline
        bl_mean, bl_std = compute_zscore_baseline(
            recording.data[ch], fs, window_sec=10.0, percentile=15
        )
        baseline_rms = bl_mean
    else:
        bl_mean = baseline_rms
        bl_std = baseline_rms * 0.3

    # Z-score: (feature - mean) / std — proper z-score using baseline std
    peak_ll_z = float(np.max((ll - bl_mean) / bl_std)) if len(ll) > 0 and bl_std > 1e-10 else 0.0
    peak_en_z = float(np.max((en - bl_mean) / bl_std)) if len(en) > 0 and bl_std > 1e-10 else 0.0

    # ── Spectral features ────────────────────────────────────────────
    nperseg = min(int(2 * fs), len(filtered))
    freqs, psd = welch(filtered, fs=fs, nperseg=nperseg)

    # Spectral entropy (Shannon entropy of normalized PSD)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    # Dominant frequency
    dominant_freq = float(freqs[np.argmax(psd)])

    # Band power ratios
    band_powers = {}
    for name, (f_lo, f_hi) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        band_powers[name] = float(np.trapezoid(psd[mask], dx=freq_res)) if np.any(mask) else 0.0

    theta_power = band_powers.get("theta", 0.0)
    delta_power = band_powers.get("delta", 1e-12)
    theta_delta_ratio = theta_power / delta_power

    # Signal-to-baseline ratio
    event_rms = float(np.sqrt(np.mean(filtered ** 2)))
    signal_to_baseline = event_rms / baseline_rms

    return {
        "peak_ll_zscore": peak_ll_z,
        "peak_energy_zscore": peak_en_z,
        "spectral_entropy": spectral_entropy,
        "dominant_freq_hz": dominant_freq,
        "theta_delta_ratio": theta_delta_ratio,
        "signal_to_baseline_ratio": signal_to_baseline,
        "event_rms": event_rms,
        "baseline_rms": baseline_rms,
    }


def compute_local_baseline_ratio(
    recording: EEGRecording,
    event: DetectedEvent,
    local_start_sec: float = -20.0,
    local_end_sec: float = -5.0,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
    all_events: list[DetectedEvent] | None = None,
    trim_pct: float = 0.0,
    max_lookback_sec: float = 120.0,
) -> float:
    """Compute RMS ratio of event signal to a pre-ictal local baseline window.

    Two-pass refinement (when *all_events* is provided):
    1. If any other detected events overlap the baseline window, shift the
       window further back until a clean window is found (up to
       *max_lookback_sec* before the event).
    2. If *trim_pct* > 0, remove the top *trim_pct* % of amplitude samples
       from the baseline segment before computing RMS. This discards
       stray high-amplitude spikes that contaminate the baseline.

    Parameters
    ----------
    recording : EEGRecording
    event : DetectedEvent
    local_start_sec, local_end_sec : float
        Offsets relative to event onset (both should be negative).
        Default: -20 to -5 seconds before onset.
    bandpass_low, bandpass_high : float
    all_events : list[DetectedEvent] | None
        All detected events (for overlap checking). If None, no
        event-aware shifting is performed (pass-1 behaviour).
    trim_pct : float
        Percentage (0–100) of top-amplitude samples to remove from the
        baseline before computing RMS.  0 = no trimming.
    max_lookback_sec : float
        Maximum seconds before event onset to search for a clean window.

    Returns
    -------
    float
        event_rms / local_baseline_rms.  Values >>1 indicate the event
        stands out clearly from the immediately preceding activity.
        Returns 0.0 if the local baseline window is invalid.
    """
    fs = recording.fs
    ch = event.channel
    window_dur = local_start_sec - local_end_sec  # negative - negative = duration

    # ── Step 1: find a clean baseline window ────────────────────
    bl_start_sec = event.onset_sec + local_start_sec
    bl_end_sec = event.onset_sec + local_end_sec

    if all_events is not None:
        # Check for overlapping events and shift back if needed
        same_ch_events = [
            e for e in all_events
            if e.channel == ch and e is not event
        ]
        min_onset = event.onset_sec - max_lookback_sec
        max_shifts = 20  # safety limit

        for _ in range(max_shifts):
            if bl_start_sec < min_onset or bl_start_sec < 0:
                break  # can't go further back

            # Check overlap with any other event
            has_overlap = False
            for other in same_ch_events:
                # Other event overlaps our baseline window?
                if other.onset_sec < bl_end_sec and other.offset_sec > bl_start_sec:
                    has_overlap = True
                    # Shift window to end before this other event
                    bl_end_sec = other.onset_sec - 1.0  # 1s gap
                    bl_start_sec = bl_end_sec + window_dur  # window_dur is negative
                    break

            if not has_overlap:
                break

    # ── Step 2: extract and filter baseline segment ─────────────
    bl_start = max(0, int(bl_start_sec * fs))
    bl_end = max(0, int(bl_end_sec * fs))

    if bl_end - bl_start < int(fs * 0.5):  # need at least 0.5s
        return 0.0

    bl_segment = recording.data[ch, bl_start:bl_end]
    bl_filtered = bandpass_filter(bl_segment, fs, bandpass_low, bandpass_high)

    # ── Step 3: trimmed RMS ─────────────────────────────────────
    if trim_pct > 0 and len(bl_filtered) > 10:
        abs_vals = np.abs(bl_filtered)
        cutoff = np.percentile(abs_vals, 100.0 - trim_pct)
        mask = abs_vals <= cutoff
        if mask.sum() > int(fs * 0.5):  # enough samples left
            bl_filtered = bl_filtered[mask]

    bl_rms = float(np.sqrt(np.mean(bl_filtered ** 2)))

    if bl_rms < 1e-10:
        return 0.0

    # ── Step 4: event RMS ───────────────────────────────────────
    ev_start = int(event.onset_sec * fs)
    ev_end = min(recording.n_samples, int(event.offset_sec * fs))
    ev_segment = recording.data[ch, ev_start:ev_end]
    if len(ev_segment) < int(fs * 0.5):
        return 0.0
    ev_filtered = bandpass_filter(ev_segment, fs, bandpass_low, bandpass_high)
    ev_rms = float(np.sqrt(np.mean(ev_filtered ** 2)))

    return ev_rms / bl_rms


def compute_top_spike_amplitude(event: DetectedEvent) -> float:
    """Return the mean amplitude of the top 10% of spikes (×baseline).

    Uses the max_amplitude_x_baseline from features as a proxy when
    per-spike amplitudes are not stored.  In practice, this is a measure
    of the *strongest* spikes in the event, not diluted by many weak ones.

    Returns
    -------
    float
        Top-percentile spike amplitude as ×baseline.  0.0 if unavailable.
    """
    # If per-spike amplitudes are stored in features, use them
    spike_amps = event.features.get("spike_amplitudes_x")
    if spike_amps and len(spike_amps) > 0:
        arr = np.array(spike_amps)
        n_top = max(1, int(len(arr) * 0.1))
        return float(np.mean(np.sort(arr)[-n_top:]))

    # Fallback: use max amplitude (approximation)
    return float(event.features.get("max_amplitude_x_baseline", 0.0))


def compute_confidence_score(quality_metrics: dict) -> float:
    """Derive a 0–1 confidence score from quality metrics.

    Higher scores indicate the event is more likely a true seizure.
    The scoring is amplitude-weighted: it prioritises signal strength
    (local baseline ratio, signal-to-baseline) over pure spectral features.

    Weights:
    - Local baseline ratio (30%) — strongest discriminator of true events
    - Signal-to-baseline ratio (20%) — global baseline comparison
    - LL z-score (15%)
    - Energy z-score (10%)
    - Spectral entropy (10%)
    - Dominant frequency in seizure range (10%)
    - Theta/delta ratio (5%)

    Parameters
    ----------
    quality_metrics : dict
        Output from compute_event_quality(), optionally with
        ``local_baseline_ratio`` added.

    Returns
    -------
    float
        Confidence score in [0, 1].
    """
    scores = []

    # Local baseline ratio — most important for distinguishing true events
    # from periods of elevated noise.  Ratio of ~1 = same as background.
    lbr = quality_metrics.get("local_baseline_ratio", 0.0)
    if lbr > 0:
        scores.append(min(1.0, (lbr - 1.0) / 3.0) * 0.30)
    else:
        # Not available — redistribute weight to signal-to-baseline
        sbr = quality_metrics.get("signal_to_baseline_ratio", 1.0)
        scores.append(min(1.0, (sbr - 1.0) / 5.0) * 0.30)

    # Signal-to-baseline ratio (global)
    sbr = quality_metrics.get("signal_to_baseline_ratio", 1.0)
    scores.append(min(1.0, (sbr - 1.0) / 5.0) * 0.20)

    # LL z-score contribution (sigmoid-like, saturates around 10)
    ll_z = quality_metrics.get("peak_ll_zscore", 0.0)
    scores.append(min(1.0, ll_z / 10.0) * 0.15)

    # Energy z-score contribution
    en_z = quality_metrics.get("peak_energy_zscore", 0.0)
    scores.append(min(1.0, en_z / 10.0) * 0.10)

    # Spectral entropy — moderate is best (seizures have structured spectra)
    se = quality_metrics.get("spectral_entropy", 0.0)
    if 2.0 <= se <= 5.0:
        se_score = 1.0
    elif se < 2.0:
        se_score = se / 2.0
    else:
        se_score = max(0.0, 1.0 - (se - 5.0) / 3.0)
    scores.append(se_score * 0.10)

    # Dominant frequency in seizure range (3–30 Hz)
    dom_freq = quality_metrics.get("dominant_freq_hz", 0.0)
    if 3.0 <= dom_freq <= 30.0:
        freq_score = 1.0
    elif dom_freq < 3.0:
        freq_score = dom_freq / 3.0
    else:
        freq_score = max(0.0, 1.0 - (dom_freq - 30.0) / 20.0)
    scores.append(freq_score * 0.10)

    # Theta/delta ratio (elevated in many seizure types)
    tdr = quality_metrics.get("theta_delta_ratio", 0.0)
    scores.append(min(1.0, tdr / 3.0) * 0.05)

    confidence = sum(scores)
    return max(0.0, min(1.0, confidence))


def apply_quality_filter(
    events: list[DetectedEvent],
    recording: EEGRecording,
    baseline_rms: float | None = None,
    min_confidence: float = 0.3,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
    metric_filters: dict | None = None,
) -> list[DetectedEvent]:
    """Score all events and filter by minimum confidence.

    Computes quality_metrics and confidence for each event. Events below
    min_confidence are removed. All surviving events have their
    quality_metrics dict populated.

    Parameters
    ----------
    events : list[DetectedEvent]
    recording : EEGRecording
    baseline_rms : float | None
    min_confidence : float
        Minimum confidence to keep (default 0.3).
    bandpass_low, bandpass_high : float
    metric_filters : dict | None
        Optional per-metric filter thresholds. Supported keys:
        - min_ll_zscore, min_energy_zscore, min_signal_to_baseline_ratio
        - min_spectral_entropy, max_spectral_entropy

    Returns
    -------
    list[DetectedEvent]
        Filtered events with quality_metrics populated.
    """
    if metric_filters is None:
        metric_filters = {}

    filtered = []
    for event in events:
        qm = compute_event_quality(
            recording, event,
            baseline_rms=baseline_rms,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
        )
        event.quality_metrics = qm
        event.confidence = compute_confidence_score(qm)

        if event.confidence < min_confidence:
            continue

        # Per-metric filters (quality_metrics + event features)
        if not _passes_metric_filters(qm, metric_filters, event):
            continue

        filtered.append(event)
    return filtered


def _passes_metric_filters(
    qm: dict, filters: dict, event: DetectedEvent | None = None
) -> bool:
    """Check whether quality metrics pass all individual thresholds."""
    if not filters:
        return True

    checks = [
        ("min_ll_zscore", "peak_ll_zscore", "min"),
        ("min_energy_zscore", "peak_energy_zscore", "min"),
        ("min_signal_to_baseline_ratio", "signal_to_baseline_ratio", "min"),
        ("min_spectral_entropy", "spectral_entropy", "min"),
        ("max_spectral_entropy", "spectral_entropy", "max"),
    ]

    for filter_key, metric_key, direction in checks:
        threshold = filters.get(filter_key)
        if threshold is None:
            continue
        value = qm.get(metric_key, 0.0)
        if direction == "min" and value < threshold:
            return False
        if direction == "max" and value > threshold:
            return False

    # Spike frequency filter (from event.features, not quality_metrics)
    min_spike_freq = filters.get("min_spike_frequency")
    if min_spike_freq is not None and event is not None:
        freq = event.features.get("mean_spike_frequency_hz", 0.0)
        if freq < min_spike_freq:
            return False

    return True


def _empty_metrics() -> dict:
    """Return zeroed quality metrics for too-short segments."""
    return {
        "peak_ll_zscore": 0.0,
        "peak_energy_zscore": 0.0,
        "spectral_entropy": 0.0,
        "dominant_freq_hz": 0.0,
        "theta_delta_ratio": 0.0,
        "signal_to_baseline_ratio": 0.0,
        "local_baseline_ratio": 0.0,
        "top_spike_amplitude_x": 0.0,
        "event_rms": 0.0,
        "baseline_rms": 0.0,
    }
