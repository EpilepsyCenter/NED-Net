"""Detector-agnostic feature extraction for the seizure event re-ranker.

The re-ranker is a learned confidence/filter layer: given a *candidate* seizure
event proposed by any classical detector, it scores P(real) so false candidates
can be down-ranked. To make candidates from different detectors comparable —
most labelled events came from the autocorrelation detector, which does NOT
store the spike-train morphology features — every feature here is **computed
fresh from the raw signal**, independent of how the event was originally found.

The SAME function (``extract_event_features``) runs at training time (over
annotated events, one EDF window each) and at inference time (over freshly
detected events on the already-loaded recording). Keeping a single code path is
the most important correctness property: train/inference feature parity.

All features are plain floats; missing/uncomputable context (edge-of-recording,
truncated post-ictal window) is returned as NaN on purpose — the downstream
HistGradientBoosting model handles NaN natively, so no imputation is needed.
"""
from __future__ import annotations

import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent
from eeg_seizure_analyzer.detection.confidence import (
    compute_event_quality,
    compute_local_baseline_ratio,
)
from eeg_seizure_analyzer.io.base import EEGRecording
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

# Canonical, ORDERED feature list. Training and inference both build their
# matrix from this so column order can never drift between the two paths.
FEATURE_NAMES: list[str] = [
    "duration_sec",
    "log_duration",
    # within-event morphology / spectral (recomputed from signal)
    "peak_ll_zscore",
    "peak_energy_zscore",
    "spectral_entropy",
    "dominant_freq_hz",
    "theta_delta_ratio",
    "signal_to_baseline_ratio",
    "event_rms",
    # pre-ictal contrast
    "local_baseline_ratio",
    "local_baseline_ratio_trim",
    "onset_sharpness",
    # post-ictal depression (the part no detector currently computes)
    "postictal_rms_ratio",
    "postictal_suppression_depth",
    "postictal_recovery_ratio",
]

_NAN = float("nan")


def _band_rms(
    recording: EEGRecording,
    ch: int,
    t0_sec: float,
    t1_sec: float,
    low: float = 1.0,
    high: float = 50.0,
    trim_pct: float = 0.0,
    min_sec: float = 0.5,
) -> float:
    """Band-passed RMS of ``recording`` channel ``ch`` over [t0, t1] seconds.

    Returns NaN if the window falls outside the recording or is too short —
    that propagates to the feature as a genuine "not available", which the
    tree model routes around.
    """
    fs = recording.fs
    a = int(round(t0_sec * fs))
    b = int(round(t1_sec * fs))
    a = max(0, a)
    b = min(recording.n_samples, b)
    if b - a < int(fs * min_sec):
        return _NAN
    seg = recording.data[ch, a:b]
    seg = bandpass_filter(seg, fs, low, high)
    if trim_pct > 0 and len(seg) > 10:
        cutoff = np.percentile(np.abs(seg), 100.0 - trim_pct)
        kept = seg[np.abs(seg) <= cutoff]
        if len(kept) > int(fs * min_sec):
            seg = kept
    rms = float(np.sqrt(np.mean(seg ** 2)))
    return rms if rms > 1e-12 else _NAN


def extract_event_features(
    recording: EEGRecording,
    event: DetectedEvent,
    all_events: list[DetectedEvent] | None = None,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
) -> dict[str, float]:
    """Compute the uniform re-ranker feature vector for one event.

    ``event`` times are interpreted in ``recording``'s own time frame (so at
    training time read an EDF window and pass an event whose onset/offset are
    relative to that window). Returns a dict keyed by ``FEATURE_NAMES``.
    """
    feats: dict[str, float] = {k: _NAN for k in FEATURE_NAMES}

    dur = float(event.offset_sec - event.onset_sec)
    feats["duration_sec"] = dur
    feats["log_duration"] = float(np.log10(dur + 1.0)) if dur >= 0 else _NAN

    # ── within-event spectral / amplitude (detector-agnostic) ──────────
    qm = compute_event_quality(recording, event, None, bandpass_low, bandpass_high)
    for k in ("peak_ll_zscore", "peak_energy_zscore", "spectral_entropy",
              "dominant_freq_hz", "theta_delta_ratio", "signal_to_baseline_ratio",
              "event_rms"):
        v = qm.get(k)
        if v is not None and np.isfinite(v):
            feats[k] = float(v)

    # ── pre-ictal contrast ─────────────────────────────────────────────
    lbr = compute_local_baseline_ratio(recording, event, all_events=all_events)
    feats["local_baseline_ratio"] = lbr if lbr > 0 else _NAN
    lbr_t = compute_local_baseline_ratio(
        recording, event, all_events=all_events, trim_pct=25.0
    )
    feats["local_baseline_ratio_trim"] = lbr_t if lbr_t > 0 else _NAN

    # Pre-ictal baseline RMS (shared by onset sharpness + post-ictal ratios)
    pre_rms = _band_rms(recording, event.channel,
                        event.onset_sec - 20.0, event.onset_sec - 5.0,
                        bandpass_low, bandpass_high, trim_pct=25.0)

    # Onset sharpness: RMS of the first 2 s of the event vs the pre-ictal baseline.
    onset_rms = _band_rms(recording, event.channel,
                          event.onset_sec, event.onset_sec + min(2.0, max(dur, 0.5)),
                          bandpass_low, bandpass_high)
    if np.isfinite(onset_rms) and np.isfinite(pre_rms):
        feats["onset_sharpness"] = onset_rms / pre_rms

    # ── post-ictal depression ──────────────────────────────────────────
    # Behavioural arrest / EEG suppression after a real (esp. convulsive)
    # seizure: the signal drops below pre-ictal baseline, then recovers.
    post_early = _band_rms(recording, event.channel,
                           event.offset_sec + 1.0, event.offset_sec + 11.0,
                           bandpass_low, bandpass_high)
    post_late = _band_rms(recording, event.channel,
                          event.offset_sec + 11.0, event.offset_sec + 31.0,
                          bandpass_low, bandpass_high)
    if np.isfinite(post_early) and np.isfinite(pre_rms):
        ratio = post_early / pre_rms
        feats["postictal_rms_ratio"] = ratio
        feats["postictal_suppression_depth"] = float(np.clip(1.0 - ratio, 0.0, 1.0))
    if np.isfinite(post_late) and np.isfinite(post_early):
        feats["postictal_recovery_ratio"] = post_late / post_early

    return feats


def features_to_row(feats: dict[str, float]) -> list[float]:
    """Flatten a feature dict to a row in canonical ``FEATURE_NAMES`` order."""
    return [float(feats.get(k, _NAN)) for k in FEATURE_NAMES]
