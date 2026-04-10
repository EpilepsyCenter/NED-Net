"""Ensemble seizure detection.

Runs multiple sub-detectors on the same recording and combines their
results via temporal overlap voting.  An event survives if at least
``voting_threshold`` methods detect it.
"""

from __future__ import annotations

import numpy as np

from eeg_seizure_analyzer.config import EnsembleParams
from eeg_seizure_analyzer.detection.base import DetectedEvent, DetectorBase
from eeg_seizure_analyzer.io.base import EEGRecording


class EnsembleDetector(DetectorBase):
    """Combine results from multiple detectors using vote-based merging.

    Unlike other detectors, ``detect()`` is not used directly.
    Call ``detect_ensemble()`` instead, passing pre-run event lists.
    """

    def detect(
        self,
        recording: EEGRecording,
        channel: int,
        **params,
    ) -> list[DetectedEvent]:
        raise NotImplementedError(
            "EnsembleDetector does not support detect() directly. "
            "Use detect_ensemble() with pre-computed per-method event lists."
        )

    def detect_ensemble(
        self,
        method_events: dict[str, list[DetectedEvent]],
        params: EnsembleParams | None = None,
    ) -> list[DetectedEvent]:
        """Combine events from multiple methods using temporal overlap voting.

        Parameters
        ----------
        method_events : dict[str, list[DetectedEvent]]
            Mapping of method name → list of events from that method.
            Example: {"spike_train": [...], "spectral_band": [...]}
        params : EnsembleParams, optional

        Returns
        -------
        list[DetectedEvent]
            Events that pass the voting threshold, with merged boundaries.
        """
        if params is None:
            params = EnsembleParams()

        # Collect all events with their source method
        tagged: list[tuple[str, DetectedEvent]] = []
        for method_name, events in method_events.items():
            for ev in events:
                tagged.append((method_name, ev))

        if not tagged:
            return []

        # For each event, count how many other methods have an overlapping event
        survivors: list[DetectedEvent] = []
        used_methods_per_event: list[set[str]] = []

        for method_name, ev in tagged:
            supporting = {method_name}
            for other_method, other_ev in tagged:
                if other_method == method_name:
                    continue
                if other_ev.channel != ev.channel:
                    continue
                # Check temporal overlap
                if _events_overlap(ev, other_ev):
                    supporting.add(other_method)

            if len(supporting) >= params.voting_threshold:
                survivors.append(ev)
                used_methods_per_event.append(supporting)

        if not survivors:
            return []

        # Merge overlapping survivors into single events
        merged = _merge_voted_events(
            survivors,
            used_methods_per_event,
            merge_strategy=params.merge_strategy,
            confidence_merge=params.confidence_merge,
        )

        # Store detection info
        self._last_detection_info = {
            "channel": survivors[0].channel if survivors else 0,
            "methods_used": list(method_events.keys()),
            "voting_threshold": params.voting_threshold,
            "n_candidates": len(tagged),
            "n_survivors": len(merged),
            # Empty spike/SBI lists for inspector compatibility
            "baseline_mean": 0,
            "baseline_std": 0,
            "threshold": 0,
            "all_spike_times": [],
            "all_spike_amplitudes": [],
            "all_spike_samples": [],
        }

        # Collect spike info from all contributing events
        all_spike_times = []
        all_spike_amps = []
        all_spike_samples = []
        for ev in merged:
            all_spike_times.extend(ev.features.get("spike_times", []))
            all_spike_amps.extend(ev.features.get("spike_amplitudes", []))
            all_spike_samples.extend(ev.features.get("spike_samples", []))
        self._last_detection_info["all_spike_times"] = all_spike_times
        self._last_detection_info["all_spike_amplitudes"] = all_spike_amps
        self._last_detection_info["all_spike_samples"] = all_spike_samples

        return sorted(merged, key=lambda e: e.onset_sec)


# ── Helpers ─────────────────────────────────────────────────────────


def _events_overlap(a: DetectedEvent, b: DetectedEvent) -> bool:
    """Check if two events have any temporal overlap."""
    return a.onset_sec < b.offset_sec and b.onset_sec < a.offset_sec


def _merge_voted_events(
    events: list[DetectedEvent],
    methods_per_event: list[set[str]],
    merge_strategy: str = "union",
    confidence_merge: str = "mean",
) -> list[DetectedEvent]:
    """Merge overlapping events that passed the vote into single events.

    Groups events by channel and temporal overlap, then merges each group.
    """
    if not events:
        return []

    # Sort by channel then onset
    indexed = list(enumerate(events))
    indexed.sort(key=lambda x: (x[1].channel, x[1].onset_sec))

    groups: list[list[int]] = []  # groups of original indices
    current_group = [indexed[0][0]]
    current_end = indexed[0][1].offset_sec
    current_ch = indexed[0][1].channel

    for orig_idx, ev in indexed[1:]:
        if ev.channel == current_ch and ev.onset_sec < current_end:
            current_group.append(orig_idx)
            current_end = max(current_end, ev.offset_sec)
        else:
            groups.append(current_group)
            current_group = [orig_idx]
            current_end = ev.offset_sec
            current_ch = ev.channel
    groups.append(current_group)

    merged: list[DetectedEvent] = []

    for group_indices in groups:
        group_events = [events[i] for i in group_indices]
        group_methods: set[str] = set()
        for i in group_indices:
            group_methods.update(methods_per_event[i])

        # Merge boundaries
        if merge_strategy == "intersection":
            onset = max(ev.onset_sec for ev in group_events)
            offset = min(ev.offset_sec for ev in group_events)
            if offset <= onset:
                # No intersection — fall back to union
                onset = min(ev.onset_sec for ev in group_events)
                offset = max(ev.offset_sec for ev in group_events)
        else:  # "union"
            onset = min(ev.onset_sec for ev in group_events)
            offset = max(ev.offset_sec for ev in group_events)

        duration = offset - onset

        # Merge confidence
        confidences = [ev.confidence for ev in group_events]
        if confidence_merge == "max":
            confidence = max(confidences)
        else:  # "mean"
            confidence = sum(confidences) / len(confidences)

        # Severity from duration
        if duration > 45:
            severity = "severe"
        elif duration > 15:
            severity = "moderate"
        else:
            severity = "mild"

        # Merge features: collect spike info from all contributing events
        all_spike_times = []
        all_spike_amps = []
        all_spike_samples = []
        for ev in group_events:
            all_spike_times.extend(ev.features.get("spike_times", []))
            all_spike_amps.extend(ev.features.get("spike_amplitudes", []))
            all_spike_samples.extend(ev.features.get("spike_samples", []))

        # Deduplicate spike times (within 5ms tolerance)
        if all_spike_times:
            all_spike_times, all_spike_amps, all_spike_samples = _dedupe_spikes(
                all_spike_times, all_spike_amps, all_spike_samples, tol=0.005
            )

        n_spikes = max(
            (ev.features.get("n_spikes", 0) for ev in group_events), default=0
        )

        merged.append(
            DetectedEvent(
                onset_sec=round(onset, 4),
                offset_sec=round(offset, 4),
                duration_sec=round(duration, 4),
                channel=group_events[0].channel,
                event_type="seizure",
                confidence=round(confidence, 3),
                severity=severity,
                features=_build_ensemble_features(
                    group_methods, n_spikes,
                    all_spike_times, all_spike_amps, all_spike_samples,
                ),
            )
        )

    return merged


def _build_ensemble_features(
    group_methods: set[str],
    n_spikes: int,
    spike_times: list[float],
    spike_amps: list[float],
    spike_samples: list[int],
) -> dict:
    """Build features dict for an ensemble event."""
    # ISI stats from merged spike times
    isis_ms = []
    if len(spike_times) > 1:
        sorted_times = sorted(spike_times)
        for i in range(len(sorted_times) - 1):
            isis_ms.append((sorted_times[i + 1] - sorted_times[i]) * 1000)
    mean_isi = float(np.mean(isis_ms)) if isis_ms else 0.0
    isi_cv = float(np.std(isis_ms) / np.mean(isis_ms)) if isis_ms and np.mean(isis_ms) > 0 else 0.0

    return {
        "detection_method": "ensemble",
        "seizure_subtype": "seizure",
        "contributing_methods": sorted(group_methods),
        "n_methods": len(group_methods),
        "n_spikes": n_spikes,
        "mean_spike_frequency_hz": round(float(n_spikes / max(1e-6, sorted(spike_times)[-1] - sorted(spike_times)[0])), 2) if len(spike_times) > 1 else 0.0,
        "mean_isi_ms": round(mean_isi, 2),
        "spike_regularity": round(isi_cv, 3),
        "mean_amplitude_uv": round(float(np.mean(spike_amps)), 2) if spike_amps else 0.0,
        "max_amplitude_uv": round(float(np.max(spike_amps)), 2) if spike_amps else 0.0,
        "spike_times": spike_times,
        "spike_amplitudes": spike_amps,
        "spike_samples": spike_samples,
    }


def _dedupe_spikes(
    times: list[float],
    amps: list[float],
    samples: list[int],
    tol: float = 0.005,
) -> tuple[list[float], list[float], list[int]]:
    """Deduplicate spikes within tolerance (keep highest amplitude)."""
    if not times:
        return times, amps, samples

    # Sort by time
    order = sorted(range(len(times)), key=lambda i: times[i])
    t = [times[i] for i in order]
    a = [amps[i] for i in order]
    s = [samples[i] for i in order]

    deduped_t = [t[0]]
    deduped_a = [a[0]]
    deduped_s = [s[0]]

    for i in range(1, len(t)):
        if t[i] - deduped_t[-1] < tol:
            # Duplicate — keep higher amplitude
            if a[i] > deduped_a[-1]:
                deduped_t[-1] = t[i]
                deduped_a[-1] = a[i]
                deduped_s[-1] = s[i]
        else:
            deduped_t.append(t[i])
            deduped_a.append(a[i])
            deduped_s.append(s[i])

    return deduped_t, deduped_a, deduped_s
