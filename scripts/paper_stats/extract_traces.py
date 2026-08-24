#!/usr/bin/env python
"""Extract representative EEG traces + NED-Net model output for Supplementary Fig. 3.

For each chosen event, reads the raw EDF window, runs the trained U-Net over it
exactly as batch detection does (decimate to 250 Hz, z-score per window, 60 s
windows with 15 s overlap, average overlapping predictions), and saves the
signal together with the per-sample seizure and convulsive probabilities.

Also extracts one window of rule-based interictal-spike detections.

    python scripts/paper_stats/extract_traces.py

Writes ``traces.npz`` next to this script.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import torch

from eeg_seizure_analyzer.io.edf_reader import read_edf_window
from eeg_seizure_analyzer.ml.dataset import _downsample, _normalize_channels, _pad_or_trim
from eeg_seizure_analyzer.ml.train import load_trained_model

HERE = Path(__file__).parent
EDF_ROOT = "/Users/marcoledri/Software/edf/SV2A_2024/"
SEIZ_DB = os.path.expanduser("~/.eeg_seizure_analyzer/projects/lunarc_detect_wk1-6_final.db")
SPIKE_DB = os.path.expanduser("~/.eeg_seizure_analyzer/projects/sv2a_spikes_wk1-6.db")
MODEL = "UNetv2_20260615"

# (label, chunk_id, channel, event_start_sec, event_duration_sec, pad_sec)
EVENTS = [
    ("convulsive", 695, 3, 4379.700, 19.600, 20.0),
    ("nonconvulsive", 1364, 4, 1703.864, 16.076, 20.0),
]
# Interictal-spike window: (chunk_id, channel, start_sec, duration_sec).
# NOTE: chunk_id here indexes the SPIKE database, whose numbering is independent
# of the seizure database's.
# Animal 355669 (SV2A). Must be an animal INCLUDED in the analysis — 30, 355676
# and 372837 are excluded everywhere as too noisy, so no panel may show them.
SPIKE_WINDOW = (86, 6, 51.68, 11.5)

TARGET_FS = 250
WINDOW_SEC = 60
OVERLAP_SEC = 15


def _local_path(db: str, chunk_id: int) -> str:
    conn = sqlite3.connect(db)
    path = conn.execute("SELECT path FROM chunks WHERE id=?", (chunk_id,)).fetchone()[0]
    return EDF_ROOT + path.split("edf_data/", 1)[1]


def _run_model(model, device, edf_path: str, channel: int,
               start_sec: float, span_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signal, seizure_prob, convulsive_prob) at 250 Hz over the span."""
    n_out = int(span_sec * TARGET_FS)
    pred = np.zeros(n_out); conv = np.zeros(n_out); cnt = np.zeros(n_out)
    stride = WINDOW_SEC - OVERLAP_SEC
    n_chunks = max(1, int(np.ceil((span_sec - WINDOW_SEC) / stride)) + 1)
    target_samples = TARGET_FS * WINDOW_SEC

    for i in range(n_chunks):
        w0 = i * stride
        rec = read_edf_window(edf_path, channels=[channel],
                              start_sec=start_sec + w0, duration_sec=WINDOW_SEC)
        eeg = _pad_or_trim(_normalize_channels(_downsample(rec.data, rec.fs, TARGET_FS)),
                           target_samples)
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.from_numpy(eeg).unsqueeze(0).to(device))
            ).squeeze(0).cpu().numpy()
        s = int(w0 * TARGET_FS)
        n = min(probs.shape[1], n_out - s)
        if n <= 0:
            continue
        pred[s:s + n] += probs[0, :n]
        conv[s:s + n] += probs[1, :n]
        cnt[s:s + n] += 1.0

    cnt[cnt == 0] = 1.0
    rec = read_edf_window(edf_path, channels=[channel],
                          start_sec=start_sec, duration_sec=span_sec)
    sig = _downsample(rec.data, rec.fs, TARGET_FS)[0][:n_out]
    return sig, pred / cnt, conv / cnt


def main() -> None:
    model, meta = load_trained_model(MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"model {MODEL} on {device}")

    out: dict[str, np.ndarray] = {}
    for label, cid, ch, ev_start, ev_dur, pad in EVENTS:
        path = _local_path(SEIZ_DB, cid)
        start = max(0.0, ev_start - pad)
        span = ev_dur + 2 * pad
        sig, p, c = _run_model(model, device, path, ch, start, span)
        out[f"{label}_signal"] = sig.astype(np.float32)
        out[f"{label}_prob"] = p.astype(np.float32)
        out[f"{label}_conv"] = c.astype(np.float32)
        out[f"{label}_event"] = np.array([ev_start - start, ev_start - start + ev_dur])
        print(f"{label}: {span:.0f}s span, peak p={p.max():.3f}, "
              f"mean conv p={c[int((ev_start-start)*TARGET_FS):int((ev_start-start+ev_dur)*TARGET_FS)].mean():.3f}")

    # Interictal spikes — rule-based detections from the spike DB
    cid, ch, s0, dur = SPIKE_WINDOW
    path = _local_path(SPIKE_DB, cid)
    rec = read_edf_window(path, channels=[ch], start_sec=s0, duration_sec=dur)
    out["spike_signal"] = _downsample(rec.data, rec.fs, TARGET_FS)[0].astype(np.float32)
    conn = sqlite3.connect(SPIKE_DB)
    rows = conn.execute(
        "SELECT start_sec, duration_sec FROM events WHERE chunk_id=? AND channel=? "
        "AND start_sec>=? AND start_sec<? AND excluded=0 ORDER BY start_sec",
        (cid, ch, s0, s0 + dur)).fetchall()
    out["spike_times"] = np.array([r[0] - s0 + r[1] / 2 for r in rows])
    print(f"spikes: {len(rows)} detections in {dur:.0f}s window")

    out["fs"] = np.array([TARGET_FS])
    np.savez(HERE / "traces.npz", **out)
    print(f"wrote {HERE / 'traces.npz'}")


if __name__ == "__main__":
    main()
