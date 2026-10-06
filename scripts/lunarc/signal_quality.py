#!/usr/bin/env python3
"""Per-channel signal-quality metrics, computed from the raw EEG only.

Purpose: provide a **model-independent** basis for excluding unusable recordings.
Excluding a channel because a detector fired on it a lot is circular -- performance
-driven exclusion. These metrics are computed from the signal alone and are blind
to any detection output, so a threshold set on them can be pre-registered and
applied to future cohorts.

Metrics, per (recording, EEG channel):

  rhythmic_duty_cycle   fraction of sampled windows whose 4-20 Hz spectral peak
                        exceeds `--peak-ratio` times the 1-45 Hz median power.
                        This is the discriminator that matters: seizures are rare
                        events (a few per day, tens of seconds each -- well under
                        1% duty cycle), whereas rhythmic artefact is persistent.
                        A channel rhythmic for 30% of its recording is not a
                        seizing animal.
  peak_freq_hz          median frequency of that 4-20 Hz peak -- the reviewer
                        observed regular ~10 Hz spiking on the bad channels.
  line_noise_ratio      50 Hz band power / 1-45 Hz median power (mains pickup).
  rms, baseline_iqr     overall amplitude and its spread.

Windows are sampled, not exhaustive: `--windows-per-channel` 10 s windows drawn at
even spacing. A duty-cycle estimate needs far less than the whole recording and
this keeps the sweep cheap (~11% of the samples at the default 60 windows).

Writes one CSV row per (file, channel). Analyse the distribution and set the
threshold at a natural gap BEFORE using it to exclude anything.

Usage:
    python scripts/lunarc/signal_quality.py \
        --edf-dir /lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025 \
        --path-include 'Batch_[1-4]_Recordings' \
        --out ~/signal_quality_ramgdnf.csv --workers 48
"""
from __future__ import annotations
import argparse, csv, os, re, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

WIN_SEC = 10.0
TARGET_FS = 250.0


def _resample_to(x: np.ndarray, fs: float, target: float) -> tuple[np.ndarray, float]:
    if fs <= target:
        return x, fs
    step = int(round(fs / target))
    return x[::step], fs / step


def _file_metrics(path: str, n_windows: int, peak_ratio: float) -> list[dict]:
    import pyedflib
    from scipy.signal import welch

    out: list[dict] = []
    try:
        r = pyedflib.EdfReader(path)
    except Exception as e:                      # unreadable file: record and move on
        return [{"path": path, "channel": -1, "error": str(e)[:120]}]
    try:
        labels = r.getSignalLabels()
        # EEG channels only: the Biopot amplifier labels them '<n> Biopot';
        # activity channels are '<n> Act' and are not EEG.
        eeg = [i for i, l in enumerate(labels) if "biopot" in l.lower()]
        if not eeg:
            eeg = list(range(min(8, r.signals_in_file)))
        for ch in eeg:
            fs = float(r.getSampleFrequency(ch))
            n = int(r.getNSamples()[ch])
            win = int(WIN_SEC * fs)
            if n < win * 2:
                continue
            starts = np.linspace(0, n - win, num=min(n_windows, max(1, n // win)),
                                 dtype=np.int64)
            ratios, peaks, lines, rmss = [], [], [], []
            for s in starts:
                try:
                    seg = r.readSignal(ch, start=int(s), n=win).astype(np.float64)
                except Exception:
                    continue
                if not np.isfinite(seg).all() or seg.std() == 0:
                    continue
                seg, fs2 = _resample_to(seg, fs, TARGET_FS)
                f, p = welch(seg, fs=fs2, nperseg=min(len(seg), int(fs2 * 2)))
                band = (f >= 1) & (f <= 45)
                if not band.any():
                    continue
                med = float(np.median(p[band])) or 1e-20
                sel = (f >= 4) & (f <= 20)
                if sel.any():
                    i = int(np.argmax(p[sel]))
                    ratios.append(float(p[sel][i]) / med)
                    peaks.append(float(f[sel][i]))
                ln = (f >= 48) & (f <= 52)
                if ln.any():
                    lines.append(float(np.max(p[ln])) / med)
                rmss.append(float(np.sqrt(np.mean(seg ** 2))))
            if not ratios:
                continue
            ratios = np.asarray(ratios)
            out.append({
                "path": path, "channel": ch, "label": labels[ch],
                "n_windows": len(ratios),
                "rhythmic_duty_cycle": round(float(np.mean(ratios > peak_ratio)), 4),
                "peak_ratio_median": round(float(np.median(ratios)), 3),
                "peak_ratio_p95": round(float(np.percentile(ratios, 95)), 3),
                "peak_freq_hz": round(float(np.median(peaks)), 2) if peaks else None,
                "line_noise_ratio": round(float(np.median(lines)), 3) if lines else None,
                "rms": round(float(np.median(rmss)), 6),
                "baseline_iqr": round(float(np.subtract(*np.percentile(rmss, [75, 25]))), 6),
                "error": "",
            })
    finally:
        try:
            r.close()
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--path-include", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--windows-per-channel", type=int, default=60)
    ap.add_argument("--peak-ratio", type=float, default=10.0,
                    help="a window counts as rhythmic when its 4-20 Hz peak exceeds "
                         "this multiple of the 1-45 Hz median power")
    a = ap.parse_args()

    pat = re.compile(a.path_include) if a.path_include else None
    files = [str(p) for p in Path(a.edf_dir).rglob("*.edf")
             if pat is None or pat.search(str(p))]
    files.sort()
    print(f"{len(files)} EDFs | {a.workers} workers | "
          f"{a.windows_per_channel} windows/channel | peak_ratio={a.peak_ratio}",
          flush=True)

    fields = ["path", "channel", "label", "n_windows", "rhythmic_duty_cycle",
              "peak_ratio_median", "peak_ratio_p95", "peak_freq_hz",
              "line_noise_ratio", "rms", "baseline_iqr", "error"]
    done = 0
    with open(os.path.expanduser(a.out), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_file_metrics, f, a.windows_per_channel, a.peak_ratio): f
                    for f in files}
            for fut in as_completed(futs):
                for row in fut.result():
                    w.writerow(row)
                done += 1
                if done % 100 == 0:
                    fp.flush()
                    print(f"  {done}/{len(files)} files", flush=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
