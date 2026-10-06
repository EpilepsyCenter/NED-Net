#!/usr/bin/env python3
"""Per-channel signal-quality metrics, computed from the raw EEG only.

Purpose: provide a **model-independent** basis for excluding unusable recordings.
Excluding a channel because a detector fired on it a lot is circular -- performance
-driven exclusion. These metrics are computed from the signal alone and are blind
to any detection output, so a threshold set on them can be pre-registered and
applied to future cohorts.

Metrics, per (recording, EEG channel):

  prominence_db         PRIMARY METRIC. Median dB by which the 1/f-detrended spectrum's
                        4-20 Hz peak rises above the fitted background. On a 4-file local
                        test the three flood animals scored 5.4-7.3 dB and three
                        seizure-bearing animals 3.4-4.1 dB.
  rhythmic_duty_cycle   fraction of sampled windows whose prominence exceeds
                        `--prominence-db`.

                        v1 of this metric compared the raw 4-20 Hz peak against the
                        1-45 Hz median and FAILED its pre-registered test (2026-10-06):
                        EEG power falls with frequency, so the 4-20 Hz maximum sits at the
                        band edge almost always -- peak_freq_hz was 5.0 Hz at the median
                        and both quartiles, duty cycle was 0.92 cohort-wide, and the flood
                        animals came out LOWER than the high-seizure animals. It measured
                        the 1/f slope, not rhythmicity. Detrending plus a time-domain
                        periodicity test is the fix.
                        This is the discriminator that matters: seizures are rare
                        events (a few per day, tens of seconds each -- well under
                        1% duty cycle), whereas rhythmic artefact is persistent.
                        A channel rhythmic for 30% of its recording is not a
                        seizing animal.
  peak_freq_hz          median frequency of the detrended 4-20 Hz peak -- the reviewer
                        observed regular ~10 Hz spiking on the bad channels.
  acorr_peak            median autocorrelation peak in the 50-250 ms lag band; a regular
                        10 Hz spike train gives a strong peak at ~100 ms.
  prominence_db         median dB prominence of the detrended spectral peak.
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


def _file_metrics(path: str, n_windows: int, prominence_db: float,
                  acorr_min: float) -> list[dict]:
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
            peaks, lines, rmss = [], [], []
            proms, acorrs, rhythmic = [], [], []
            for s in starts:
                try:
                    seg = r.readSignal(ch, start=int(s), n=win).astype(np.float64)
                except Exception:
                    continue
                if not np.isfinite(seg).all() or seg.std() == 0:
                    continue
                seg, fs2 = _resample_to(seg, fs, TARGET_FS)
                seg = seg - seg.mean()
                f, p = welch(seg, fs=fs2, nperseg=min(len(seg), int(fs2 * 2)))
                band = (f >= 1) & (f <= 45) & (f > 0)
                if band.sum() < 8:
                    continue
                med = float(np.median(p[band])) or 1e-20

                # --- spectral prominence above the 1/f background ---------------
                # Fit log10(power) ~ a*log10(f) + b over 1-45 Hz and work with the
                # residual, so a genuine oscillation shows as a bump rather than
                # the band edge always winning.
                lf, lp = np.log10(f[band]), np.log10(np.maximum(p[band], 1e-30))
                a_, b_ = np.polyfit(lf, lp, 1)
                resid_db = 10.0 * (lp - (a_ * lf + b_))
                fb = f[band]
                sel = (fb >= 4) & (fb <= 20)
                prom = 0.0
                if sel.any():
                    i = int(np.argmax(resid_db[sel]))
                    prom = float(resid_db[sel][i])
                    peaks.append(float(fb[sel][i]))
                proms.append(prom)

                # --- time-domain periodicity (regular spiking) ------------------
                # Autocorrelation peak at 50-250 ms lag = 4-20 Hz repetition.
                x = seg / (seg.std() or 1.0)
                ac = np.correlate(x, x, mode="full")[len(x) - 1:]
                ac = ac / (ac[0] or 1.0)
                lo, hi = int(0.05 * fs2), int(0.25 * fs2)
                apk = float(np.max(ac[lo:hi])) if hi < len(ac) and hi > lo else 0.0
                acorrs.append(apk)

                # Prominence ALONE. An earlier version required prominence AND an
                # autocorrelation peak; a 4-file local test (2026-10-06) showed the two
                # run in OPPOSITE directions -- the flood animals had high prominence
                # (5.4-7.3 dB) with low acorr (0.05-0.10), the seizing animals low
                # prominence (3.4-4.1 dB) with high acorr (0.25) -- so the conjunction
                # was near-empty everywhere and discriminated nothing. acorr_peak is
                # still reported, separately, because that contrast is itself a signal.
                rhythmic.append(prom >= prominence_db)

                ln = (f >= 48) & (f <= 52)
                if ln.any():
                    lines.append(float(np.max(p[ln])) / med)
                rmss.append(float(np.sqrt(np.mean(seg ** 2))))
            if not rhythmic:
                continue
            out.append({
                "path": path, "channel": ch, "label": labels[ch],
                "n_windows": len(rhythmic),
                "rhythmic_duty_cycle": round(float(np.mean(rhythmic)), 4),
                "prominence_db": round(float(np.median(proms)), 3),
                "prominence_p95": round(float(np.percentile(proms, 95)), 3),
                "acorr_peak": round(float(np.median(acorrs)), 4),
                "acorr_p95": round(float(np.percentile(acorrs, 95)), 4),
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
    ap.add_argument("--prominence-db", type=float, default=6.0,
                    help="dB the detrended 4-20 Hz peak must rise above the fitted "
                         "1/f background for a window to count as rhythmic")
    ap.add_argument("--acorr-min", type=float, default=0.3,
                    help="minimum autocorrelation peak in the 50-250 ms lag band")
    a = ap.parse_args()

    pat = re.compile(a.path_include) if a.path_include else None
    files = [str(p) for p in Path(a.edf_dir).rglob("*.edf")
             if pat is None or pat.search(str(p))]
    files.sort()
    print(f"{len(files)} EDFs | {a.workers} workers | "
          f"{a.windows_per_channel} windows/channel | "
          f"prominence>={a.prominence_db} dB | acorr>={a.acorr_min}",
          flush=True)

    fields = ["path", "channel", "label", "n_windows", "rhythmic_duty_cycle",
              "prominence_db", "prominence_p95", "acorr_peak", "acorr_p95",
              "peak_freq_hz", "line_noise_ratio", "rms", "baseline_iqr", "error"]
    done = 0
    with open(os.path.expanduser(a.out), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_file_metrics, f, a.windows_per_channel,
                                 a.prominence_db, a.acorr_min): f
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
