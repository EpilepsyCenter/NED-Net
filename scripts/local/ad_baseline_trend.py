#!/usr/bin/env python
"""Does the AD spike-rate rise reflect activity, or a falling noise floor?

The classical detector is baseline-relative: the amplitude threshold is
``baseline_mean + z * baseline_std`` and the dominant post-filter is local SNR,
both measured against the background. So a background that quietens over the
three weeks — electrode encapsulation is the usual cause and develops on
exactly this timescale — raises the spike count with no change in the
underlying activity. This measures the background directly so the two
explanations can be told apart.

For each sampled file it recomputes exactly what the detector computes:
bandpass 3-50 Hz, then compute_zscore_baseline(window 30 s, percentile 25),
giving baseline_mean (≈ RMS of the quiet periods) and the resulting detection
threshold. If baseline_mean is flat while spike rate climbs, the trend is real.

Sampling: one file per day by default (the same file index each day, so
time-of-day is held roughly constant) — 21 files rather than 319, because the
EDFs stream over a ~10 MB/s share and the full set is 50 GB.

    python scripts/local/ad_baseline_trend.py
    python scripts/local/ad_baseline_trend.py --file-index 1 5 --out baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

DEFAULT_ROOT = ("/Volumes/research/LU26D1055-epicenter/Data/KAHA recordings/"
                "AD_Animals_Recordings")
# code channel -> animal ID, as in ad_metadata.csv.
CHANNELS = {0: "1", 3: "2", 5: "3", 6: "4"}
# Detector settings the LUNARC run used (detect_spikes_batch._build_params).
BP_LOW, BP_HIGH = 3.0, 50.0
BL_WINDOW_SEC, BL_PERCENTILE = 30.0, 25
ZSCORE = 4.0

_DAY_RE = re.compile(r"/(W(\d+)_D(\d+))/")
_IDX_RE = re.compile(r"\((\d+)\)\.edf$", re.IGNORECASE)


def pick_files(root: str, indices: list[int]) -> list[tuple[str, str, int, str]]:
    """-> [(week, day, day_number, path)] for the chosen per-day file indices."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.edf"), recursive=True)):
        m = _DAY_RE.search(path)
        i = _IDX_RE.search(path)
        if not (m and i) or int(i.group(1)) not in indices:
            continue
        week, wk_n, day_n = m.group(1), int(m.group(2)), int(m.group(3))
        out.append((f"Week_{wk_n}", week, (wk_n - 1) * 7 + day_n, path))
    return sorted(out, key=lambda t: (t[2], t[3]))


def measure(path: str) -> dict[int, tuple[float, float]]:
    """-> {channel: (baseline_mean, baseline_std)} for the mapped channels."""
    from eeg_seizure_analyzer.io.edf_reader import read_edf
    from eeg_seizure_analyzer.processing.features import compute_zscore_baseline
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

    rec = read_edf(path, channels=sorted(CHANNELS))
    out = {}
    for pos, ch in enumerate(sorted(CHANNELS)):
        data = rec.get_channel_data(pos)
        filt = bandpass_filter(data, rec.fs, BP_LOW, BP_HIGH)
        out[ch] = compute_zscore_baseline(
            filt, rec.fs, window_sec=BL_WINDOW_SEC, percentile=BL_PERCENTILE)
    return out


def spearman(xs, ys):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    if len(xs) < 3:
        return None
    r = spearmanr(xs, ys)
    return float(r.statistic), float(r.pvalue)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--file-index", type=int, nargs="+", default=[1],
                    help="which per-day file number(s) to sample, e.g. 1 5")
    ap.add_argument("--out", default="ad_baseline.csv")
    args = ap.parse_args()

    files = pick_files(args.root, args.file_index)
    if not files:
        print(f"No EDFs matching index {args.file_index} under {args.root}",
              file=sys.stderr)
        return 1
    print(f"Sampling {len(files)} files (index {args.file_index} of each day)\n")

    rows = []
    per_week = defaultdict(list)   # (week, animal) -> [baseline_mean]
    per_day = defaultdict(list)    # animal -> [(day_number, baseline_mean)]
    for week, day, day_n, path in files:
        try:
            res = measure(path)
        except Exception as e:  # one unreadable file must not stop the sweep
            print(f"  !! {os.path.basename(path)}: {e!r}")
            continue
        for ch, (mean, std) in sorted(res.items()):
            animal = CHANNELS[ch]
            rows.append({"week": week, "day": day, "day_number": day_n,
                         "animal": animal, "channel": ch,
                         "baseline_mean_uv": round(mean, 4),
                         "baseline_std_uv": round(std, 4),
                         "threshold_uv": round(mean + ZSCORE * std, 4),
                         "file": os.path.basename(path)})
            per_week[(week, animal)].append(mean)
            per_day[animal].append((day_n, mean))
        print(f"  {day}: " + "  ".join(
            f"a{CHANNELS[c]}={res[c][0]:.1f}uV" for c in sorted(res)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    weeks = sorted({r["week"] for r in rows})
    animals = sorted({r["animal"] for r in rows})

    print("\nBaseline amplitude (uV, mean of quiet-period |signal|), by week")
    print(f"  {'animal':<8}" + "".join(f"{w:>10}" for w in weeks) + f"{'W1->W3':>10}")
    for a in animals:
        means = []
        for w in weeks:
            vals = per_week.get((w, a), [])
            means.append(sum(vals) / len(vals) if vals else 0.0)
        chg = f"{(means[-1] / means[0] - 1) * 100:+.0f}%" if means[0] else "-"
        print(f"  {a:<8}" + "".join(f"{m:>10.2f}" for m in means) + f"{chg:>10}")

    print("\nBaseline trend across days (Spearman rho vs day number)")
    for a in animals:
        pts = sorted(per_day[a])
        s = spearman([p[0] for p in pts], [p[1] for p in pts])
        if s is None:
            print(f"  animal {a}: (scipy unavailable)")
        else:
            rho, p = s
            print(f"  animal {a}: rho={rho:+.2f}  p={p:.3f}"
                  f"{' *' if p < 0.05 else ''}  (n={len(pts)})")

    print(f"\nWrote {args.out}")
    print("Read it as: baseline FALLING while spike rate RISES => the rate trend "
          "is at least partly a moving threshold, not more activity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
