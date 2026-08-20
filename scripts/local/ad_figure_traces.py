#!/usr/bin/env python
"""Extract representative raw traces for the AD Extended Data figure.

Exports plain CSV for Prism — no plotting here.

Per week it picks ONE 30-min window whose local spike rate is as close as
possible to that week's cohort mean rate, so the panel is representative
rather than the best-looking stretch, and records what it chose (file, offset,
rate) so the choice is reproducible and reportable in the legend.

Animal 4 is the default: its weekly rates (9.2 / 9.9 / 15.0 per h) track the
cohort means (8.2 / 12.6 / 16.4) within ~20% and it shows the increase, so a
single animal can carry all three panels — a within-animal comparison is
stronger than three different animals.

Outputs (--outdir, default ad_figure/):
    week{N}_long.csv    time_s, raw_uv, filt_uv  — 30 min, decimated
    week{N}_zoom.csv    time_s, raw_uv, filt_uv  — 10 s at full rate,
                        centred on the median-confidence spike in that window
    spike_times.csv     week, t_sec (relative to window start) — for tick marks
    windows.csv         what was selected, for the figure legend / methods

    python scripts/local/ad_figure_traces.py
    python scripts/local/ad_figure_traces.py --animal 2 --long-fs 125
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sqlite3
import sys

DEFAULT_DB = os.path.expanduser("~/.eeg_seizure_analyzer/projects/ad_spikes.db")
DEFAULT_ROOT = ("/Volumes/research/LU26D1055-epicenter/Data/KAHA recordings/"
                "AD_Animals_Recordings")
ANIMAL_TO_CH = {"1": 0, "2": 3, "3": 5, "4": 6}
BP_LOW, BP_HIGH = 3.0, 50.0
_WEEK_RE = re.compile(r"/(Week_\d+)/")


def cohort_week_rates(db: str) -> dict[str, float]:
    """Cohort mean spikes/hour per week — the target each window matches."""
    conn = sqlite3.connect(db)
    try:
        hours, spikes = {}, {}
        for path, sec in conn.execute(
                "SELECT c.path, SUM(f.valid_sec) FROM file_animals f "
                "JOIN chunks c ON f.chunk_id = c.id GROUP BY c.path"):
            m = _WEEK_RE.search(path)
            if m:
                hours[m.group(1)] = hours.get(m.group(1), 0) + (sec or 0) / 3600
        for path, n in conn.execute(
                "SELECT c.path, COUNT(*) FROM events e "
                "JOIN chunks c ON e.chunk_id = c.id "
                "WHERE e.type='interictal_spike' GROUP BY c.path"):
            m = _WEEK_RE.search(path)
            if m:
                spikes[m.group(1)] = spikes.get(m.group(1), 0) + n
    finally:
        conn.close()
    return {w: spikes.get(w, 0) / h for w, h in hours.items() if h}


def animal_events(db: str, animal: str) -> dict[str, list[tuple[float, float]]]:
    """-> {file_path: [(onset_sec, confidence)]} for one animal."""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT c.path, e.start_sec, e.cnn_confidence FROM events e "
            "JOIN chunks c ON e.chunk_id = c.id "
            "WHERE e.type='interictal_spike' AND e.animal_id = ? "
            "ORDER BY c.path, e.start_sec", (animal,)).fetchall()
    finally:
        conn.close()
    out: dict[str, list[tuple[float, float]]] = {}
    for path, t, conf in rows:
        out.setdefault(path, []).append((t, conf or 0.0))
    return out


def pick_window(events: dict, target_rate: float, week: str,
                win_sec: float, step_sec: float, file_len_sec: float):
    """Best (path, start_sec, spikes_in_window) for one week.

    "Best" = windowed rate closest to the cohort mean for that week; ties go to
    the window with more spikes, so the panel is not an empty stretch.
    """
    best = None
    for path, evs in events.items():
        if not _WEEK_RE.search(path) or _WEEK_RE.search(path).group(1) != week:
            continue
        start = 0.0
        while start + win_sec <= file_len_sec:
            inside = [t for t, _c in evs if start <= t < start + win_sec]
            rate = len(inside) / (win_sec / 3600.0)
            key = (abs(rate - target_rate), -len(inside))
            if best is None or key < best[0]:
                best = (key, path, start, len(inside), rate)
            start += step_sec
    return best


def load_window(path: str, ch: int, start: float, dur: float):
    """-> (fs, raw, filtered) for one channel over [start, start+dur)."""
    from eeg_seizure_analyzer.io.edf_reader import read_edf
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

    rec = read_edf(path, channels=[ch])
    fs = rec.fs
    data = rec.get_channel_data(0)
    # Filter the whole record, then slice: filtering a short slice would ring at
    # the edges of the window we are about to plot.
    filt = bandpass_filter(data, fs, BP_LOW, BP_HIGH)
    i0, i1 = int(start * fs), int((start + dur) * fs)
    return fs, data[i0:i1], filt[i0:i1]


def write_trace(out_path: str, fs: float, raw, filt, out_fs: float | None):
    import numpy as np
    if out_fs and out_fs < fs:
        from scipy.signal import decimate
        q = int(round(fs / out_fs))
        # Anti-aliased decimation, not naive slicing, so spike shape survives.
        raw, filt = decimate(raw, q, ftype="fir"), decimate(filt, q, ftype="fir")
        fs = fs / q
    t = np.arange(len(raw)) / fs
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "raw_uv", "filt_uv"])
        for i in range(len(raw)):
            w.writerow([f"{t[i]:.4f}", f"{raw[i]:.3f}", f"{filt[i]:.3f}"])
    return fs, len(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--animal", default="4", choices=sorted(ANIMAL_TO_CH))
    ap.add_argument("--outdir", default="ad_figure")
    ap.add_argument("--long-sec", type=float, default=1800.0)
    ap.add_argument("--zoom-sec", type=float, default=10.0)
    ap.add_argument("--long-fs", type=float, default=250.0,
                    help="decimate the long trace to this (0 = keep 2 kHz)")
    ap.add_argument("--step-sec", type=float, default=300.0,
                    help="window search stride")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1
    os.makedirs(args.outdir, exist_ok=True)

    ch = ANIMAL_TO_CH[args.animal]
    targets = cohort_week_rates(args.db)
    events = animal_events(args.db, args.animal)
    weeks = sorted(targets)
    print(f"animal {args.animal} (code ch{ch})   cohort targets: "
          + ", ".join(f"{w}={targets[w]:.1f}/h" for w in weeks) + "\n")

    spike_rows, win_rows = [], []
    for week in weeks:
        pick = pick_window(events, targets[week], week, args.long_sec,
                           args.step_sec, 5400.0)
        if not pick:
            print(f"  !! no window found for {week}")
            continue
        _key, db_path, start, n_sp, rate = pick
        base = os.path.basename(db_path)
        local = glob.glob(os.path.join(args.root, "**", base), recursive=True)
        if not local:
            print(f"  !! {base} not found under {args.root}")
            continue
        print(f"  {week}: {base} +{start:.0f}s  {n_sp} spikes  "
              f"{rate:.1f}/h (target {targets[week]:.1f})")

        fs, raw, filt = load_window(local[0], ch, start, args.long_sec)
        out_fs, n = write_trace(os.path.join(args.outdir, f"{week}_long.csv"),
                                fs, raw, filt, args.long_fs or None)
        print(f"      long: {n} rows @ {out_fs:.0f} Hz")

        evs = [(t, c) for t, c in events[db_path]
               if start <= t < start + args.long_sec]
        for t, _c in evs:
            spike_rows.append({"week": week, "t_sec": round(t - start, 4)})

        # Zoom on the MEDIAN-confidence spike: a representative example, not the
        # most impressive one.
        if evs:
            mid = sorted(evs, key=lambda e: e[1])[len(evs) // 2][0]
            z0 = max(0.0, mid - args.zoom_sec / 2)
            fs, zraw, zfilt = load_window(local[0], ch, z0, args.zoom_sec)
            _fs, zn = write_trace(
                os.path.join(args.outdir, f"{week}_zoom.csv"), fs, zraw, zfilt, None)
            print(f"      zoom: {zn} rows @ {fs:.0f} Hz, spike at "
                  f"t={mid - z0:.2f}s")
        else:
            mid = None

        win_rows.append({"week": week, "animal": args.animal, "channel": ch,
                         "file": base, "window_start_sec": round(start, 1),
                         "window_sec": args.long_sec, "spikes_in_window": n_sp,
                         "window_rate_per_h": round(rate, 2),
                         "cohort_target_per_h": round(targets[week], 2),
                         "zoom_spike_sec": None if mid is None else round(mid, 3)})

    for name, rows in (("spike_times.csv", spike_rows), ("windows.csv", win_rows)):
        if rows:
            with open(os.path.join(args.outdir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"\nWrote {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
