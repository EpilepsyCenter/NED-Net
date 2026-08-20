#!/usr/bin/env python
"""Build the raw-trace panels for the AD Extended Data figure (vector PDF).

Per week it picks ONE 30-min window whose local spike rate is as close as
possible to that week's cohort mean rate, so each panel is representative
rather than the best-looking stretch, and writes windows.csv recording what it
chose (file, offset, rate) for the legend and methods.

Animal 4 is the default: its weekly rates (9.2 / 9.9 / 15.0 per h) track the
cohort means (8.2 / 12.6 / 16.4) within ~20% and it shows the increase, so one
animal carries all three panels — a within-animal comparison is stronger than
three different animals.

Layout: three rows (Week 1-3). Left, the 30-min compressed trace with a tick
above each detected spike; right, a 10 s zoom on the median-confidence spike in
that window — representative morphology, not the most striking example. All
panels share one y-scale, so the rows are visually comparable; scale bars
replace axes, as is conventional for EEG traces.

The long traces are drawn as a min/max envelope (two points per pixel column)
rather than 3.6 M samples: visually identical at print size, but the PDF stays
small and vector, so text and scale bars stay editable in Illustrator.

    python scripts/local/ad_figure_traces.py
    python scripts/local/ad_figure_traces.py --animal 2 --out fig3_traces.pdf
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
FILE_LEN_SEC = 5400.0
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
                win_sec: float, step_sec: float):
    """Best (path, start_sec, n_spikes, rate) for one week.

    "Best" = windowed rate closest to the cohort mean for that week; ties go to
    the window with more spikes, so a panel is never an empty stretch.
    """
    best = None
    for path, evs in events.items():
        m = _WEEK_RE.search(path)
        if not m or m.group(1) != week:
            continue
        start = 0.0
        while start + win_sec <= FILE_LEN_SEC:
            inside = [t for t, _c in evs if start <= t < start + win_sec]
            rate = len(inside) / (win_sec / 3600.0)
            key = (abs(rate - target_rate), -len(inside))
            if best is None or key < best[0]:
                best = (key, path, start, len(inside), rate)
            start += step_sec
    return best


_CACHE: dict[tuple[str, int], tuple[float, "object"]] = {}


def uv_scale(path: str) -> float:
    """Factor converting this EDF's physical units to µV.

    These recordings declare 'mV' in the header and read_edf returns the
    physical units unchanged, so values come back 1000x smaller than the µV
    that EEG scale bars are conventionally drawn in.
    """
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256].decode("ascii", "replace").strip())
        f.read(16 * ns)      # labels
        f.read(80 * ns)      # transducer
        dim = f.read(8).decode("ascii", "replace").strip().lower()
    return {"mv": 1000.0, "uv": 1.0, "µv": 1.0, "v": 1e6}.get(dim, 1.0)


def load_channel(path: str, ch: int):
    """-> (fs, whole bandpassed channel), cached.

    Cached because reading these EDFs off the mounted share runs at ~1 MB/s
    (pyedflib does strided per-channel reads, which over SMB become many small
    round-trips), and every file is needed twice — long window and zoom.
    """
    from eeg_seizure_analyzer.io.edf_reader import read_edf
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

    key = (path, ch)
    if key not in _CACHE:
        rec = read_edf(path, channels=[ch])
        # Filter the whole record then slice: filtering a short slice would ring
        # at the edges of exactly the window we are about to plot.
        filt = bandpass_filter(rec.get_channel_data(0), rec.fs, BP_LOW, BP_HIGH)
        _CACHE[key] = (rec.fs, filt * uv_scale(path))
    return _CACHE[key]


def load_window(path: str, ch: int, start: float, dur: float):
    """-> (fs, filtered slice) for one channel over [start, start+dur)."""
    fs, filt = load_channel(path, ch)
    i0, i1 = int(start * fs), int((start + dur) * fs)
    return fs, filt[i0:i1]


def envelope(y, n_bins: int):
    """Min/max per bin -> (x_index, y) drawing ~2 points per pixel column."""
    import numpy as np
    n = len(y) // n_bins * n_bins
    if n == 0:
        return np.arange(len(y)), y
    blocks = y[:n].reshape(n_bins, -1)
    lo, hi = blocks.min(axis=1), blocks.max(axis=1)
    # Interleave lo/hi so the polyline sweeps the full excursion of each bin.
    xs = np.repeat(np.arange(n_bins), 2)
    ys = np.empty(n_bins * 2)
    ys[0::2], ys[1::2] = lo, hi
    return xs, ys


def _scalebar(ax, x, y, dx, dy, xlabel, ylabel, fs=6):
    """L-shaped scale bar in data coordinates."""
    ax.plot([x, x, x + dx], [y + dy, y, y], color="black", lw=1.0,
            solid_capstyle="butt", clip_on=False)
    ax.text(x + dx / 2, y - dy * 0.12, xlabel, ha="center", va="top", fontsize=fs)
    ax.text(x - dx * 0.02, y + dy / 2, ylabel, ha="right", va="center",
            fontsize=fs, rotation=90)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--animal", default="4", choices=sorted(ANIMAL_TO_CH))
    ap.add_argument("--outdir", default="ad_figure")
    ap.add_argument("--out", default="ED_Fig3_traces.pdf")
    ap.add_argument("--long-sec", type=float, default=1800.0)
    ap.add_argument("--zoom-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=300.0)
    ap.add_argument("--width-mm", type=float, default=180.0,
                    help="figure width; 180 mm = full journal page width")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # TrueType, not Type 3, so the text stays editable in Illustrator.
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                                "font.family": "sans-serif",
                                "font.sans-serif": ["Arial", "Helvetica",
                                                    "DejaVu Sans"]})

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

    # ---- Collect the data for every panel before drawing, so the shared
    # y-scale can be computed across all of them. ----
    panels, win_rows = [], []
    for week in weeks:
        pick = pick_window(events, targets[week], week, args.long_sec,
                           args.step_sec)
        if not pick:
            print(f"  !! no window found for {week}")
            continue
        _key, db_path, start, n_sp, rate = pick
        base = os.path.basename(db_path)
        local = glob.glob(os.path.join(args.root, "**", base), recursive=True)
        if not local:
            print(f"  !! {base} not found under {args.root} — is the share mounted?")
            continue
        print(f"  {week}: {base} +{start:.0f}s  {n_sp} spikes  "
              f"{rate:.1f}/h (target {targets[week]:.1f})", flush=True)

        fs, y_long = load_window(local[0], ch, start, args.long_sec)
        evs = [(t, c) for t, c in events[db_path]
               if start <= t < start + args.long_sec]
        # Median-confidence spike: representative, not the most impressive one.
        mid = sorted(evs, key=lambda e: e[1])[len(evs) // 2][0] if evs else None
        y_zoom = None
        if mid is not None:
            z0 = max(0.0, mid - args.zoom_sec / 2)
            _fs, y_zoom = load_window(local[0], ch, z0, args.zoom_sec)
            zoom_spikes = [t - z0 for t, _c in evs if z0 <= t < z0 + args.zoom_sec]
        else:
            zoom_spikes = []

        panels.append({"week": week, "fs": fs, "long": y_long, "zoom": y_zoom,
                       "spikes": [t - start for t, _c in evs],
                       "zoom_spikes": zoom_spikes, "n": n_sp, "rate": rate})
        win_rows.append({"week": week, "animal": args.animal, "channel": ch,
                         "file": base, "window_start_sec": round(start, 1),
                         "window_sec": args.long_sec, "spikes_in_window": n_sp,
                         "window_rate_per_h": round(rate, 2),
                         "cohort_target_per_h": round(targets[week], 2),
                         "zoom_spike_sec": None if mid is None else round(mid, 3)})

    if not panels:
        print("No panels built.", file=sys.stderr)
        return 1

    # One y-scale for every row, or the comparison between weeks is meaningless.
    # 99.8th percentile, so a single artefact does not flatten the traces.
    ylim = max(np.percentile(np.abs(p["long"]), 99.8) for p in panels) * 1.6
    zlim = max(np.percentile(np.abs(p["zoom"]), 99.9) for p in panels
               if p["zoom"] is not None) * 1.25

    mm = 1 / 25.4
    fig = plt.figure(figsize=(args.width_mm * mm, 26 * len(panels) * mm))
    gs = GridSpec(len(panels), 2, width_ratios=[3.4, 1], figure=fig,
                  hspace=0.55, wspace=0.12,
                  left=0.06, right=0.99, top=0.95, bottom=0.06)

    for row, p in enumerate(panels):
        wk = p["week"].replace("_", " ")
        # ---- long compressed trace ----
        ax = fig.add_subplot(gs[row, 0])
        n_bins = 3000
        xs, ys = envelope(p["long"], n_bins)
        t = xs / n_bins * (args.long_sec / 60.0)      # minutes
        ax.plot(t, ys, lw=0.25, color="#222222")
        for s in p["spikes"]:
            ax.plot([s / 60.0, s / 60.0], [ylim * 0.72, ylim * 0.92],
                    lw=0.7, color="#c1272d", solid_capstyle="butt")
        ax.set_xlim(0, args.long_sec / 60.0)
        ax.set_ylim(-ylim, ylim)
        ax.axis("off")
        ax.text(0, ylim * 0.98, f"{wk}", fontsize=8, fontweight="bold",
                ha="left", va="top")
        ax.text(1.0, ylim * 0.98,
                f"{p['n']} spikes / {args.long_sec / 60:.0f} min "
                f"({p['rate']:.1f} h$^{{-1}}$)",
                fontsize=6.5, ha="left", va="top", color="#555555")
        if row == len(panels) - 1:
            _scalebar(ax, 0.2, -ylim * 0.92, 5.0, ylim * 0.5,
                      "5 min", f"{ylim * 0.5:.0f} µV")

        # ---- zoom ----
        az = fig.add_subplot(gs[row, 1])
        if p["zoom"] is not None:
            tz = np.arange(len(p["zoom"])) / p["fs"]
            az.plot(tz, p["zoom"], lw=0.4, color="#222222")
            for s in p["zoom_spikes"]:
                az.plot([s], [zlim * 0.82], marker="v", ms=2.5,
                        color="#c1272d", clip_on=False)
            az.set_xlim(0, args.zoom_sec)
            az.set_ylim(-zlim, zlim)
            if row == len(panels) - 1:
                _scalebar(az, 0.3, -zlim * 0.92, 2.0, zlim * 0.5,
                          "2 s", f"{zlim * 0.5:.0f} µV")
        az.axis("off")

    out_pdf = os.path.join(args.outdir, args.out)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.outdir, "windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0]))
        w.writeheader()
        w.writerows(win_rows)

    size_kb = os.path.getsize(out_pdf) / 1024
    print(f"\nWrote {out_pdf} ({size_kb:.0f} kB, vector)")
    print(f"      {os.path.join(args.outdir, 'windows.csv')} "
          f"(what each panel shows — for the legend)")
    print(f"      shared y-scale +/-{ylim:.0f} µV (long), "
          f"+/-{zlim:.0f} µV (zoom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
