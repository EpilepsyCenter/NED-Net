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


def animal_week_rates(db: str, animal: str) -> dict[str, float]:
    """One animal's own mean spikes/hour per week.

    For an animal whose rates sit far from the cohort mean (animal 2 runs 3.8
    -> 21.9 against a cohort 8.2 -> 16.4), matching windows to the COHORT mean
    would force a week-1 window several times busier than that animal's own
    week 1 — i.e. cherry-picking. Matching its own rate keeps each panel
    representative of the animal being shown.
    """
    conn = sqlite3.connect(db)
    try:
        hours, spikes = {}, {}
        for path, sec in conn.execute(
                "SELECT c.path, f.valid_sec FROM file_animals f "
                "JOIN chunks c ON f.chunk_id = c.id WHERE f.animal_id = ?",
                (animal,)):
            m = _WEEK_RE.search(path)
            if m:
                hours[m.group(1)] = hours.get(m.group(1), 0) + (sec or 0) / 3600
        for path, n in conn.execute(
                "SELECT c.path, COUNT(*) FROM events e "
                "JOIN chunks c ON e.chunk_id = c.id "
                "WHERE e.type='interictal_spike' AND e.animal_id = ? "
                "GROUP BY c.path", (animal,)):
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

    "Best" = windowed rate closest to the target for that week; ties go to the
    window with more spikes, so a panel is never an empty stretch. This is a
    first pass over spike TIMES only — no signal is read, so it can range over
    every file in the week cheaply.
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


def _spike_amps(y, fs, times, t0):
    """Signed peak of each spike, measured on the loaded window."""
    import numpy as np
    out = []
    for t in times:
        seg = y[max(0, int((t - t0 - 0.05) * fs)):int((t - t0 + 0.15) * fs)]
        if len(seg):
            out.append(float(seg[np.argmax(np.abs(seg))]))
    return out


def refine_window(path: str, ch: int, evs, target_rate: float, win_sec: float,
                  step_sec: float, tol: float = 0.3):
    """Re-pick the window WITHIN one already-loaded file, on amplitude too.

    The rate-only pass can land on a window whose spikes are far bigger than
    that animal's typical spike (animal 2's week-1 window held two ~1000 µV
    spikes against a ~360 µV median). Under one shared y-scale that squashes
    every other row, so among windows whose rate is still within `tol` of the
    target, prefer the one whose largest spike is closest to the file's median
    spike amplitude. Costs nothing extra: the whole channel is already cached.
    """
    import numpy as np
    fs, full = load_channel(path, ch)
    all_amps = np.abs(_spike_amps(full, fs, [t for t, _c in evs], 0.0))
    if not len(all_amps):
        return None
    median_amp = float(np.median(all_amps))

    best = None
    start = 0.0
    while start + win_sec <= FILE_LEN_SEC:
        inside = [t for t, _c in evs if start <= t < start + win_sec]
        rate = len(inside) / (win_sec / 3600.0)
        if inside and (target_rate == 0 or
                       abs(rate - target_rate) <= tol * max(target_rate, 1e-9)):
            amps = np.abs(_spike_amps(full, fs, inside, 0.0))
            if len(amps):
                key = (abs(float(amps.max()) - median_amp), abs(rate - target_rate))
                if best is None or key < best[0]:
                    best = (key, start, len(inside), rate, float(amps.max()))
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


def _pack(panels):
    """panels -> flat npz-able dict (npz cannot hold a list of dicts)."""
    import json
    out = {"meta": json.dumps([{k: v for k, v in p.items()
                                if k not in ("long", "zoom")} for p in panels])}
    for i, p in enumerate(panels):
        out[f"long{i}"] = p["long"]
        if p["zoom"] is not None:
            out[f"zoom{i}"] = p["zoom"]
    return out


def _unpack(z):
    import json
    meta = json.loads(str(z["meta"]))
    panels = []
    for i, m in enumerate(meta):
        m = dict(m)
        m["long"] = z[f"long{i}"]
        m["zoom"] = z[f"zoom{i}"] if f"zoom{i}" in z else None
        panels.append(m)
    win_rows = [{"week": p["week"], "spikes_in_window": p["n"],
                 "window_rate_per_h": p["rate"]} for p in panels]
    return panels, win_rows


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
    ap.add_argument("--ylim-pct", type=float, default=100.0,
                    help="percentile of |signal| setting the long-trace y-limit; "
                         "100 = never clip a spike, lower = fatter background "
                         "band but truncated peaks")
    ap.add_argument("--target", choices=("cohort", "animal"), default="cohort",
                    help="match each window to the cohort's mean rate for that "
                         "week (default) or to the shown animal's own rate")
    ap.add_argument("--per-row-scale", action="store_true",
                    help="scale each week independently, with its own scale bar. "
                         "Use when spike amplitude changes a lot between weeks "
                         "(animal 2 falls 4x) and a shared scale flattens a row. "
                         "The panel then shows RATE, not amplitude — say so in "
                         "the legend.")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the amplitude-typicality refinement pass")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="ignore/refresh the cached extracted windows")
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
    targets = (animal_week_rates(args.db, args.animal) if args.target == "animal"
               else cohort_week_rates(args.db))
    events = animal_events(args.db, args.animal)
    weeks = sorted(targets)
    print(f"animal {args.animal} (code ch{ch})   {args.target} targets: "
          + ", ".join(f"{w}={targets[w]:.1f}/h" for w in weeks) + "\n")

    # Extraction reads ~170 MB per week off a ~1 MB/s share, so cache the
    # extracted windows: re-rendering with different styling is then instant.
    cache_path = os.path.join(
        args.outdir,
        f"_traces_a{args.animal}_{args.target}_{int(args.long_sec)}s.npz")
    panels, win_rows = [], []
    if args.cache and os.path.exists(cache_path):
        panels, win_rows = _unpack(np.load(cache_path, allow_pickle=True))
        print(f"  loaded cached windows from {cache_path}")

    # ---- Collect the data for every panel before drawing, so the shared
    # y-scale can be computed across all of them. ----
    for week in [] if panels else weeks:
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
        # Second pass inside the chosen file: same rate, more typical spike
        # amplitudes. The file is loaded once and cached, so this is free.
        if not args.no_refine:
            ref = refine_window(local[0], ch, events[db_path], targets[week],
                                args.long_sec, args.step_sec)
            if ref and abs(ref[1] - start) > 1.0:
                _k, start, n_sp, rate, pk = ref
                print(f"  {week}: refined to +{start:.0f}s "
                      f"(largest spike {pk:.0f} µV)", flush=True)
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
                         "target_per_h": round(targets[week], 2),
                         "target_basis": args.target,
                         "zoom_spike_sec": None if mid is None else round(mid, 3)})

    if not panels:
        print("No panels built.", file=sys.stderr)
        return 1
    if args.cache:
        np.savez_compressed(cache_path, **_pack(panels))
        print(f"  cached extracted windows -> {cache_path}")

    # One y-scale for every row, or the comparison between weeks is meaningless.
    # Default 100th percentile: a truncated spike reads as an artefact in print,
    # so nothing is clipped even though it costs a thinner background band.
    def _lim(arrs):
        return max(np.percentile(np.abs(a), args.ylim_pct) for a in arrs) * 1.04

    if args.per_row_scale:
        # Each row on its own scale; the scale bar moves to every row so the
        # reader is never misled into comparing amplitudes across weeks.
        ylims = [_lim([p["long"]]) for p in panels]
        zlims = [_lim([p["zoom"]]) if p["zoom"] is not None else 1.0
                 for p in panels]
    else:
        ylims = [_lim([p["long"] for p in panels])] * len(panels)
        zlims = [_lim([p["zoom"] for p in panels if p["zoom"] is not None])] * len(panels)
    ylim, zlim = ylims[0], zlims[0]
    zlim = max(np.percentile(np.abs(p["zoom"]), 99.9) for p in panels
               if p["zoom"] is not None) * 1.25

    mm = 1 / 25.4
    n_rows = len(panels)
    row_mm = 34 if args.per_row_scale else 30
    fig = plt.figure(figsize=(args.width_mm * mm, row_mm * n_rows * mm))
    # Two sub-rows per week: a thin raster row carrying the spike ticks, then
    # the trace. Ticks inside the trace band would be invisible against it.
    gs = GridSpec(n_rows * 2, 2, width_ratios=[3.4, 1],
                  height_ratios=[0.22, 1] * n_rows, figure=fig,
                  hspace=0.30 if args.per_row_scale else 0.0, wspace=0.14,
                  left=0.07, right=0.99, top=0.96, bottom=0.08)

    for row, p in enumerate(panels):
        wk = p["week"].replace("_", " ")
        x_max = args.long_sec / 60.0
        ylim, zlim = ylims[row], zlims[row]
        show_bar = args.per_row_scale or row == n_rows - 1

        # ---- spike raster + labels ----
        at = fig.add_subplot(gs[row * 2, 0])
        for s in p["spikes"]:
            at.plot([s / 60.0, s / 60.0], [0.05, 0.55], lw=0.8,
                    color="#c1272d", solid_capstyle="butt")
        at.set_xlim(0, x_max)
        at.set_ylim(0, 1)
        at.axis("off")
        at.text(0, 0.72, wk, fontsize=8, fontweight="bold", ha="left", va="bottom")
        at.text(x_max, 0.72,
                f"{p['n']} spikes / {args.long_sec / 60:.0f} min "
                f"({p['rate']:.1f} h$^{{-1}}$)",
                fontsize=6.5, ha="right", va="bottom", color="#555555")

        # ---- long compressed trace ----
        ax = fig.add_subplot(gs[row * 2 + 1, 0])
        n_bins = 3000
        xs, ys = envelope(p["long"], n_bins)
        ax.plot(xs / n_bins * x_max, ys, lw=0.2, color="#333333")
        ax.set_xlim(0, x_max)
        ax.set_ylim(-ylim, ylim)
        ax.axis("off")
        if show_bar:
            # In per-row mode the bar must sit INSIDE the row: rows are packed
            # tight, so a bar at the axis floor overlaps the next week's label.
            y0 = -ylim * (0.55 if args.per_row_scale else 0.95)
            _scalebar(ax, 0.4, y0, 5.0, ylim * 0.4,
                      "5 min", f"{ylim * 0.4:.0f} µV")

        # ---- zoom, spanning both sub-rows ----
        az = fig.add_subplot(gs[row * 2:row * 2 + 2, 1])
        if p["zoom"] is not None:
            tz = np.arange(len(p["zoom"])) / p["fs"]
            az.plot(tz, p["zoom"], lw=0.4, color="#333333")
            for s in p["zoom_spikes"]:
                az.plot([s], [zlim * 0.88], marker="v", ms=2.5,
                        color="#c1272d", clip_on=False)
            az.set_xlim(0, args.zoom_sec)
            az.set_ylim(-zlim, zlim)
            if show_bar:
                z0 = -zlim * (0.55 if args.per_row_scale else 0.95)
                _scalebar(az, 0.3, z0, 2.0, zlim * 0.4,
                          "2 s", f"{zlim * 0.4:.0f} µV")
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
    if args.per_row_scale:
        print("      per-row y-scale: "
              + ", ".join(f"{p['week']}=+/-{y:.0f}" for p, y in zip(panels, ylims))
              + " µV")
    else:
        print(f"      shared y-scale +/-{ylims[0]:.0f} µV (long), "
              f"+/-{zlims[0]:.0f} µV (zoom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
