#!/usr/bin/env python
"""Which AD animal has the largest, most up-going spikes? (figure selection)

The Extended Data trace panels read better with large positive-going spikes,
but polarity and amplitude are properties of the electrode, not something to
choose blind. This measures both for every animal from the same recordings —
all four animals share each EDF, so one file read characterises all of them.

For each detected spike it takes the signed extremum over the event window and
reports, per animal: median |amplitude|, the fraction that are positive-going,
and the median amplitude of each polarity separately.

    python scripts/local/ad_spike_polarity.py
    python scripts/local/ad_spike_polarity.py --files "AD_W3_D1_27072026(1).edf"
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys

DEFAULT_DB = os.path.expanduser("~/.eeg_seizure_analyzer/projects/ad_spikes.db")
DEFAULT_ROOT = ("/Volumes/research/LU26D1055-epicenter/Data/KAHA recordings/"
                "AD_Animals_Recordings")
ANIMAL_TO_CH = {"1": 0, "2": 3, "3": 5, "4": 6}
BP_LOW, BP_HIGH = 3.0, 50.0
# One file per week by default — the same ones the figure windows came from.
DEFAULT_FILES = ["AD_W1_D1_13072026(1).edf",
                 "AD_W2_D1_20072026(1).edf",
                 "AD_W3_D1_27072026(1).edf"]


def uv_scale(path: str) -> float:
    """These EDFs declare mV; read_edf returns physical units unchanged."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256].decode("ascii", "replace").strip())
        f.read(16 * ns)
        f.read(80 * ns)
        dim = f.read(8).decode("ascii", "replace").strip().lower()
    return {"mv": 1000.0, "uv": 1.0, "µv": 1.0, "v": 1e6}.get(dim, 1.0)


def spikes_for(db: str, basename: str):
    """-> [(animal, channel, start_sec, end_sec)] for one file."""
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT e.animal_id, e.channel, e.start_sec, e.end_sec FROM events e "
            "JOIN chunks c ON e.chunk_id = c.id "
            "WHERE c.path LIKE ? AND e.type='interictal_spike'",
            ("%/" + basename,)).fetchall()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    args = ap.parse_args()

    import numpy as np
    from eeg_seizure_analyzer.io.edf_reader import read_edf
    from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

    amps: dict[str, list[float]] = {a: [] for a in ANIMAL_TO_CH}
    for basename in args.files:
        hits = glob.glob(os.path.join(args.root, "**", basename), recursive=True)
        if not hits:
            print(f"!! {basename} not found under {args.root}", file=sys.stderr)
            continue
        path = hits[0]
        evs = spikes_for(args.db, basename)
        if not evs:
            print(f"   {basename}: no spikes in DB")
            continue
        print(f"   reading {basename} ({len(evs)} spikes)...", flush=True)
        k = uv_scale(path)
        chans = sorted(ANIMAL_TO_CH.values())
        rec = read_edf(path, channels=chans)
        filt = {ch: bandpass_filter(rec.get_channel_data(pos), rec.fs,
                                    BP_LOW, BP_HIGH) * k
                for pos, ch in enumerate(chans)}
        for animal, ch, t0, t1 in evs:
            y = filt.get(ch)
            if y is None:
                continue
            i0, i1 = int(t0 * rec.fs), int(max(t1, t0 + 0.02) * rec.fs)
            seg = y[max(0, i0):min(len(y), i1)]
            if len(seg) == 0:
                continue
            # Signed extremum: whichever way the spike actually goes.
            amps[animal].append(float(seg[np.argmax(np.abs(seg))]))

    print(f"\n{'animal':<8}{'n':>6}{'median |amp|':>14}{'% up-going':>12}"
          f"{'median up':>12}{'median down':>13}")
    for a in sorted(amps):
        v = np.array(amps[a])
        if not len(v):
            print(f"{a:<8}{'0':>6}")
            continue
        up, dn = v[v > 0], v[v < 0]
        print(f"{a:<8}{len(v):>6}{np.median(np.abs(v)):>13.0f}µV"
              f"{100 * len(up) / len(v):>11.0f}%"
              f"{(np.median(up) if len(up) else float('nan')):>11.0f}µV"
              f"{(np.median(dn) if len(dn) else float('nan')):>12.0f}µV")
    print("\nUp-going and large = best candidate for the trace panels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
