#!/usr/bin/env python
"""Cross-check NED-Net's AD spike detections against the MATLAB IED/ISI run.

Two AD files were analysed both ways: detect_5xFAD_IED_ISI.m wrote a
*_IED_ISI_results/ folder next to each of them, and the LUNARC batch run put
its own events in ad_spikes.db. Same recordings, same animals — so the two
counts should be relatable, and if they are not, one of them is wrong.

The headline gap is expected: MATLAB reports raw CANDIDATES (its own header
says "CANDIDATE interictal spikes until visually validated"), while the batch
run additionally applies the Spikes-tab operating point (confidence >= 0.7,
local SNR >= 10, amplitude >= 15x baseline). --rerun quantifies exactly that by
re-detecting the same files locally with the filters off and reporting the
funnel, so you can see how much each filter removes rather than assuming.

Usage:
    # DB counts vs MATLAB (fast, no EDF access needed beyond the summary CSVs)
    python scripts/local/compare_ad_matlab.py

    # ...plus a local unfiltered re-detection to show the filter funnel
    #    (~2 min/file: 1.5 h x 2 kHz x 4 channels)
    python scripts/local/compare_ad_matlab.py --rerun
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sqlite3
import sys

DEFAULT_DB = os.path.expanduser("~/.eeg_seizure_analyzer/projects/ad_spikes.db")
DEFAULT_ROOT = ("/Volumes/research/LU26D1055-epicenter/Data/KAHA recordings/"
                "AD_Animals_Recordings")

# code channel index -> (MATLAB channel label, animal ID) — see ad_metadata.csv.
CHANNELS = {0: ("Ch1", "1"), 3: ("Ch4", "2"), 5: ("Ch6", "3"), 6: ("Ch7", "4")}

# The operating point the LUNARC run used (detect_spikes_all.sh defaults).
FILTERS = (("conf>=0.7", "confidence", 0.7),
           ("snr>=10", "local_snr", 10.0),
           ("xbl>=15", "amplitude_x_baseline", 15.0))


def find_matlab_runs(root: str) -> list[tuple[str, str]]:
    """-> [(edf_basename, summary_csv_path)] for every MATLAB results folder."""
    out = []
    pat = os.path.join(root, "**", "*_IED_ISI_results", "*_IED_ISI_summary.csv")
    for csv_path in sorted(glob.glob(pat, recursive=True)):
        folder = os.path.basename(os.path.dirname(csv_path))
        stem = folder[: -len("_IED_ISI_results")]
        out.append((stem + ".edf", csv_path))
    return out


def read_matlab(csv_path: str) -> dict[str, dict]:
    """-> {'Ch1': {'spikes': int, 'hours': float}} keyed by MATLAB channel."""
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            # 'Animal_Ch4' -> 'Ch4'; EDF_Channel is the same number, 1-based.
            label = row["Animal"].replace("Animal_", "")
            rows[label] = {"spikes": int(float(row["CandidateSpikes"])),
                           "hours": float(row["RecordingHours"])}
    return rows


def read_db(db_path: str, basename: str) -> dict[int, int]:
    """-> {code_channel: n_spikes} for one file, from the project DB."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """SELECT e.channel, COUNT(*)
                 FROM events e JOIN chunks c ON e.chunk_id = c.id
                WHERE c.path LIKE ? AND e.type = 'interictal_spike'
                GROUP BY e.channel""",
            ("%/" + basename,),
        ).fetchall()
    finally:
        conn.close()
    return {ch: n for ch, n in rows}


def find_edf(root: str, basename: str) -> str | None:
    hits = glob.glob(os.path.join(root, "**", basename), recursive=True)
    return hits[0] if hits else None


def rerun_unfiltered(edf_path: str) -> dict[int, list[int]]:
    """Re-detect locally with filters off -> {channel: [raw, after each filter]}.

    Mirrors analysis.process_spike_chunk_classical's detection path and the
    batch run's parameters, but keeps every stage so the funnel is visible.
    """
    from eeg_seizure_analyzer.analysis import (_filter_spike_events,
                                               auto_pair_channels,
                                               scan_edf_channels)
    from eeg_seizure_analyzer.config import SpikeDetectionParams
    from eeg_seizure_analyzer.detection.base import detect_chunked
    from eeg_seizure_analyzer.detection.spike import SpikeDetector

    # detect_spikes_batch._build_params defaults = the Spikes-tab operating
    # point, which is what the LUNARC run used. Keep these in sync with it.
    params = SpikeDetectionParams(
        bandpass_low=3.0, bandpass_high=50.0, amplitude_threshold_zscore=4.0,
        spike_min_amplitude_uv=0.0, spike_prominence_x_baseline=6.0,
        max_duration_ms=300.0, min_duration_ms=10.0, refractory_ms=750.0,
        baseline_method="percentile", baseline_percentile=25,
        baseline_rms_window_sec=30.0,
        isolation_window_sec=2.0, isolation_max_neighbours=1,
    )
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, _act, _pair = auto_pair_channels(ch_info)
    want = [c for c in eeg_idx if c in CHANNELS]
    spikes, _info = detect_chunked(
        SpikeDetector(), path=edf_path, channels=want,
        chunk_duration_sec=1800.0, overlap_sec=10.0, params=params)

    out: dict[int, list[int]] = {}
    for ch in want:
        ch_sp = [e for e in spikes if e.channel == ch]
        # Cumulative, in the same order _filter_spike_events applies them.
        stages = [len(ch_sp)]
        kw = {}
        for _label, key, val in FILTERS:
            kw[{"confidence": "min_confidence",
                "local_snr": "min_local_snr",
                "amplitude_x_baseline": "min_amplitude_x_baseline"}[key]] = val
            stages.append(len(_filter_spike_events(ch_sp, **kw)))
        out[ch] = stages
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="AD_Animals_Recordings folder (mounted research share)")
    ap.add_argument("--rerun", action="store_true",
                    help="re-detect locally with filters off to show the funnel")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}\n"
              "scp it back from LUNARC first.", file=sys.stderr)
        return 1
    runs = find_matlab_runs(args.root)
    if not runs:
        print(f"No *_IED_ISI_results folders under {args.root}", file=sys.stderr)
        return 1

    for basename, csv_path in runs:
        mat = read_matlab(csv_path)
        db_counts = read_db(args.db, basename)
        hours = next(iter(mat.values()))["hours"] if mat else 0.0
        print(f"\n{basename}   ({hours:.2f} h)")

        funnel = {}
        if args.rerun:
            edf = find_edf(args.root, basename)
            if edf:
                print("  re-detecting locally (filters off)...", flush=True)
                funnel = rerun_unfiltered(edf)
            else:
                print(f"  !! EDF not found under {args.root} — skipping rerun")

        head = f"  {'animal':<7}{'ch':<5}{'MATLAB':>9}{'NED-Net':>9}{'ratio':>8}"
        if funnel:
            head += f"{'raw':>9}" + "".join(f"{lbl:>10}" for lbl, _k, _v in FILTERS)
        print(head)
        for ch, (label, animal) in sorted(CHANNELS.items()):
            m = mat.get(label, {}).get("spikes")
            n = db_counts.get(ch, 0)
            ratio = f"{m / n:.1f}x" if m and n else "-"
            line = (f"  {animal:<7}{label:<5}"
                    f"{('-' if m is None else m):>9}{n:>9}{ratio:>8}")
            if funnel:
                stages = funnel.get(ch, [])
                line += "".join(f"{s:>9}" if i == 0 else f"{s:>10}"
                                for i, s in enumerate(stages))
            print(line)

        if funnel:
            # The DB count and the final funnel stage are the same computation
            # on the same data; a mismatch means the params drifted apart.
            for ch in sorted(CHANNELS):
                stages = funnel.get(ch)
                if stages and stages[-1] != db_counts.get(ch, 0):
                    print(f"  !! ch{ch}: local rerun ends at {stages[-1]} but the "
                          f"DB has {db_counts.get(ch, 0)} — parameters differ "
                          f"between this script and the batch run.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
