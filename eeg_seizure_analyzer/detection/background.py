"""Propose seizure-free background stretches for review.

Training needs examples of "no seizure", but unreviewed recording time can hide
seizures nobody annotated. This proposes stretches that look quiet by a simple
per-channel rule; the user then confirms or rejects them in the Training tab
(Background mode), and training can use the confirmed ones as negatives.

Rule, per channel and recording (no model needed, so it works for a first model):
  1. 1-s epochs; RMS (epoch mean removed) and line length, each divided by the
     channel's median over the recording.
  2. An epoch is busy if either ratio exceeds ``k`` (default 2); busy epochs are
     widened by ``pad_sec`` each side (seizure onsets and offsets are lower
     amplitude than their middle).
  3. Candidate stretches are ``stretch_sec`` of consecutive quiet epochs, at least
     ``margin_sec`` away from any interval to avoid (seizure candidates and
     annotations of any label).
  4. Up to ``n_per_hour`` stretches per channel per recording hour (so longer
     recordings get more), spread over the recording: one random candidate per
     equal part of the recording.

On held-out recordings in our tests, k = 2 left 96% of annotated and 87% of
unannotated seizure time out of the quiet epochs, at the cost of 30% of non-seizure
time. The proposals are for review, not labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eeg_seizure_analyzer.detection.base import DetectedEvent

DEFAULTS = {"k": 2.0, "stretch_sec": 30.0, "n_per_hour": 10.0, "pad_sec": 5.0,
            "margin_sec": 10.0}


def epoch_ratios(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Per 1-s epoch: RMS and line length, each relative to the channel median."""
    step = max(1, int(round(fs)))
    n = len(x) // step
    if n == 0:
        return np.zeros(0), np.zeros(0)
    e = np.asarray(x[: n * step], dtype=np.float64).reshape(n, step)
    e = e - e.mean(axis=1, keepdims=True)
    rms = np.sqrt((e ** 2).mean(axis=1))
    ll = np.abs(np.diff(e, axis=1)).mean(axis=1)
    return rms / max(np.median(rms), 1e-12), ll / max(np.median(ll), 1e-12)


def quiet_epochs(x: np.ndarray, fs: float, k: float = DEFAULTS["k"],
                 pad_sec: float = DEFAULTS["pad_sec"]) -> np.ndarray:
    """Boolean per 1-s epoch: True where the channel is quiet by the rule."""
    rr, lr = epoch_ratios(x, fs)
    busy = (rr > k) | (lr > k)
    pad = int(round(pad_sec))
    if pad > 0 and busy.any():
        busy = np.convolve(busy.astype(int), np.ones(2 * pad + 1, int), "same") > 0
    return ~busy


def propose_background(
    x: np.ndarray,
    fs: float,
    channel: int,
    avoid: list[tuple[float, float]] | tuple = (),
    k: float = DEFAULTS["k"],
    stretch_sec: float = DEFAULTS["stretch_sec"],
    n_per_hour: float = DEFAULTS["n_per_hour"],
    pad_sec: float = DEFAULTS["pad_sec"],
    margin_sec: float = DEFAULTS["margin_sec"],
    animal_id: str = "",
    seed: int = 0,
) -> list[DetectedEvent]:
    """Propose up to ``n_per_hour`` quiet stretches of ``stretch_sec`` per hour of
    recording on one channel.

    ``x`` is the channel's signal at ``fs``; ``avoid`` holds (onset, offset) in
    seconds of anything that may be a seizure (detector candidates, annotations).
    """
    quiet = quiet_epochs(x, fs, k=k, pad_sec=pad_sec)
    n = len(quiet)
    ok = quiet.copy()
    for on, off in avoid:
        a = max(0, int(np.floor(on - margin_sec)))
        b = min(n, int(np.ceil(off + margin_sec)))
        if b > a:
            ok[a:b] = False
    L = int(round(stretch_sec))
    n_max = max(1, int(round(n_per_hour * n / 3600.0))) if n_per_hour > 0 else 0
    if L <= 0 or n < L or n_max <= 0:
        return []
    # starts[i] is True if epochs i .. i+L-1 are all eligible
    run = np.convolve(ok.astype(int), np.ones(L, int), "valid") == L
    starts = np.flatnonzero(run)
    if len(starts) == 0:
        return []
    rng = np.random.default_rng(seed + 7919 * int(channel))
    bounds = np.linspace(0, n, n_max + 1)
    chosen: list[int] = []
    for i in range(n_max):
        cand = starts[(starts >= bounds[i]) & (starts < bounds[i + 1])]
        if chosen:
            cand = cand[cand >= chosen[-1] + L]
        if len(cand):
            chosen.append(int(rng.choice(cand)))
    rr, lr = epoch_ratios(x, fs)
    events = []
    for s0 in chosen:
        events.append(DetectedEvent(
            onset_sec=float(s0), offset_sec=float(s0 + L), duration_sec=float(L),
            channel=int(channel), event_type="background", confidence=0.0,
            animal_id=animal_id,
            features={"detection_method": "background_rule",
                      "rms_ratio_max": round(float(rr[s0:s0 + L].max()), 3),
                      "ll_ratio_max": round(float(lr[s0:s0 + L].max()), 3),
                      "k": k},
        ))
    return events


# ---------------------------------------------------------------------------
# Command line: propose background for a folder of EDFs (writes pending
# proposals to each recording's _ned_background_annotations.json).
# ---------------------------------------------------------------------------

def _seizure_avoid(edf_path: str) -> dict[int, list[tuple[float, float]]]:
    """Every seizure annotation of any label, per channel (pending ones may be
    seizures nobody reviewed yet)."""
    from eeg_seizure_analyzer.io.annotation_store import load_annotations
    out: dict[int, list[tuple[float, float]]] = {}
    for a in load_annotations(edf_path) or []:
        out.setdefault(int(a.channel), []).append((a.onset_sec, a.offset_sec))
    return out


def propose_for_edf(edf_path: str, channels: list[int] | None = None,
                    **params) -> dict:
    from eeg_seizure_analyzer.io.annotation_store import write_background_proposals
    from eeg_seizure_analyzer.io.edf_reader import read_edf_window, scan_edf_channels
    info = scan_edf_channels(edf_path)
    max_fs = max(c["fs"] for c in info)
    eeg = [c for c in info if c["fs"] == max_fs]
    if channels is not None:
        eeg = [c for c in eeg if c["index"] in set(channels)]
    ch_file = Path(edf_path).with_name(Path(edf_path).stem + "_ned_channels.json")
    ids = {}
    if ch_file.is_file():
        ids = {int(k): v for k, v in json.loads(ch_file.read_text()).get("channel_ids", {}).items()}
    avoid = _seizure_avoid(edf_path)
    events = []
    for c in eeg:
        dur = c["n_samples"] / c["fs"]
        rec = read_edf_window(edf_path, channels=[c["index"]], start_sec=0.0,
                              duration_sec=dur - 0.5)
        events += propose_background(rec.data[0], rec.fs, c["index"],
                                     avoid=avoid.get(c["index"], []),
                                     animal_id=ids.get(c["index"], ""), **params)
    settings = {**DEFAULTS, **params, "method": "background_rule"}
    return write_background_proposals(edf_path, events, channel_ids=ids,
                                      proposal_settings=settings)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", required=True, help="folder of EDFs (recursive)")
    ap.add_argument("--k", type=float, default=DEFAULTS["k"])
    ap.add_argument("--stretch-sec", type=float, default=DEFAULTS["stretch_sec"])
    ap.add_argument("--n-per-hour", type=float, default=DEFAULTS["n_per_hour"],
                    help="stretches per channel per recording hour")
    ap.add_argument("--only-annotated", action="store_true",
                    help="only recordings that already have a seizure annotation file")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    edfs = sorted(str(p) for p in Path(args.folder).rglob("*.edf"))
    if args.only_annotated:
        edfs = [e for e in edfs if Path(e).with_name(Path(e).stem + "_ned_annotations.json").is_file()]
    params = {"k": args.k, "stretch_sec": args.stretch_sec, "n_per_hour": args.n_per_hour}
    n_new = 0
    jobs = [(e, params) for e in edfs]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_one, jobs))
    else:
        results = [_one(j) for j in jobs]
    for e, r, err in results:
        if err:
            print(f"  [ERR] {Path(e).name}: {err}", file=sys.stderr)
        else:
            n_new += r["n_new"]
    print(f"{len(edfs)} recordings, {n_new} background stretches proposed")
    return 0


def _one(arg):
    e, params = arg
    try:
        return e, propose_for_edf(e, **params), None
    except Exception as exc:  # one bad file must not stop the folder
        return e, None, repr(exc)


if __name__ == "__main__":
    sys.exit(main())
