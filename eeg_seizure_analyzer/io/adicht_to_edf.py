"""Convert .adicht files to EDF+ format (Windows-only CLI tool).

Usage:
    python -m eeg_seizure_analyzer.io.adicht_to_edf input.adicht output.edf
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np

ProgressCallback = Callable[[str, float], None]


def convert_adicht_to_edf(
    adicht_path: str,
    edf_path: str,
    mode: str = "fast",
    progress_cb: ProgressCallback | None = None,
) -> None:
    """Convert a .adicht file to EDF+ with annotations preserved.

    Parameters
    ----------
    adicht_path : str
        Input .adicht file path.
    edf_path : str
        Output .edf file path.
    mode : {"fast", "blocked"}
        "fast" hands the full per-channel arrays to pyedflib in a single
        writeSamples call (C-level blocking, ~10-50x faster).
        "blocked" keeps the original Python-side 1-second loop as a fallback
        if the fast path misbehaves on a given recording.
    progress_cb : callable, optional
        ``progress_cb(stage, fraction)`` invoked periodically.
        ``stage`` is one of ``"reading"``, ``"writing"``, ``"done"``.
        ``fraction`` is in ``[0.0, 1.0]``.
    """
    def _emit(stage: str, fraction: float) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, float(fraction))
        except Exception:
            pass

    if sys.platform != "win32":
        raise RuntimeError("adicht conversion is only available on Windows.")

    try:
        import adi
    except ImportError:
        raise ImportError("adi-reader not installed. pip install adi-reader")

    import pyedflib

    # Read the adicht file
    from eeg_seizure_analyzer.io.adicht_reader import read_adicht

    _emit("reading", 0.0)
    recording = read_adicht(adicht_path)
    _emit("reading", 1.0)

    # Create EDF+ writer
    writer = pyedflib.EdfWriter(edf_path, recording.n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)

    try:
        # Set file-level metadata
        if recording.start_time:
            writer.setStartdatetime(recording.start_time)

        # Set channel headers
        for i in range(recording.n_channels):
            writer.setLabel(i, recording.channel_names[i])
            writer.setPhysicalDimension(i, recording.units[i])
            writer.setSamplefrequency(i, recording.fs)

            ch_data = recording.data[i]
            phys_min = float(np.min(ch_data))
            phys_max = float(np.max(ch_data))
            # Avoid zero range
            if phys_min == phys_max:
                phys_min -= 1.0
                phys_max += 1.0
            writer.setPhysicalMinimum(i, phys_min)
            writer.setPhysicalMaximum(i, phys_max)
            writer.setDigitalMinimum(i, -32768)
            writer.setDigitalMaximum(i, 32767)

        if mode == "fast":
            # Single C-level call; pyedflib handles internal block packing.
            # EDF stores fixed-duration records, so pad each channel up to a
            # whole-second multiple (matches the blocked path's behaviour).
            samples_per_block = int(recording.fs)
            remainder = recording.n_samples % samples_per_block
            pad = (samples_per_block - remainder) if remainder else 0
            if pad:
                padded = np.pad(recording.data, ((0, 0), (0, pad)))
            else:
                padded = recording.data
            _emit("writing", 0.0)
            writer.writeSamples([padded[ch] for ch in range(recording.n_channels)])
            _emit("writing", 1.0)
        elif mode == "blocked":
            # Original Python-side 1-second block loop (slow, kept as fallback).
            samples_per_block = int(recording.fs)
            n_blocks = int(np.ceil(recording.n_samples / samples_per_block))

            # Emit at most ~100 progress updates regardless of duration.
            emit_every = max(1, n_blocks // 100)

            _emit("writing", 0.0)
            for block_idx in range(n_blocks):
                start = block_idx * samples_per_block
                end = min(start + samples_per_block, recording.n_samples)
                block_data = []
                for ch in range(recording.n_channels):
                    segment = recording.data[ch, start:end]
                    if len(segment) < samples_per_block:
                        segment = np.pad(segment, (0, samples_per_block - len(segment)))
                    block_data.append(segment)
                writer.writeSamples(block_data)

                if block_idx % emit_every == 0 or block_idx == n_blocks - 1:
                    _emit("writing", (block_idx + 1) / n_blocks)
        else:
            raise ValueError(f"Unknown mode: {mode!r} (expected 'fast' or 'blocked')")

        # Write annotations
        for ann in recording.annotations:
            duration = ann.duration_sec if ann.duration_sec else -1
            writer.writeAnnotation(ann.onset_sec, duration, ann.text)

    finally:
        writer.close()

    _emit("done", 1.0)

    print(f"Converted: {adicht_path} -> {edf_path} [mode={mode}]")
    print(f"  Channels: {recording.n_channels}")
    print(f"  Duration: {recording.duration_sec:.1f}s ({recording.duration_sec / 3600:.2f}h)")
    print(f"  Sampling rate: {recording.fs} Hz")
    print(f"  Annotations: {len(recording.annotations)}")


def main():
    parser = argparse.ArgumentParser(description="Convert .adicht to EDF+")
    parser.add_argument("input", help="Input .adicht file")
    parser.add_argument("output", help="Output .edf file")
    parser.add_argument(
        "--mode",
        choices=("fast", "blocked"),
        default="fast",
        help="Write path: 'fast' (single C-level writeSamples, default) or "
             "'blocked' (legacy 1-second Python loop, fallback if 'fast' fails).",
    )
    args = parser.parse_args()
    convert_adicht_to_edf(args.input, args.output, mode=args.mode)


if __name__ == "__main__":
    main()
