"""Convert .adicht files to EDF+ format (Windows-only CLI tool).

Usage:
    python -m eeg_seizure_analyzer.io.adicht_to_edf input.adicht output.edf
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def convert_adicht_to_edf(adicht_path: str, edf_path: str) -> None:
    """Convert a .adicht file to EDF+ with annotations preserved.

    Parameters
    ----------
    adicht_path : str
        Input .adicht file path.
    edf_path : str
        Output .edf file path.
    """
    if sys.platform != "win32":
        raise RuntimeError("adicht conversion is only available on Windows.")

    try:
        import adi
    except ImportError:
        raise ImportError("adi-reader not installed. pip install adi-reader")

    import pyedflib

    # Read the adicht file
    from eeg_seizure_analyzer.io.adicht_reader import read_adicht

    recording = read_adicht(adicht_path)

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

        # Write data in 1-second blocks
        samples_per_block = int(recording.fs)
        n_blocks = int(np.ceil(recording.n_samples / samples_per_block))

        for block_idx in range(n_blocks):
            start = block_idx * samples_per_block
            end = min(start + samples_per_block, recording.n_samples)
            block_data = []
            for ch in range(recording.n_channels):
                segment = recording.data[ch, start:end]
                # Pad last block if needed
                if len(segment) < samples_per_block:
                    segment = np.pad(segment, (0, samples_per_block - len(segment)))
                block_data.append(segment)
            writer.writeSamples(block_data)

        # Write annotations
        for ann in recording.annotations:
            duration = ann.duration_sec if ann.duration_sec else -1
            writer.writeAnnotation(ann.onset_sec, duration, ann.text)

    finally:
        writer.close()

    print(f"Converted: {adicht_path} -> {edf_path}")
    print(f"  Channels: {recording.n_channels}")
    print(f"  Duration: {recording.duration_sec:.1f}s ({recording.duration_sec / 3600:.2f}h)")
    print(f"  Sampling rate: {recording.fs} Hz")
    print(f"  Annotations: {len(recording.annotations)}")


def main():
    parser = argparse.ArgumentParser(description="Convert .adicht to EDF+")
    parser.add_argument("input", help="Input .adicht file")
    parser.add_argument("output", help="Output .edf file")
    args = parser.parse_args()
    convert_adicht_to_edf(args.input, args.output)


if __name__ == "__main__":
    main()
