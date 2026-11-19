"""Read LabChart .adicht files and convert them to EDF on any operating system.

Uses the file-format parsers of adichtion (``io/_adichtion``, GPL-3.0-or-later) and
needs neither LabChart nor the ADInstruments library, so it also runs on macOS and
Linux. Each LabChart record (the stretch between a start and a stop of sampling)
becomes one EDF+ file, named the way LabChart names its own exports:
``<name>(1).edf``, ``<name>(2).edf``, ... or ``<name>.edf`` for a single record.

Every channel keeps its own sampling rate, so EEG at 2 kHz and an activity channel
at 2 Hz go into the same EDF, as in a LabChart export. Integer (ADC) channels are
written count for count; nothing is resampled or rescaled.

    python -m eeg_seizure_analyzer.io.adicht_native recording.adicht [output_folder]
"""
from __future__ import annotations

import math
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from eeg_seizure_analyzer.io._adichtion import chartdata as _chartdata
from eeg_seizure_analyzer.io._adichtion.blocks import (
    FLAG_32BIT, FLAG_PACKED, FLAG_UNPACKED, HEADER_SIZE, SubBlock, decode_subblock,
)
from eeg_seizure_analyzer.io._adichtion.container import AdichtContainer
from eeg_seizure_analyzer.io._adichtion.errors import AdichtionError

ProgressCallback = Callable[[str, float], None]


class AdichtReadError(ValueError):
    """The file is not a LabChart file this reader understands."""


@dataclass
class ChannelInfo:
    index: int              # 0-based channel slot in the LabChart file
    name: str
    fs: float               # samples per second
    unit: str
    n_samples: int
    is_integer: bool        # stored as int16 ADC counts
    scale: float            # value = scale * stored sample + offset, in ``unit``
    offset: float


def _subblock_at(view: memoryview, off: int) -> SubBlock:
    """Header of the sample block at ``off`` (after adichtion's ``file.subblock_at``)."""
    if off < 0 or off + HEADER_SIZE > len(view):
        raise AdichtReadError(f"block index points outside the sample data ({off})")
    n, _check, negn, flags = struct.unpack_from("<iiii", view, off)
    if negn != -n or n < 12:
        raise AdichtReadError(f"no sample block at offset {off}")
    body = off + HEADER_SIZE
    width = 4 if flags & FLAG_32BIT else 2
    if flags & FLAG_PACKED and not flags & FLAG_UNPACKED:
        if body + 4 > len(view):
            raise AdichtReadError(f"sample block at {off} is cut short")
        raw, = struct.unpack_from("<i", view, body)
        if raw < 0:
            raise AdichtReadError(f"sample block at {off} has a negative size")
        n_samples = raw // width
    else:
        n_samples = (n - 12) // width
    return SubBlock(index=0, offset=off, body_offset=body, body_size=n - 12,
                    flags=flags, n_samples=n_samples)


class AdichtFile:
    """A LabChart file opened for reading. Use as a context manager."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        try:
            self._container = AdichtContainer(self.path)
        except AdichtionError as e:
            raise AdichtReadError(f"{Path(self.path).name}: not a LabChart file ({e})") from e
        try:
            map_entry = self._container.find_stream("Map")
            blocks_entry = self._container.find_stream("Blocks_0")
            if map_entry is None or blocks_entry is None:
                raise AdichtReadError("not a LabChart data file (no Map / Blocks_0 stream)")
            self.chart = _chartdata.parse_sequential(self._container.read_stream(map_entry))
            view = self._container.stream_view(blocks_entry)
            self._view = view if view is not None else memoryview(
                self._container.read_stream(blocks_entry))
        except AdichtionError as e:
            self._container.close()
            raise AdichtReadError(f"{Path(self.path).name}: {e}") from e
        except Exception:
            self._container.close()
            raise

    def __enter__(self) -> "AdichtFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._view = memoryview(b"")
        self._container.close()

    @property
    def n_records(self) -> int:
        return len(self.chart.records)

    def record_start(self, rec: int) -> datetime:
        """Start of the record as LabChart writes it into its own EDF exports."""
        return datetime.fromtimestamp(self.chart.records[rec].start_utc,
                                      tz=timezone.utc).replace(tzinfo=None)

    def comments(self, rec: int) -> list[tuple[float, str]]:
        r = self.chart.records[rec]
        return [(c.tick / r.tick_rate_hz, c.text) for c in r.comments]

    def channels(self, rec: int) -> list[ChannelInfo]:
        """Channels that hold samples in this record."""
        out = []
        for k in self.chart.records[rec].channels:
            n = sum(b.ticks for b in k.blocks)
            if n <= 0 or k.tick_rate_hz <= 0:
                continue
            u = k.units
            scale, offset, unit = 1.0, 0.0, ""
            if u is not None:
                lsb = u.lsb if math.isfinite(u.lsb) else 1.0
                if u.enabled:
                    scale, offset, unit = u.slope * lsb, u.offset, (u.units or u.base_units)
                else:
                    scale, unit = lsb, u.base_units
            if unit == "V":         # LabChart shows and exports biopotentials in mV
                scale, offset, unit = scale * 1000.0, offset * 1000.0, "mV"
            titles = self.chart.titles
            name = titles[k.channel] if k.channel < len(titles) and titles[k.channel] \
                else f"Channel {k.channel + 1}"
            out.append(ChannelInfo(index=k.channel, name=name, fs=float(k.tick_rate_hz),
                                   unit=unit, n_samples=n, is_integer=k.dtype == "int16",
                                   scale=scale, offset=offset))
        return out

    def stored(self, rec: int, channel: int) -> np.ndarray:
        """Samples as stored: int32 ADC counts, or float32 for calculated channels."""
        k = self.chart.records[rec].channels[channel]
        parts = []
        for b in k.blocks:
            sb = _subblock_at(self._view, b.offset)
            if not sb.is_packed and (b.dtype == "float32") != sb.is_float:
                # unpacked blocks do not say their width; the block index does
                width = 4 if b.dtype == "float32" else 2
                sb = SubBlock(index=0, offset=sb.offset, body_offset=sb.body_offset,
                              body_size=sb.body_size,
                              flags=(sb.flags | FLAG_32BIT) if width == 4 else (sb.flags & ~FLAG_32BIT),
                              n_samples=sb.body_size // width)
            try:
                parts.append(decode_subblock(
                    self._view[sb.body_offset: sb.body_offset + sb.body_size], sb))
            except AdichtionError as e:
                raise AdichtReadError(f"{Path(self.path).name}: {e}") from e
        if not parts:
            return np.zeros(0, np.int32)
        return np.concatenate(parts)

    def values(self, rec: int, channel: int) -> np.ndarray:
        """Samples in the channel's unit (see :meth:`channels`)."""
        info = next(c for c in self.channels(rec) if c.index == channel)
        return self.stored(rec, channel).astype(np.float64) * info.scale + info.offset


def edf_names(adicht_path: str, n_records: int, out_dir: str | None = None) -> list[str]:
    """Output paths for the records of a file, LabChart style."""
    p = Path(adicht_path)
    folder = Path(out_dir) if out_dir else p.parent
    if n_records == 1:
        return [str(folder / f"{p.stem}.edf")]
    return [str(folder / f"{p.stem}({i + 1}).edf") for i in range(n_records)]


def _write_record(f: AdichtFile, rec: int, edf_path: str) -> bool:
    """Write one record to ``edf_path``. False when the record holds no samples."""
    import pyedflib

    chans = [c for c in f.channels(rec) if c.fs == int(c.fs)]
    if not chans:
        return False
    # Data-record length as in LabChart's exports: the slowest channel gets one sample
    # per data record (0.5 s for a 2 Hz channel). The record is padded to a whole number
    # of data records, counted from its length in ticks.
    g = math.gcd(*(int(c.fs) for c in chans))
    rec_dur = 1.0 / g if g <= 50 else 1.0
    r = f.chart.records[rec]
    n_datarecords = max(1, math.ceil(r.ticks / r.tick_rate_hz / rec_dur - 1e-9))
    headers, data, digital = [], [], True
    for c in chans:
        x = f.stored(rec, c.index)
        want = int(round(n_datarecords * rec_dur * c.fs))
        if c.is_integer and c.scale > 0:
            x = x.astype(np.int32)
            # LabChart pads the last second with count 0; so do we
            x = np.pad(x[:want], (0, max(0, want - len(x))))
            lo, hi = c.scale * -32768 + c.offset, c.scale * 32767 + c.offset
            dlo, dhi = -32768, 32767
        else:
            digital = False
            x = x.astype(np.float64) * c.scale + c.offset
            x = np.pad(x[:want], (0, max(0, want - len(x))), mode="edge" if len(x) else "constant")
            lo, hi = (float(np.nanmin(x)), float(np.nanmax(x))) if len(x) else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            dlo, dhi = -32768, 32767
        headers.append({"label": c.name[:16], "dimension": c.unit[:8],
                        "sample_frequency": int(c.fs), "physical_min": lo, "physical_max": hi,
                        "digital_min": dlo, "digital_max": dhi, "transducer": "", "prefilter": ""})
        data.append(x)
    if not digital:                     # one mode per file: convert integer channels too
        for i, c in enumerate(chans):
            if data[i].dtype == np.int32:
                data[i] = data[i].astype(np.float64) * c.scale + c.offset

    writer = pyedflib.EdfWriter(edf_path, len(chans), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(f.record_start(rec).replace(microsecond=0))
        writer.setSignalHeaders(headers)
        if rec_dur != 1.0:
            writer.setDatarecordDuration(rec_dur)
        writer.writeSamples(data, digital=digital)
        for onset, text in f.comments(rec):
            writer.writeAnnotation(float(onset), -1, text)
    finally:
        writer.close()
    return True


def convert_adicht_native(
    adicht_path: str,
    out_dir: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> list[str]:
    """Convert every record of a LabChart file to its own EDF+. Returns the files written.

    An EDF appears under its final name only once it is complete. ``progress_cb(stage,
    fraction)`` is called between records and may raise to stop the conversion.
    """
    written: list[str] = []
    with AdichtFile(adicht_path) as f:
        names = edf_names(adicht_path, f.n_records, out_dir)
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for rec, edf in enumerate(names):
            if progress_cb is not None:
                progress_cb("writing", rec / max(f.n_records, 1))
            part = edf + ".part"
            try:
                if _write_record(f, rec, part):
                    os.replace(part, edf)
                    written.append(edf)
            finally:
                if os.path.exists(part):
                    os.remove(part)
    if progress_cb is not None:
        progress_cb("done", 1.0)
    return written


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = convert_adicht_native(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                                lambda s, fr: print(f"  {s} {100 * fr:.0f}%", flush=True))
    print(f"wrote {len(out)} EDF file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
