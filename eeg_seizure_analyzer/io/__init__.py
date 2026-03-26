"""IO module: read EEG data from .adicht and .edf files."""

from eeg_seizure_analyzer.io.base import Annotation, EEGRecording, RecordInfo
from eeg_seizure_analyzer.io.edf_reader import read_edf

__all__ = ["Annotation", "EEGRecording", "RecordInfo", "read_edf"]

# Conditionally export adicht reader (Windows-only)
try:
    from eeg_seizure_analyzer.io.adicht_reader import read_adicht

    __all__.append("read_adicht")
except ImportError:
    pass
