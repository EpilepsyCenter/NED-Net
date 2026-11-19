"""Parts of adichtion 1.1.0, the LabChart .adicht reader by Emil Ingerslev Walsted.

Source: https://github.com/emilwalsted/adichtion (GPL-3.0-or-later, see LICENSE here).
Copied unchanged: container.py, blocks.py, model.py, serialization.py, errors.py.
Changed: chartdata.py, one check in ``_parse_channel_entry`` relaxed so that files
with channels at different sampling rates can be read (marked "NED-Net change").
The rest of adichtion (viewer, exporters, the high-level ``AdichtFile``) is not included;
NED-Net reads samples through ``eeg_seizure_analyzer.io.adicht_native``.
"""
