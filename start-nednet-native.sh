#!/bin/bash
# Linux launcher — open NED-Net as a native window (GTK/Qt WebKit).
# First time only: make it executable with  chmod +x start-nednet-native.sh
# Requires the desktop extra with a Linux webview backend:
#     pip install -e ".[desktop]"     # pulls pywebview[qt]
# (Qt bundles its own web engine; no system WebKit packages needed.)
cd "$(dirname "$0")" || exit 1
echo "============================================================"
echo "  NED-Net  |  native desktop app (Linux WebKit)"
echo "  The app opens in its own window in a few seconds."
echo "  Close the app window (or press Ctrl+C here) to stop."
echo "============================================================"
echo
exec .venv/bin/python -m eeg_seizure_analyzer.dash_app.desktop
