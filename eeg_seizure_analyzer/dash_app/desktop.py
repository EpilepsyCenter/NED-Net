"""Launch NED-Net as a standalone desktop window using pywebview.

Usage
-----
    python -m eeg_seizure_analyzer.dash_app.desktop

This opens NED-Net in its own native window (no browser needed).
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path


def _wait_for_server(url: str, timeout: float = 15.0):
    """Block until the Dash server responds or timeout."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.2)
    return False


def main():
    import webview

    # Resolve icon path
    assets_dir = Path(__file__).parent / "assets"
    icns_path = assets_dir / "nednet.icns"
    png_path = assets_dir / "nednet_logo.png"
    icon_path = str(icns_path if icns_path.exists() else png_path)

    # Set macOS dock icon and app name
    if sys.platform == "darwin":
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info:
                info["CFBundleName"] = "NED-Net"
                info["CFBundleDisplayName"] = "NED-Net"
        except ImportError:
            pass

        try:
            from AppKit import NSApplication, NSImage
            ns_app = NSApplication.sharedApplication()
            icon = NSImage.alloc().initWithContentsOfFile_(icon_path)
            if icon:
                ns_app.setApplicationIconImage_(icon)
        except ImportError:
            pass

    # Start Dash server in a background thread
    from eeg_seizure_analyzer.dash_app.main import app

    server_thread = threading.Thread(
        target=lambda: app.run(debug=False, port=8050, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    # Wait for server to be ready before opening the window
    print("Waiting for server to start...")
    if not _wait_for_server("http://127.0.0.1:8050"):
        print("Warning: server did not respond in time, opening window anyway")

    # Open native window — this blocks until the window is closed
    window = webview.create_window(
        "NED-Net",
        "http://127.0.0.1:8050",
        width=1400,
        height=900,
        min_size=(1024, 700),
    )
    webview.start(gui="cef" if sys.platform == "linux" else None)


if __name__ == "__main__":
    main()
