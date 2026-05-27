"""Native desktop window for NED-Net.

Serves the Dash app with a production WSGI server (waitress) on a background
thread and displays it inside a native OS window — WebView2 on Windows,
WKWebView on macOS, Qt WebKit on Linux — instead of a browser tab.

Launch with:
    python -m eeg_seizure_analyzer.dash_app.desktop
    nednet                      # console script (after `pip install -e ".[desktop]"`)
or the platform launcher:
    Windows : "Start NED-Net (Native).bat"
    macOS   : NED-Net.app  (double-click in Finder)
    Linux   : "start-nednet-native.sh"
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from pathlib import Path

import webview
from waitress import serve

from eeg_seizure_analyzer.dash_app.main import app

HOST = "127.0.0.1"
PREFERRED_PORT = 8051  # native app; falls back to a free port if busy
TITLE = "NED-Net"
WIN_WIDTH = 1440
WIN_HEIGHT = 900
MIN_SIZE = (960, 640)
SERVER_THREADS = 8  # waitress worker threads

_ASSETS = Path(__file__).parent / "assets"


def _icon_path() -> str | None:
    """Return the platform-appropriate window/dock icon, or None if missing.

    Windows needs a real ``.ico``; macOS uses ``.icns`` (also used by the
    NED-Net.app bundle); Linux loads the ``.png``.
    """
    if sys.platform.startswith("win"):
        name = "nednet_icon.ico"
    elif sys.platform == "darwin":
        name = "nednet.icns"
    else:
        name = "nednet_icon.png"
    icon = _ASSETS / name
    return str(icon) if icon.is_file() else None


def _set_macos_app_identity(icon: str | None) -> None:
    """Set the macOS dock icon and app name when launched via ``python -m``.

    (When launched from NED-Net.app, the bundle's Info.plist already does this,
    but running the module directly otherwise shows up as a generic "Python".)
    """
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = TITLE
            info["CFBundleDisplayName"] = TITLE
    except ImportError:
        pass

    if icon:
        try:
            from AppKit import NSApplication, NSImage
            img = NSImage.alloc().initWithContentsOfFile_(icon)
            if img:
                NSApplication.sharedApplication().setApplicationIconImage_(img)
        except ImportError:
            pass


def _pick_port(preferred: int) -> int:
    """Return ``preferred`` if free, otherwise an OS-assigned free port.

    Lets the native launcher coexist with the browser launcher (or a second
    instance) instead of crashing on "address already in use".
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, preferred))
            return preferred
        except OSError:
            pass  # taken — fall through to an ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _run_server(port: int) -> None:
    """Serve the Dash WSGI app via waitress (blocking) — runs in a daemon thread."""
    logging.getLogger("waitress").setLevel(logging.ERROR)  # quiet startup chatter
    serve(app.server, host=HOST, port=port, threads=SERVER_THREADS)


def _wait_until_up(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the server accepts TCP connections, or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.15)
    return False


def main() -> None:
    icon = _icon_path()
    if sys.platform == "darwin":
        _set_macos_app_identity(icon)

    port = _pick_port(PREFERRED_PORT)

    server_thread = threading.Thread(
        target=_run_server, args=(port,), daemon=True, name="ned-net-server"
    )
    server_thread.start()

    if not _wait_until_up(HOST, port):
        raise RuntimeError(
            f"NED-Net server did not come up on {HOST}:{port} within 30s."
        )

    webview.create_window(
        TITLE,
        f"http://{HOST}:{port}",
        width=WIN_WIDTH,
        height=WIN_HEIGHT,
        min_size=MIN_SIZE,
    )
    # Blocks on the native GUI event loop; returns when the window is closed.
    # The server thread is a daemon, so it dies with the process on return.
    webview.start(icon=icon)


if __name__ == "__main__":
    main()
