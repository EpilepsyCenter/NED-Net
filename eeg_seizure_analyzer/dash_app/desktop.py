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

import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import webview
from webview.menu import Menu, MenuAction, MenuSeparator
from waitress import serve

from eeg_seizure_analyzer import __version__
# The Dash app (and with it numpy, torch, ...) is imported in _boot, after
# the window shows its splash: a cold first start can take a while (e.g. while
# Windows Defender scans torch).

HOST = "127.0.0.1"
PREFERRED_PORT = 8051  # native app; falls back to a free port if busy
TITLE = "NED-Net"
MIN_SIZE = (960, 640)
# Window background while the page loads (matches --ned-bg in style.css),
# so the window doesn't flash white on start-up.
BACKGROUND = "#F3F2EE"
# Default window: a comfortable fraction of the primary screen so it fills a
# typical laptop display (e.g. ~1730x970 on 1920x1080) without going edge-to-edge.
WIN_SCREEN_FRACTION = 0.90
WIN_MAX = (2200, 1400)        # cap so it stays sane on large/ultrawide monitors
WIN_FALLBACK = (1600, 1000)   # used only if the screen size can't be detected
SERVER_THREADS = 8  # waitress worker threads

_ASSETS = Path(__file__).parent / "assets"
_WIN_STATE_PATH = Path.home() / ".eeg_seizure_analyzer" / "window.json"
_WEBVIEW_STORAGE = Path.home() / ".eeg_seizure_analyzer" / "webview"


def _bring_to_front() -> None:
    """Make NED-Net the active app with the window in front.

    Started from a terminal (``python -m``), macOS leaves the new window
    behind the terminal and not focused, so the first clicks only activated it.
    """
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication
        from PyObjCTools import AppHelper
        AppHelper.callAfter(
            lambda: NSApplication.sharedApplication().activateIgnoringOtherApps_(True))
    except Exception:
        pass


def _default_window_size() -> tuple[int, int]:
    """A comfortably large default: ~90% of the primary screen, capped so it
    stays reasonable on big/ultrawide monitors and never below MIN_SIZE.

    Falls back to a fixed size if the screen can't be detected (e.g. a
    pywebview build without ``screens``).
    """
    try:
        screen = webview.screens[0]
        w = min(int(int(screen.width) * WIN_SCREEN_FRACTION), WIN_MAX[0])
        h = min(int(int(screen.height) * WIN_SCREEN_FRACTION), WIN_MAX[1])
    except Exception:
        w, h = WIN_FALLBACK
    return (max(w, MIN_SIZE[0]), max(h, MIN_SIZE[1]))


def _load_window_size() -> tuple[int, int]:
    """Restore the last saved window size (clamped to MIN_SIZE); otherwise a
    screen-sized default."""
    try:
        d = json.loads(_WIN_STATE_PATH.read_text())
        return (max(int(d["width"]), MIN_SIZE[0]),
                max(int(d["height"]), MIN_SIZE[1]))
    except Exception:
        return _default_window_size()


def _save_window_size(width: int, height: int) -> None:
    """Persist the window size so the next launch reopens at the same size."""
    try:
        _WIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WIN_STATE_PATH.write_text(
            json.dumps({"width": int(width), "height": int(height)}))
    except Exception:
        pass


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


# ── Native menu bar ───────────────────────────────────────────────────
# macOS shows it in the global menu bar (next to the app menu with About /
# Hide / Quit); Windows and Linux show it inside the window. Menu actions
# drive the page through window.nednet (assets/app_shell.js).

# Go menu: (sidebar section id, label)
_GO_ITEMS = [
    ("upload", "Load"),
    ("viewer", "Viewer"),
    (None, None),
    ("seizures", "Seizure Detection"),
    ("spikes", "Interictal Spike Detection"),
    (None, None),
    ("training", "Seizure Annotation"),
    ("training_spikes", "Spike Annotation"),
    ("training_background", "Background Annotation"),
    (None, None),
    ("ml_datasets", "Dataset & Model"),
    ("blind_review", "Blind Review"),
    ("analysis", "Batch Analysis"),
    ("results", "Results"),
]

_ABOUT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body {{ margin: 0; padding: 28px 24px; text-align: center; background: #f3f2ee;
         color: #15171b; font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI",
         system-ui, Cantarell, Ubuntu, sans-serif; user-select: none; cursor: default; }}
  img {{ width: 180px; margin-bottom: 14px; }}
  .v {{ color: #55585f; margin: 4px 0 14px; }}
  p {{ margin: 0 auto; max-width: 300px; color: #55585f; line-height: 1.45; }}
</style></head><body>
  <img src="{base}/assets/nednet_logo.png" alt="NED-Net">
  <div class="v">Version {version}</div>
  <p>Detection and annotation of seizures and interictal spikes in long-term EEG recordings.</p>
</body></html>"""


def _run_js(window, code: str) -> None:
    """Run JavaScript in the page; ignore failures (e.g. page still loading)."""
    try:
        window.evaluate_js(code)
    except Exception:
        pass


def _build_menu(window, urls: dict) -> list:
    """``urls['base']`` is the app's address, set once the server runs."""
    def go(tab):
        return lambda: _run_js(window, f"window.nednet && window.nednet.go({json.dumps(tab)})")

    def open_recording(kind):
        return lambda: _run_js(window, f"window.nednet && window.nednet.open({json.dumps(kind)})")

    def show_manual():
        if not urls.get("base"):
            return  # still starting
        webview.create_window("NED-Net Manual", f"{urls['base']}/assets/manual.html",
                              width=980, height=800, background_color=BACKGROUND)

    def show_about():
        if not urls.get("base"):
            return
        webview.create_window(
            "About NED-Net",
            html=_ABOUT_HTML.format(base=urls["base"], version=__version__),
            width=380, height=300, resizable=False, background_color=BACKGROUND,
        )

    file_items = [
        MenuAction("Open Recording…", open_recording("file")),
        MenuAction("Open Several Recordings…", open_recording("multiple")),
    ]
    if sys.platform != "darwin":  # macOS has Quit in the app menu
        file_items += [MenuSeparator(), MenuAction("Exit", window.destroy)]

    go_items = [MenuSeparator() if tab is None else MenuAction(label, go(tab))
                for tab, label in _GO_ITEMS]
    go_items += [
        MenuSeparator(),
        MenuAction("Show/Hide Settings Panel",
                   lambda: _run_js(window, "window.nednet && window.nednet.togglePanel()")),
        MenuAction("Toggle Dark Mode",
                   lambda: _run_js(window, "window.nednet && window.nednet.toggleTheme()")),
    ]

    return [
        Menu("File", file_items),
        Menu("Go", go_items),
        Menu("Tools", [
            MenuAction("Video Converter", go("video_converter")),
            MenuAction("ADICHT → EDF Converter", go("adicht_converter")),
        ]),
        Menu("Help", [
            MenuAction("NED-Net Manual", show_manual),
            MenuSeparator(),
            MenuAction("About NED-Net", show_about),
        ]),
    ]


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


def _run_server(app, port: int) -> None:
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


# ── Splash screen while the app starts ───────────────────────────────

_SPLASH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root {{ --bg: #f3f2ee; --text: #15171b; --muted: #55585f; --accent: #13788c;
           --track: rgba(20, 22, 28, 0.10); --danger: #b5332a; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #0e1114; --text: #e6e8ec; --muted: #9ba2ad; --accent: #6cc3d5;
             --track: rgba(255, 255, 255, 0.12); --danger: #f0857c; }}
  }}
  html, body {{ height: 100%; margin: 0; }}
  body {{ display: flex; flex-direction: column; align-items: center; justify-content: center;
         background: var(--bg); color: var(--text); user-select: none; cursor: default;
         font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Cantarell,
               Ubuntu, sans-serif; }}
  img {{ width: 240px; margin-bottom: 28px; padding: 16px 22px; border-radius: 18px; }}
  @media (prefers-color-scheme: dark) {{ img {{ background: #f3f2ee; }} }}  /* navy logo text */
  .bar {{ position: relative; width: 220px; height: 3px; border-radius: 2px;
          background: var(--track); overflow: hidden; }}
  .bar::after {{ content: ""; position: absolute; top: 0; bottom: 0; width: 40%;
                 border-radius: 2px; background: var(--accent);
                 animation: slide 1.1s ease-in-out infinite; }}
  @keyframes slide {{ from {{ left: -40%; }} to {{ left: 100%; }} }}
  #status {{ margin-top: 14px; color: var(--muted); }}
  #hint {{ margin-top: 6px; font-size: 11px; color: var(--muted); opacity: 0.8; }}
  .v {{ position: fixed; bottom: 16px; font-size: 11px; color: var(--muted); }}
  body.error .bar {{ display: none; }}
  body.error #status {{ color: var(--danger); max-width: 520px; text-align: center; }}
</style></head><body>
  <img src="{logo}" alt="NED-Net">
  <div class="bar"></div>
  <div id="status">Starting NED-Net…</div>
  <div id="hint">The first start after installing or updating can take a minute.</div>
  <div class="v">Version {version}</div>
  <script>
    window.setStatus = function (t) {{ document.getElementById('status').textContent = t; }};
    window.showError = function (t) {{
      document.body.classList.add('error');
      document.getElementById('status').textContent = t;
      document.getElementById('hint').textContent =
        'Close this window and try again. Details are in the log.';
    }};
  </script>
</body></html>"""


def _splash_html() -> str:
    """The splash page, with the logo embedded (it shows before the server runs)."""
    import base64
    logo = ""
    try:
        data = (_ASSETS / "nednet_logo.png").read_bytes()
        logo = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        pass
    return _SPLASH_HTML.format(logo=logo, version=__version__)


def _boot(window, urls: dict) -> None:
    """Start the app behind the splash (runs in pywebview's start-up thread):
    import the Dash app, start its server, then load it into the window."""
    def status(text):
        _run_js(window, f"window.setStatus && window.setStatus({json.dumps(text)})")

    try:
        status("Loading analysis libraries…")
        from eeg_seizure_analyzer.dash_app.main import app

        status("Starting…")
        port = _pick_port(PREFERRED_PORT)
        threading.Thread(target=_run_server, args=(app, port), daemon=True,
                         name="ned-net-server").start()
        if not _wait_until_up(HOST, port):
            raise RuntimeError(f"the app's server did not start on {HOST}:{port} within 30 s")

        urls["base"] = f"http://{HOST}:{port}"
        window.load_url(urls["base"])
    except Exception as exc:  # show it on the splash instead of a dead window
        logging.exception("NED-Net failed to start")
        _run_js(window, "window.showError && window.showError("
                        + json.dumps(f"NED-Net could not start: {exc}") + ")")


def main() -> None:
    icon = _icon_path()
    if sys.platform == "darwin":
        _set_macos_app_identity(icon)

    width, height = _load_window_size()
    size = {"w": width, "h": height}
    urls: dict = {"base": None}  # set by _boot once the server runs

    # The window opens at once with the splash; _boot then loads the app.
    window = webview.create_window(
        TITLE,
        html=_splash_html(),
        width=width,
        height=height,
        min_size=MIN_SIZE,
        background_color=BACKGROUND,
    )
    # Started from a terminal, macOS would leave the window behind it
    window.events.shown += lambda *_a: _bring_to_front()

    # File/folder pickers open on this window: fast, and with keyboard focus.
    from eeg_seizure_analyzer.dash_app import file_dialogs
    file_dialogs.set_native_window(window)

    # Track the latest size — pywebview's resized event passes (width, height).
    def _on_resized(*args):
        if len(args) >= 2:
            size["w"], size["h"] = args[0], args[1]

    try:
        window.events.resized += _on_resized
    except Exception:
        pass  # older pywebview without the resized event — size just won't update

    # Blocks on the native GUI event loop; returns when the window is closed.
    # The server thread is a daemon, so it dies with the process on return.
    try:
        _WEBVIEW_STORAGE.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # private_mode=False with our own storage folder: the page's settings
    # (e.g. the settings panel's width and visibility) survive a restart.
    webview.start(
        _boot, (window, urls),
        icon=icon,
        menu=_build_menu(window, urls),
        private_mode=False,
        storage_path=str(_WEBVIEW_STORAGE),
    )

    # Window closed — remember its final size for the next launch.
    _save_window_size(size["w"], size["h"])


if __name__ == "__main__":
    main()
