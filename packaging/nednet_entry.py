"""Entry point of the packaged NED-Net app (PyInstaller, see nednet.spec).

Besides starting the desktop app, it covers what a bundled app lacks:
- worker processes (dataset building, background proposals, DataLoader
  workers) re-launch this executable: multiprocessing.freeze_support();
- Windows file dialogs run a small tkinter snippet in a second instance of the
  app (file_dialogs._tk_frozen): ``--nednet-run-script <code>``;
- a windowed app has no console: output and errors go to a log file.
"""
import multiprocessing
import os
import sys


def _log_path() -> str:
    """Per-platform log file location for the packaged app."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        folder = os.path.join(home, "Library", "Logs")
    elif sys.platform.startswith("win"):
        folder = os.path.join(os.environ.get("LOCALAPPDATA", home), "NED-Net")
    else:
        folder = os.path.join(os.environ.get("XDG_STATE_HOME",
                                             os.path.join(home, ".local", "state")), "nednet")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "NED-Net.log")


def main() -> None:
    multiprocessing.freeze_support()

    if len(sys.argv) >= 3 and sys.argv[1] == "--nednet-run-script":
        # Helper mode (file dialogs on Windows): run the given snippet and exit.
        exec(compile(sys.argv[2], "<nednet-helper>", "exec"), {"__name__": "__main__"})
        return

    # No console in a windowed app: keep output in a log file instead.
    try:
        log = open(_log_path(), "a", buffering=1, encoding="utf-8")
        sys.stdout = log
        sys.stderr = log
        import datetime
        print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] NED-Net starting "
              f"({sys.executable})", flush=True)
    except Exception:
        pass

    from eeg_seizure_analyzer.dash_app.desktop import main as desktop_main
    desktop_main()


if __name__ == "__main__":
    main()
