"""Native OS file/folder dialogs for the in-window desktop app.

The webview window has no browser file-input or download capability, so every
picker uses a native dialog:

* desktop app, macOS -> the pywebview window's own dialog (``set_native_window``).
                        It belongs to the app's window, so it opens at once and
                        with keyboard focus.
* macOS (browser)    -> AppleScript via ``osascript``
* Linux              -> ``zenity`` (GTK) or ``kdialog`` (KDE), whichever exists
* Windows / fallback -> ``tkinter``

Each call blocks until the user responds and returns the chosen path(s), or
``None`` / ``[]`` on cancel (or if no backend is available). Dialogs run in a
subprocess so a crash in a native toolkit can't take down the Dash server.

Note on Linux: tkinter is a separate system package (``python3-tk``) that is
frequently absent, which is why ``zenity``/``kdialog`` are tried first — they
ship with most desktops and need no extra setup.

Why not osascript in the desktop app: it runs as a separate background process,
so its dialog opened without keyboard focus (a first click only focused it).
Making it ``activate`` first made it slower to appear and still unfocused, since
macOS no longer lets a background process take focus from the active app.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

_TIMEOUT = 120  # seconds a dialog may stay open before we give up

# The desktop app's pywebview window, when running as the desktop app
# (set by desktop.main). None in a browser: the other backends are used then.
_native_window = None


def set_native_window(window) -> None:
    """Route every dialog through this pywebview window (desktop app, macOS only).

    macOS only, on purpose:
    - The focus problem was macOS-specific (osascript is a separate background
      process). On Windows the tkinter dialog is topmost and focused; on Linux
      zenity/kdialog are focused by the window manager.
    - pywebview's Windows backend shows the dialog on the calling thread, which
      here is a server thread, not the GUI thread. When that fails it logs the
      error and returns None, which looks like Cancel, so Browse would silently
      do nothing. That cannot be tested from macOS, so the proven backends stay.
    """
    global _native_window
    _native_window = window if platform.system() == "Darwin" else None


def _webview_dialog(kind: str, title: str = "", exts=(), multiple: bool = False,
                    save_name: str = ""):
    """Open the dialog on the app's own window.

    Returns a list of paths, or None when there is no window or the dialog failed,
    so the caller falls back to the other backends. pywebview runs the dialog on
    its GUI thread and blocks this (callback) thread until the user answers.
    """
    if _native_window is None:
        return None
    try:
        import webview
        dtype = {"open": webview.FileDialog.OPEN, "folder": webview.FileDialog.FOLDER,
                 "save": webview.FileDialog.SAVE}[kind]
        types = ()
        if exts and kind == "open":
            # pywebview accepts only letters, digits, _ and spaces in the label
            # ("EDF/ADICHT files" was rejected, so the dialog fell back to the old
            # unfocused one). A filter that still fails validation is dropped:
            # an unfiltered fast dialog beats the slow one.
            from webview.util import parse_file_type
            names = [e.upper() for e in exts]
            label = " and ".join(names) if len(names) == 2 else " ".join(names)
            cand = (f"{label} files ({';'.join('*.' + e for e in exts)})",
                    "All files (*.*)")
            ok = []
            for t in cand:
                try:
                    parse_file_type(t)
                    ok.append(t)
                except ValueError:
                    pass
            types = tuple(ok) if len(ok) == len(cand) else ()
        res = _native_window.create_file_dialog(
            dtype, allow_multiple=multiple, save_filename=save_name, file_types=types)
    except Exception:
        return None
    if not res:
        return []
    return [res] if isinstance(res, str) else [str(r) for r in res]


# ── subprocess plumbing ─────────────────────────────────────────────────

def _stdout(cmd: list[str]) -> str:
    """Run *cmd*, returning its raw stdout ('' on any error, cancel, or timeout)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT
        ).stdout
    except Exception:
        return ""


def _osascript(script: str) -> str:
    """Run an AppleScript dialog (macOS in a browser; see the module docstring)."""
    return _stdout(["osascript", "-e", script])


def _lines(text: str) -> list[str]:
    return [p for p in text.splitlines() if p.strip()]


# ── filter formatting per backend ───────────────────────────────────────

def _norm_exts(exts) -> list[str]:
    """Strip any leading ``*.`` / ``.`` so callers can pass either form."""
    out = []
    for e in (exts or ()):
        e = e.strip()
        if e.startswith("*."):
            e = e[2:]
        e = e.lstrip(".")
        if e and e != "*":
            out.append(e)
    return out


def _zenity_filters(exts: list[str]) -> list[str]:
    """``--file-filter`` args for zenity (specific filter first, then 'All')."""
    if not exts:
        return []
    label = "/".join(e.upper() for e in exts)
    pattern = " ".join(f"*.{e}" for e in exts)
    return [f"--file-filter={label} files | {pattern}",
            "--file-filter=All files | *"]


def _kdialog_filter(exts: list[str]) -> str:
    if not exts:
        return "*"
    pattern = " ".join(f"*.{e}" for e in exts)
    label = "/".join(e.upper() for e in exts)
    return f"{pattern}|{label} files"


def _tk_filetypes(exts: list[str]) -> list[tuple[str, str]]:
    if exts:
        label = "/".join(e.upper() for e in exts) + " files"
        pattern = " ".join(f"*.{e}" for e in exts)
        return [(label, pattern), ("All files", "*.*")]
    return [("All files", "*.*")]


def _osascript_of_type(exts: list[str]) -> str:
    if not exts:
        return ""
    quoted = ", ".join('"' + e + '"' for e in exts)
    return f" of type {{{quoted}}}"


# ── tkinter fallback (Windows; last resort everywhere) ──────────────────

def _tk(body: str) -> str:
    """Run a tkinter snippet in a child interpreter; return its stdout.

    *body* must assign the result to ``out`` (a str; newline-joined for
    multi-select)."""
    script = "\n".join([
        "import tkinter as tk",
        "from tkinter import filedialog",
        "root = tk.Tk(); root.withdraw()",
        "try:",
        "    root.attributes('-topmost', True)",
        "except Exception:",
        "    pass",
        "root.update()",
        "try:",
        "    root.lift(); root.focus_force()",
        "except Exception:",
        "    pass",
        body,
        "root.destroy()",
        "print(out or '')",
    ])
    return _stdout([sys.executable, "-c", script])


# ── public API ──────────────────────────────────────────────────────────

def pick_folder(title: str = "Select folder") -> str | None:
    """Open a folder picker. Returns the chosen path, or None if cancelled."""
    res = _webview_dialog("folder", title)
    if res is not None:
        return res[0].rstrip("/") if res else None
    system = platform.system()
    if system == "Darwin":
        out = _osascript(f'POSIX path of (choose folder with prompt "{title}")')
        return out.strip().rstrip("/") or None
    if system == "Linux":
        z = shutil.which("zenity")
        if z:
            return _stdout([z, "--file-selection", "--directory",
                            f"--title={title}"]).strip() or None
        k = shutil.which("kdialog")
        if k:
            return _stdout([k, "--title", title, "--getexistingdirectory",
                            os.path.expanduser("~")]).strip() or None
    return _tk(f'out = filedialog.askdirectory(title="{title}")').strip() or None


def pick_file(title: str = "Select file", exts=()) -> str | None:
    """Open a single-file picker. *exts* is the allowed extensions without the
    dot (empty = any file). Returns the chosen path, or None if cancelled."""
    exts = _norm_exts(exts)
    res = _webview_dialog("open", title, exts)
    if res is not None:
        return res[0] if res else None
    system = platform.system()
    if system == "Darwin":
        out = _osascript(f'POSIX path of (choose file with prompt "{title}"'
                         f'{_osascript_of_type(exts)})')
        return out.strip() or None
    if system == "Linux":
        z = shutil.which("zenity")
        if z:
            cmd = [z, "--file-selection", f"--title={title}"] + _zenity_filters(exts)
            return _stdout(cmd).strip() or None
        k = shutil.which("kdialog")
        if k:
            return _stdout([k, "--title", title, "--getopenfilename",
                            os.path.expanduser("~"),
                            _kdialog_filter(exts)]).strip() or None
    body = (f'out = filedialog.askopenfilename(title="{title}", '
            f'filetypes={_tk_filetypes(exts)!r})')
    return _tk(body).strip() or None


def pick_files(title: str = "Select files", exts=()) -> list[str]:
    """Open a multi-file picker. Returns the list of chosen paths (possibly
    empty)."""
    exts = _norm_exts(exts)
    res = _webview_dialog("open", title, exts, multiple=True)
    if res is not None:
        return res
    system = platform.system()
    if system == "Darwin":
        script = "\n".join([
            f'set theFiles to (choose file with prompt "{title}"'
            f'{_osascript_of_type(exts)} with multiple selections allowed)',
            'set out to ""',
            'repeat with f in theFiles',
            '    set out to out & POSIX path of f & linefeed',
            'end repeat',
            'return out',
        ])
        return _lines(_osascript(script))
    if system == "Linux":
        z = shutil.which("zenity")
        if z:
            cmd = [z, "--file-selection", "--multiple", "--separator=\n",
                   f"--title={title}"] + _zenity_filters(exts)
            return _lines(_stdout(cmd))
        k = shutil.which("kdialog")
        if k:
            return _lines(_stdout([k, "--title", title, "--multiple",
                                   "--separate-output", "--getopenfilename",
                                   os.path.expanduser("~"),
                                   _kdialog_filter(exts)]))
    body = (f'fs = filedialog.askopenfilenames(title="{title}", '
            f'filetypes={_tk_filetypes(exts)!r})\n'
            f'out = "\\n".join(fs)')
    return _lines(_tk(body))


def save_file(default_name: str, title: str = "Save file",
              default_ext: str | None = None) -> str | None:
    """Open a 'Save as' dialog. *default_ext* (e.g. ``".csv"``) is appended when
    the user-typed name has no extension; if None it's taken from *default_name*.
    Returns the chosen path, or None if cancelled."""
    if default_ext is None:
        default_ext = os.path.splitext(default_name)[1]  # ".csv" or ""

    def _ensure_ext(path: str) -> str | None:
        path = path.strip()
        if path and default_ext and not os.path.splitext(path)[1]:
            path += default_ext
        return path or None

    res = _webview_dialog("save", title, save_name=default_name)
    if res is not None:
        return _ensure_ext(res[0]) if res else None

    system = platform.system()
    if system == "Darwin":
        out = _osascript(f'POSIX path of (choose file name with prompt "{title}" '
                         f'default name "{default_name}")')
        return out.strip() or None
    if system == "Linux":
        z = shutil.which("zenity")
        if z:
            return _ensure_ext(_stdout([z, "--file-selection", "--save",
                                        "--confirm-overwrite",
                                        f"--filename={default_name}",
                                        f"--title={title}"]))
        k = shutil.which("kdialog")
        if k:
            return _ensure_ext(_stdout([k, "--title", title,
                                        "--getsavefilename", default_name]))
    body = (f'out = filedialog.asksaveasfilename(title="{title}", '
            f'initialfile="{default_name}", defaultextension="{default_ext}")')
    return _tk(body).strip() or None
