"""Native OS file/folder dialogs for the in-window desktop app.

The webview window has no browser file-input or download capability, so every
picker shells out to the platform's native dialog:

* macOS              -> AppleScript via ``osascript``
* Linux              -> ``zenity`` (GTK) or ``kdialog`` (KDE), whichever exists
* Windows / fallback -> ``tkinter``

Each call blocks until the user responds and returns the chosen path(s), or
``None`` / ``[]`` on cancel (or if no backend is available). Dialogs run in a
subprocess so a crash in a native toolkit can't take down the Dash server.

Note on Linux: tkinter is a separate system package (``python3-tk``) that is
frequently absent, which is why ``zenity``/``kdialog`` are tried first — they
ship with most desktops and need no extra setup.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

_TIMEOUT = 120  # seconds a dialog may stay open before we give up


# ── subprocess plumbing ─────────────────────────────────────────────────

def _stdout(cmd: list[str]) -> str:
    """Run *cmd*, returning its raw stdout ('' on any error, cancel, or timeout)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT
        ).stdout
    except Exception:
        return ""


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
        body,
        "root.destroy()",
        "print(out or '')",
    ])
    return _stdout([sys.executable, "-c", script])


# ── public API ──────────────────────────────────────────────────────────

def pick_folder(title: str = "Select folder") -> str | None:
    """Open a folder picker. Returns the chosen path, or None if cancelled."""
    system = platform.system()
    if system == "Darwin":
        out = _stdout(["osascript", "-e",
                       f'POSIX path of (choose folder with prompt "{title}")'])
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
    system = platform.system()
    if system == "Darwin":
        out = _stdout(["osascript", "-e",
                       f'POSIX path of (choose file with prompt "{title}"'
                       f'{_osascript_of_type(exts)})'])
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
        return _lines(_stdout(["osascript", "-e", script]))
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

    system = platform.system()
    if system == "Darwin":
        out = _stdout(["osascript", "-e",
                       f'POSIX path of (choose file name with prompt "{title}" '
                       f'default name "{default_name}")'])
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
