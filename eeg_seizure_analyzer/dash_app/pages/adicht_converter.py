"""Tools subtab – ADICHT → EDF Converter (Windows only)."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert


# ── Helpers ─────────────────────────────────────────────────────────

_convert_progress: dict[str, dict] = {}

_IS_WINDOWS = sys.platform == "win32"


def _browse_files() -> list[str]:
    """Open a native file picker for .adicht files."""
    import subprocess as _sp

    if sys.platform == "darwin":
        script = (
            'set theFiles to choose file of type {"adicht"} '
            'with prompt "Select ADICHT files" with multiple selections allowed\n'
            "set output to \"\"\n"
            "repeat with f in theFiles\n"
            '    set output to output & POSIX path of f & "\\n"\n'
            "end repeat\n"
            "return output"
        )
        result = _sp.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        paths = [p for p in result.stdout.strip().split("\n") if p]
        return paths

    # Windows / Linux: tkinter
    script = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
root.update()
files = filedialog.askopenfilenames(
    title="Select ADICHT files",
    filetypes=[("ADICHT files", "*.adicht"), ("All files", "*.*")],
)
root.destroy()
for f in files:
    print(f)
"""
    result = _sp.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    return [p for p in result.stdout.strip().split("\n") if p]


# ── Layout ──────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the ADICHT converter tab layout."""

    return html.Div(
        style={"padding": "24px", "maxWidth": "900px"},
        children=[
            html.H4("ADICHT → EDF Converter", style={"marginBottom": "4px"}),
            html.Div(
                [
                    html.Span(
                        "Windows only",
                        style={
                            "backgroundColor": "#da3633",
                            "color": "#fff",
                            "padding": "2px 8px",
                            "borderRadius": "4px",
                            "fontSize": "0.75rem",
                            "fontWeight": "600",
                            "marginRight": "8px",
                        },
                    ),
                    html.Span(
                        "Requires adi-reader package (pip install adi-reader)",
                        style={"color": "var(--ned-text-muted)", "fontSize": "0.82rem"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),

            html.P(
                "Convert LabChart .adicht files to EDF+ format. "
                "EDF is required for detection, training, ML inference, "
                "and saving/loading results.",
                style={"color": "var(--ned-text-muted)", "fontSize": "0.9rem",
                       "marginBottom": "24px"},
            ),

            # Platform check
            html.Div(id="ac-platform-check", children=_platform_check()),

            # Input files
            html.Label("ADICHT files", style={"fontSize": "0.82rem",
                                                "color": "var(--ned-text-muted)"}),
            dbc.InputGroup([
                dbc.Textarea(
                    id="ac-file-list",
                    placeholder="Paste .adicht file paths here (one per line), or click Browse",
                    style={"minHeight": "80px", "fontFamily": "monospace",
                           "fontSize": "0.82rem"},
                ),
            ], className="mb-2"),

            dbc.Button(
                "Browse", id="ac-browse-btn",
                className="btn-ned-secondary",
                style={"marginBottom": "16px"},
            ),

            # Output folder
            html.Label("Output folder (leave empty to save next to originals)",
                       style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
            dbc.Input(
                id="ac-output-folder",
                placeholder="/path/to/output/folder (optional)",
                type="text",
                className="mb-3",
            ),

            # Convert button
            dbc.Button(
                "Convert to EDF",
                id="ac-convert-btn",
                className="btn-ned-primary",
                disabled=not _IS_WINDOWS,
            ),

            # Progress
            dcc.Interval(id="ac-progress-interval", interval=1000,
                         disabled=True),
            html.Div(id="ac-progress", style={"marginTop": "16px"}),

            # Status
            html.Div(id="ac-status", style={"marginTop": "16px"}),
        ],
    )


def _platform_check():
    if _IS_WINDOWS:
        try:
            import adi  # noqa: F401
            return html.Div(
                "adi-reader found — ready to convert",
                style={"color": "var(--ned-success)", "fontSize": "0.82rem",
                       "marginBottom": "12px"},
            )
        except ImportError:
            return alert(
                "adi-reader not installed. Install with: pip install adi-reader",
                "danger",
            )
    return alert(
        "ADICHT conversion requires Windows with the adi-reader package. "
        "This tool is not available on macOS or Linux.",
        "warning",
    )


# ── Browse callback ─────────────────────────────────────────────────


@callback(
    Output("ac-file-list", "value"),
    Input("ac-browse-btn", "n_clicks"),
    State("ac-file-list", "value"),
    prevent_initial_call=True,
)
def browse_adicht(n_clicks, current):
    if not n_clicks:
        return no_update
    try:
        paths = _browse_files()
        if not paths:
            return no_update
        # Append to existing list
        existing = [p.strip() for p in (current or "").split("\n") if p.strip()]
        combined = existing + [p for p in paths if p not in existing]
        return "\n".join(combined)
    except Exception:
        return no_update


# ── Convert callback ────────────────────────────────────────────────


@callback(
    Output("ac-status", "children"),
    Output("ac-progress-interval", "disabled"),
    Output("ac-convert-btn", "disabled"),
    Input("ac-convert-btn", "n_clicks"),
    State("ac-file-list", "value"),
    State("ac-output-folder", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_convert(n_clicks, file_list, output_folder, sid):
    if not n_clicks:
        return no_update, no_update, no_update

    if not _IS_WINDOWS:
        return alert("Conversion is only available on Windows.", "danger"), True, False

    # Parse file list
    paths = [p.strip() for p in (file_list or "").split("\n") if p.strip()]
    if not paths:
        return alert("No files specified.", "warning"), True, False

    # Validate files exist and are .adicht
    for p in paths:
        if not os.path.isfile(p):
            return alert(f"File not found: {p}", "danger"), True, False
        if not p.lower().endswith(".adicht"):
            return alert(f"Not an ADICHT file: {p}", "danger"), True, False

    # Validate output folder if specified
    if output_folder and not os.path.isdir(output_folder):
        return alert(f"Output folder not found: {output_folder}", "danger"), True, False

    # Build conversion list: (adicht_path, edf_path)
    conversions = []
    for p in paths:
        stem = Path(p).stem
        if output_folder:
            edf = os.path.join(output_folder, stem + ".edf")
        else:
            edf = str(Path(p).with_suffix(".edf"))

        if os.path.exists(edf):
            return alert(f"Output already exists: {edf}", "warning"), True, False
        conversions.append((p, edf))

    # Init progress
    _convert_progress[sid] = {
        "current": 0,
        "total": len(conversions),
        "current_file": "",
        "done": False,
        "error": None,
        "results": [],
    }

    t = threading.Thread(
        target=_run_conversions,
        args=(sid, conversions),
        daemon=True,
    )
    t.start()

    return (
        html.Div("Conversion started...", style={"color": "var(--ned-accent)"}),
        False,
        True,
    )


def _run_conversions(sid: str, conversions: list[tuple[str, str]]):
    """Background: convert each ADICHT file to EDF."""
    from eeg_seizure_analyzer.io.adicht_to_edf import convert_adicht_to_edf

    progress = _convert_progress[sid]

    for i, (adicht_path, edf_path) in enumerate(conversions):
        progress["current"] = i
        progress["current_file"] = os.path.basename(adicht_path)

        try:
            convert_adicht_to_edf(adicht_path, edf_path)
            progress["results"].append({
                "file": os.path.basename(adicht_path),
                "output": edf_path,
                "status": "ok",
            })
        except Exception as e:
            progress["error"] = f"{os.path.basename(adicht_path)}: {e}"
            progress["done"] = True
            return

    progress["current"] = len(conversions)
    progress["done"] = True


# ── Progress polling ────────────────────────────────────────────────


@callback(
    Output("ac-progress", "children"),
    Output("ac-progress-interval", "disabled", allow_duplicate=True),
    Output("ac-convert-btn", "disabled", allow_duplicate=True),
    Input("ac-progress-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_progress(n_intervals, sid):
    progress = _convert_progress.get(sid)
    if not progress:
        return no_update, no_update, no_update

    total = progress["total"]
    current = progress["current"]
    pct = int((current / total) * 100) if total else 0

    bar = dbc.Progress(
        value=pct,
        label=f"{current}/{total}",
        striped=True,
        animated=not progress["done"],
        className="mb-2",
    )

    if progress["error"]:
        _convert_progress.pop(sid, None)
        return html.Div([
            bar,
            alert(f"Error: {progress['error']}", "danger"),
        ]), True, False

    if progress["done"]:
        _convert_progress.pop(sid, None)
        results = progress["results"]
        rows = []
        for r in results:
            rows.append(html.Tr([
                html.Td(r["file"], style={"padding": "4px 12px"}),
                html.Td(r["output"], style={"padding": "4px 12px",
                                             "fontFamily": "monospace",
                                             "fontSize": "0.8rem"}),
            ]))

        return html.Div([
            dbc.Progress(value=100, label="100%", color="success",
                         className="mb-2"),
            alert(f"Successfully converted {len(results)} file(s) to EDF.", "success"),
            html.Table(
                [html.Thead(html.Tr([
                    html.Th("Source", style={"padding": "4px 12px"}),
                    html.Th("Output", style={"padding": "4px 12px"}),
                ]))] + [html.Tbody(rows)],
                style={"width": "100%", "fontSize": "0.85rem",
                       "marginTop": "12px"},
            ) if rows else None,
        ]), True, False

    status = f"Converting {progress['current_file']}..." if progress["current_file"] else "Starting..."

    return html.Div([
        bar,
        html.Div(status, style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)"}),
    ]), False, True
