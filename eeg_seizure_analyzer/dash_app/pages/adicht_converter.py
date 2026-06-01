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

            # Write mode
            dbc.Checkbox(
                id="ac-fast-write",
                label="Fast write mode (single C-level call, ~10–50× faster)",
                value=True,
                className="mb-3",
            ),
            html.Div(
                "Disable to fall back to the legacy 1-second block loop if "
                "the fast path produces a malformed EDF.",
                style={"color": "var(--ned-text-muted)", "fontSize": "0.78rem",
                       "marginTop": "-8px", "marginBottom": "16px"},
            ),

            # Convert / Stop buttons
            dbc.Button(
                "Convert to EDF",
                id="ac-convert-btn",
                className="btn-ned-primary",
                disabled=not _IS_WINDOWS,
            ),
            dbc.Button(
                "Stop",
                id="ac-stop-btn",
                className="btn-ned-secondary",
                style={"marginLeft": "8px"},
                disabled=True,
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
    Output("ac-stop-btn", "disabled"),
    Input("ac-convert-btn", "n_clicks"),
    State("ac-file-list", "value"),
    State("ac-output-folder", "value"),
    State("ac-fast-write", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_convert(n_clicks, file_list, output_folder, fast_write, sid):
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    if not _IS_WINDOWS:
        return alert("Conversion is only available on Windows.", "danger"), True, False, True

    # Parse file list
    paths = [p.strip() for p in (file_list or "").split("\n") if p.strip()]
    if not paths:
        return alert("No files specified.", "warning"), True, False, True

    # Validate files exist and are .adicht
    for p in paths:
        if not os.path.isfile(p):
            return alert(f"File not found: {p}", "danger"), True, False, True
        if not p.lower().endswith(".adicht"):
            return alert(f"Not an ADICHT file: {p}", "danger"), True, False, True

    # Validate output folder if specified
    if output_folder and not os.path.isdir(output_folder):
        return alert(f"Output folder not found: {output_folder}", "danger"), True, False, True

    # Build conversion list: (adicht_path, edf_path)
    conversions = []
    for p in paths:
        stem = Path(p).stem
        if output_folder:
            edf = os.path.join(output_folder, stem + ".edf")
        else:
            edf = str(Path(p).with_suffix(".edf"))

        if os.path.exists(edf):
            return alert(f"Output already exists: {edf}", "warning"), True, False, True
        conversions.append((p, edf))

    # Init progress
    _convert_progress[sid] = {
        "current": 0,
        "total": len(conversions),
        "current_file": "",
        "file_stage": "",
        "file_pct": 0.0,
        "done": False,
        "error": None,
        "cancel_requested": False,
        "cancelled": False,
        "results": [],
    }

    mode = "fast" if fast_write else "blocked"
    t = threading.Thread(
        target=_run_conversions,
        args=(sid, conversions, mode),
        daemon=True,
    )
    t.start()

    return (
        html.Div("Conversion started...", style={"color": "var(--ned-accent)"}),
        False,
        True,
        False,
    )


def _run_conversions(sid: str, conversions: list[tuple[str, str]], mode: str = "fast"):
    """Background: convert each ADICHT file to EDF."""
    from eeg_seizure_analyzer.io.adicht_to_edf import (
        ConversionCancelled,
        convert_adicht_to_edf,
    )

    progress = _convert_progress[sid]

    def _cb(stage: str, fraction: float) -> None:
        if progress.get("cancel_requested"):
            raise ConversionCancelled()
        progress["file_stage"] = stage
        progress["file_pct"] = fraction

    for i, (adicht_path, edf_path) in enumerate(conversions):
        if progress.get("cancel_requested"):
            progress["cancelled"] = True
            progress["done"] = True
            return

        progress["current"] = i
        progress["current_file"] = os.path.basename(adicht_path)
        progress["file_stage"] = "starting"
        progress["file_pct"] = 0.0

        try:
            convert_adicht_to_edf(adicht_path, edf_path, mode=mode, progress_cb=_cb)
            progress["results"].append({
                "file": os.path.basename(adicht_path),
                "output": edf_path,
                "status": "ok",
            })
        except ConversionCancelled:
            # Delete the partial output — it's almost certainly invalid.
            try:
                if os.path.exists(edf_path):
                    os.remove(edf_path)
            except OSError:
                pass
            progress["cancelled"] = True
            progress["done"] = True
            return
        except Exception as e:
            progress["error"] = f"{os.path.basename(adicht_path)}: {e}"
            progress["done"] = True
            return

    progress["current"] = len(conversions)
    progress["file_stage"] = "done"
    progress["file_pct"] = 1.0
    progress["done"] = True


# ── Progress polling ────────────────────────────────────────────────


@callback(
    Output("ac-progress", "children"),
    Output("ac-progress-interval", "disabled", allow_duplicate=True),
    Output("ac-convert-btn", "disabled", allow_duplicate=True),
    Output("ac-stop-btn", "disabled", allow_duplicate=True),
    Input("ac-progress-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_progress(n_intervals, sid):
    progress = _convert_progress.get(sid)
    if not progress:
        return no_update, no_update, no_update, no_update

    total = progress["total"]
    current = progress["current"]
    file_pct = float(progress.get("file_pct", 0.0))
    file_stage = progress.get("file_stage", "")

    # Overall pct counts completed files + fraction of current file.
    if progress["done"]:
        overall_frac = 1.0
    elif total:
        overall_frac = min(1.0, (current + file_pct) / total)
    else:
        overall_frac = 0.0
    overall_pct = int(overall_frac * 100)

    bar = dbc.Progress(
        value=overall_pct,
        label=f"{overall_pct}% ({current}/{total})",
        striped=True,
        animated=not progress["done"],
        className="mb-2",
    )

    # Per-file bar (only while a single file is being processed).
    file_bar = None
    if not progress["done"] and total:
        file_bar = dbc.Progress(
            value=int(file_pct * 100),
            label=f"{int(file_pct * 100)}%",
            striped=True,
            animated=True,
            color="info",
            className="mb-2",
            style={"height": "12px"},
        )

    if progress["error"]:
        _convert_progress.pop(sid, None)
        return html.Div([
            bar,
            alert(f"Error: {progress['error']}", "danger"),
        ]), True, False, True

    if progress.get("cancelled"):
        results = progress["results"]
        _convert_progress.pop(sid, None)
        completed_msg = (
            f"Stopped. {len(results)} file(s) finished before cancellation; "
            f"the in-progress file's partial output was deleted."
            if results
            else "Stopped before any file completed. Partial output deleted."
        )
        return html.Div([
            dbc.Progress(
                value=overall_pct,
                label=f"Cancelled at {overall_pct}%",
                color="warning",
                className="mb-2",
            ),
            alert(completed_msg, "warning"),
        ]), True, False, True

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
        ]), True, False, True

    if progress["current_file"]:
        stage_label = {
            "starting": "starting",
            "reading": "reading",
            "writing": "writing EDF",
            "done": "finishing",
        }.get(file_stage, file_stage or "...")
        status = f"[{current + 1}/{total}] {progress['current_file']} — {stage_label}"
    else:
        status = "Starting..."

    if progress.get("cancel_requested"):
        status = f"Stopping... ({status})"

    children = [bar]
    if file_bar is not None:
        children.append(file_bar)
    children.append(
        html.Div(status, style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)"})
    )

    # Stop button stays enabled until cancellation has actually taken effect.
    stop_disabled = bool(progress.get("cancel_requested"))
    return html.Div(children), False, True, stop_disabled


# ── Stop callback ───────────────────────────────────────────────────


@callback(
    Output("ac-stop-btn", "disabled", allow_duplicate=True),
    Input("ac-stop-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stop_convert(n_clicks, sid):
    """Signal the background thread to cancel at the next chunk boundary."""
    if not n_clicks:
        return no_update
    progress = _convert_progress.get(sid)
    if progress is None:
        return True
    progress["cancel_requested"] = True
    return True
