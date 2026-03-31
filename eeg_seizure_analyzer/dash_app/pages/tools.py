"""Tools tab – Video Converter (WMV Records → single MP4)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert, metric_card


# ── Layout ───────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the video converter tab layout."""
    state = server_state.get_session(sid)

    # Auto-detect video folder from loaded recording
    auto_folder = ""
    auto_output = ""
    if state.recording and state.recording.source_path:
        src = state.recording.source_path
        base = os.path.splitext(src)[0]
        candidate = base + " movies"
        if os.path.isdir(candidate):
            auto_folder = candidate
            auto_output = base + ".mp4"

    return html.Div(
        style={"padding": "24px", "maxWidth": "900px"},
        children=[
            html.H4("Video Converter", style={"marginBottom": "8px"}),
            html.P(
                "Merge LabChart Record WMV files into a single MP4 for "
                "time-synced video playback alongside EEG.",
                style={"color": "#8b949e", "fontSize": "0.9rem",
                       "marginBottom": "24px"},
            ),

            # ffmpeg check
            html.Div(id="vc-ffmpeg-status",
                      children=_check_ffmpeg()),

            # Input folder
            html.Label("Video folder", style={"fontSize": "0.82rem",
                                               "color": "#8b949e"}),
            dbc.InputGroup([
                dbc.Input(
                    id="vc-folder-input",
                    placeholder="/path/to/Recording movies",
                    value=auto_folder,
                    type="text",
                ),
                dbc.Button("Browse", id="vc-browse-btn",
                           className="btn-ned-secondary"),
                dbc.Button("Scan", id="vc-scan-btn",
                           className="btn-ned-primary"),
            ], className="mb-3"),

            # Output path
            html.Label("Output MP4 path", style={"fontSize": "0.82rem",
                                                   "color": "#8b949e"}),
            dbc.Input(
                id="vc-output-input",
                placeholder="/path/to/Recording.mp4",
                value=auto_output,
                type="text",
                className="mb-3",
            ),

            # Scan results
            html.Div(id="vc-scan-results"),

            # Convert button + progress
            html.Div(
                id="vc-convert-area",
                style={"display": "none"},
                children=[
                    dbc.Button("Convert & Merge", id="vc-convert-btn",
                               className="btn-ned-primary",
                               style={"marginTop": "16px"}),
                ],
            ),

            # Progress
            dcc.Interval(id="vc-progress-interval", interval=1000,
                         disabled=True),
            html.Div(id="vc-progress", style={"marginTop": "16px"}),

            # Status
            html.Div(id="vc-status", style={"marginTop": "16px"}),
        ],
    )


def _check_ffmpeg():
    """Check if ffmpeg is available."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return html.Div(
            "ffmpeg found",
            style={"color": "#3fb950", "fontSize": "0.82rem",
                   "marginBottom": "12px"},
        )
    return alert(
        "ffmpeg not found on PATH. Install ffmpeg to use the video converter. "
        "Download from https://ffmpeg.org/download.html",
        "danger",
    )


# ── Browse folder callback ───────────────────────────────────────────


@callback(
    Output("vc-folder-input", "value"),
    Output("vc-output-input", "value", allow_duplicate=True),
    Input("vc-browse-btn", "n_clicks"),
    State("vc-folder-input", "value"),
    State("vc-output-input", "value"),
    prevent_initial_call=True,
)
def browse_folder(n_clicks, current_folder, current_output):
    """Open a native folder picker dialog via subprocess."""
    if not n_clicks:
        return no_update, no_update

    try:
        import sys

        # Run tkinter dialog in a separate process to avoid blocking Dash
        script = """
import tkinter as tk
from tkinter import filedialog
import sys
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
root.update()
folder = filedialog.askdirectory(title="Select video folder")
root.destroy()
print(folder or "")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        folder = result.stdout.strip()

        if not folder:
            return no_update, no_update

        # Auto-fill output path from folder name
        # e.g., "/path/to/Marco movies" → "/path/to/Marco.mp4"
        output = current_output
        folder_name = os.path.basename(folder)
        if folder_name.endswith(" movies"):
            base = folder_name[:-7]  # strip " movies"
            output = os.path.join(os.path.dirname(folder), base + ".mp4")

        return folder, output

    except Exception:
        return no_update, no_update


# ── Scan callback ────────────────────────────────────────────────────


@callback(
    Output("vc-scan-results", "children"),
    Output("vc-convert-area", "style"),
    Input("vc-scan-btn", "n_clicks"),
    State("vc-folder-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def scan_folder(n_clicks, folder, sid):
    """Scan a folder for Record*.wmv files."""
    if not n_clicks or not folder:
        return no_update, no_update

    if not os.path.isdir(folder):
        return alert(f"Folder not found: {folder}", "danger"), {"display": "none"}

    # Find Record files, sorted numerically
    wmv_files = []
    for f in os.listdir(folder):
        m = re.match(r"^Record(\d+)\.wmv$", f, re.IGNORECASE)
        if m:
            wmv_files.append((int(m.group(1)), f))

    wmv_files.sort(key=lambda x: x[0])

    if not wmv_files:
        return alert("No Record*.wmv files found in this folder.", "warning"), \
               {"display": "none"}

    # Get durations via ffprobe
    rows = []
    total_dur = 0
    has_ffprobe = shutil.which("ffprobe") is not None

    for num, fname in wmv_files:
        fpath = os.path.join(folder, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        dur_str = "—"
        if has_ffprobe:
            dur = _get_duration(fpath)
            if dur is not None:
                total_dur += dur
                h, rem = divmod(int(dur), 3600)
                m, s = divmod(rem, 60)
                dur_str = f"{h}:{m:02d}:{s:02d}"

        rows.append(html.Tr([
            html.Td(str(num), style={"padding": "4px 12px"}),
            html.Td(fname, style={"padding": "4px 12px"}),
            html.Td(f"{size_mb:.0f} MB", style={"padding": "4px 12px"}),
            html.Td(dur_str, style={"padding": "4px 12px"}),
        ]))

    # Save file list in state for convert step
    state = server_state.get_session(sid)
    state.extra["vc_files"] = [(num, os.path.join(folder, fname))
                                for num, fname in wmv_files]

    th, trem = divmod(int(total_dur), 3600)
    tm, ts = divmod(trem, 60)

    table = html.Div([
        dbc.Row([
            dbc.Col(metric_card("Records", str(len(wmv_files))), width=4),
            dbc.Col(metric_card("Total Duration",
                                f"{th}:{tm:02d}:{ts:02d}"), width=4),
            dbc.Col(metric_card("Total Size",
                                f"{sum(os.path.getsize(os.path.join(folder, f)) for _, f in wmv_files) / (1024**2):.0f} MB"),
                    width=4),
        ], className="g-3 mb-3"),
        html.Table(
            [html.Thead(html.Tr([
                html.Th("#", style={"padding": "4px 12px"}),
                html.Th("File", style={"padding": "4px 12px"}),
                html.Th("Size", style={"padding": "4px 12px"}),
                html.Th("Duration", style={"padding": "4px 12px"}),
            ]))] + [html.Tbody(rows)],
            style={"width": "100%", "fontSize": "0.85rem"},
        ),
    ])

    return table, {"display": "block"}


def _get_duration(path: str) -> float | None:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _has_videotoolbox() -> bool:
    """Check if ffmpeg supports h264_videotoolbox (macOS hardware encoder)."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_videotoolbox" in result.stdout
    except Exception:
        return False


# ── Convert callback ─────────────────────────────────────────────────

# Shared progress state (per session)
_convert_progress: dict[str, dict] = {}


@callback(
    Output("vc-status", "children"),
    Output("vc-progress-interval", "disabled"),
    Output("vc-convert-btn", "disabled"),
    Input("vc-convert-btn", "n_clicks"),
    State("vc-folder-input", "value"),
    State("vc-output-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_convert(n_clicks, folder, output_path, sid):
    """Start the ffmpeg conversion in a background thread."""
    if not n_clicks:
        return no_update, no_update, no_update

    if not output_path:
        return alert("Please specify an output MP4 path.", "warning"), True, False

    if not shutil.which("ffmpeg"):
        return alert("ffmpeg not found.", "danger"), True, False

    state = server_state.get_session(sid)
    files = state.extra.get("vc_files", [])
    if not files:
        return alert("No files to convert. Scan a folder first.", "warning"), \
               True, False

    if os.path.exists(output_path):
        return alert(f"Output file already exists: {output_path}", "warning"), \
               True, False

    # Init progress
    _convert_progress[sid] = {
        "current": 0,
        "total": len(files),
        "status": "Starting...",
        "done": False,
        "error": None,
    }

    # Run in background thread
    t = threading.Thread(
        target=_run_conversion,
        args=(sid, files, output_path),
        daemon=True,
    )
    t.start()

    return html.Div("Conversion started...",
                     style={"color": "#58a6ff"}), False, True


def _run_conversion(sid: str, files: list[tuple[int, str]], output_path: str):
    """Background: convert each WMV to temp MP4, then concatenate."""
    import tempfile

    progress = _convert_progress[sid]
    temp_dir = tempfile.mkdtemp(prefix="nednet_vc_")
    temp_mp4s = []

    try:
        # Detect hardware encoder availability (macOS VideoToolbox)
        hw_ok = _has_videotoolbox()

        # Step 1: Convert each WMV to MP4
        for i, (num, wmv_path) in enumerate(files):
            progress["current"] = i
            enc_label = "HW" if hw_ok else "SW"
            progress["status"] = (
                f"Converting Record{num}.wmv ({i+1}/{len(files)}) [{enc_label}]"
            )

            temp_mp4 = os.path.join(temp_dir, f"record_{num:04d}.mp4")
            temp_mp4s.append(temp_mp4)

            if hw_ok:
                # Apple VideoToolbox hardware encoding — much faster
                cmd = [
                    "ffmpeg", "-y",
                    "-i", wmv_path,
                    "-c:v", "h264_videotoolbox",
                    "-q:v", "65",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    temp_mp4,
                ]
            else:
                # Software fallback
                cmd = [
                    "ffmpeg", "-y",
                    "-i", wmv_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    temp_mp4,
                ]

            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=7200,
            )
            if result.returncode != 0:
                progress["error"] = f"Failed on Record{num}: {result.stderr[-500:]}"
                progress["done"] = True
                return

        # Step 2: Concatenate all MP4s
        progress["status"] = "Merging all records into single MP4..."
        progress["current"] = len(files)

        # Create concat list file
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for mp4 in temp_mp4s:
                f.write(f"file '{mp4}'\n")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            progress["error"] = f"Merge failed: {result.stderr[-500:]}"
            progress["done"] = True
            return

        progress["status"] = "Done!"
        progress["done"] = True

    except Exception as e:
        progress["error"] = str(e)
        progress["done"] = True
    finally:
        # Clean up temp files
        import shutil as _shutil
        _shutil.rmtree(temp_dir, ignore_errors=True)


# ── Progress polling callback ────────────────────────────────────────


@callback(
    Output("vc-progress", "children"),
    Output("vc-progress-interval", "disabled", allow_duplicate=True),
    Output("vc-convert-btn", "disabled", allow_duplicate=True),
    Input("vc-progress-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_progress(n_intervals, sid):
    """Poll conversion progress."""
    progress = _convert_progress.get(sid)
    if not progress:
        return no_update, no_update, no_update

    total = progress["total"]
    current = progress["current"]
    status = progress["status"]
    pct = int((current / (total + 1)) * 100) if total else 0

    bar = dbc.Progress(
        value=pct,
        label=f"{pct}%",
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
        return html.Div([
            dbc.Progress(value=100, label="100%", color="success",
                         className="mb-2"),
            alert("Conversion complete! MP4 file ready for playback.", "success"),
        ]), True, False

    return html.Div([
        bar,
        html.Div(status, style={"fontSize": "0.85rem", "color": "#8b949e"}),
    ]), False, True
