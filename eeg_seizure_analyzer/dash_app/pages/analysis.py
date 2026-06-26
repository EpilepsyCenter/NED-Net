"""Analysis tab — unified CNN seizure detection with Single/Batch/Live modes.

Replaces the old ML detection tab. All modes share one ``process_chunk()``
function from ``eeg_seizure_analyzer.analysis``.  Results are written to
SQLite via the ``db`` module.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert, metric_card, collapsible_section
from eeg_seizure_analyzer.ml.train import list_models, MODELS_DIR
from eeg_seizure_analyzer.ml.spike_train import list_spike_models
from eeg_seizure_analyzer import db, analysis
from eeg_seizure_analyzer.analysis import ClassificationParams


# ── Helpers ────────────────────────────────────────────────────────────


def _browse_folder(title: str = "Select folder") -> str | None:
    """Native folder picker."""
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 f'POSIX path of (choose folder with prompt "{title}")'],
                capture_output=True, text=True, timeout=120,
            )
            folder = r.stdout.strip().rstrip("/")
            return folder if folder else None
        except Exception:
            pass
    try:
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk()", "root.withdraw()",
                "root.attributes('-topmost', True)", "root.update()",
                f'folder = filedialog.askdirectory(title="{title}")',
                "root.destroy()", "print(folder or '')",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        folder = r.stdout.strip()
        return folder if folder else None
    except Exception:
        return None


def _browse_file(title: str = "Select EDF file",
                 file_types: tuple[str, ...] = ("edf",)) -> str | None:
    """Native file picker. ``file_types`` is the list of allowed extensions
    (without the dot); an empty tuple allows any file."""
    exts = [e.lstrip(".") for e in (file_types or ())]
    if platform.system() == "Darwin":
        try:
            of_type = (f' of type {{{", ".join(chr(34)+e+chr(34) for e in exts)}}}'
                       if exts else "")
            r = subprocess.run(
                ["osascript", "-e",
                 f'POSIX path of (choose file with prompt "{title}"{of_type})'],
                capture_output=True, text=True, timeout=120,
            )
            path = r.stdout.strip()
            return path if path else None
        except Exception:
            pass
    try:
        if exts:
            label = "/".join(e.upper() for e in exts) + " files"
            pattern = " ".join(f"*.{e}" for e in exts)
            filetypes = [(label, pattern), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk()", "root.withdraw()",
                "root.attributes('-topmost', True)", "root.update()",
                'path = filedialog.askopenfilename(',
                f'    title="{title}",',
                f'    filetypes={filetypes!r},',
                ')',
                "root.destroy()", "print(path or '')",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        path = r.stdout.strip()
        return path if path else None
    except Exception:
        return None


def _save_file(default_name: str, title: str = "Save file") -> str | None:
    """Native 'Save as' dialog (the in-window webview can't do browser
    downloads, so templates are written to a path the user picks). Returns the
    chosen path or None if cancelled."""
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 f'POSIX path of (choose file name with prompt "{title}" '
                 f'default name "{default_name}")'],
                capture_output=True, text=True, timeout=120,
            )
            return r.stdout.strip() or None
        except Exception:
            pass
    try:
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk(); root.withdraw()",
                "root.attributes('-topmost', True); root.update()",
                f'p = filedialog.asksaveasfilename(title="{title}", '
                f'initialfile="{default_name}", defaultextension=".csv")',
                "root.destroy(); print(p or '')",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _model_options(model_type: str = "seizure") -> list[dict]:
    """Build dropdown options for available models."""
    if model_type == "spike":
        models = list_spike_models()
    else:
        models = list_models()
    options = []
    for m in models:
        # Convulsive classifiers are Stage-2 cascade models, not seizure
        # detectors — exclude them from the detector list.
        if m.get("architecture") == "convulsive_classifier":
            continue
        f1 = m.get("best_event_f1", 0)
        ds = m.get("dataset", "")
        arch = m.get("architecture", "unet").upper()
        arch_tag = f" [{arch}]" if arch != "UNET" else ""
        label = (f"{m['name']}{arch_tag}  —  F1: {f1:.3f}  ({ds})"
                 if ds else f"{m['name']}{arch_tag}  —  F1: {f1:.3f}")
        options.append({"label": label, "value": m["name"]})
    return options


def _convulsive_model_options() -> list[dict]:
    """Dropdown options for Stage-2 convulsive classifiers (cascade)."""
    opts = [{"label": "(use detector ch1)", "value": ""}]
    for m in list_models():
        if m.get("architecture") != "convulsive_classifier":
            continue
        f1 = m.get("best_event_f1", 0)
        opts.append({"label": f"{m['name']}  —  F1: {f1:.3f}", "value": m["name"]})
    return opts


def _get_analysis_store(state) -> dict:
    """Read store-analysis from session state."""
    return state.extra.get("store_analysis", {})


def _set_analysis_store(state, data: dict):
    """Write store-analysis to session state."""
    state.extra["store_analysis"] = data


def _parse_boundary(val, threshold: float):
    """U-Net hysteresis boundary threshold: float below ``threshold``, else None.

    Blank, unparseable, or >= the detection threshold disables growing (None).
    """
    try:
        b = float(val)
    except (TypeError, ValueError):
        return None
    return b if b < threshold else None


# ── Layout ─────────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Build the full Analysis tab layout."""
    state = server_state.get_session(sid)
    store = _get_analysis_store(state)

    prev_det_type = store.get("detection_type", "seizure")
    models = _model_options(prev_det_type)
    conv_models = _convulsive_model_options()
    prev_conv_model = store.get("convulsive_model_name", "")
    prev_model = store.get("model_path")
    prev_threshold = store.get("confidence_threshold", 0.5)
    prev_conv_threshold = store.get("convulsive_threshold", 0.5)
    prev_bnd_threshold = store.get("boundary_threshold", 0.1)
    prev_min_dur = store.get("min_duration_sec", 5.0)
    prev_merge_gap = store.get("merge_gap_sec", 2.0)
    prev_hvsw_freq = store.get("hvsw_max_freq_hz", 4.0)
    prev_hvsw_swi = store.get("hvsw_min_slow_wave_index", 0.5)
    prev_hpd_freq = store.get("hpd_min_freq_hz", 15.0)
    prev_hpd_hfi = store.get("hpd_min_hf_index", 0.3)
    prev_mode = store.get("mode", "single")

    # Pre-populated file path (from currently loaded recording)
    loaded_path = ""
    if state.recording and hasattr(state.recording, "source_path"):
        loaded_path = state.recording.source_path or ""

    has_models = len(models) > 0

    return html.Div(
        style={"padding": "24px", "maxWidth": "1100px"},
        children=[
            html.H4("Analysis", style={"marginBottom": "8px"}),
            html.P(
                "Run CNN detection on single files, batches, or "
                "monitor a folder for new recordings in real time.",
                style={"color": "var(--ned-text-muted)", "fontSize": "0.9rem",
                       "marginBottom": "16px"},
            ),

            # ── Project database ──────────────────────────────────
            html.Div(
                style={"marginBottom": "16px", "padding": "12px",
                       "border": "1px solid #30363d", "borderRadius": "6px"},
                children=[
                    html.Label(
                        "Project database",
                        style={"fontSize": "0.82rem", "fontWeight": "600",
                               "color": "var(--ned-text-muted)"}),
                    html.P(
                        "Analysis writes to the active project; Results reads "
                        "from it. Switch projects to keep experiments separate, "
                        "or add cohorts to an existing one.",
                        style={"color": "var(--ned-text-muted)",
                               "fontSize": "0.76rem", "margin": "2px 0 8px"}),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id="an-project-select",
                            options=[{"label": p, "value": p}
                                     for p in db.list_projects()],
                            value=db.get_active_project(),
                            clearable=False,
                        ), width=4),
                        dbc.Col(dbc.Input(
                            id="an-project-new-name", type="text",
                            placeholder="New project name…", size="sm",
                        ), width=4),
                        dbc.Col(dbc.Button(
                            "Create project", id="an-project-create-btn",
                            className="btn-ned-secondary", size="sm",
                        ), width="auto"),
                        dbc.Col(dbc.Button(
                            "Delete project", id="an-project-delete-btn",
                            className="btn-ned-danger", size="sm",
                        ), width="auto"),
                    ], className="g-2", align="center"),
                    html.Div(id="an-project-status",
                             style={"fontSize": "0.78rem", "marginTop": "6px"}),
                    dcc.ConfirmDialog(id="an-project-delete-confirm"),
                ],
            ),

            # ── Detection type ────────────────────────────────────
            dbc.RadioItems(
                id="an-detection-type",
                options=[
                    {"label": " Seizures", "value": "seizure"},
                    {"label": " Interictal Spikes", "value": "spike"},
                ],
                value=prev_det_type,
                inline=True,
                className="mb-3",
                style={"fontSize": "0.95rem", "fontWeight": "600"},
            ),

            # ── Top controls (always visible) ──────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("Trained model",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dcc.Dropdown(
                        id="an-model",
                        options=models,
                        value=prev_model if prev_model and any(
                            o["value"] == prev_model for o in models) else None,
                        placeholder="Select a trained model..." if has_models else "No models trained yet",
                        clearable=True,
                        disabled=not has_models,
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Seizure threshold (ch0)",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dbc.Input(
                        id="an-threshold", type="text",
                        value=prev_threshold, min=0.05, max=1.0, step=0.01,
                        size="sm",
                        style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                               "border": "1px solid var(--ned-border)"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Convulsive threshold (ch1)",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dbc.Input(
                        id="an-conv-threshold", type="text",
                        value=prev_conv_threshold, min=0.05, max=1.0, step=0.01,
                        size="sm",
                        style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                               "border": "1px solid var(--ned-border)"},
                    ),
                ], width=3),
            ], className="g-3 mb-3"),

            # ── Convulsive classifier (Stage-2 cascade, optional) ──
            dbc.Row([
                dbc.Col([
                    html.Label("Convulsive classifier (optional)",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dcc.Dropdown(
                        id="an-conv-model",
                        options=conv_models,
                        value=prev_conv_model if any(
                            o["value"] == prev_conv_model for o in conv_models) else "",
                        clearable=False,
                    ),
                    html.Div(
                        "When set, overrides the detector's ch1 flag for the "
                        "convulsive decision.",
                        style={"fontSize": "0.72rem", "color": "var(--ned-text-muted)",
                               "marginTop": "2px"},
                    ),
                ], width=6),
            ], className="g-3 mb-3"),

            # ── Detection parameters ──────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("Boundary threshold (hysteresis)",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dbc.Input(
                        id="an-bnd-threshold", type="text",
                        value=prev_bnd_threshold, min=0.05, max=0.9, step=0.05,
                        size="sm",
                        style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                               "border": "1px solid var(--ned-border)"},
                    ),
                    html.Div(
                        "Grows event onset/offset out to this lower prob (U-Net "
                        "only). Below the seizure threshold; blank/≥ it = off.",
                        style={"fontSize": "0.72rem", "color": "var(--ned-text-muted)",
                               "marginTop": "2px"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Min event duration (s)",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dbc.Input(
                        id="an-min-duration", type="text",
                        value=prev_min_dur, min=0, step=0.5,
                        size="sm",
                        style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                               "border": "1px solid var(--ned-border)"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Merge gap (s)",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dbc.Input(
                        id="an-merge-gap", type="text",
                        value=prev_merge_gap, min=0, step=0.5,
                        size="sm",
                        style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                               "border": "1px solid var(--ned-border)"},
                    ),
                ], width=4),
            ], className="g-3 mb-3"),

            dbc.Checklist(
                id="an-overwrite",
                options=[{
                    "label": " Re-analyze files already in the database "
                             "(overwrite previous results for this detector)",
                    "value": "overwrite",
                }],
                value=[],
                switch=True,
                style={"fontSize": "0.82rem", "marginBottom": "12px"},
            ),

            # ── HVSW / HPD classification thresholds ──────────────
            collapsible_section(
                "HVSW / HPD classification", "an-cls",
                default_open=True,
                children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("HVSW — max dominant freq (Hz)",
                                       className="param-label",
                                       style={"fontSize": "0.8rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(
                                id="an-hvsw-freq", type="text",
                                value=prev_hvsw_freq, min=1, max=10, step=0.5,
                                size="sm",
                                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                                       "border": "1px solid var(--ned-border)"},
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("HVSW — min slow-wave index",
                                       className="param-label",
                                       style={"fontSize": "0.8rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(
                                id="an-hvsw-swi", type="text",
                                value=prev_hvsw_swi, min=0, max=1, step=0.05,
                                size="sm",
                                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                                       "border": "1px solid var(--ned-border)"},
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("HPD — min dominant freq (Hz)",
                                       className="param-label",
                                       style={"fontSize": "0.8rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(
                                id="an-hpd-freq", type="text",
                                value=prev_hpd_freq, min=5, max=50, step=1,
                                size="sm",
                                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                                       "border": "1px solid var(--ned-border)"},
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("HPD — min HF index",
                                       className="param-label",
                                       style={"fontSize": "0.8rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(
                                id="an-hpd-hfi", type="text",
                                value=prev_hpd_hfi, min=0, max=1, step=0.05,
                                size="sm",
                                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                                       "border": "1px solid var(--ned-border)"},
                            ),
                        ], width=3),
                    ], className="g-3"),
                    html.Div(
                        "HVSW: slow, high-amplitude waves (< max freq, high slow-wave index). "
                        "HPD: fast periodic discharges (> min freq, high HF index).",
                        style={"fontSize": "0.75rem", "color": "var(--ned-text-muted)",
                               "marginTop": "8px"},
                    ),
                ],
            ),

            # Model info card
            html.Div(id="an-model-info", className="mb-3"),

            html.Hr(style={"borderColor": "#21262d"}),

            # ── Mode selector ──────────────────────────────────────
            html.Label("Mode", style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)",
                                       "marginBottom": "4px"}),
            dbc.RadioItems(
                id="an-mode",
                options=[
                    {"label": " Single file", "value": "single"},
                    {"label": " Batch", "value": "batch"},
                    {"label": " Live monitoring", "value": "live"},
                ],
                value=prev_mode,
                inline=True,
                className="mb-3",
                style={"fontSize": "0.9rem"},
            ),

            # ── Mode panels (toggled by mode selector) ─────────────
            html.Div(id="an-single-panel", children=_single_panel(loaded_path, store)),
            html.Div(id="an-batch-panel", children=_batch_panel(store),
                     style={"display": "none"}),
            html.Div(id="an-live-panel", children=_live_panel(store),
                     style={"display": "none"}),

            # ── Results summary (below modes) ──────────────────────
            html.Hr(style={"borderColor": "#21262d", "marginTop": "24px"}),
            html.Div(id="an-results-summary", children=_results_summary()),

            # ── Stores & intervals ─────────────────────────────────
            dcc.Store(id="an-store", storage_type="session"),
            dcc.Interval(id="an-poll", interval=2000, disabled=True),
        ],
    )


# ── Single file panel ──────────────────────────────────────────────────


def _single_panel(loaded_path: str, store: dict) -> list:
    prev_path = store.get("single_file_path", loaded_path)
    return [
        html.Label("EDF file",
                   style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
        dbc.InputGroup([
            dbc.Input(
                id="an-single-path",
                value=prev_path,
                placeholder="Path to EDF file...",
                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                       "border": "1px solid var(--ned-border)"},
            ),
            dbc.Button("Browse", id="an-single-browse",
                       outline=True, color="secondary", size="sm"),
        ], className="mb-3"),

        # Already processed warning
        html.Div(id="an-single-warning"),

        dbc.Button(
            "Run analysis",
            id="an-single-run",
            style={"backgroundColor": "#238636", "border": "1px solid #2ea043",
                   "color": "#fff", "fontWeight": "600"},
            className="mb-3",
        ),

        # Progress area
        html.Div(id="an-single-progress"),
    ]


# ── Batch panel ────────────────────────────────────────────────────────


def _batch_panel(store: dict) -> list:
    prev_folder = store.get("batch_folder", "")
    prev_sub = store.get("batch_include_sub", True)
    prev_meta = store.get("batch_metadata_path", "")
    return [
        html.Label("Folder",
                   style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
        dbc.InputGroup([
            dbc.Input(
                id="an-batch-folder",
                value=prev_folder,
                placeholder="Path to folder with EDF files...",
                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                       "border": "1px solid var(--ned-border)"},
            ),
            dbc.Button("Browse", id="an-batch-browse",
                       outline=True, color="secondary", size="sm"),
        ], className="mb-2"),

        dbc.Checkbox(
            id="an-batch-sub", label="Include subfolders",
            value=prev_sub,
            style={"fontSize": "0.85rem"},
            className="mb-2",
        ),

        # Batch metadata Excel
        html.Label("Batch metadata (optional)",
                   style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)",
                          "marginTop": "4px"}),
        dbc.InputGroup([
            dbc.Input(
                id="an-batch-meta-path",
                value=prev_meta,
                placeholder="batch_metadata.csv / .xlsx — cohort, group, animal IDs",
                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                       "border": "1px solid var(--ned-border)"},
                size="sm",
            ),
            dbc.Button("Browse", id="an-batch-meta-browse",
                       outline=True, color="secondary", size="sm"),
            dbc.Button("Generate template", id="an-batch-meta-gen",
                       outline=True, color="info", size="sm"),
            dbc.Button("Save empty template", id="an-batch-meta-dl",
                       outline=True, color="secondary", size="sm"),
        ], className="mb-1"),
        html.Div(id="an-batch-meta-status",
                 style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
                        "marginBottom": "12px"}),

        dbc.Button("Scan folder", id="an-batch-scan",
                   outline=True, color="info", size="sm", className="me-2"),
        html.Div(id="an-batch-scan-result", className="mb-3",
                 style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)"}),

        dbc.ButtonGroup([
            dbc.Button("Run batch analysis", id="an-batch-run",
                       style={"backgroundColor": "#238636",
                              "border": "1px solid #2ea043",
                              "color": "#fff", "fontWeight": "600"}),
            dbc.Button("Pause", id="an-batch-pause",
                       outline=True, color="warning", disabled=True),
            dbc.Button("Cancel", id="an-batch-cancel",
                       outline=True, color="danger", disabled=True),
        ], className="mb-3"),

        html.Div(id="an-batch-progress"),
    ]


# ── Live panel ─────────────────────────────────────────────────────────


def _live_panel(store: dict) -> list:
    prev_folder = store.get("live_folder", "")
    prev_wait = store.get("live_wait_sec", 30)
    prev_backlog = store.get("live_process_backlog", True)
    return [
        html.Label("Watch folder",
                   style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
        dbc.InputGroup([
            dbc.Input(
                id="an-live-folder",
                value=prev_folder,
                placeholder="Path to shared folder...",
                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                       "border": "1px solid var(--ned-border)"},
            ),
            dbc.Button("Browse", id="an-live-browse",
                       outline=True, color="secondary", size="sm"),
        ], className="mb-2"),

        dbc.Checkbox(
            id="an-live-backlog", label="Process backlog on startup",
            value=prev_backlog,
            style={"fontSize": "0.85rem"},
            className="mb-2",
        ),

        dbc.Row([
            dbc.Col([
                html.Label("Wait before processing (seconds)",
                           style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                dbc.Input(
                    id="an-live-wait", type="text",
                    value=prev_wait, min=5, max=300, step=5,
                    style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                           "border": "1px solid var(--ned-border)"},
                    size="sm",
                ),
                html.Small(
                    "Wait for LabChart to finish writing before reading",
                    style={"color": "var(--ned-text-muted)", "fontSize": "0.72rem"},
                ),
            ], width=4),
        ], className="mb-3"),

        # Per-channel metadata template, applied to every incoming file
        # (the live montage is fixed for a session, so metadata is by channel).
        html.Label("Channel metadata template (optional)",
                   style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
        dbc.InputGroup([
            dbc.Input(
                id="an-live-template",
                value=store.get("live_template_path", ""),
                placeholder="Path to filled live_template.csv...",
                style={"backgroundColor": "var(--ned-bg)", "color": "var(--ned-text)",
                       "border": "1px solid var(--ned-border)"},
                size="sm",
            ),
            dbc.Button("Browse", id="an-live-template-browse",
                       outline=True, color="secondary", size="sm"),
            dbc.Button("Save empty template", id="an-live-template-dl",
                       outline=True, color="secondary", size="sm"),
        ], className="mb-1"),
        html.Small(
            "Per-channel Animal ID / Cohort / Group applied to every incoming "
            "file.",
            style={"color": "var(--ned-text-muted)", "fontSize": "0.72rem"},
        ),
        html.Div(id="an-live-template-status",
                 style={"fontSize": "0.78rem", "marginTop": "4px",
                        "marginBottom": "12px"}),

        dbc.Button(
            "Start monitoring", id="an-live-start",
            style={"backgroundColor": "#238636", "border": "1px solid #2ea043",
                   "color": "#fff", "fontWeight": "600"},
            className="me-2",
        ),
        dbc.Button(
            "Stop monitoring", id="an-live-stop",
            outline=True, color="danger",
            style={"display": "none"},
        ),

        html.Div(id="an-live-status", className="mt-3"),
    ]


# ── Results summary panel ──────────────────────────────────────────────


def _results_summary() -> list:
    """Build results summary from SQLite."""
    try:
        summary = db.get_summary()
    except Exception:
        return [html.P("No analysis run yet.",
                       style={"color": "var(--ned-text-muted)", "fontSize": "0.85rem"})]

    if summary["n_files"] == 0:
        return [html.P("No analysis run yet.",
                       style={"color": "var(--ned-text-muted)", "fontSize": "0.85rem"})]

    n_total = summary["total_events"]
    n_conv = summary["n_convulsive"]
    n_nonconv = summary["n_nonconvulsive"]
    n_hvsw = summary["n_hvsw"]
    n_hpd = summary["n_hpd"]
    n_flagged = summary["n_flagged"]

    pct_conv = f"({round(100 * n_conv / n_total)}%)" if n_total else ""
    pct_nonconv = f"({round(100 * n_nonconv / n_total)}%)" if n_total else ""

    return [
        html.H6("Results — last analysis run",
                style={"color": "var(--ned-accent)", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(metric_card("Files", str(summary["n_files"])), width=2),
            dbc.Col(metric_card("Animals", str(summary["n_animals"])), width=2),
            dbc.Col(metric_card("Total events", str(n_total), accent=True), width=2),
            dbc.Col(metric_card("Convulsive", f"{n_conv} {pct_conv}"), width=2),
            dbc.Col(metric_card("Non-conv", f"{n_nonconv} {pct_nonconv}"), width=2),
            dbc.Col(metric_card("Flagged", str(n_flagged)), width=2),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(metric_card("HVSW", str(n_hvsw)), width=2),
            dbc.Col(metric_card("HPD", str(n_hpd)), width=2),
        ], className="g-2 mb-3"),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════


# ── Mode selector: toggle panel visibility ─────────────────────────────


@callback(
    Output("an-project-select", "options"),
    Output("an-project-select", "value"),
    Output("an-project-status", "children"),
    Output("an-results-summary", "children", allow_duplicate=True),
    Input("an-project-select", "value"),
    Input("an-project-create-btn", "n_clicks"),
    State("an-project-new-name", "value"),
    prevent_initial_call=True,
)
def an_switch_or_create_project(selected, create_clicks, new_name):
    """Switch the active project DB, or create a new one, then refresh the
    summary. The active project is app-wide (shared with the Results tab)."""
    if ctx.triggered_id == "an-project-create-btn":
        try:
            name = db.create_project(new_name)
        except ValueError as e:
            return (no_update, no_update,
                    html.Span(str(e), style={"color": "#d29922"}), no_update)
        status = html.Span(f"Created and switched to '{name}'.",
                           style={"color": "var(--ned-success)"})
    else:
        name = db.set_active_project(selected)
        status = html.Span(f"Active project: '{name}'.",
                           style={"color": "var(--ned-text-muted)"})

    options = [{"label": p, "value": p} for p in db.list_projects()]
    return options, db.get_active_project(), status, _results_summary()


@callback(
    Output("an-project-delete-confirm", "displayed"),
    Output("an-project-delete-confirm", "message"),
    Output("an-project-status", "children", allow_duplicate=True),
    Input("an-project-delete-btn", "n_clicks"),
    State("an-project-select", "value"),
    prevent_initial_call=True,
)
def ask_delete_project(n_clicks, selected):
    """Pop a confirm dialog before deleting a project DB."""
    if not n_clicks:
        return no_update, no_update, no_update
    if not selected or selected == "Default":
        return (False, "",
                html.Span("The Default project can't be deleted.",
                          style={"color": "#d29922"}))
    return (True,
            f"Delete project '{selected}' and all its analysis results? "
            f"This cannot be undone.",
            no_update)


@callback(
    Output("an-project-select", "options", allow_duplicate=True),
    Output("an-project-select", "value", allow_duplicate=True),
    Output("an-project-status", "children", allow_duplicate=True),
    Output("an-results-summary", "children", allow_duplicate=True),
    Input("an-project-delete-confirm", "submit_n_clicks"),
    State("an-project-select", "value"),
    prevent_initial_call=True,
)
def do_delete_project(submit_n, selected):
    """Delete the project after the user confirms; fall back to Default."""
    if not submit_n or not selected:
        return no_update, no_update, no_update, no_update
    try:
        db.delete_project(selected)
    except ValueError as e:
        return (no_update, no_update,
                html.Span(str(e), style={"color": "#d29922"}), no_update)
    options = [{"label": p, "value": p} for p in db.list_projects()]
    status = html.Span(f"Deleted '{selected}'. Active project is now Default.",
                       style={"color": "var(--ned-success)"})
    return options, db.get_active_project(), status, _results_summary()


@callback(
    Output("an-single-panel", "style"),
    Output("an-batch-panel", "style"),
    Output("an-live-panel", "style"),
    Input("an-mode", "value"),
)
def toggle_mode_panels(mode):
    show = {"display": "block"}
    hide = {"display": "none"}
    if mode == "single":
        return show, hide, hide
    elif mode == "batch":
        return hide, show, hide
    else:
        return hide, hide, show


# ── Detection type: update model dropdown ─────────────────────────────


@callback(
    Output("an-model", "options"),
    Output("an-model", "value"),
    Output("an-model", "placeholder"),
    Input("an-detection-type", "value"),
)
def update_model_list(det_type):
    models = _model_options(det_type)
    has = len(models) > 0
    label = "spike" if det_type == "spike" else "seizure"
    placeholder = (
        f"Select a trained {label} model..."
        if has else f"No {label} models trained yet"
    )
    return models, None, placeholder


@callback(
    Output("an-conv-model", "options"),
    Output("an-conv-model", "disabled"),
    Input("an-detection-type", "value"),
)
def update_conv_model_list(det_type):
    """Populate the Stage-2 classifier dropdown (seizure detection only)."""
    if det_type == "spike":
        return [{"label": "(use detector ch1)", "value": ""}], True
    return _convulsive_model_options(), False


# ── Model info card ────────────────────────────────────────────────────


@callback(
    Output("an-model-info", "children"),
    Input("an-model", "value"),
    prevent_initial_call=True,
)
def show_model_info(model_name):
    if not model_name:
        return html.Div()

    meta_path = MODELS_DIR / model_name / "metadata.json"
    if not meta_path.exists():
        return html.Div()

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return html.Div()

    best = meta.get("best_metrics", {})
    dc = meta.get("dataset_config", {})

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(metric_card("Dataset", meta.get("dataset_name", "—")), width=2),
                dbc.Col(metric_card("Event F1",
                                    f"{best.get('best_event_f1', best.get('event_f1', 0)):.3f}",
                                    accent=True), width=2),
                dbc.Col(metric_card("Precision",
                                    f"{best.get('best_event_precision', best.get('event_precision', 0)):.3f}"), width=2),
                dbc.Col(metric_card("Recall",
                                    f"{best.get('best_event_recall', best.get('event_recall', 0)):.3f}"), width=2),
                dbc.Col(metric_card("Params",
                                    f"{meta.get('n_params', 0):,}"), width=2),
                dbc.Col(metric_card("Classes",
                                    str(meta.get("n_classes", 1))), width=2),
            ], className="g-2"),
            html.Div(
                "Metrics at the model's best threshold"
                + (f" ({best['best_threshold']})" if best.get('best_threshold') is not None else "")
                + (f"  —  Convulsive best F1: {best['conv_best_event_f1']:.3f}"
                   if best.get('conv_best_event_f1') is not None else "")
                + f"  —  Trained: {meta.get('created', '—')[:10]}"
                + f"  —  Window: {dc.get('window_sec', 60)}s @ {dc.get('target_fs', 250)}Hz",
                style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
                       "marginTop": "8px"},
            ),
        ]),
        style={"backgroundColor": "var(--ned-sidebar)", "border": "1px solid #21262d"},
    )


# ── Browse buttons ─────────────────────────────────────────────────────


@callback(
    Output("an-single-path", "value"),
    Input("an-single-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_single_file(n):
    path = _browse_file("Select EDF file for analysis")
    return path or no_update


@callback(
    Output("an-batch-folder", "value"),
    Input("an-batch-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_batch_folder(n):
    folder = _browse_folder("Select folder with EDF files")
    return folder or no_update


@callback(
    Output("an-batch-meta-path", "value"),
    Input("an-batch-meta-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_batch_meta(n):
    path = _browse_file("Select batch metadata (CSV or Excel)",
                        file_types=("csv", "xlsx"))
    return path or no_update


@callback(
    Output("an-batch-meta-status", "children"),
    Input("an-batch-meta-gen", "n_clicks"),
    State("an-batch-folder", "value"),
    State("an-batch-sub", "value"),
    prevent_initial_call=True,
)
def generate_batch_template(n, folder, include_sub):
    if not n or not folder or not os.path.isdir(folder):
        return "Select a folder first."
    try:
        from eeg_seizure_analyzer.io.batch_metadata import generate_template
        scan = analysis.scan_folder(folder, include_sub)
        out = generate_template(folder, scan["files"])
        return html.Span(
            f"Template saved: {out}",
            style={"color": "#2ea043"},
        )
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "var(--ned-danger)"})


@callback(
    Output("an-live-folder", "value"),
    Input("an-live-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_live_folder(n):
    folder = _browse_folder("Select watch folder")
    return folder or no_update


@callback(
    Output("an-live-template", "value"),
    Input("an-live-template-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_live_template(n):
    path = _browse_file("Select live channel template (CSV or Excel)",
                        file_types=("csv", "xlsx"))
    return path or no_update


# Write the template to a path the user picks (the in-window webview can't do
# browser downloads). CSV needs no extra deps; opens in Excel fine.
@callback(
    Output("an-batch-meta-status", "children", allow_duplicate=True),
    Input("an-batch-meta-dl", "n_clicks"),
    prevent_initial_call=True,
)
def download_batch_template(n_clicks):
    """Save an empty batch-metadata template (.csv) to a chosen path."""
    if not n_clicks:
        return no_update
    path = _save_file("batch_metadata.csv", "Save batch metadata template")
    if not path:
        return no_update
    from eeg_seizure_analyzer.io.batch_metadata import empty_batch_template
    try:
        empty_batch_template().to_csv(path, index=False)
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "var(--ned-danger)"})
    return html.Span(f"Template saved: {path}", style={"color": "#2ea043"})


@callback(
    Output("an-live-template-status", "children"),
    Input("an-live-template-dl", "n_clicks"),
    prevent_initial_call=True,
)
def download_live_template(n_clicks):
    """Save an empty live channel template (.csv) to a chosen path."""
    if not n_clicks:
        return no_update
    path = _save_file("live_template.csv", "Save live channel template")
    if not path:
        return no_update
    from eeg_seizure_analyzer.io.batch_metadata import empty_live_template
    try:
        empty_live_template().to_csv(path, index=False)
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "var(--ned-danger)"})
    return html.Span(f"Template saved: {path}", style={"color": "#2ea043"})


# ── Single file: check if already processed ────────────────────────────


@callback(
    Output("an-single-warning", "children"),
    Input("an-single-path", "value"),
    prevent_initial_call=True,
)
def check_single_processed(path):
    if not path:
        return html.Div()
    try:
        processed = db.get_processed_paths()
        if str(path) in processed:
            return alert(
                "This file has already been analysed. Run again to overwrite.",
                "warning",
            )
    except Exception:
        pass
    return html.Div()


# ── Single file: run analysis ──────────────────────────────────────────


@callback(
    Output("an-single-progress", "children"),
    Output("an-poll", "disabled", allow_duplicate=True),
    Output("an-single-run", "disabled"),
    Input("an-single-run", "n_clicks"),
    State("an-single-path", "value"),
    State("an-model", "value"),
    State("an-conv-model", "value"),
    State("an-threshold", "value"),
    State("an-conv-threshold", "value"),
    State("an-bnd-threshold", "value"),
    State("an-min-duration", "value"),
    State("an-merge-gap", "value"),
    State("an-hvsw-freq", "value"),
    State("an-hvsw-swi", "value"),
    State("an-hpd-freq", "value"),
    State("an-hpd-hfi", "value"),
    State("an-detection-type", "value"),
    State("an-overwrite", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_single(n_clicks, edf_path, model_name, conv_model_name, threshold,
               conv_threshold, bnd_threshold, min_dur, merge_gap, hvsw_freq,
               hvsw_swi, hpd_freq, hpd_hfi, det_type, overwrite_val, sid):
    if not n_clicks:
        return no_update, no_update, no_update

    if not model_name:
        return alert("Select a model first.", "warning"), True, False
    if not edf_path or not os.path.isfile(edf_path):
        return alert("Select a valid EDF file.", "warning"), True, False

    threshold = float(threshold or 0.5)
    conv_threshold = float(conv_threshold or 0.5)
    bnd = _parse_boundary(bnd_threshold, threshold)
    conv_model_name = conv_model_name or None
    overwrite = bool(overwrite_val and "overwrite" in overwrite_val)
    is_spike = det_type == "spike"

    cls_params = ClassificationParams(
        hvsw_max_freq_hz=float(hvsw_freq or 4.0),
        hvsw_min_slow_wave_index=float(hvsw_swi or 0.5),
        hpd_min_freq_hz=float(hpd_freq or 15.0),
        hpd_min_hf_index=float(hpd_hfi or 0.3),
    )

    # Persist settings
    state = server_state.get_session(sid)
    store = _get_analysis_store(state)
    store.update({
        "mode": "single",
        "detection_type": det_type,
        "model_path": model_name,
        "convulsive_model_name": conv_model_name or "",
        "confidence_threshold": threshold,
        "convulsive_threshold": conv_threshold,
        "boundary_threshold": bnd_threshold,
        "single_file_path": edf_path,
        "min_duration_sec": min_dur,
        "merge_gap_sec": merge_gap,
        "hvsw_max_freq_hz": hvsw_freq,
        "hvsw_min_slow_wave_index": hvsw_swi,
        "hpd_min_freq_hz": hpd_freq,
        "hpd_min_hf_index": hpd_hfi,
    })
    _set_analysis_store(state, store)

    # Reset status and launch background thread
    analysis.reset_status()
    analysis._update_status(
        running=True, mode="single",
        total_files=1, processed_files=0,
        current_file=Path(edf_path).name,
    )

    def _worker():
        try:
            def _prog(current, total):
                analysis._update_status(
                    file_progress_current=current,
                    file_progress_total=total,
                )

            if is_spike:
                res = analysis.process_spike_chunk(
                    edf_path=edf_path,
                    model_name=model_name,
                    confidence_threshold=threshold,
                    min_duration_sec=float(min_dur or 0.002),
                    merge_gap_sec=float(merge_gap or 0.05),
                    mode="single",
                    progress_callback=_prog,
                    overwrite=overwrite,
                )
            else:
                res = analysis.process_chunk(
                    edf_path=edf_path,
                    model_name=model_name,
                    confidence_threshold=threshold,
                    boundary_threshold=bnd,
                    convulsive_threshold=float(conv_threshold or 0.5),
                    min_duration_sec=float(min_dur or 5.0),
                    merge_gap_sec=float(merge_gap or 2.0),
                    mode="single",
                    classification_params=cls_params,
                    progress_callback=_prog,
                    overwrite=overwrite,
                    convulsive_model_name=conv_model_name,
                )
            if res and res.get("skipped"):
                analysis._update_status(
                    running=False, processed_files=0,
                    last_error="This file is already in the database. "
                               "Tick 'Re-analyze files already in the database' "
                               "to overwrite.",
                )
            else:
                analysis._update_status(running=False, processed_files=1)
        except Exception as e:
            analysis._update_status(
                running=False,
                last_error=str(e),
            )

    threading.Thread(target=_worker, daemon=True).start()

    progress = html.Div([
        dbc.Progress(
            value=0, striped=True, animated=True,
            style={"height": "24px", "marginBottom": "8px"},
        ),
        html.Div("Starting analysis...",
                 style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                        "textAlign": "center"}),
    ])

    return progress, False, True  # enable polling, disable button


# ── Batch: scan folder ─────────────────────────────────────────────────


@callback(
    Output("an-batch-scan-result", "children"),
    Input("an-batch-scan", "n_clicks"),
    State("an-batch-folder", "value"),
    State("an-batch-sub", "value"),
    prevent_initial_call=True,
)
def scan_batch_folder(n, folder, include_sub):
    if not folder or not os.path.isdir(folder):
        return alert("Select a valid folder.", "warning")

    scan = analysis.scan_folder(folder, include_sub)
    return html.Div([
        html.Span(f"Found: {scan['total']} EDF files", style={"fontWeight": "600"}),
        html.Br(),
        html.Span(f"Already processed: {scan['already_processed']}"),
        html.Br(),
        html.Span(f"To process: {scan['to_process']}",
                  style={"color": "var(--ned-accent)"}),
    ])


# ── Batch: run / pause / cancel ───────────────────────────────────────


@callback(
    Output("an-batch-progress", "children"),
    Output("an-poll", "disabled", allow_duplicate=True),
    Output("an-batch-run", "disabled"),
    Output("an-batch-pause", "disabled"),
    Output("an-batch-cancel", "disabled"),
    Input("an-batch-run", "n_clicks"),
    State("an-batch-folder", "value"),
    State("an-batch-sub", "value"),
    State("an-model", "value"),
    State("an-conv-model", "value"),
    State("an-threshold", "value"),
    State("an-conv-threshold", "value"),
    State("an-bnd-threshold", "value"),
    State("an-min-duration", "value"),
    State("an-merge-gap", "value"),
    State("an-hvsw-freq", "value"),
    State("an-hvsw-swi", "value"),
    State("an-hpd-freq", "value"),
    State("an-hpd-hfi", "value"),
    State("an-detection-type", "value"),
    State("an-batch-meta-path", "value"),
    State("an-overwrite", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_batch(n, folder, include_sub, model_name, conv_model_name, threshold,
              conv_threshold, bnd_threshold, min_dur, merge_gap, hvsw_freq,
              hvsw_swi, hpd_freq, hpd_hfi, det_type, meta_path, overwrite_val, sid):
    if not n:
        return no_update, no_update, no_update, no_update, no_update
    if not model_name:
        return alert("Select a model first.", "warning"), True, False, True, True
    if not folder or not os.path.isdir(folder):
        return alert("Select a valid folder.", "warning"), True, False, True, True

    threshold = float(threshold or 0.5)
    conv_threshold = float(conv_threshold or 0.5)
    bnd = _parse_boundary(bnd_threshold, threshold)
    conv_model_name = conv_model_name or None
    is_spike = det_type == "spike"

    cls_params = ClassificationParams(
        hvsw_max_freq_hz=float(hvsw_freq or 4.0),
        hvsw_min_slow_wave_index=float(hvsw_swi or 0.5),
        hpd_min_freq_hz=float(hpd_freq or 15.0),
        hpd_min_hf_index=float(hpd_hfi or 0.3),
    )

    # Persist
    state = server_state.get_session(sid)
    store = _get_analysis_store(state)
    store.update({
        "mode": "batch",
        "detection_type": det_type,
        "model_path": model_name,
        "convulsive_model_name": conv_model_name or "",
        "confidence_threshold": threshold,
        "convulsive_threshold": conv_threshold,
        "boundary_threshold": bnd_threshold,
        "min_duration_sec": min_dur,
        "merge_gap_sec": merge_gap,
        "hvsw_max_freq_hz": hvsw_freq,
        "hvsw_min_slow_wave_index": hvsw_swi,
        "hpd_min_freq_hz": hpd_freq,
        "hpd_min_hf_index": hpd_hfi,
        "batch_folder": folder,
        "batch_include_sub": include_sub,
        "batch_metadata_path": meta_path or "",
    })
    _set_analysis_store(state, store)

    # Launch batch in background
    batch_kwargs = {
        "confidence_threshold": threshold,
        "convulsive_threshold": float(conv_threshold or 0.5),
        "overwrite": bool(overwrite_val and "overwrite" in overwrite_val),
        "min_duration_sec": float(min_dur or (0.002 if is_spike else 5.0)),
        "merge_gap_sec": float(merge_gap or (0.05 if is_spike else 2.0)),
        "include_subfolders": include_sub,
        "detection_type": det_type,
    }
    if not is_spike:
        batch_kwargs["classification_params"] = cls_params
        batch_kwargs["convulsive_model_name"] = conv_model_name
        batch_kwargs["boundary_threshold"] = bnd
    if meta_path and os.path.isfile(meta_path):
        batch_kwargs["metadata_path"] = meta_path

    threading.Thread(
        target=analysis.run_batch,
        args=(folder, model_name),
        kwargs=batch_kwargs,
        daemon=True,
    ).start()

    progress = html.Div([
        dbc.Progress(value=0, striped=True, animated=True,
                     style={"height": "24px", "marginBottom": "8px"}),
        html.Div("Starting batch...",
                 style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                        "textAlign": "center"}),
    ])
    return progress, False, True, False, False  # enable poll, disable run, enable pause/cancel


@callback(
    Output("an-batch-pause", "children"),
    Input("an-batch-pause", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_batch_pause(n):
    if not n:
        return no_update
    status = analysis.get_status()
    if status.get("paused"):
        analysis.request_resume()
        return "Pause"
    else:
        analysis.request_pause()
        return "Resume"


@callback(
    Output("an-batch-cancel", "children", allow_duplicate=True),
    Input("an-batch-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_batch(n):
    if n:
        analysis.request_cancel()
    return "Cancelling..."


# ── Live monitoring: start / stop ──────────────────────────────────────


@callback(
    Output("an-live-start", "style"),
    Output("an-live-stop", "style"),
    Output("an-poll", "disabled", allow_duplicate=True),
    Input("an-live-start", "n_clicks"),
    State("an-live-folder", "value"),
    State("an-live-backlog", "value"),
    State("an-live-wait", "value"),
    State("an-model", "value"),
    State("an-conv-model", "value"),
    State("an-threshold", "value"),
    State("an-conv-threshold", "value"),
    State("an-bnd-threshold", "value"),
    State("an-min-duration", "value"),
    State("an-merge-gap", "value"),
    State("an-hvsw-freq", "value"),
    State("an-hvsw-swi", "value"),
    State("an-hpd-freq", "value"),
    State("an-hpd-hfi", "value"),
    State("an-detection-type", "value"),
    State("an-live-template", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_live(n, folder, backlog, wait_sec, model_name, conv_model_name,
               threshold, conv_threshold, bnd_threshold, min_dur, merge_gap,
               hvsw_freq, hvsw_swi, hpd_freq, hpd_hfi, det_type, template_path, sid):
    if not n:
        return no_update, no_update, no_update
    if not model_name or not folder:
        return no_update, no_update, no_update

    threshold = float(threshold or 0.5)
    conv_threshold = float(conv_threshold or 0.5)
    bnd = _parse_boundary(bnd_threshold, threshold)
    conv_model_name = conv_model_name or None
    cls_params = ClassificationParams(
        hvsw_max_freq_hz=float(hvsw_freq or 4.0),
        hvsw_min_slow_wave_index=float(hvsw_swi or 0.5),
        hpd_min_freq_hz=float(hpd_freq or 15.0),
        hpd_min_hf_index=float(hpd_hfi or 0.3),
    )

    # Persist
    state = server_state.get_session(sid)
    store = _get_analysis_store(state)
    store.update({
        "mode": "live",
        "model_path": model_name,
        "convulsive_model_name": conv_model_name or "",
        "confidence_threshold": threshold,
        "convulsive_threshold": conv_threshold,
        "boundary_threshold": bnd_threshold,
        "min_duration_sec": min_dur,
        "merge_gap_sec": merge_gap,
        "hvsw_max_freq_hz": hvsw_freq,
        "hvsw_min_slow_wave_index": hvsw_swi,
        "hpd_min_freq_hz": hpd_freq,
        "hpd_min_hf_index": hpd_hfi,
        "live_folder": folder,
        "live_wait_sec": wait_sec or 30,
        "live_process_backlog": bool(backlog),
        "live_template_path": template_path or "",
    })
    _set_analysis_store(state, store)

    # Init DB
    db.init_db()

    # Load the per-channel live template (applied to every incoming file).
    live_template = None
    if template_path:
        try:
            from eeg_seizure_analyzer.io.batch_metadata import load_live_template
            live_template = load_live_template(template_path)
        except Exception:
            live_template = None

    analysis.start_live_monitoring(
        watch_folder=folder,
        model_name=model_name,
        confidence_threshold=threshold,
        boundary_threshold=bnd,
        convulsive_threshold=float(conv_threshold or 0.5),
        min_duration_sec=float(min_dur or 5.0),
        merge_gap_sec=float(merge_gap or 2.0),
        wait_sec=int(wait_sec or 30),
        process_backlog=bool(backlog),
        classification_params=cls_params,
        live_template=live_template,
        convulsive_model_name=conv_model_name,
    )

    hide = {"display": "none"}
    show_stop = {
        "display": "inline-block",
        "backgroundColor": "#da3633",
        "border": "1px solid #f85149",
        "color": "#fff",
        "fontWeight": "600",
    }
    return hide, show_stop, False  # hide start, show stop, enable polling


@callback(
    Output("an-live-start", "style", allow_duplicate=True),
    Output("an-live-stop", "style", allow_duplicate=True),
    Input("an-live-stop", "n_clicks"),
    prevent_initial_call=True,
)
def stop_live(n):
    if n:
        analysis.stop_live_monitoring()
    show_start = {
        "backgroundColor": "#238636", "border": "1px solid #2ea043",
        "color": "#fff", "fontWeight": "600",
    }
    hide = {"display": "none"}
    return show_start, hide


# ── Poll progress (shared across all modes) ────────────────────────────


@callback(
    Output("an-single-progress", "children", allow_duplicate=True),
    Output("an-batch-progress", "children", allow_duplicate=True),
    Output("an-live-status", "children"),
    Output("an-results-summary", "children"),
    Output("an-poll", "disabled", allow_duplicate=True),
    Output("an-single-run", "disabled", allow_duplicate=True),
    Output("an-batch-run", "disabled", allow_duplicate=True),
    Output("an-batch-pause", "disabled", allow_duplicate=True),
    Output("an-batch-cancel", "disabled", allow_duplicate=True),
    Input("an-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_progress(n):
    status = analysis.get_status()
    mode = status.get("mode")
    running = status.get("running", False)

    single_progress = no_update
    batch_progress = no_update
    live_status = no_update
    results = no_update
    disable_poll = no_update
    single_run_disabled = no_update
    batch_run_disabled = no_update
    batch_pause_disabled = no_update
    batch_cancel_disabled = no_update

    if mode == "single":
        if running:
            current = status.get("file_progress_current", 0)
            total = status.get("file_progress_total", 1) or 1
            pct = int(100 * current / total)
            single_progress = html.Div([
                dbc.Progress(
                    value=pct, striped=True, animated=True,
                    label=f"{current}/{total}",
                    style={"height": "24px", "marginBottom": "8px"},
                ),
                html.Div(
                    f"Processing {status.get('current_file', '')}... ({pct}%)",
                    style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                           "textAlign": "center"},
                ),
            ])
        else:
            # Done or error
            err = status.get("last_error")
            if err:
                single_progress = html.Div([
                    dbc.Progress(value=100, color="danger",
                                style={"height": "24px", "marginBottom": "8px"}),
                    alert(f"Error: {err}", "danger"),
                ])
            elif status.get("processed_files", 0) > 0:
                single_progress = html.Div([
                    dbc.Progress(value=100, color="success", label="Complete",
                                style={"height": "24px", "marginBottom": "8px"}),
                ])
            disable_poll = True
            single_run_disabled = False
            results = _results_summary()

    elif mode == "batch":
        processed = status.get("processed_files", 0)
        total = status.get("total_files", 0) or 1
        pct = int(100 * processed / total) if total else 0
        mean_sec = status.get("mean_file_sec")
        remaining = ""
        if mean_sec and processed < total:
            est_min = round(mean_sec * (total - processed) / 60, 1)
            remaining = f"  ~{est_min} min remaining"

        if running:
            file_cur = status.get("file_progress_current", 0)
            file_total = status.get("file_progress_total", 1) or 1

            batch_progress = html.Div([
                dbc.Progress(
                    value=pct, striped=True, animated=True,
                    label=f"{processed}/{total}",
                    style={"height": "24px", "marginBottom": "4px"},
                ),
                html.Div(
                    f"Current: {status.get('current_file', '')}"
                    f"  ({file_cur}/{file_total} windows){remaining}",
                    style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                           "textAlign": "center"},
                ),
            ])
            batch_pause_disabled = False
            batch_cancel_disabled = False
        else:
            err = status.get("last_error")
            color = "success" if not err else "warning"
            batch_progress = html.Div([
                dbc.Progress(value=100, color=color,
                             label=f"Done — {processed}/{status.get('total_files', 0)}",
                             style={"height": "24px", "marginBottom": "8px"}),
                *([] if not err else [
                    html.Div(f"Last error: {err}",
                             style={"fontSize": "0.8rem", "color": "var(--ned-warning)"}),
                ]),
            ])
            disable_poll = True
            batch_run_disabled = False
            batch_pause_disabled = True
            batch_cancel_disabled = True
            results = _results_summary()

    elif mode == "live":
        import time as _time

        is_running = analysis.is_live_running()
        processed = status.get("processed_files", 0)
        start = status.get("start_time")
        elapsed = ""
        if start:
            mins = int((_time.time() - start) / 60)
            elapsed = f"{mins} min" if mins else "< 1 min"

        current = status.get("current_file", "")
        err = status.get("last_error")

        live_items = [
            html.Div([
                dbc.Badge("LIVE", color="success" if is_running else "secondary",
                          className="me-2"),
                html.Span(
                    f"Monitoring: {current}" if is_running else "Stopped",
                    style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)"},
                ),
            ], className="mb-2"),
            html.Div(f"Chunks processed: {processed}  |  Uptime: {elapsed}",
                     style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
        ]
        if err:
            live_items.append(
                html.Div(f"Last error: {err}",
                         style={"fontSize": "0.8rem", "color": "var(--ned-warning)",
                                "marginTop": "4px"}),
            )

        live_status = html.Div(live_items)

        if not is_running:
            disable_poll = True

        # Always refresh results for live mode
        results = _results_summary()

    return (single_progress, batch_progress, live_status, results,
            disable_poll, single_run_disabled, batch_run_disabled,
            batch_pause_disabled, batch_cancel_disabled)


# ── Collapsible section toggle ────────────────────────────────────────

@callback(
    Output("an-cls-collapse", "is_open"),
    Output("an-cls-chevron", "children"),
    Input("an-cls-header", "n_clicks"),
    State("an-cls-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_cls_section(n, is_open):
    return not is_open, "\u25BC" if not is_open else "\u25B6"
