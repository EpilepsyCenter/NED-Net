"""Results tab — reads from SQLite, shows summary stats, daily burden,
circadian analysis, event table with filters, and event inspector.

All data queries go through the ``db`` module — no raw SQL here.
Clicking an event row opens an inspector with EEG trace, PSD,
spectrogram (power over time), and all measured/computed parameters.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch, spectrogram as scipy_spectrogram

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    alert,
    get_plotly_theme,
    metric_card,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter
from eeg_seizure_analyzer import db

import json as _json

# ── Per-project Results filter persistence ──────────────────────────────
# Remember each database's Results-tab filter selections (cohort/group/animal/
# file/date/type/mode/min-conf/category/detector/normalise/panels) so reopening
# the same project restores them. Keyed by project name. (Per-animal exclusions
# already persist in the DB's animal_status table.)
_RESULTS_STATE_PATH = Path.home() / ".eeg_seizure_analyzer" / "results_state.json"


def _load_results_state(project: str) -> dict:
    if not project:
        return {}
    try:
        return _json.loads(_RESULTS_STATE_PATH.read_text()).get(project, {})
    except Exception:
        return {}


def _save_results_state(project: str, state: dict) -> None:
    if not project:
        return
    try:
        _RESULTS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            all_state = _json.loads(_RESULTS_STATE_PATH.read_text())
        except Exception:
            all_state = {}
        all_state[project] = state
        _RESULTS_STATE_PATH.write_text(_json.dumps(all_state, indent=2))
    except Exception:
        pass


def _save_file(default_name: str, title: str = "Save file") -> str | None:
    """Native 'Save as' dialog (the in-window webview can't do browser
    downloads). Returns the chosen path, or None if cancelled."""
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


# ── Layout ─────────────────────────────────────────────────────────────


# Friendly labels for the `source` (specific detector) column / filter.
_DETECTOR_LABELS = {
    "seizure_unet": "U-Net",
    "seizure_bendr": "BENDR",
    "seizure_cnn": "CNN (legacy)",
    "spike_unet": "U-Net (spike)",
    "spike_bendr": "BENDR (spike)",
    "spike_cnn": "CNN spike (legacy)",
    "spike_train": "Spike-Train",
    "spectral_band": "Spectral Band",
    "autocorrelation": "Autocorrelation",
    "ensemble": "Ensemble",
}


def layout(sid: str | None) -> html.Div:
    """Build Results tab with filter controls and data panels."""
    try:
        animals = db.get_all_animals()
        date_min, date_max = db.get_date_range()
        files = db.get_all_files()
    except Exception:
        animals = []
        date_min = date_max = ""
        files = []

    file_options = [
        {"label": Path(f["path"]).name, "value": str(f["id"])}
        for f in files
    ]

    return html.Div(
        style={"padding": "24px"},
        children=[
            html.H4("Results", style={"marginBottom": "8px"}),
            html.P(
                "Analysis results from all modes (single, batch, live). "
                "Use filters to scope what is shown. Click an event to inspect.",
                style={"color": "var(--ned-text-muted)", "fontSize": "0.9rem",
                       "marginBottom": "16px"},
            ),

            # ── Project database (read-only selector) ─────────────
            html.Div(
                style={"marginBottom": "16px", "padding": "12px",
                       "border": "1px solid #30363d", "borderRadius": "6px"},
                children=[
                    html.Label(
                        "Project database",
                        style={"fontSize": "0.82rem", "fontWeight": "600",
                               "color": "var(--ned-text-muted)"}),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id="res-project-select",
                            options=[{"label": p, "value": p}
                                     for p in db.list_projects()],
                            value=db.get_active_project(),
                            clearable=False,
                        ), width=4),
                    ], className="g-2", align="center"),
                ],
            ),

            # ── Event category selector ───────────────────────────
            dbc.RadioItems(
                id="res-source-selector",
                options=[
                    {"label": " Seizures", "value": "seizure_cnn"},
                    {"label": " Interictal Spikes", "value": "spike_cnn"},
                ],
                value="seizure_cnn",
                inline=True,
                className="mb-3",
                style={"fontSize": "0.95rem", "fontWeight": "600"},
            ),

            # ── Filter controls ────────────────────────────────────
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Source file",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-file-filter",
                                options=file_options,
                                multi=True,
                                placeholder="All files",
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Date range",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Row([
                                dbc.Col(dbc.Input(
                                    id="res-date-start", type="text",
                                    placeholder="Start",
                                    value=date_min or "", size="sm",
                                    style={"backgroundColor": "var(--ned-bg)",
                                           "color": "var(--ned-text)",
                                           "border": "1px solid var(--ned-border)"},
                                ), width=6),
                                dbc.Col(dbc.Input(
                                    id="res-date-end", type="text",
                                    placeholder="End",
                                    value=date_max or "", size="sm",
                                    style={"backgroundColor": "var(--ned-bg)",
                                           "color": "var(--ned-text)",
                                           "border": "1px solid var(--ned-border)"},
                                ), width=6),
                            ], className="g-1"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Mode",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Checklist(
                                id="res-mode-filter",
                                options=[
                                    {"label": "Single", "value": "single"},
                                    {"label": "Batch", "value": "batch"},
                                    {"label": "Live", "value": "live"},
                                ],
                                value=["single", "batch", "live"],
                                inline=True,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Animals",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-animal-filter",
                                options=[{"label": a, "value": a}
                                         for a in animals],
                                multi=True,
                                placeholder="All",
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Detector",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-detector-filter",
                                options=([{"label": "All detectors", "value": ""}]
                                         + [{"label": v, "value": k}
                                            for k, v in _DETECTOR_LABELS.items()]),
                                value="",
                                clearable=False,
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Cohort",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-cohort-filter",
                                options=[{"label": c, "value": c}
                                         for c in db.get_all_cohorts()],
                                placeholder="All", clearable=True, multi=True,
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Group",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="res-group-filter",
                                options=[{"label": g, "value": g}
                                         for g in db.get_all_groups()],
                                placeholder="All", clearable=True, multi=True,
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Event type",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Checklist(
                                id="res-type-filter",
                                options=[
                                    {"label": "Conv", "value": "convulsive"},
                                    {"label": "Non-conv", "value": "non_convulsive"},
                                ],
                                value=["convulsive", "non_convulsive"],
                                inline=True,
                                style={"fontSize": "0.82rem"},
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Min conf",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dbc.Input(
                                id="res-min-conf", type="number",
                                value=0, min=0, max=1, step=0.05, size="sm",
                                style={"backgroundColor": "var(--ned-bg)",
                                       "color": "var(--ned-text)",
                                       "border": "1px solid var(--ned-border)"},
                            ),
                        ], width=1),
                    ], className="g-2"),
                    dbc.Button(
                        "Apply filters", id="res-apply",
                        size="sm", outline=True, color="info",
                        className="mt-2",
                    ),
                ]),
                style={"backgroundColor": "var(--ned-sidebar)",
                       "border": "1px solid #21262d",
                       "marginBottom": "20px"},
            ),

            # ── Summary cards ──────────────────────────────────────
            html.Div(id="res-summary"),

            # ── Panel visibility ───────────────────────────────────
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "12px", "marginBottom": "8px"},
                children=[
                    html.Span("Show:", style={"fontSize": "0.82rem",
                                              "fontWeight": "600",
                                              "color": "var(--ned-text-muted)"}),
                    dbc.Checklist(
                        id="res-panels-toggle",
                        options=[
                            {"label": "Daily burden", "value": "daily"},
                            {"label": "Circadian", "value": "circadian"},
                            {"label": "Group comparison", "value": "groups"},
                            {"label": "Per-animal", "value": "animals"},
                            {"label": "Distributions", "value": "dist"},
                            {"label": "Longitudinal", "value": "long"},
                        ],
                        value=["daily", "circadian", "groups", "animals",
                               "dist", "long"],
                        inline=True,
                        style={"fontSize": "0.82rem"},
                    ),
                ],
            ),

            # ── Longitudinal (cohort progression — post-hoc) ───────
            html.Div(id="res-panel-long", children=[
                dcc.Graph(id="res-longitudinal", className="mb-3",
                          config={"responsive": True}),
            ]),

            # ── Cross-group / per-animal analysis ──────────────────
            html.Div(id="res-panel-groups", children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center",
                           "gap": "16px", "marginTop": "8px"},
                    children=[
                        html.H6("Group & cohort comparison",
                                style={"color": "var(--ned-accent)",
                                       "margin": 0}),
                        html.Span("Normalise:",
                                  style={"fontSize": "0.82rem",
                                         "color": "var(--ned-text-muted)"}),
                        dbc.RadioItems(
                            id="res-normalize",
                            options=[
                                {"label": " Raw counts", "value": "raw"},
                                {"label": " Per animal-hour",
                                 "value": "per_hour"},
                            ],
                            value="raw", inline=True,
                            style={"fontSize": "0.82rem"},
                        ),
                    ],
                ),
                html.P(
                    "Rates use each animal's recorded time (file length per "
                    "file it was recorded in). In the per-animal table below, "
                    "untick Include to drop an animal from these views, or set "
                    "Valid until to censor it after a date (e.g. died "
                    "mid-experiment). Low coverage / early-ended animals are "
                    "flagged for review.",
                    style={"color": "var(--ned-text-muted)",
                           "fontSize": "0.78rem", "margin": "4px 0 8px 0"},
                ),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="res-group-compare",
                                      config={"responsive": True},
                                      style={"height": "380px"}), width=6),
                    dbc.Col(html.Div(id="res-group-table"), width=6),
                ], className="mb-3"),
            ]),

            html.Div(id="res-panel-animals", children=[
                html.H6("Per-animal summary",
                        style={"color": "var(--ned-accent)",
                               "marginTop": "8px"}),
                html.Div(id="res-animal-table", className="mb-3"),
            ]),

            html.Div(id="res-panel-dist", children=[
                html.H6("Distributions",
                        style={"color": "var(--ned-accent)",
                               "marginTop": "8px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="res-dist-duration",
                                      config={"responsive": True}), width=6),
                    dbc.Col(dcc.Graph(id="res-dist-confidence",
                                      config={"responsive": True}), width=6),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="res-dist-duration-group",
                                      config={"responsive": True}), width=12),
                ], className="mb-3"),
            ]),

            # ── Interictal-spike analysis (shown only for the Spikes view) ──
            html.Div(id="res-panel-is", style={"display": "none"}, children=[
                html.H6("Interictal spike analysis",
                        style={"color": "var(--ned-accent)",
                               "marginTop": "8px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="res-is-isi",
                                      config={"responsive": True}), width=6),
                    dbc.Col(dcc.Graph(id="res-is-isi-cdf",
                                      config={"responsive": True}), width=6),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="res-is-diurnal",
                                      config={"responsive": True}), width=12),
                ], className="mb-3"),
            ]),

            # ── Daily burden + circadian (timeline — live monitoring) ──
            dbc.Row([
                dbc.Col(dcc.Graph(id="res-daily-burden",
                                  config={"responsive": True}),
                        id="res-col-daily", width=6),
                dbc.Col(dcc.Graph(id="res-circadian",
                                  config={"responsive": True}),
                        id="res-col-circadian", width=6),
            ], className="mb-3"),

            # ── Events table ───────────────────────────────────────
            html.H6("Events", style={"color": "var(--ned-accent)", "marginTop": "16px"}),
            html.Div(id="res-events-table"),

            # ── Event inspector ────────────────────────────────────
            html.Div(id="res-inspector", style={"marginTop": "16px"}),

            # Hidden store for selected event data
            dcc.Store(id="res-selected-event"),
            # Bumped when an event's Exclude checkbox is toggled, to re-render.
            dcc.Store(id="res-exclude-signal", data=0),
            dcc.Store(id="res-state-saved"),
            # Bumped when a per-animal Include / Valid-until edit is persisted.
            dcc.Store(id="res-animal-signal", data=0),

            # ── Export (raw per-event CSV for Prism etc.) ──────────
            dbc.Button("Export filtered CSV", id="res-export-csv",
                       outline=True, color="secondary", size="sm",
                       className="mt-3"),
            dbc.Button("Export all events CSV", id="res-export-all",
                       outline=True, color="secondary", size="sm",
                       className="mt-3 ms-2"),
            dbc.Button("Export graph data (XLSX)", id="res-export-graphdata",
                       outline=True, color="secondary", size="sm",
                       className="mt-3 ms-2"),
            html.Span(id="res-export-status",
                      style={"fontSize": "0.78rem", "marginLeft": "10px"}),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════


# ── Main filter callback ───────────────────────────────────────────────


@callback(
    Output("res-animal-filter", "options"),
    Output("res-animal-filter", "value"),
    Output("res-file-filter", "options"),
    Output("res-file-filter", "value"),
    Output("res-date-start", "value"),
    Output("res-date-end", "value"),
    Output("res-cohort-filter", "options"),
    Output("res-cohort-filter", "value"),
    Output("res-group-filter", "options"),
    Output("res-group-filter", "value"),
    Output("res-source-selector", "value"),
    Output("res-detector-filter", "value"),
    Output("res-normalize", "value"),
    Output("res-mode-filter", "value"),
    Output("res-type-filter", "value"),
    Output("res-min-conf", "value"),
    Output("res-panels-toggle", "value"),
    Input("res-project-select", "value"),
)
def res_on_project_switch(project):
    """Refresh DB-derived filter options and RESTORE the project's saved
    Results-tab filter selections (or defaults if none).

    Fires on tab load and whenever the active project changes, so opening a
    database brings back the filters last used on it. Per-animal exclusions are
    already persisted in the DB and apply independently."""
    if project:
        db.set_active_project(project)
    try:
        animals = db.get_all_animals()
        files = db.get_all_files()
        date_min, date_max = db.get_date_range()
    except Exception:
        animals, files, date_min, date_max = [], [], "", ""
    animal_opts = [{"label": a, "value": a} for a in animals]
    file_opts = [{"label": Path(f["path"]).name, "value": str(f["id"])}
                 for f in files]
    try:
        cohort_opts = [{"label": c, "value": c} for c in db.get_all_cohorts()]
        group_opts = [{"label": g, "value": g} for g in db.get_all_groups()]
    except Exception:
        cohort_opts, group_opts = [], []

    # Restore saved selections for this project (defaults if none / unknown).
    s = _load_results_state(project)
    return (
        animal_opts, s.get("animals", []),
        file_opts, s.get("files", []),
        s.get("date_start", date_min or ""), s.get("date_end", date_max or ""),
        cohort_opts, s.get("cohort", None),
        group_opts, s.get("group", None),
        s.get("source", "seizure_cnn"),
        s.get("detector", None),
        s.get("normalize", "raw"),
        s.get("modes", ["single", "batch", "live"]),
        s.get("types", ["convulsive", "non_convulsive"]),
        s.get("min_conf", 0),
        s.get("panels",
              ["daily", "circadian", "groups", "animals", "dist", "long"]),
    )


@callback(
    Output("res-state-saved", "data"),
    Input("res-source-selector", "value"),
    Input("res-detector-filter", "value"),
    Input("res-normalize", "value"),
    Input("res-date-start", "value"),
    Input("res-date-end", "value"),
    Input("res-mode-filter", "value"),
    Input("res-animal-filter", "value"),
    Input("res-type-filter", "value"),
    Input("res-min-conf", "value"),
    Input("res-file-filter", "value"),
    Input("res-cohort-filter", "value"),
    Input("res-group-filter", "value"),
    Input("res-panels-toggle", "value"),
    State("res-project-select", "value"),
    prevent_initial_call=True,
)
def res_save_filter_state(source, detector, normalize, ds, de, modes, animals,
                          types, min_conf, files, cohort, group, panels, project):
    """Persist the current Results-tab filter selections for the active project
    so they're restored next time it's opened."""
    if not project:
        return no_update
    _save_results_state(project, {
        "source": source, "detector": detector, "normalize": normalize,
        "date_start": ds, "date_end": de, "modes": modes, "animals": animals,
        "types": types, "min_conf": min_conf, "files": files,
        "cohort": cohort, "group": group, "panels": panels,
    })
    return no_update


@callback(
    Output("res-summary", "children"),
    Output("res-daily-burden", "figure"),
    Output("res-circadian", "figure"),
    Output("res-events-table", "children"),
    Output("res-group-compare", "figure"),
    Output("res-group-table", "children"),
    Output("res-animal-table", "children"),
    Output("res-dist-duration", "figure"),
    Output("res-dist-confidence", "figure"),
    Output("res-dist-duration-group", "figure"),
    Output("res-longitudinal", "figure"),
    Output("res-panel-is", "style"),
    Output("res-is-isi", "figure"),
    Output("res-is-isi-cdf", "figure"),
    Output("res-is-diurnal", "figure"),
    Input("res-apply", "n_clicks"),
    Input("res-source-selector", "value"),
    Input("res-project-select", "value"),
    Input("res-detector-filter", "value"),
    Input("res-exclude-signal", "data"),
    Input("res-animal-signal", "data"),
    Input("res-normalize", "value"),
    State("res-date-start", "value"),
    State("res-date-end", "value"),
    State("res-mode-filter", "value"),
    State("res-animal-filter", "value"),
    State("res-type-filter", "value"),
    State("res-min-conf", "value"),
    State("res-file-filter", "value"),
    State("res-cohort-filter", "value"),
    State("res-group-filter", "value"),
)
def update_results(n, source, project, detector, excl_signal, animal_signal,
                   normalize, date_start, date_end, modes, animals, types,
                   min_conf, file_ids, cohort, group_id):
    """Re-query SQLite and update all panels."""
    # Honour the active project DB (app-wide; shared with the Analysis tab).
    if project and project != db.get_active_project():
        db.set_active_project(project)
    # On a project switch, the carried-over control values belong to the
    # previous project. Apply THIS project's saved filters for this render;
    # res_on_project_switch restores the visible controls to match in parallel.
    if ctx.triggered_id == "res-project-select":
        s = _load_results_state(project)
        source = s.get("source", "seizure_cnn")
        detector = s.get("detector", None)
        normalize = s.get("normalize", "raw")
        date_start = s.get("date_start") or None
        date_end = s.get("date_end") or None
        modes = s.get("modes", ["single", "batch", "live"])
        animals = s.get("animals") or None
        types = s.get("types", ["convulsive", "non_convulsive"])
        min_conf = s.get("min_conf", 0)
        file_ids = s.get("files") or None
        cohort = s.get("cohort") or None
        group_id = s.get("group") or None
    # The Seizures/Spikes radio picks the high-level category; the Detector
    # dropdown narrows to a specific source within it (ML or classical).
    category = "spike" if source == "spike_cnn" else "seizure"
    detector = detector or None
    cohort = cohort or None
    group_id = group_id or None
    animal_id = animals[0] if animals and len(animals) == 1 else None
    event_type = types[0] if types and len(types) == 1 else None
    min_confidence = float(min_conf) if min_conf and float(min_conf) > 0 else None

    filter_kw = {
        "date_start": date_start or None,
        "date_end": date_end or None,
        "animal_id": animal_id,
        "min_confidence": min_confidence,
        "event_type": event_type,
        "category": category,
        "source": detector,
        "cohort": cohort,
        "group_id": group_id,
    }
    if modes and len(modes) < 3:
        filter_kw["mode"] = modes[0] if len(modes) == 1 else None

    try:
        events = db.get_events(**filter_kw)
        # Daily burden is computed in Python from the filtered events (so it can
        # split by group and honour exclude/censor); circadian stays DB-side.
        circadian = db.get_circadian(
            animal_id=animal_id, min_confidence=min_confidence,
            source=detector, category=category, cohort=cohort, group_id=group_id)
    except Exception as e:
        empty_fig = go.Figure()
        apply_fig_theme(empty_fig)
        return (alert(f"Database error: {e}", "danger"),
                empty_fig, empty_fig, html.Div(),
                empty_fig, html.Div(), html.Div(),
                empty_fig, empty_fig, empty_fig, empty_fig,
                {"display": "none"}, empty_fig, empty_fig, empty_fig)

    # Post-filter by file IDs
    if file_ids:
        chunk_ids = {int(fid) for fid in file_ids}
        events = [e for e in events if e.get("chunk_id") in chunk_ids]

    # Post-filter by multiple animals
    if animals and len(animals) > 1:
        events = [e for e in events if e.get("animal_id") in animals]
    if types and len(types) < 2:
        events = [e for e in events if e.get("type") in types]

    circ_fig = _panel_legend(_build_circadian(circadian))
    filters_active = bool(
        detector or cohort or group_id or animal_id
        or (animals and len(animals) > 1)
        or (types and len(types) < 2)
        or min_confidence
        or file_ids
        or (modes and len(modes) < 3))
    table = _build_events_table(events, filters_active=filters_active)

    # Cross-group / per-animal views — computed from the filtered event list
    # (drops per-event excludes) so they inherit every filter applied above.
    is_seizure = category == "seizure"
    agg_events = _active_events(events)

    # Observation rows (exact recording time) + per-animal review status.
    try:
        fa_rows = db.get_file_animals(
            date_start=date_start or None, date_end=date_end or None,
            animal_id=animal_id, mode=filter_kw.get("mode"),
            cohort=cohort, group_id=group_id)
        status = db.get_animal_status()
    except Exception:
        fa_rows, status = [], {}
    if file_ids:
        fa_rows = [r for r in fa_rows if r.get("chunk_id") in chunk_ids]
    if animals and len(animals) > 1:
        fa_rows = [r for r in fa_rows if r.get("animal_id") in animals]

    # Censor after each animal's valid_until, then split off excluded animals
    # (kept in the per-animal table, dropped from aggregations/plots).
    agg_events, fa_rows = _apply_censor(agg_events, fa_rows, status)
    excluded = {a for a, s in status.items() if s.get("excluded")}
    vis_events = [e for e in agg_events
                  if (e.get("animal_id") or "") not in excluded]
    vis_fa = [r for r in fa_rows if (r.get("animal_id") or "") not in excluded]

    # Summary cards — computed from the SAME filtered + censored set the plots
    # use (vis_events / vis_fa), so every filter (including multi-animal and the
    # file filter, which get_summary ignored) changes the cards, not just the
    # graphs.
    files_set = ({r.get("chunk_id") for r in vis_fa}
                 or {e.get("chunk_id") for e in vis_events})
    animals_set = ({r.get("animal_id") for r in vis_fa if r.get("animal_id")}
                   or {e.get("animal_id") for e in vis_events if e.get("animal_id")})
    n_total = len(vis_events)
    n_conv = sum(1 for e in vis_events if e.get("type") == "convulsive")
    n_nonconv = n_total - n_conv
    n_flagged = sum(1 for e in vis_events if e.get("movement_flag"))
    if category == "spike":
        summary_cards = dbc.Row([
            dbc.Col(metric_card("Files", str(len(files_set))), width=2),
            dbc.Col(metric_card("Animals", str(len(animals_set))), width=2),
            dbc.Col(metric_card("Total spikes", str(n_total), accent=True), width=2),
            dbc.Col(metric_card("Flagged", str(n_flagged)), width=2),
        ], className="g-2 mb-3")
    else:
        pct_c = f"({round(100*n_conv/n_total)}%)" if n_total else ""
        pct_nc = f"({round(100*n_nonconv/n_total)}%)" if n_total else ""
        summary_cards = dbc.Row([
            dbc.Col(metric_card("Files", str(len(files_set))), width=2),
            dbc.Col(metric_card("Animals", str(len(animals_set))), width=2),
            dbc.Col(metric_card("Total events", str(n_total), accent=True), width=2),
            dbc.Col(metric_card("Convulsive", f"{n_conv} {pct_c}"), width=2),
            dbc.Col(metric_card("Non-conv", f"{n_nonconv} {pct_nc}"), width=2),
            dbc.Col(metric_card("Flagged", str(n_flagged)), width=2),
        ], className="g-2 mb-3")

    group_rollup = _group_rollup(vis_events, vis_fa)
    animal_rollup = _animal_rollup(agg_events, fa_rows, status)
    # Per-animal day-1 reference (override or earliest recorded date) so the
    # longitudinal view aligns cohorts with different calendar starts.
    animal_starts = {r["animal"]: r["rec_start"]
                     for r in animal_rollup if r["rec_start"]}
    # Daily burden: per-group when >1 group, else convulsive/non-convulsive.
    daily_fig = _panel_legend(_build_daily_burden(vis_events))
    group_fig = _panel_legend(
        _build_group_comparison(group_rollup, normalize, is_seizure))
    group_table = _build_group_table(group_rollup, is_seizure)
    animal_table = _build_animal_table(animal_rollup, is_seizure)
    dur_fig = _panel_legend(_build_duration_hist(vis_events, is_seizure))
    conf_fig = _panel_legend(_build_confidence_hist(vis_events, is_seizure))
    dur_grp_fig = _panel_legend(_build_duration_by_group(vis_events, is_seizure))
    long_fig = _panel_legend(
        _build_longitudinal(vis_events, animal_starts, vis_fa, normalize))

    # Interictal-spike-specific panels — only for the Spikes view; hidden
    # (and left blank) otherwise so the seizure layout is unchanged.
    if category == "spike":
        is_style = {"display": "block"}
        isi_fig = _panel_legend(_build_isi_distribution(vis_events))
        isi_cdf_fig = _panel_legend(_build_isi_cdf(vis_events))
        diurnal_fig = _panel_legend(_build_spike_diurnal(vis_events))
    else:
        is_style = {"display": "none"}
        _blank = go.Figure()
        apply_fig_theme(_blank)
        isi_fig = isi_cdf_fig = diurnal_fig = _blank

    return (summary_cards, daily_fig, circ_fig, table,
            group_fig, group_table, animal_table,
            dur_fig, conf_fig, dur_grp_fig, long_fig,
            is_style, isi_fig, isi_cdf_fig, diurnal_fig)


@callback(
    Output("res-exclude-signal", "data"),
    Input("res-grid", "cellValueChanged"),
    State("res-exclude-signal", "data"),
    prevent_initial_call=True,
)
def res_toggle_exclude(changed, sig):
    """Persist an event's Exclude checkbox to the active project DB and bump the
    signal so summaries/plots re-render (excluded events drop out, but the row
    stays in the table so it can be toggled back)."""
    if not changed:
        return no_update
    changes = changed if isinstance(changed, list) else [changed]
    n = 0
    for ch in changes:
        if not isinstance(ch, dict) or ch.get("colId") != "Exclude":
            continue
        row = ch.get("data") or {}
        eid = row.get("_event_id")
        if eid is None:
            continue
        db.set_event_excluded(int(eid), bool(ch.get("value")))
        n += 1
    return (sig or 0) + 1 if n else no_update


@callback(
    Output("res-animal-signal", "data"),
    Input("res-animal-grid", "cellValueChanged"),
    State("res-animal-signal", "data"),
    prevent_initial_call=True,
)
def res_edit_animal(changed, sig):
    """Persist per-animal Include / Valid-until edits to animal_status and bump
    the signal so the comparison/per-animal/plot panels re-render."""
    if not changed:
        return no_update
    changes = changed if isinstance(changed, list) else [changed]
    n = 0
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        row = ch.get("data") or {}
        aid = row.get("_animal")
        if not aid:
            continue
        col = ch.get("colId")
        if col == "Include":
            db.set_animal_status(aid, excluded=not bool(ch.get("value")))
            n += 1
        elif col == "Valid until":
            db.set_animal_status(aid, valid_until=(ch.get("value") or "").strip())
            n += 1
        elif col == "Rec start":
            db.set_animal_status(
                aid, recording_start_date=(ch.get("value") or "").strip())
            n += 1
    return (sig or 0) + 1 if n else no_update


@callback(
    Output("res-col-daily", "style"),
    Output("res-col-circadian", "style"),
    Output("res-panel-groups", "style"),
    Output("res-panel-animals", "style"),
    Output("res-panel-dist", "style"),
    Output("res-panel-long", "style"),
    Input("res-panels-toggle", "value"),
)
def res_toggle_panels(shown):
    """Show/hide each results panel from the 'Show:' checklist."""
    shown = shown or []

    def st(key):
        return {} if key in shown else {"display": "none"}

    return (st("daily"), st("circadian"), st("groups"),
            st("animals"), st("dist"), st("long"))


# ── Event selection from AG Grid ───────────────────────────────────────


@callback(
    Output("res-selected-event", "data"),
    Input("res-grid", "selectedRows"),
    prevent_initial_call=True,
)
def select_event(selected):
    if not selected:
        return no_update
    row = selected[0]
    return {
        "path": row.get("_path", ""),
        "start_sec": row.get("Start (s)", 0),
        "end_sec": row.get("End (s)", 0),
        "duration": row.get("Duration", 0),
        "animal": row.get("Animal", ""),
        "type": row.get("Type", ""),
        "subtype": row.get("Subtype", ""),
        "confidence": row.get("Confidence", 0),
        "conv_pct": row.get("Conv %", 0),
        "flagged": row.get("Flagged", ""),
        "hour": row.get("Hour", ""),
        "mode": row.get("Mode", ""),
        "date": row.get("Date", ""),
        "file": row.get("File", ""),
        "_channel_idx": row.get("_channel_idx", 0),
    }


# ── Event inspector ────────────────────────────────────────────────────


@callback(
    Output("res-inspector", "children"),
    Input("res-selected-event", "data"),
    prevent_initial_call=True,
)
def show_inspector(ev_data):
    if not ev_data:
        return html.Div()

    edf_path = ev_data.get("path", "")
    if not edf_path or not os.path.isfile(edf_path):
        return _inspector_params_only(ev_data)

    try:
        return _build_full_inspector(edf_path, ev_data)
    except Exception as e:
        return html.Div([
            _inspector_params_panel(ev_data),
            alert(f"Could not load EEG data: {e}", "warning"),
        ])


def _inspector_params_only(ev_data: dict):
    """Show just the parameters table when EDF file is not available."""
    return html.Div([
        html.H6("Event Inspector",
                style={"color": "var(--ned-accent)", "marginBottom": "12px"}),
        alert("EDF file not found — showing parameters only.", "info"),
        _inspector_params_panel(ev_data),
    ])


def _build_full_inspector(edf_path: str, ev_data: dict):
    """Build inspector with EEG trace, PSD, spectrogram, and params."""
    from eeg_seizure_analyzer.io.edf_reader import read_edf_window, scan_edf_channels, auto_pair_channels

    onset = float(ev_data["start_sec"])
    offset = float(ev_data["end_sec"])
    context_sec = 10.0
    bp_low, bp_high = 1.0, 50.0

    # Determine channel index
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, _, _ = auto_pair_channels(ch_info)
    channel_idx = int(ev_data.get("_channel_idx", 0))
    if channel_idx not in eeg_idx and eeg_idx:
        channel_idx = eeg_idx[0]

    # Read window around event
    win_start = max(0, onset - context_sec)
    win_end = offset + context_sec
    rec = read_edf_window(edf_path, channels=[channel_idx],
                          start_sec=win_start, duration_sec=win_end - win_start)

    data = rec.data[0].astype(np.float64)
    fs = rec.fs
    data_filt = bandpass_filter(data, fs, bp_low, bp_high)
    time_axis = np.linspace(win_start, win_start + len(data) / fs, len(data))

    # ── EEG trace ──────────────────────────────────────────────────
    ds_time, ds_data = _minmax_downsample(time_axis, data_filt)

    _eeg_color = "#1b2a4a" if get_plotly_theme() == "light" else "#58a6ff"
    fig_eeg = go.Figure()
    fig_eeg.add_trace(go.Scattergl(
        x=ds_time, y=ds_data, mode="lines",
        line=dict(width=0.8, color=_eeg_color),
        name="EEG",
    ))
    fig_eeg.add_shape(
        type="rect", x0=onset, x1=offset, y0=0, y1=1, yref="paper",
        fillcolor="rgba(88,166,255,0.15)",
        line=dict(color="#58a6ff", width=1.5), layer="below",
    )
    fig_eeg.update_layout(
        height=280, xaxis_title="Time (s)", yaxis_title="Amplitude",
        showlegend=False, dragmode="zoom",
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_eeg)

    # ── PSD of event window ────────────────────────────────────────
    event_start_idx = max(0, int((onset - win_start) * fs))
    event_end_idx = min(len(data_filt), int((offset - win_start) * fs))
    event_data = data_filt[event_start_idx:event_end_idx]

    nperseg_psd = min(int(2 * fs), len(event_data))
    nperseg_psd = max(nperseg_psd, 64)
    freqs_psd, psd_vals = welch(event_data, fs=fs, nperseg=nperseg_psd)
    psd_mask = freqs_psd <= 100
    freqs_psd, psd_vals = freqs_psd[psd_mask], psd_vals[psd_mask]

    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(
        x=freqs_psd, y=10 * np.log10(psd_vals + 1e-12),
        mode="lines", line=dict(color="#58a6ff"),
        name="PSD",
    ))
    fig_psd.update_layout(
        height=250, xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)",
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_psd)

    # ── Spectrogram (power over time) ──────────────────────────────
    nperseg_spec = min(int(1.0 * fs), len(data_filt) // 4)
    nperseg_spec = max(nperseg_spec, 64)
    noverlap = int(nperseg_spec * 0.9)
    f_spec, t_spec, Sxx = scipy_spectrogram(
        data_filt, fs=fs, nperseg=nperseg_spec, noverlap=noverlap)
    t_spec = t_spec + win_start
    freq_mask = f_spec <= 100
    f_spec, Sxx = f_spec[freq_mask], Sxx[freq_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig_spec = go.Figure(go.Heatmap(
        x=t_spec, y=f_spec, z=Sxx_db, colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8),
    ))
    fig_spec.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_spec.update_layout(
        height=250, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_spec)

    # ── Band power over time ───────────────────────────────────────
    bands = {
        "Delta (0.5-4)": (0.5, 4, "#1f77b4"),
        "Theta (4-8)": (4, 8, "#ff7f0e"),
        "Alpha (8-13)": (8, 13, "#2ca02c"),
        "Beta (13-30)": (13, 30, "#d62728"),
        "Gamma (30-50)": (30, 50, "#9467bd"),
    }
    win_samples = int(2.0 * fs)
    step_samples = int(1.0 * fs)
    band_power_data = {name: [] for name in bands}
    bp_times = []
    for start_s in range(0, max(1, len(data_filt) - win_samples), step_samples):
        seg = data_filt[start_s:start_s + win_samples]
        bp_times.append(win_start + (start_s + win_samples / 2) / fs)
        f_w, psd_w = welch(seg, fs=fs, nperseg=min(win_samples, len(seg)))
        for name, (flo, fhi, _) in bands.items():
            mask = (f_w >= flo) & (f_w <= fhi)
            bp = np.trapezoid(psd_w[mask], f_w[mask]) if mask.sum() > 1 else 0.0
            band_power_data[name].append(bp)

    fig_bp = go.Figure()
    for name, (_, _, color) in bands.items():
        fig_bp.add_trace(go.Scatter(
            x=bp_times, y=band_power_data[name],
            name=name, mode="lines", line=dict(color=color),
            stackgroup="bands",
        ))
    fig_bp.add_vline(x=onset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_bp.add_vline(x=offset, line=dict(color="#f85149", width=1.5, dash="dash"))
    fig_bp.update_layout(
        height=250, xaxis_title="Time (s)", yaxis_title="Power",
        yaxis_rangemode="tozero", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_fig_theme(fig_bp)

    # ── Computed features from event ───────────────────────────────
    # Compute spectral features on the event window
    computed = {}
    if len(event_data) > 0 and len(psd_vals) > 0:
        total_power = np.sum(psd_vals)
        if total_power > 0:
            dominant_freq = freqs_psd[np.argmax(psd_vals)]
            computed["Dominant freq (Hz)"] = f"{dominant_freq:.1f}"

            # Band powers
            for bname, (flo, fhi) in [("Delta", (0.5, 4)), ("Theta", (4, 8)),
                                       ("Alpha", (8, 13)), ("Beta", (13, 30)),
                                       ("Gamma", (30, 50))]:
                mask = (freqs_psd >= flo) & (freqs_psd <= fhi)
                rel = np.sum(psd_vals[mask]) / total_power * 100
                computed[f"{bname} power (%)"] = f"{rel:.1f}"

            # Spectral entropy
            psd_norm = psd_vals / total_power
            psd_norm = psd_norm[psd_norm > 0]
            spec_entropy = -np.sum(psd_norm * np.log2(psd_norm))
            computed["Spectral entropy"] = f"{spec_entropy:.2f}"

        # RMS amplitude
        rms = np.sqrt(np.mean(event_data ** 2))
        computed["RMS amplitude"] = f"{rms:.2f}"

        # Peak-to-peak
        ptp = float(np.ptp(event_data))
        computed["Peak-to-peak"] = f"{ptp:.2f}"

    # Build layout
    return html.Div([
        html.Hr(style={"borderColor": "#58a6ff", "margin": "16px 0"}),
        html.H6("Event Inspector",
                style={"color": "var(--ned-accent)", "marginBottom": "12px"}),

        # EEG trace
        dcc.Graph(figure=fig_eeg, config={"displayModeBar": False}),

        # PSD + Spectrogram side by side
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_psd, config={"displayModeBar": False}),
                    width=6),
            dbc.Col(dcc.Graph(figure=fig_spec, config={"displayModeBar": False}),
                    width=6),
        ]),

        # Band power over time
        dcc.Graph(figure=fig_bp, config={"displayModeBar": False}),

        # Parameters panel
        _inspector_params_panel(ev_data, computed),
    ])


def _inspector_params_panel(ev_data: dict, computed: dict | None = None):
    """Build a card showing all event parameters."""
    params = {
        "File": ev_data.get("file", Path(ev_data.get("path", "")).name),
        "Animal": ev_data.get("animal", ""),
        "Date": ev_data.get("date", ""),
        "Onset (s)": ev_data.get("start_sec", ""),
        "Offset (s)": ev_data.get("end_sec", ""),
        "Duration (s)": ev_data.get("duration", ""),
        "Type": ev_data.get("type", ""),
        "Subtype": ev_data.get("subtype", ""),
        "CNN confidence": ev_data.get("confidence", ""),
        "Convulsive %": ev_data.get("conv_pct", ""),
        "Movement flagged": ev_data.get("flagged", "No"),
        "Hour of day": ev_data.get("hour", ""),
        "Analysis mode": ev_data.get("mode", ""),
    }

    # Merge computed spectral features
    if computed:
        params.update(computed)

    rows = []
    for k, v in params.items():
        if v == "" or v is None:
            continue
        rows.append(
            html.Tr([
                html.Td(k, style={"color": "var(--ned-text-muted)", "fontSize": "0.82rem",
                                   "paddingRight": "16px", "whiteSpace": "nowrap"}),
                html.Td(str(v), style={"color": "var(--ned-text)", "fontSize": "0.82rem"}),
            ])
        )

    return dbc.Card(
        dbc.CardBody([
            html.H6("Parameters", style={"color": "var(--ned-text-muted)",
                                          "fontSize": "0.82rem",
                                          "marginBottom": "8px"}),
            html.Table(
                html.Tbody(rows),
                style={"width": "100%"},
            ),
        ]),
        style={"backgroundColor": "var(--ned-sidebar)", "border": "1px solid #21262d",
               "marginTop": "12px"},
    )


# ═══════════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════════


def _panel_legend(fig):
    """Place the legend below the plot (left-aligned title above it) so the
    title and legend never overlap. Call AFTER apply_fig_theme, which resets the
    margins — this re-sets them with headroom for the title and footroom for the
    horizontal legend."""
    fig.update_layout(
        title=dict(x=0.0, xanchor="left", y=0.97, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=-0.28, x=0,
                    font=dict(size=10)),
        margin=dict(l=60, r=20, t=46, b=78),
    )
    return fig


def _build_daily_burden(events) -> go.Figure:
    """Events per calendar date. Split by GROUP when more than one group is
    present (stacked bars per group); otherwise split convulsive vs
    non-convulsive (the severity view)."""
    fig = go.Figure()
    dated = [e for e in events if _ev_date(e)]
    if not dated:
        apply_fig_theme(fig)
        fig.update_layout(title="Daily Seizure Burden", xaxis_title="Date",
                          yaxis_title="Events")
        return fig

    groups = sorted({(e.get("group_id") or "(unlabeled)") for e in dated})
    if len(groups) > 1:
        # Per-group stacked bars by date.
        for i, g in enumerate(groups):
            by_date: dict = {}
            for e in dated:
                if (e.get("group_id") or "(unlabeled)") != g:
                    continue
                d = _ev_date(e)
                by_date[d] = by_date.get(d, 0) + 1
            dts = sorted(by_date)
            fig.add_trace(go.Bar(
                x=dts, y=[by_date[d] for d in dts], name=g,
                marker_color=_GROUP_PALETTE[i % len(_GROUP_PALETTE)]))
    else:
        # Single group → convulsive / non-convulsive.
        conv: dict = {}
        nonconv: dict = {}
        for e in dated:
            d = _ev_date(e)
            tgt = conv if e.get("type") == "convulsive" else nonconv
            tgt[d] = tgt.get(d, 0) + 1
        if conv:
            dts = sorted(conv)
            fig.add_trace(go.Bar(x=dts, y=[conv[d] for d in dts],
                                 name="Convulsive", marker_color=_CONV_COLOR))
        if nonconv:
            dts = sorted(nonconv)
            fig.add_trace(go.Bar(x=dts, y=[nonconv[d] for d in dts],
                                 name="Non-convulsive", marker_color=_NONCONV_COLOR))

    fig.update_layout(
        barmode="stack", title="Daily Seizure Burden",
        xaxis_title="Date", yaxis_title="Events",
        legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


def _build_circadian(circadian: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not circadian:
        apply_fig_theme(fig)
        fig.update_layout(title="Circadian Distribution")
        return fig

    conv_by_hour = [0] * 24
    nonconv_by_hour = [0] * 24
    for row in circadian:
        h = row["hour_of_day"]
        if h is None:
            continue
        if row["type"] == "convulsive":
            conv_by_hour[h] += row["n_events"]
        else:
            nonconv_by_hour[h] += row["n_events"]

    hours = [f"{h:02d}:00" for h in range(24)]
    fig.add_trace(go.Bar(x=hours, y=conv_by_hour,
                         name="Convulsive", marker_color="#f85149"))
    fig.add_trace(go.Bar(x=hours, y=nonconv_by_hour,
                         name="Non-convulsive", marker_color="#58a6ff"))

    fig.update_layout(
        barmode="stack", title="Circadian Distribution",
        xaxis_title="Hour of day", yaxis_title="Events",
        legend=dict(orientation="h", y=1.1),
    )
    apply_fig_theme(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Interictal-spike-specific views (ISI distribution / CDF / diurnal profile)
# ═══════════════════════════════════════════════════════════════════════
#
# These reuse the already-filtered event list (fast: ~0.4s even for ~2M spikes)
# rather than re-querying. ISI = inter-spike interval, computed per animal WITHIN
# each file (chunk) so gaps between files don't create spurious huge intervals.

_ISI_MIN_S = 0.1      # clamp ISIs to a sane plotting range (refractory ~0.75s)
_ISI_MAX_S = 1000.0
_ISI_NBINS = 50       # log-spaced bins; shared by the plot and the export


def _isi_bins():
    """Log-spaced ISI bin edges and centres — the single source of truth for
    both the distribution plot and its exported table."""
    edges = np.logspace(np.log10(_ISI_MIN_S), np.log10(_ISI_MAX_S), _ISI_NBINS)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def _isi_distribution_table(isis: dict):
    """Per-group probability DENSITY per ISI bin: (count / total) / bin_width.

    Dividing by bin width is essential with log-spaced bins — plotting raw
    per-bin probability makes the geometrically wider high-ISI bins accumulate
    more counts and pushes the apparent peak toward the median (a binning
    artifact). Density puts the peak at the true mode. Integrates to 1 over ISI.
    Returns (edges, centers, {group: density}).
    """
    edges, centers = _isi_bins()
    widths = np.diff(edges)
    out: dict = {}
    for g in sorted(isis):
        arr = np.asarray(isis[g])
        arr = arr[(arr >= _ISI_MIN_S) & (arr <= _ISI_MAX_S)]
        if arr.size == 0:
            continue
        h, _ = np.histogram(arr, bins=edges)
        tot = h.sum()
        if tot:
            out[g] = (h / tot) / widths
    return edges, centers, out


def _isi_cdf_table(isis: dict, at):
    """Empirical cumulative probability P(ISI <= x) per group, evaluated at the
    points ``at`` (the ISI bin edges) so the CDF is compact and on the same
    grid as the distribution — no need to ship ~2M raw intervals."""
    out: dict = {}
    for g in sorted(isis):
        arr = np.sort(np.asarray(isis[g]))
        if arr.size:
            out[g] = np.searchsorted(arr, at, side="right") / arr.size
    return out


def _isis_by_group(events) -> dict:
    """Map group_id -> list of inter-spike intervals (seconds).

    ISIs are diffs of sorted spike onset times within each (animal, file); the
    file's group label is used so SV2A vs Control can be compared directly.
    """
    from collections import defaultdict
    onsets: dict = defaultdict(list)
    grp_of: dict = {}
    for e in events:
        key = (e.get("animal_id") or "", e.get("chunk_id"))
        onsets[key].append(e.get("start_sec") or 0.0)
        grp_of[key] = e.get("group_id") or "(unlabeled)"
    out: dict = defaultdict(list)
    for key, ts in onsets.items():
        if len(ts) < 2:
            continue
        ts.sort()
        g = grp_of.get(key, "(unlabeled)")
        prev = ts[0]
        for t in ts[1:]:
            d = t - prev
            prev = t
            if d > 0:
                out[g].append(d)
    return out


def _build_isi_distribution(events) -> go.Figure:
    """Inter-spike-interval frequency distribution, one density curve per group
    (log-spaced bins, normalised to probability density so the peak reflects the
    true mode, not the bin width, and groups compare regardless of n)."""
    fig = go.Figure()
    _edges, centers, dens = _isi_distribution_table(_isis_by_group(events))
    if not dens:
        apply_fig_theme(fig)
        fig.update_layout(title="Inter-spike interval distribution",
                          xaxis_title="ISI (s)",
                          yaxis_title="Probability density (1/s)")
        return fig
    for i, g in enumerate(sorted(dens)):
        fig.add_trace(go.Scatter(
            x=centers, y=dens[g], mode="lines", name=g,
            line=dict(color=_GROUP_PALETTE[i % len(_GROUP_PALETTE)], width=2)))
    fig.update_layout(
        title="Inter-spike interval distribution",
        xaxis_title="ISI (s)", yaxis_title="Probability density (1/s)",
        legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(type="log")
    apply_fig_theme(fig)
    return fig


def _build_isi_cdf(events) -> go.Figure:
    """Cumulative probability of inter-spike intervals, one curve per group.
    A rightward shift = longer intervals (sparser firing); good for KS-style
    visual comparison between SV2A and Control."""
    fig = go.Figure()
    isis = _isis_by_group(events)
    if not any(isis.values()):
        apply_fig_theme(fig)
        fig.update_layout(title="ISI cumulative probability",
                          xaxis_title="ISI (s)",
                          yaxis_title="Cumulative probability")
        return fig
    for i, g in enumerate(sorted(isis)):
        arr = np.sort(np.asarray(isis[g]))
        if arr.size == 0:
            continue
        y = np.arange(1, arr.size + 1) / arr.size
        if arr.size > 4000:  # thin for the browser; shape is preserved
            idx = np.linspace(0, arr.size - 1, 4000).astype(int)
            arr, y = arr[idx], y[idx]
        fig.add_trace(go.Scatter(
            x=arr, y=y, mode="lines", name=g,
            line=dict(color=_GROUP_PALETTE[i % len(_GROUP_PALETTE)], width=2)))
    fig.update_layout(
        title="ISI cumulative probability",
        xaxis_title="ISI (s)", yaxis_title="Cumulative probability",
        legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(type="log")
    apply_fig_theme(fig)
    return fig


def _build_spike_diurnal(events) -> go.Figure:
    """Diurnal spike profile: mean spikes per animal in each hour of day, one
    line per group. Reveals circadian modulation of interictal activity; the
    per-animal mean controls for differing group sizes."""
    from collections import defaultdict
    fig = go.Figure()
    dated = [e for e in events if e.get("hour_of_day") is not None]
    if not dated:
        apply_fig_theme(fig)
        fig.update_layout(title="Diurnal spike profile",
                          xaxis_title="Hour of day",
                          yaxis_title="Spikes per animal")
        return fig
    cnt: dict = defaultdict(int)
    animals_in: dict = defaultdict(set)
    for e in dated:
        g = e.get("group_id") or "(unlabeled)"
        a = e.get("animal_id") or ""
        cnt[(g, a, e.get("hour_of_day"))] += 1
        animals_in[g].add(a)
    hours = list(range(24))
    labels = [f"{h:02d}:00" for h in hours]
    for i, g in enumerate(sorted(animals_in)):
        animals = animals_in[g]
        n = len(animals) or 1
        y = [sum(cnt.get((g, a, h), 0) for a in animals) / n for h in hours]
        fig.add_trace(go.Scatter(
            x=labels, y=y, mode="lines+markers", name=g,
            line=dict(color=_GROUP_PALETTE[i % len(_GROUP_PALETTE)], width=2)))
    fig.update_layout(
        title="Diurnal spike profile (mean per animal)",
        xaxis_title="Hour of day", yaxis_title="Spikes per animal",
        legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Cross-group / per-animal analysis
# ═══════════════════════════════════════════════════════════════════════
#
# These views are computed in Python from the already-filtered event list so
# they honour every filter the callback applies (file IDs, multi-animal, type,
# exclude) without re-implementing the SQL WHERE clause.
#
# Recording time / coverage comes from the per-file observation rows
# (db.get_file_animals): one row per animal per file, valid_sec = the file's
# recording length, written at analysis time from the channel→animal map. This
# is exact even for animals with zero detected events. A group's animal-hours is
# the sum of valid_sec over its rows. Older projects without observation data
# fall back to raw counts (rates show as "—").

_CONV_COLOR = "#f85149"
_NONCONV_COLOR = "#58a6ff"
_GROUP_PALETTE = ["#58a6ff", "#f85149", "#3fb950", "#d29922", "#bc8cff",
                  "#39c5cf", "#ff7b72", "#a5d6ff"]


def _active_events(events: list[dict]) -> list[dict]:
    """Drop excluded events — aggregations never count them."""
    return [e for e in events if not e.get("excluded")]


def _ev_date(e: dict) -> str:
    return e.get("date") or e.get("chunk_date") or ""


def _apply_censor(events, fa_rows, status):
    """Drop events and observation that fall after an animal's valid_until
    cutoff (an animal that died/dropped mid-experiment is counted only for the
    period it was still recorded)."""
    vu = {a: s["valid_until"] for a, s in status.items() if s.get("valid_until")}
    if not vu:
        return events, fa_rows
    kept_ev = [e for e in events
               if not (vu.get(e.get("animal_id") or "")
                       and _ev_date(e) > vu[e.get("animal_id") or ""])]
    kept_fa = [r for r in fa_rows
               if not (vu.get(r.get("animal_id") or "")
                       and (r.get("date") or "") > vu[r.get("animal_id") or ""])]
    return kept_ev, kept_fa


def _group_rollup(events: list[dict], fa_rows: list[dict]) -> list[dict]:
    """Per-group event counts plus recording time from observation rows.

    Event counts come from ``events``; animals, files and recording hours come
    from ``fa_rows`` (one row per animal per file) so quiet animals still count
    toward the denominator. Older projects without observation data fall back to
    animals/files derived from events, with hours = 0 (rates show as raw)."""
    groups: dict = {}

    def _st(g):
        return groups.setdefault(g, {
            "group": g, "n": 0, "conv": 0, "nonconv": 0,
            "animals": set(), "files": set(), "rec_sec": 0.0})

    for e in events:
        st = _st(e.get("group_id") or "(unlabeled)")
        st["n"] += 1
        if e.get("type") == "convulsive":
            st["conv"] += 1
        else:
            st["nonconv"] += 1
        if e.get("animal_id"):
            st["animals"].add(e["animal_id"])
        st["files"].add(e.get("chunk_id"))
    for r in fa_rows:
        st = _st(r.get("group_id") or "(unlabeled)")
        if r.get("animal_id"):
            st["animals"].add(r["animal_id"])
        st["files"].add(r.get("chunk_id"))
        st["rec_sec"] += float(r.get("valid_sec") or 0)

    out = []
    for st in groups.values():
        hours = st["rec_sec"] / 3600.0
        out.append({
            "group": st["group"], "n": st["n"],
            "conv": st["conv"], "nonconv": st["nonconv"],
            "n_animals": len(st["animals"]), "n_files": len(st["files"]),
            "rec_hours": hours,
            "rate": (st["n"] / hours) if hours > 0 else None,
            "conv_rate": (st["conv"] / hours) if hours > 0 else None,
            "nonconv_rate": (st["nonconv"] / hours) if hours > 0 else None,
        })
    out.sort(key=lambda d: d["group"])
    return out


def _animal_rollup(events, fa_rows, status) -> list[dict]:
    """Per-animal totals, recording hours, coverage and review status.

    Universe is every animal seen in events OR recorded (fa_rows), so quiet and
    excluded animals still appear. Coverage % and the dropout flag are relative
    to the animal's own group."""
    A: dict = {}

    def _st(a):
        return A.setdefault(a, {
            "animal": a, "groups": set(), "cohorts": set(), "files": set(),
            "n": 0, "conv": 0, "nonconv": 0, "dur": 0.0, "rec_sec": 0.0,
            "dates": set()})

    for e in events:
        st = _st(e.get("animal_id") or "(unknown)")
        st["n"] += 1
        if e.get("type") == "convulsive":
            st["conv"] += 1
        else:
            st["nonconv"] += 1
        if e.get("group_id"):
            st["groups"].add(e["group_id"])
        if e.get("cohort"):
            st["cohorts"].add(e["cohort"])
        st["files"].add(e.get("chunk_id"))
        st["dur"] += float(e.get("duration_sec") or 0)
        if _ev_date(e):
            st["dates"].add(_ev_date(e))
    for r in fa_rows:
        st = _st(r.get("animal_id") or "(unknown)")
        if r.get("group_id"):
            st["groups"].add(r["group_id"])
        if r.get("cohort"):
            st["cohorts"].add(r["cohort"])
        st["files"].add(r.get("chunk_id"))
        st["rec_sec"] += float(r.get("valid_sec") or 0)
        if r.get("date"):
            st["dates"].add(r["date"])

    out = []
    for st in A.values():
        hours = st["rec_sec"] / 3600.0
        dates = sorted(st["dates"])
        sstat = status.get(st["animal"], {})
        first_date = dates[0] if dates else ""
        out.append({
            "animal": st["animal"],
            "groups": ", ".join(sorted(st["groups"])),
            "primary_group": (sorted(st["groups"])[0]
                              if st["groups"] else "(unlabeled)"),
            "cohorts": ", ".join(sorted(st["cohorts"])),
            "n_files": len(st["files"]),
            "n": st["n"], "conv": st["conv"], "nonconv": st["nonconv"],
            "mean_dur": (st["dur"] / st["n"]) if st["n"] else 0.0,
            "rec_hours": hours,
            "rate": (st["n"] / hours) if hours > 0 else None,
            "rec_days": len(st["dates"]),
            "first_date": first_date,
            "last_date": dates[-1] if dates else "",
            "excluded": bool(sstat.get("excluded")),
            "valid_until": sstat.get("valid_until", ""),
            "notes": sstat.get("notes", ""),
            # Day-1 reference for the longitudinal view: explicit override if
            # set, else the animal's earliest recorded date. Aligns cohorts
            # that started on different calendar dates.
            "rec_start_override": sstat.get("recording_start_date", ""),
            "rec_start": sstat.get("recording_start_date") or first_date,
        })

    # Coverage % and dropout flags, relative to each animal's primary group.
    by_g: dict = {}
    for r in out:
        by_g.setdefault(r["primary_group"], []).append(r)
    for rows in by_g.values():
        max_h = max((r["rec_hours"] for r in rows), default=0.0)
        last = max((r["last_date"] for r in rows if r["last_date"]), default="")
        for r in rows:
            r["coverage"] = (round(100 * r["rec_hours"] / max_h)
                             if max_h > 0 else None)
            flags = []
            if r["coverage"] is not None and r["coverage"] < 70:
                flags.append("low coverage")
            if last and r["last_date"] and r["last_date"] < last:
                flags.append(f"ended {r['last_date']}")
            if r["valid_until"]:
                flags.append(f"censored {r['valid_until']}")
            r["flag"] = ", ".join(flags)
    out.sort(key=lambda d: d["animal"])
    return out


def _build_group_comparison(rollup, normalize, is_seizure) -> go.Figure:
    fig = go.Figure()
    if not rollup:
        apply_fig_theme(fig)
        fig.update_layout(title="Group comparison")
        return fig

    per_hour = normalize == "per_hour"
    note = ""
    if per_hour and not any((r["rec_hours"] or 0) > 0 for r in rollup):
        per_hour = False
        note = "  (no recording-time data — showing counts)"

    groups = [r["group"] for r in rollup]
    if is_seizure:
        if per_hour:
            conv_y = [r["conv_rate"] or 0 for r in rollup]
            nonconv_y = [r["nonconv_rate"] or 0 for r in rollup]
        else:
            conv_y = [r["conv"] for r in rollup]
            nonconv_y = [r["nonconv"] for r in rollup]
        fig.add_trace(go.Bar(x=groups, y=conv_y, name="Convulsive",
                             marker_color=_CONV_COLOR))
        fig.add_trace(go.Bar(x=groups, y=nonconv_y, name="Non-convulsive",
                             marker_color=_NONCONV_COLOR))
        fig.update_layout(barmode="stack")
    else:
        y = [(r["rate"] or 0) if per_hour else r["n"] for r in rollup]
        fig.add_trace(go.Bar(x=groups, y=y, name="Events",
                             marker_color=_NONCONV_COLOR))

    ytitle = "Events / animal-hour" if per_hour else "Events"
    fig.update_layout(
        title="Group comparison" + note, xaxis_title="Group",
        yaxis_title=ytitle, legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


def _build_group_table(rollup, is_seizure):
    if not rollup:
        return html.Div()
    rows = []
    for r in rollup:
        row = {
            "Group": r["group"],
            "Animals": r["n_animals"],
            "Files": r["n_files"],
            "Rec h": round(r["rec_hours"], 1),
            "Events": r["n"],
            "Events/h": round(r["rate"], 3) if r["rate"] is not None else "—",
        }
        if is_seizure:
            row["Conv"] = r["conv"]
            row["Non-conv"] = r["nonconv"]
        rows.append(row)
    cols = [{"field": "Group", "width": 130}, {"field": "Animals", "width": 90},
            {"field": "Files", "width": 80}, {"field": "Rec h", "width": 90}]
    if is_seizure:
        cols += [{"field": "Conv", "width": 80},
                 {"field": "Non-conv", "width": 95}]
    cols += [{"field": "Events", "width": 90}, {"field": "Events/h", "width": 95}]
    return dag.AgGrid(
        rowData=rows, columnDefs=cols,
        defaultColDef={"sortable": True, "resizable": True},
        style={"height": f"{min(60 + len(rows) * 42, 320)}px"},
        className="ag-theme-alpine-dark",
    )


def _build_animal_table(rollup, is_seizure):
    if not rollup:
        return html.P("No events found.",
                      style={"color": "var(--ned-text-muted)",
                             "fontSize": "0.85rem"})
    rows = []
    for r in rollup:
        span = (f"{r['first_date']} – {r['last_date']}"
                if r["first_date"] else "")
        row = {
            "Include": not r["excluded"],
            "Animal": r["animal"],
            "Group": r["groups"],
            "Cohort": r["cohorts"],
            "Files": r["n_files"],
            "Events": r["n"],
            "Mean dur (s)": round(r["mean_dur"], 1),
            "Rec days": r["rec_days"],
            "Rec h": round(r["rec_hours"], 1),
            "Coverage %": r["coverage"] if r["coverage"] is not None else "—",
            "Span": span,
            "Events/h": round(r["rate"], 3) if r["rate"] is not None else "—",
            "Rec start": r["rec_start"],
            "Valid until": r["valid_until"],
            "Flag": r["flag"],
            # hidden
            "_animal": r["animal"],
            "Excluded": r["excluded"],
        }
        if is_seizure:
            row["Conv"] = r["conv"]
            row["Non-conv"] = r["nonconv"]
        rows.append(row)

    cols = [
        {"field": "Include", "width": 90, "editable": True,
         "cellDataType": "boolean",
         "headerTooltip": "Untick to drop this animal from group "
                          "comparison, rates and plots"},
        {"field": "Animal", "width": 100},
        {"field": "Group", "width": 100},
        {"field": "Cohort", "width": 100},
        {"field": "Files", "width": 70},
    ]
    if is_seizure:
        cols += [{"field": "Conv", "width": 70},
                 {"field": "Non-conv", "width": 85}]
    cols += [
        {"field": "Events", "width": 80},
        {"field": "Mean dur (s)", "width": 110},
        {"field": "Rec days", "width": 90},
        {"field": "Rec h", "width": 75},
        {"field": "Coverage %", "width": 100},
        {"field": "Span", "width": 160},
        {"field": "Events/h", "width": 85},
        {"field": "Rec start", "width": 115, "editable": True,
         "headerTooltip": "Day-1 reference for the longitudinal plot "
                          "(YYYY-MM-DD); blank = earliest recorded date"},
        {"field": "Valid until", "width": 110, "editable": True,
         "headerTooltip": "Censor after this date (YYYY-MM-DD); clear to undo"},
        {"field": "Flag", "width": 180},
        {"field": "_animal", "hide": True},
        {"field": "Excluded", "hide": True},
    ]
    return dag.AgGrid(
        id="res-animal-grid",
        rowData=rows, columnDefs=cols,
        defaultColDef={
            "sortable": True, "filter": True, "resizable": True,
            # Grey out rows for excluded animals (kept visible, toggleable).
            "cellStyle": {"styleConditions": [
                {"condition": "params.data.Excluded",
                 "style": {"color": "#6e7681"}}]},
        },
        style={"height": f"{min(60 + len(rows) * 42, 380)}px"},
        dashGridOptions={"animateRows": False},
        className="ag-theme-alpine-dark",
    )


def _hist_bar(values, color, nbins: int = 60) -> go.Bar:
    """Server-side histogram as a go.Bar — ships ~nbins bars instead of the raw
    values. go.Histogram bins client-side, so at ~2M spikes it would serialise
    tens of MB to the browser and freeze the tab; this keeps it to a few KB."""
    arr = np.asarray(values, dtype=float)
    counts, edges = np.histogram(arr, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]),
                  marker_color=color, showlegend=False)


def _build_duration_hist(events, is_seizure) -> go.Figure:
    fig = go.Figure()
    durs = [float(e["duration_sec"]) for e in events if e.get("duration_sec")]
    if not durs:
        apply_fig_theme(fig)
        fig.update_layout(title="Event duration")
        return fig
    if is_seizure:
        conv = [float(e["duration_sec"]) for e in events
                if e.get("type") == "convulsive" and e.get("duration_sec")]
        nonconv = [float(e["duration_sec"]) for e in events
                   if e.get("type") != "convulsive" and e.get("duration_sec")]
        fig.add_trace(go.Histogram(x=conv, name="Convulsive",
                                   marker_color=_CONV_COLOR, opacity=0.65))
        fig.add_trace(go.Histogram(x=nonconv, name="Non-convulsive",
                                   marker_color=_NONCONV_COLOR, opacity=0.65))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(_hist_bar(durs, _NONCONV_COLOR))
    fig.update_layout(title="Event duration", xaxis_title="Duration (s)",
                      yaxis_title="Count", legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


def _build_duration_by_group(events, is_seizure) -> go.Figure:
    """Box plot of event duration by group, split by event type.

    For seizures: grouped boxes per group (Convulsive vs Non-convulsive), so
    you can compare each type's duration both within and across groups. For
    spikes (single type): one box per group.
    """
    fig = go.Figure()
    rows = [(e.get("group_id") or "(unlabeled)",
             "Convulsive" if e.get("type") == "convulsive" else "Non-convulsive",
             float(e["duration_sec"]))
            for e in events if e.get("duration_sec")]
    if not rows:
        apply_fig_theme(fig)
        fig.update_layout(title="Event duration by group")
        return fig

    groups = sorted({r[0] for r in rows})
    if is_seizure:
        for tname, color in (("Convulsive", _CONV_COLOR),
                             ("Non-convulsive", _NONCONV_COLOR)):
            xs = [r[0] for r in rows if r[1] == tname]
            ys = [r[2] for r in rows if r[1] == tname]
            if ys:
                fig.add_trace(go.Box(x=xs, y=ys, name=tname, marker_color=color,
                                     boxmean=True))
        fig.update_layout(boxmode="group")
        title = "Event duration by group & type"
    else:
        # Precompute quartiles server-side and pass them to go.Box (not raw y):
        # at ~2M spikes, raw y would serialise tens of MB to the browser.
        for i, g in enumerate(groups):
            ys = np.asarray([r[2] for r in rows if r[0] == g], dtype=float)
            if ys.size == 0:
                continue
            q1, med, q3 = (float(v) for v in np.percentile(ys, [25, 50, 75]))
            iqr = q3 - q1
            lo = float(ys[ys >= q1 - 1.5 * iqr].min())
            hi = float(ys[ys <= q3 + 1.5 * iqr].max())
            fig.add_trace(go.Box(
                name=g, q1=[q1], median=[med], q3=[q3],
                lowerfence=[lo], upperfence=[hi], mean=[float(ys.mean())],
                boxmean=True,
                marker_color=_GROUP_PALETTE[i % len(_GROUP_PALETTE)]))
        title = "Spike duration by group"
    fig.update_layout(title=title, xaxis_title="Group",
                      yaxis_title="Duration (s)",
                      legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


def _build_confidence_hist(events, is_seizure) -> go.Figure:
    fig = go.Figure()
    confs = [float(e["cnn_confidence"]) for e in events
             if e.get("cnn_confidence") is not None]
    if not confs:
        apply_fig_theme(fig)
        fig.update_layout(title="Confidence")
        return fig
    if is_seizure:
        conv = [float(e["cnn_confidence"]) for e in events
                if e.get("type") == "convulsive"
                and e.get("cnn_confidence") is not None]
        nonconv = [float(e["cnn_confidence"]) for e in events
                   if e.get("type") != "convulsive"
                   and e.get("cnn_confidence") is not None]
        fig.add_trace(go.Histogram(x=conv, name="Convulsive",
                                   marker_color=_CONV_COLOR, opacity=0.65))
        fig.add_trace(go.Histogram(x=nonconv, name="Non-convulsive",
                                   marker_color=_NONCONV_COLOR, opacity=0.65))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(_hist_bar(confs, _NONCONV_COLOR))
    fig.update_layout(title="Detector confidence", xaxis_title="Confidence",
                      yaxis_title="Count", legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


def _build_longitudinal(events, starts, fa_rows=None, normalize=None) -> go.Figure:
    """Per recording-day distribution across animals, grouped by group.

    For each recording day (day 1 = each animal's own recording start) and
    group, a bar shows the MEAN per-animal value and the individual animals are
    overlaid as points. The per-animal value is that animal's event count that
    day (raw) or events per animal-hour that day (``normalize == "per_hour"``);
    animals that recorded that day with no events count as 0 so means aren't
    inflated. Cohorts with different calendar starts align on a common day 1."""
    from datetime import date as _date

    fig = go.Figure()

    def _parse(d):
        try:
            return _date.fromisoformat(d)
        except (ValueError, TypeError):
            return None

    start_dt = {a: _parse(s) for a, s in (starts or {}).items()}
    per_hour = normalize == "per_hour"
    # Per-(animal, day) event counts and recorded hours.
    group_of: dict = {}
    counts: dict = {}
    for e in events:
        dt = _parse(_ev_date(e))
        a = e.get("animal_id") or ""
        s = start_dt.get(a)
        if dt is None or s is None:
            continue
        di = (dt - s).days + 1
        counts[(a, di)] = counts.get((a, di), 0) + 1
        if e.get("group_id"):
            group_of[a] = e["group_id"]
    hours: dict = {}
    if fa_rows:
        for r in fa_rows:
            dt = _parse(r.get("date"))
            a = r.get("animal_id") or ""
            s = start_dt.get(a)
            if dt is None or s is None:
                continue
            di = (dt - s).days + 1
            hours[(a, di)] = hours.get((a, di), 0.0) + (r.get("valid_sec") or 0) / 3600.0
            group_of.setdefault(a, r.get("group_id") or "")

    def _by_group_day(use_hour):
        """(group, day) -> list of per-animal values; 0 for recorded-but-empty
        animal-days. Universe is recording presence (hours) when available."""
        res: dict = {}
        universe = (set(hours) if (use_hour and hours)
                    else (set(hours) | set(counts)))
        for (a, di) in universe:
            h = hours.get((a, di), 0.0)
            n = counts.get((a, di), 0)
            if use_hour:
                if h <= 0:
                    continue
                v = n / h
            else:
                v = n
            g = group_of.get(a) or "(unlabeled)"
            res.setdefault((g, di), []).append(v)
        return res

    by_gd = _by_group_day(per_hour)
    # Fall back to raw counts if per-animal-hour yields nothing (no hour data).
    if per_hour and not by_gd:
        per_hour = False
        by_gd = _by_group_day(False)

    if not by_gd:
        apply_fig_theme(fig)
        fig.update_layout(title="Longitudinal (no date data)",
                          xaxis_title="Recording day", yaxis_title="Events")
        return fig

    groups = sorted({g for (g, _) in by_gd})
    days = sorted({di for (_, di) in by_gd})
    n_g = max(len(groups), 1)
    bw = 0.8 / n_g  # width allotted to each group within a day slot

    for gi, g in enumerate(groups):
        off = (gi - (n_g - 1) / 2.0) * bw
        color = _GROUP_PALETTE[gi % len(_GROUP_PALETTE)]
        means = [(sum(by_gd[(g, d)]) / len(by_gd[(g, d)]))
                 if by_gd.get((g, d)) else 0.0 for d in days]
        fig.add_trace(go.Bar(
            x=[d + off for d in days], y=means, width=bw, name=g,
            marker_color=color, opacity=0.45))
        # Individual animal points, deterministic jitter within the bar.
        px, py = [], []
        for d in days:
            vals = by_gd.get((g, d), [])
            k = len(vals)
            for j, v in enumerate(vals):
                px.append(d + off + (((j + 1) / (k + 1)) - 0.5) * bw * 0.7)
                py.append(v)
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="markers", name=g, showlegend=False,
            marker=dict(color=color, size=5,
                        line=dict(width=0.5, color="rgba(0,0,0,0.45)"))))

    fig.update_layout(
        barmode="overlay",
        title="Longitudinal — per-animal "
              + ("events/animal-hour" if per_hour else "events")
              + " by recording day (bar = group mean, dots = animals)",
        xaxis_title="Recording day",
        yaxis_title="Events / animal-hour" if per_hour else "Events / animal",
        legend=dict(orientation="h", y=1.1))
    apply_fig_theme(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Events table
# ═══════════════════════════════════════════════════════════════════════


def _build_events_table(events: list[dict], filters_active: bool = False):
    tag = []
    if filters_active:
        tag = [html.Span(
            "● Filters applied — table shows the filtered subset",
            style={"fontSize": "0.78rem", "color": "var(--ned-accent)",
                   "fontWeight": "600", "marginBottom": "6px",
                   "display": "inline-block"})]
    if not events:
        return html.Div(tag + [html.P(
            "No events found.",
            style={"color": "var(--ned-text-muted)", "fontSize": "0.85rem"})])

    rows = []
    for ev in events[:500]:
        edf_path = ev.get("path", "")
        rows.append({
            "Exclude": bool(ev.get("excluded")),
            "Animal": ev.get("animal_id", ""),
            "File": Path(edf_path).name if edf_path else "",
            "Date": ev.get("date", ev.get("chunk_date", "")),
            "Start (s)": round(ev.get("start_sec", 0), 1),
            "End (s)": round(ev.get("end_sec", 0), 1),
            "Duration": round(ev.get("duration_sec", 0), 1),
            "Type": ev.get("type", ""),
            "Detector": _DETECTOR_LABELS.get(ev.get("source", ""),
                                             ev.get("source", "")),
            "Group": ev.get("group_id", "") or "",
            "Cohort": ev.get("cohort", "") or "",
            "Subtype": ev.get("subtype", "") or "",
            "Confidence": round(ev.get("cnn_confidence", 0), 3),
            "Conv %": round((ev.get("convulsive_confidence") or 0) * 100, 0),
            "Flagged": "Yes" if ev.get("movement_flag") else "",
            "Hour": ev.get("hour_of_day", ""),
            "Ch": ev.get("channel") if ev.get("channel") is not None else "",
            "Mode": ev.get("mode", ""),
            # Hidden fields for inspector / exclude toggle
            "_path": edf_path,
            # Real per-event EDF channel; older events (NULL) fall back to the
            # first EEG channel in the inspector.
            "_channel_idx": ev.get("channel") if ev.get("channel") is not None else 0,
            "_event_id": ev.get("id"),
        })

    columns = [
        {"field": "Exclude", "width": 90, "editable": True,
         "cellDataType": "boolean",
         "headerTooltip": "Exclude from summaries, plots and export"},
        {"field": "Animal", "width": 80},
        {"field": "File", "width": 150},
        {"field": "Date", "width": 95},
        {"field": "Start (s)", "width": 80},
        {"field": "End (s)", "width": 80},
        {"field": "Duration", "width": 75},
        {"field": "Type", "width": 100},
        {"field": "Detector", "width": 130},
        {"field": "Group", "width": 100},
        {"field": "Cohort", "width": 100},
        {"field": "Subtype", "width": 75},
        {"field": "Confidence", "width": 90},
        {"field": "Conv %", "width": 65},
        {"field": "Flagged", "width": 65},
        {"field": "Hour", "width": 50},
        {"field": "Ch", "width": 55},
        {"field": "Mode", "width": 65},
        # Hidden columns
        {"field": "_path", "hide": True},
        {"field": "_channel_idx", "hide": True},
        {"field": "_event_id", "hide": True},
    ]

    grid = dag.AgGrid(
        id="res-grid",
        rowData=rows,
        columnDefs=columns,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        style={"height": "400px"},
        dashGridOptions={
            "rowSelection": "single",
            "animateRows": False,
        },
        className="ag-theme-alpine-dark",
    )
    return html.Div(tag + [grid]) if tag else grid


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _minmax_downsample(time_axis, data, max_points=6000):
    """Downsample for display keeping min/max per bin."""
    n = len(data)
    if n <= max_points:
        return time_axis, data
    bin_size = n // (max_points // 2)
    n_bins = n // bin_size
    out_t, out_d = [], []
    for i in range(n_bins):
        s = i * bin_size
        e = min(s + bin_size, n)
        seg = data[s:e]
        idx_min = s + np.argmin(seg)
        idx_max = s + np.argmax(seg)
        if idx_min < idx_max:
            out_t.extend([time_axis[idx_min], time_axis[idx_max]])
            out_d.extend([data[idx_min], data[idx_max]])
        else:
            out_t.extend([time_axis[idx_max], time_axis[idx_min]])
            out_d.extend([data[idx_max], data[idx_min]])
    return np.array(out_t), np.array(out_d)


# ═══════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════


@callback(
    Output("res-export-status", "children"),
    Input("res-export-csv", "n_clicks"),
    State("res-source-selector", "value"),
    State("res-detector-filter", "value"),
    State("res-date-start", "value"),
    State("res-date-end", "value"),
    State("res-mode-filter", "value"),
    State("res-animal-filter", "value"),
    State("res-type-filter", "value"),
    State("res-min-conf", "value"),
    State("res-file-filter", "value"),
    State("res-cohort-filter", "value"),
    State("res-group-filter", "value"),
    prevent_initial_call=True,
)
def export_csv(n, source, detector, date_start, date_end, modes, animals,
               types, min_conf, file_ids, cohort, group_id):
    if not n:
        return no_update

    # Mirror the displayed table's filters exactly so the CSV == what you see.
    category = "spike" if source == "spike_cnn" else "seizure"
    animal_id = animals[0] if animals and len(animals) == 1 else None
    event_type = types[0] if types and len(types) == 1 else None
    min_confidence = float(min_conf) if min_conf and float(min_conf) > 0 else None
    filter_kw = {
        "date_start": date_start or None,
        "date_end": date_end or None,
        "animal_id": animal_id,
        "min_confidence": min_confidence,
        "event_type": event_type,
        "category": category,
        "source": detector or None,
        "cohort": cohort or None,
        "group_id": group_id or None,
    }
    if modes and len(modes) < 3:
        filter_kw["mode"] = modes[0] if len(modes) == 1 else None

    events = db.get_events(**filter_kw)
    if file_ids:
        chunk_ids = {int(fid) for fid in file_ids}
        events = [e for e in events if e.get("chunk_id") in chunk_ids]
    if animals and len(animals) > 1:
        events = [e for e in events if e.get("animal_id") in animals]
    if types and len(types) < 2:
        events = [e for e in events if e.get("type") in types]
    # Excluded events are dropped from exports too.
    events = [e for e in events if not e.get("excluded")]
    if not events:
        return html.Span("No events to export.", style={"color": "#d29922"})

    # The in-window webview can't do browser downloads — write to a chosen path.
    path = _save_file("events_filtered.csv", "Export filtered events CSV")
    if not path:
        return no_update
    return _write_events_csv(events, path)


# Raw per-event columns for CSV export (Prism-friendly: one row per event).
_EVENT_EXPORT_FIELDS = [
    "id", "animal_id", "category", "type", "subtype",
    "date", "recording_day", "hour_of_day",
    "start_sec", "end_sec", "duration_sec",
    "cnn_confidence", "convulsive_confidence", "movement_flag",
    "channel", "cohort", "group_id", "source", "mode", "excluded", "path",
]


def _write_events_csv(events, path):
    """Write a list of event dicts to ``path`` as one row per event."""
    import csv
    try:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_EVENT_EXPORT_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            for ev in events:
                writer.writerow(ev)
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "var(--ned-danger)"})
    return html.Span(f"Exported {len(events)} event(s) to {path}",
                     style={"color": "#2ea043"})


@callback(
    Output("res-export-status", "children", allow_duplicate=True),
    Input("res-export-all", "n_clicks"),
    State("res-project-select", "value"),
    prevent_initial_call=True,
)
def export_all_events(n, project):
    """Export every event in the active project DB (all categories, ignoring
    the current filters; includes excluded + category columns) so the complete
    raw dataset can be analysed externally."""
    if not n:
        return no_update
    if project and project != db.get_active_project():
        db.set_active_project(project)
    events = db.get_events()  # no filters → all events, both categories
    if not events:
        return html.Span("No events in this database.", style={"color": "#d29922"})
    path = _save_file("events_all.csv", "Export ALL events CSV")
    if not path:
        return no_update
    return _write_events_csv(events, path)


@callback(
    Output("res-export-status", "children", allow_duplicate=True),
    Input("res-export-graphdata", "n_clicks"),
    State("res-source-selector", "value"),
    State("res-detector-filter", "value"),
    State("res-date-start", "value"),
    State("res-date-end", "value"),
    State("res-mode-filter", "value"),
    State("res-animal-filter", "value"),
    State("res-type-filter", "value"),
    State("res-min-conf", "value"),
    State("res-file-filter", "value"),
    State("res-cohort-filter", "value"),
    State("res-group-filter", "value"),
    prevent_initial_call=True,
)
def export_graph_data(n, source, detector, date_start, date_end, modes, animals,
                      types, min_conf, file_ids, cohort, group_id):
    """Export the computed values BEHIND the Results graphs as a multi-sheet
    XLSX under the current filters, for stats in Prism. Sheets: PerAnimal /
    PerGroup (per-subject stats tables), Daily_counts_byGroup / Daily_rate_byGroup
    / Circadian_byGroup (WIDE grouped tables — rows = day/hour, columns = animals
    side by side under their group), and Duration_perAnimal."""
    if not n:
        return no_update
    from datetime import date as _date

    # Reproduce the same filtered + censored views the graphs use.
    category = "spike" if source == "spike_cnn" else "seizure"
    animal_id = animals[0] if animals and len(animals) == 1 else None
    event_type = types[0] if types and len(types) == 1 else None
    min_confidence = float(min_conf) if min_conf and float(min_conf) > 0 else None
    mode = (modes[0] if modes and len(modes) == 1 else None)
    filter_kw = dict(date_start=date_start or None, date_end=date_end or None,
                     animal_id=animal_id, min_confidence=min_confidence,
                     event_type=event_type, category=category,
                     source=detector or None, cohort=cohort or None,
                     group_id=group_id or None)
    if modes and len(modes) < 3:
        filter_kw["mode"] = mode
    events = db.get_events(**filter_kw)
    fa = db.get_file_animals(
        date_start=date_start or None, date_end=date_end or None,
        animal_id=animal_id, mode=(mode if modes and len(modes) < 3 else None),
        cohort=cohort or None, group_id=group_id or None)
    if file_ids:
        chunk_ids = {int(fid) for fid in file_ids}
        events = [e for e in events if e.get("chunk_id") in chunk_ids]
        fa = [r for r in fa if r.get("chunk_id") in chunk_ids]
    if animals and len(animals) > 1:
        events = [e for e in events if e.get("animal_id") in animals]
        fa = [r for r in fa if r.get("animal_id") in animals]
    if types and len(types) < 2:
        events = [e for e in events if e.get("type") in types]

    status = db.get_animal_status()
    agg = _active_events(events)
    agg, fa = _apply_censor(agg, fa, status)
    excluded = {a for a, s in status.items() if s.get("excluded")}
    vis_events = [e for e in agg if (e.get("animal_id") or "") not in excluded]
    vis_fa = [r for r in fa if (r.get("animal_id") or "") not in excluded]
    if not vis_events:
        return html.Span("No events to export.", style={"color": "#d29922"})

    animal_rollup = _animal_rollup(agg, fa, status)
    group_rollup = _group_rollup(vis_events, vis_fa)
    starts = {r["animal"]: r["rec_start"]
              for r in animal_rollup if r.get("rec_start")}

    def _pf(d):
        try:
            return _date.fromisoformat(d)
        except (ValueError, TypeError):
            return None

    # PerAnimal — one row per animal (the Prism stats table: group as factor).
    per_animal = []
    for r in animal_rollup:
        h = r.get("rec_hours") or 0
        per_animal.append({
            "animal": r["animal"], "group": r.get("primary_group", ""),
            "cohort": r.get("cohorts", ""), "rec_hours": round(h, 2),
            "rec_days": r.get("rec_days", 0), "events": r["n"],
            "convulsive": r["conv"], "non_convulsive": r["nonconv"],
            "rate_per_h": round(r["n"] / h, 4) if h else None,
            "conv_rate_per_h": round(r["conv"] / h, 4) if h else None,
            "nonconv_rate_per_h": round(r["nonconv"] / h, 4) if h else None,
            "mean_duration_s": round(r.get("mean_dur", 0), 3),
            "excluded": r.get("excluded", False),
        })
    # PerGroup — group-level aggregate (the group-comparison bars).
    per_group = [{
        "group": r["group"], "animals": r["n_animals"],
        "rec_hours": round(r.get("rec_hours") or 0, 2), "events": r["n"],
        "convulsive": r["conv"], "non_convulsive": r["nonconv"],
        "rate_per_h": round(r["rate"], 4) if r["rate"] is not None else None,
        "conv_rate_per_h": round(r["conv_rate"], 4) if r["conv_rate"] is not None else None,
        "nonconv_rate_per_h": round(r["nonconv_rate"], 4) if r["nonconv_rate"] is not None else None,
    } for r in group_rollup]
    # animal -> group (for the per-animal detail sheets).
    agrp = {r["animal"]: r.get("primary_group", "") for r in animal_rollup}

    # Daily (per animal × recording day): one row per animal per day, so each
    # animal is a subject for repeated-measures stats / per-animal trajectories.
    dc, dh = {}, {}
    for e in vis_events:
        dt, s = _pf(_ev_date(e)), _pf(starts.get(e.get("animal_id") or ""))
        if dt and s:
            dc[(e.get("animal_id") or "", (dt - s).days + 1)] = \
                dc.get((e.get("animal_id") or "", (dt - s).days + 1), 0) + 1
    for r in vis_fa:
        dt, s = _pf(r.get("date")), _pf(starts.get(r.get("animal_id") or ""))
        if dt and s:
            k = (r.get("animal_id") or "", (dt - s).days + 1)
            dh[k] = dh.get(k, 0.0) + (r.get("valid_sec") or 0) / 3600.0
    # Circadian counts (per animal × hour of day).
    cc = {}
    for e in vis_events:
        hr = e.get("hour_of_day")
        if hr is not None:
            k = (e.get("animal_id") or "", int(hr))
            cc[k] = cc.get(k, 0) + 1

    # Wide "grouped" tables for Prism: rows = recording day / hour of day,
    # columns = animals ordered by group (same-group animals side by side), with
    # two header rows (group, then animal). Far quicker to paste than the old
    # one-row-per-datapoint layout.
    def _wide(cells, xs, x_label):
        anims = sorted(agrp, key=lambda a: (agrp.get(a) or "~", a))
        anims = [a for a in anims if any((a, x) in cells for x in xs)]
        rows = [[""] + [agrp.get(a, "") for a in anims],
                [x_label] + anims]
        for x in xs:
            rows.append([x] + [cells.get((a, x)) for a in anims])
        return rows

    days = sorted({di for (_a, di) in (set(dc) | set(dh))})
    daily_rate = {k: round(dc.get(k, 0) / dh[k], 4)
                  for k in (set(dc) | set(dh)) if dh.get(k)}
    daily_counts_wide = _wide(dc, days, "recording_day")
    daily_rate_wide = _wide(daily_rate, days, "recording_day")
    circadian_wide = _wide(cc, list(range(24)), "hour_of_day")

    # Interictal-spike ISI distribution + CDF — pre-binned (the user can't
    # reasonably bin ~2M intervals in Prism), using the same log-spaced bins as
    # the plots. Columns = group (the plots pool animals within a group).
    isi_dist_wide, isi_cdf_wide = [], []
    if category == "spike":
        isis = _isis_by_group(vis_events)
        edges, centers, dens = _isi_distribution_table(isis)
        dist_groups = sorted(dens)
        if dist_groups:
            # Density (1/s) per group, with bin edges so the binning is explicit.
            isi_dist_wide.append(
                ["ISI_low_s", "ISI_high_s", "ISI_center_s"] + dist_groups)
            for i, c in enumerate(centers):
                isi_dist_wide.append(
                    [round(float(edges[i]), 4), round(float(edges[i + 1]), 4),
                     round(float(c), 4)]
                    + [round(float(dens[g][i]), 6) for g in dist_groups])
        cdf = _isi_cdf_table(isis, edges)
        cdf_groups = sorted(cdf)
        if cdf_groups:
            isi_cdf_wide.append(["ISI_s"] + cdf_groups)
            for j, x in enumerate(edges):
                isi_cdf_wide.append(
                    [round(float(x), 4)]
                    + [round(float(cdf[g][j]), 6) for g in cdf_groups])

    # Duration (per animal): mean/median per animal, split by event type — one
    # value per animal so durations can be compared statistically by group.
    import statistics
    dba = {}
    for e in vis_events:
        d = e.get("duration_sec")
        if not d:
            continue
        rec = dba.setdefault(e.get("animal_id") or "",
                             {"conv": [], "nonconv": [], "all": []})
        rec["all"].append(float(d))
        (rec["conv"] if e.get("type") == "convulsive"
         else rec["nonconv"]).append(float(d))

    def _mean(lst):
        return round(sum(lst) / len(lst), 3) if lst else None

    durations = []
    for a in sorted(dba):
        rec = dba[a]
        durations.append({
            "animal": a, "group": agrp.get(a, ""),
            "n_conv": len(rec["conv"]), "n_nonconv": len(rec["nonconv"]),
            "mean_dur_conv_s": _mean(rec["conv"]),
            "mean_dur_nonconv_s": _mean(rec["nonconv"]),
            "mean_dur_all_s": _mean(rec["all"]),
            "median_dur_all_s": (round(statistics.median(rec["all"]), 3)
                                 if rec["all"] else None),
        })

    path = _save_file("results_graph_data.xlsx", "Export graph data (XLSX)")
    if not path:
        return no_update
    try:
        import openpyxl
        wb = openpyxl.Workbook()

        def _write_dict(ws, rows):  # one row per dict, keys as the header
            if rows:
                cols = list(rows[0].keys())
                ws.append(cols)
                for row in rows:
                    ws.append([row.get(c) for c in cols])
            else:
                ws.append(["(no data)"])

        def _write_raw(ws, rows):  # pre-built wide rows (list of lists)
            for row in (rows or [["(no data)"]]):
                ws.append(row)

        # (name, kind, data). "dict" = per-animal/group stats tables; "raw" =
        # wide grouped tables (animals side by side under their group).
        sheets = [
            ("PerAnimal", "dict", per_animal),
            ("PerGroup", "dict", per_group),
            ("Daily_counts_byGroup", "raw", daily_counts_wide),
            ("Daily_rate_byGroup", "raw", daily_rate_wide),
            ("Circadian_byGroup", "raw", circadian_wide),
            ("Duration_perAnimal", "dict", durations),
        ]
        if category == "spike":
            sheets += [("ISI_distribution", "raw", isi_dist_wide),
                       ("ISI_CDF", "raw", isi_cdf_wide)]
        first = True
        for title, kind, rows in sheets:
            ws = wb.active if first else wb.create_sheet()
            ws.title = title
            first = False
            (_write_dict if kind == "dict" else _write_raw)(ws, rows)
        wb.save(path)
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "var(--ned-danger)"})
    return html.Span(
        f"Exported graph data ({len(per_animal)} animals, "
        f"{len(days)} days) to {path}", style={"color": "#2ea043"})
