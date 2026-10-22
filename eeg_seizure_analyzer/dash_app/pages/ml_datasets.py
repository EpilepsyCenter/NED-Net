"""ML Dataset Builder — curate annotation datasets for model training."""

from __future__ import annotations

import os
import re
import subprocess
import sys

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert, loading, metric_card
from eeg_seizure_analyzer.io.dataset_store import (
    scan_annotation_files,
    save_dataset,
    load_dataset,
    list_datasets,
    delete_dataset,
)
from eeg_seizure_analyzer.io.channel_ids import load_channel_ids


# ── Layout ───────────────────────────────────────────────────────────


def _param_label(text: str, tip: str | None = None,
                 font_size: str = "0.78rem") -> html.Label:
    """A muted field label with an optional ``(?)`` hover tooltip.

    Matches the convention used by ``components.param_control`` (native HTML
    ``title`` on a help-cursor span) so the explanation appears on hover.
    """
    children = [text]
    if tip:
        children.append(html.Span(
            " (?)", title=tip,
            style={"cursor": "help", "opacity": "0.5"},
        ))
    return html.Label(children,
                      style={"fontSize": font_size, "color": "var(--ned-text-muted)"})


# Background source choices shown in the UI (value = DatasetConfig.background_source,
# label, what it needs: None, "proposed" = any background stretch, "confirmed").
_BG_SOURCE_CHOICES = [
    ("random", "Random background (default)", None),
    ("none", "Rejected events only", None),
    ("proposed", "Proposed background (unreviewed)", "proposed"),
    ("confirmed", "Confirmed background", "confirmed"),
    ("confirmed+random", "Confirmed background + random", "confirmed"),
]
# What we know about each option (internal tests, 2026-10). Kept honest: one
# cohort, one or two trainings each.
_BG_SOURCE_HELP = (
    "Besides your rejected events, the model also needs examples of normal EEG. "
    "This sets where they come from."
)
_BG_SOURCE_OPTION_HELP = {
    "random": (
        "What the current models use, and the safest choice. Random stretches of "
        "your recordings are used as normal EEG. Some may contain seizures nobody "
        "annotated, which teaches the model to miss a few of them. Best when the "
        "model will run on animals you haven't annotated."),
    "none": (
        "Only your rejected events are used as negatives. The model flags more: "
        "with a lower threshold it can find more of your seizures than random "
        "background allows, but it produces many more detections to review. Its "
        "best threshold varies a lot between trainings, so check it on your own "
        "recordings. Not recommended for animals you haven't annotated. Needs a "
        "good number of rejected events."),
    "proposed": (
        "Uses the quiet stretches proposed on the Seizures page (Propose background) "
        "instead of random time, whether or not you reviewed them; your rejected "
        "background stretches are left out. In our tests it found as many seizures as "
        "random background at normal numbers of detections, and more when more "
        "detections were allowed. Confirmed background is the cleaner version of the "
        "same stretches."),
    "confirmed": (
        "Uses only the background stretches you confirmed (Training > Background): "
        "the proposed stretches, checked by you, so no hidden seizures among the "
        "negatives. The unreviewed version tested as safe in our tests; this one needs "
        "confirmed stretches from every animal, asleep and awake."),
    "confirmed+random": (
        "Uses your confirmed background first, then random stretches to make up the "
        "number. A middle ground while you have few confirmed stretches."),
}


def _bg_source_options(has_confirmed: bool, has_proposed: bool = False) -> list[dict]:
    """Select options; the background-file ones are disabled without the stretches
    they need."""
    have = {"confirmed": has_confirmed, "proposed": has_proposed or has_confirmed}
    opts = []
    for value, label, needs in _BG_SOURCE_CHOICES:
        off = needs is not None and not have[needs]
        why = " (no confirmed background)" if needs == "confirmed" else " (no background proposed)"
        opts.append({"label": label + (why if off else ""), "value": value, "disabled": off})
    return opts


def _fmt_minutes(sec: float) -> str:
    m = (sec or 0) / 60.0
    return f"{m:.0f} min" if m >= 10 else f"{m:.1f} min"


def layout(sid: str | None) -> html.Div:
    """Return the ML dataset builder layout."""
    state = server_state.get_session(sid)

    # Restore previous folder / type from session
    prev_folder = state.extra.get("ml_folder", "")
    prev_type = state.extra.get("ml_type", "seizure")

    # Available saved datasets
    saved_ds = list_datasets()
    ds_options = [{"label": n, "value": n} for n in saved_ds]

    return html.Div(
        style={"padding": "24px", "maxWidth": "1100px"},
        children=[
            html.H4("ML Dataset Builder", style={"marginBottom": "8px"}),
            html.P(
                "Select a folder containing annotated EDF recordings to "
                "build a training dataset. Annotations are discovered "
                "automatically from files saved by the annotation pages.",
                style={"color": "var(--ned-text-muted)", "fontSize": "0.9rem",
                       "marginBottom": "24px"},
            ),

            # ── Load existing dataset ────────────────────────────
            html.Div(
                style={"display": "flex", "gap": "12px",
                       "marginBottom": "24px", "alignItems": "flex-end"},
                children=[
                    html.Div(
                        style={"flex": "1", "maxWidth": "300px"},
                        children=[
                            html.Label("Saved datasets",
                                       style={"fontSize": "0.82rem",
                                              "color": "var(--ned-text-muted)"}),
                            dcc.Dropdown(
                                id="ml-load-dropdown",
                                options=ds_options,
                                placeholder="Select a dataset...",
                                clearable=True,
                            ),
                        ],
                    ),
                    dbc.Button("Load", id="ml-load-btn",
                               className="btn-ned-secondary", size="sm"),
                    dbc.Button("Delete", id="ml-delete-btn",
                               className="btn-ned-danger", size="sm"),
                ],
            ),

            # ── Annotation type ──────────────────────────────────
            _param_label(
                "Annotation type",
                "Which sidecar labels to train on: Seizure events (for the U-Net "
                "and the convulsive classifier) or Interictal Spikes (their own model).",
                font_size="0.82rem"),
            dbc.RadioItems(
                id="ml-type-radio",
                options=[
                    {"label": "Seizure", "value": "seizure"},
                    {"label": "Interictal Spike", "value": "spike"},
                ],
                value=prev_type,
                inline=True,
                className="mb-3",
                style={"fontSize": "0.9rem"},
            ),

            # ── Folder browse + scan ─────────────────────────────
            html.Label("Recordings folder",
                       style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
            dbc.InputGroup([
                dbc.Input(
                    id="ml-folder-input",
                    placeholder="/path/to/recordings",
                    value=prev_folder,
                    type="text",
                ),
                dbc.Button("Browse", id="ml-browse-btn",
                           className="btn-ned-secondary"),
                dbc.Button("Scan", id="ml-scan-btn",
                           className="btn-ned-primary"),
            ], className="mb-3"),

            # ── Scan results ─────────────────────────────────────
            loading(
                html.Div(id="ml-scan-results"),
                type="circle", color="#58a6ff",
            ),

            # ── Summary ──────────────────────────────────────────
            html.Div(id="ml-summary", style={"marginTop": "16px"}),

            # ── Save dataset ─────────────────────────────────────
            html.Div(
                id="ml-save-area",
                style={"display": "none", "marginTop": "24px"},
                children=[
                    html.Hr(style={"borderColor": "var(--ned-border)"}),
                    html.Label("Dataset name",
                               style={"fontSize": "0.82rem",
                                      "color": "var(--ned-text-muted)"}),
                    dbc.InputGroup([
                        dbc.Input(
                            id="ml-dataset-name",
                            placeholder="e.g. Study_1",
                            type="text",
                        ),
                        dbc.Button("Save Dataset", id="ml-save-btn",
                                   className="btn-ned-primary"),
                    ], style={"maxWidth": "500px"}, className="mb-3"),
                    html.Div(id="ml-save-status"),
                ],
            ),

            # ── Train model ──────────────────────────────────────
            html.Div(
                id="ml-train-area",
                style={"display": "none", "marginTop": "8px"},
                children=[
                    html.Hr(style={"borderColor": "var(--ned-border)"}),
                    html.H5("Train Model",
                            style={"marginBottom": "12px", "color": "var(--ned-accent)"}),

                    # Model name + architecture
                    dbc.Row([
                        dbc.Col([
                            _param_label(
                                "Model name",
                                "Folder name the trained model is saved under. "
                                "Pick something recognisable, e.g. study1_v1. "
                                "Reusing a name overwrites that model.",
                                font_size="0.82rem"),
                            dbc.Input(
                                id="ml-model-name",
                                placeholder="e.g. study1_v1",
                                type="text",
                                style={"marginBottom": "12px"},
                            ),
                        ], width=4),
                        dbc.Col([
                            _param_label(
                                "Architecture",
                                "What to train. U-Net = the seizure detector "
                                "(finds candidate events). Convulsive Classifier "
                                "= Stage-2 model labelling each event convulsive "
                                "vs non-convulsive.",
                                font_size="0.82rem"),
                            dbc.RadioItems(
                                id="ml-architecture",
                                options=[
                                    {"label": "U-Net", "value": "unet"},
                                    # BENDR removed from the UI 2026-06-24: shelved after
                                    # the CUDA fine-tune verdict (event_f1 ~0 vs U-Net 0.78).
                                    # Training code + CLI (train_bendr) are retained, just
                                    # not user-selectable here.
                                    {"label": "Convulsive Classifier",
                                     "value": "convulsive"},
                                    # Event Re-ranker removed from the UI 2026-06-26:
                                    # human spot-check (out-of-sample wk4-6) showed it
                                    # does NOT generalise as a precision filter — it
                                    # confidently dropped real seizures (94% of its
                                    # rejects were real). Training code/CLI retained
                                    # (could be retrained with hard negatives), just
                                    # not user-selectable. Detection uses U-Net +
                                    # hysteresis boundaries instead.
                                ],
                                value="unet",
                                inline=True,
                                style={"fontSize": "0.82rem", "marginBottom": "12px"},
                            ),
                        ], width=4),
                    ], className="g-2"),

                    # BENDR-specific training params (shown/hidden)
                    html.Div(
                        id="ml-bendr-train-params",
                        style={"display": "none"},
                        children=[
                            dbc.Row([
                                dbc.Col([
                                    _param_label(
                                        "Encoder LR",
                                        "Learning rate for the pre-trained BENDR "
                                        "encoder, kept much smaller than the head "
                                        "LR so fine-tuning doesn't wipe pre-trained "
                                        "features."),
                                    dbc.Input(
                                        id="ml-encoder-lr", type="text",
                                        value="0.00001",
                                        className="form-control", size="sm",
                                    ),
                                ], width=2),
                                dbc.Col([
                                    _param_label(
                                        "Freeze encoder (epochs)",
                                        "Train only the decoder head for this many "
                                        "epochs before unfreezing the encoder. Lets "
                                        "the head settle first so early gradients "
                                        "don't corrupt pre-trained weights."),
                                    dbc.Input(
                                        id="ml-freeze-epochs", type="text",
                                        value="5",
                                        className="form-control", size="sm",
                                    ),
                                ], width=2),
                                dbc.Col([
                                    _param_label(
                                        "Pre-trained weights",
                                        "Self-supervised BENDR checkpoint (.pt) to "
                                        "start from, found in "
                                        "~/.eeg_seizure_analyzer/pretrained/. None = "
                                        "train from scratch."),
                                    dcc.Dropdown(
                                        id="ml-pretrained-weights",
                                        options=[],
                                        placeholder="None (train from scratch)",
                                        style={"fontSize": "0.82rem"},
                                    ),
                                ], width=4),
                            ], className="g-2 mb-3"),
                            dbc.Checklist(
                                id="ml-freeze-backbone",
                                options=[{
                                    "label": " Freeze backbone — train decoder "
                                             "head only (recommended for small "
                                             "datasets; curbs overfitting)",
                                    "value": "freeze",
                                }],
                                value=[],
                                switch=True,
                                style={"fontSize": "0.82rem",
                                       "marginBottom": "8px"},
                            ),
                        ],
                    ),

                    # Config row — gradient-training hyperparameters. Hidden for
                    # the Event Re-ranker (tabular fit, no epochs/batch/LR/etc.).
                    html.Div(id="ml-train-hyperparams", children=[
                    dbc.Row([
                        dbc.Col([
                            _param_label(
                                "Epochs",
                                "Maximum passes over the training set. Training can "
                                "stop earlier once validation stops improving "
                                "(see Patience)."),
                            dbc.Input(id="ml-epochs", type="text",
                                      value="50",
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            _param_label(
                                "Batch size",
                                "Number of windows per gradient step. Larger is "
                                "faster and steadier but uses more memory — lower it "
                                "if you hit out-of-memory errors."),
                            dbc.Input(id="ml-batch-size", type="text",
                                      value="8",
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            _param_label(
                                "Learning rate",
                                "Step size for weight updates. Too high diverges, too "
                                "low trains slowly. 1e-3 is a sane default for the "
                                "U-Net."),
                            dbc.Input(id="ml-lr", type="text",
                                      value="0.001",
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            _param_label(
                                "Patience",
                                "Early stopping: halt if validation loss hasn't "
                                "improved for this many epochs. The best checkpoint "
                                "is kept, not the last."),
                            dbc.Input(id="ml-patience", type="text",
                                      value="10",
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            _param_label(
                                "Pos weight",
                                "Multiplier on the positive (seizure) class in the "
                                "loss, to counter class imbalance. Higher = more "
                                "sensitive but more false positives."),
                            dbc.Input(id="ml-pos-weight", type="text",
                                      value="5.0",
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            _param_label(
                                "Neg/Pos ratio",
                                "Background (negative) windows sampled per positive "
                                "window when building the training set. Higher = more "
                                "negatives = fewer false positives, but can dilute "
                                "the seizure signal."),
                            dbc.Input(id="ml-neg-ratio", type="text",
                                      value="2.0",
                                      className="form-control", size="sm"),
                        ], width=2),
                    ], className="g-2 mb-3"),

                    # Where "no seizure" windows come from (seizure U-Net only;
                    # DatasetConfig.background_source). "proposed" is research-only
                    # and deliberately not offered here.
                    html.Div(id="ml-bg-source-wrap", children=[
                        dbc.Row([
                            dbc.Col([
                                _param_label(
                                    "Background source",
                                    "Where the negative (no seizure) windows come "
                                    "from besides your rejected events. Rejected "
                                    "background stretches are never used as "
                                    "negatives."),
                                dbc.Select(
                                    id="ml-bg-source",
                                    options=_bg_source_options(has_confirmed=False),
                                    value="random",
                                    size="sm",
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Div(
                                    _BG_SOURCE_HELP,
                                    style={"fontSize": "0.75rem",
                                           "color": "var(--ned-text-muted)",
                                           "marginTop": "18px"},
                                ),
                                html.Div(
                                    _BG_SOURCE_OPTION_HELP["random"],
                                    id="ml-bg-source-help",
                                    style={"fontSize": "0.75rem",
                                           "color": "var(--ned-text)",
                                           "marginTop": "2px"},
                                ),
                                html.Div(
                                    id="ml-bg-source-note",
                                    style={"fontSize": "0.75rem",
                                           "color": "var(--ned-text-muted)",
                                           "marginTop": "2px"},
                                ),
                            ], width=8),
                        ], className="g-2 mb-3"),
                    ]),
                    ]),  # /ml-train-hyperparams

                    # Note shown in place of the hyperparameters for the re-ranker.
                    html.Div(
                        id="ml-reranker-note",
                        children="",
                        style={"fontSize": "0.8rem", "color": "var(--ned-text-muted)",
                               "marginBottom": "12px"},
                    ),

                    dbc.Row([
                        dbc.Col([
                            _param_label(
                                "Exclude animal IDs (comma/space-separated)",
                                "Animal IDs to leave out of training entirely, so "
                                "they stay an unseen test set. Splits are per-animal, "
                                "so excluding here guarantees no leakage from that "
                                "animal into the model."),
                            dbc.Input(id="ml-exclude-animals", type="text",
                                      value="", placeholder="e.g. A12, A17",
                                      className="form-control", size="sm"),
                        ], width=6),
                    ], className="g-2 mb-3"),

                    # Train / Stop buttons
                    html.Div(
                        style={"display": "flex", "gap": "10px",
                               "alignItems": "center"},
                        className="mb-3",
                        children=[
                            dbc.Button(
                                "Start Training",
                                id="ml-train-btn",
                                style={
                                    "backgroundColor": "#238636",
                                    "border": "1px solid #2ea043",
                                    "color": "#fff",
                                    "fontWeight": "600",
                                },
                                size="lg",
                            ),
                            dbc.Button(
                                "■ Stop",
                                id="ml-stop-btn",
                                color="danger",
                                outline=True,
                                size="lg",
                                disabled=True,
                            ),
                        ],
                    ),

                    # Progress area
                    html.Div(id="ml-train-progress"),

                    # Per-epoch metrics table (live, copyable)
                    html.Div(id="ml-train-epochs"),

                    # Training results
                    html.Div(id="ml-train-results"),
                ],
            ),

            # Stores
            dcc.Store(id="ml-scan-data"),
            dcc.Store(id="ml-train-running", data=False),
            dcc.Interval(id="ml-train-poll", interval=1500, disabled=True),
        ],
    )


# ── Browse folder ────────────────────────────────────────────────────


@callback(
    Output("ml-folder-input", "value"),
    Input("ml-browse-btn", "n_clicks"),
    prevent_initial_call=True,
)
def browse_folder(n_clicks):
    """Open a native folder picker."""
    if not n_clicks:
        return no_update
    from eeg_seizure_analyzer.dash_app.pages.upload import _browse_folder
    folder = _browse_folder("Select recordings folder")
    return folder if folder else no_update


# ── Scan folder ──────────────────────────────────────────────────────


@callback(
    Output("ml-scan-results", "children"),
    Output("ml-scan-data", "data"),
    Output("ml-save-area", "style"),
    Output("ml-train-area", "style"),
    Input("ml-scan-btn", "n_clicks"),
    State("ml-folder-input", "value"),
    State("ml-type-radio", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def scan_folder(n_clicks, folder, ann_type, sid):
    """Scan the folder for annotation files and display results."""
    if not n_clicks or not folder:
        return no_update, no_update, no_update, no_update

    if not os.path.isdir(folder):
        return (
            alert(f"Folder not found: {folder}", "danger"),
            no_update,
            {"display": "none"},
            {"display": "none"},
        )

    # Persist folder + type in session
    state = server_state.get_session(sid)
    state.extra["ml_folder"] = folder
    state.extra["ml_type"] = ann_type

    results = scan_annotation_files(folder, ann_type)

    if not results:
        type_label = "seizure" if ann_type == "seizure" else "spike"
        return (
            alert(
                f"No {type_label} annotation files found in {folder}. "
                "Annotate recordings in Seizure Annotation first.",
                "warning",
            ),
            no_update,
            {"display": "none"},
            {"display": "none"},
        )

    # Check for missing Animal IDs
    files_missing_ids = []
    for r in results:
        ch_ids = load_channel_ids(r["edf_path"])
        if not ch_ids:
            files_missing_ids.append(os.path.basename(r["edf_path"]))

    is_seizure = ann_type == "seizure"

    # Build AgGrid table
    rows = []
    for r in results:
        rows.append({
            "filename": os.path.basename(r["edf_path"]),
            "edf_path": r["edf_path"],
            "confirmed": r["n_confirmed"],
            "rejected": r["n_rejected"],
            "pending": r["n_pending"],
            "total": r["n_total"],
            # convulsive / non-convulsive split (seizures only; carried in
            # row data so the selection-based summary can re-aggregate)
            "confirmed_conv": r["n_confirmed_conv"],
            "confirmed_nonconv": r["n_confirmed_nonconv"],
            "rejected_conv": r["n_rejected_conv"],
            "rejected_nonconv": r["n_rejected_nonconv"],
            # background stretches (seizures only; 0 without a background file)
            "bg_confirmed": r.get("n_bg_confirmed", 0),
            "bg_rejected": r.get("n_bg_rejected", 0),
            "bg_pending": r.get("n_bg_pending", 0),
            "bg_confirmed_sec": r.get("bg_confirmed_sec", 0.0),
            "is_seizure": is_seizure,
        })

    col_defs = [
        {
            "field": "filename",
            "headerName": "File",
            "checkboxSelection": True,
            "headerCheckboxSelection": True,
            "flex": 3,
            "minWidth": 160,
        },
        {"field": "confirmed", "headerName": "Confirmed", "flex": 1, "minWidth": 78,
         "type": "numericColumn"},
        {"field": "rejected", "headerName": "Rejected", "flex": 1, "minWidth": 78,
         "type": "numericColumn"},
        {"field": "pending", "headerName": "Pending", "flex": 1, "minWidth": 78,
         "type": "numericColumn"},
        {"field": "total", "headerName": "Total", "flex": 1, "minWidth": 78,
         "type": "numericColumn"},
        # background stretches (Training > Background); seizure datasets only
        {"field": "bg_confirmed", "headerName": "BG confirmed", "flex": 1, "minWidth": 78,
         "type": "numericColumn", "hide": not is_seizure,
         "headerTooltip": "Confirmed seizure-free background stretches"},
        {"field": "bg_rejected", "headerName": "BG rejected", "flex": 1, "minWidth": 78,
         "type": "numericColumn", "hide": not is_seizure},
        {"field": "bg_pending", "headerName": "BG pending", "flex": 1, "minWidth": 78,
         "type": "numericColumn", "hide": not is_seizure,
         "headerTooltip": "Proposed background stretches not yet reviewed"},
        {"field": "bg_confirmed_sec", "hide": True},
        # hidden — used only to re-aggregate the summary on selection change
        {"field": "confirmed_conv", "hide": True},
        {"field": "confirmed_nonconv", "hide": True},
        {"field": "rejected_conv", "hide": True},
        {"field": "rejected_nonconv", "hide": True},
        {"field": "is_seizure", "hide": True},
    ]

    grid = dag.AgGrid(
        id="ml-file-grid",
        rowData=rows,
        columnDefs=col_defs,
        defaultColDef={"sortable": True, "resizable": True,
                       "wrapHeaderText": True, "autoHeaderHeight": True},
        # Flex columns share the grid's width and refit when the window resizes;
        # each can still be dragged wider or narrower (minimum ~700 px in total).
        dashGridOptions={
            "rowSelection": {"mode": "multiRow"},
            "suppressRowClickSelection": True,
        },
        selectedRows=rows,  # select all by default
        style={"height": f"{min(80 + len(rows) * 42, 520)}px", "width": "100%"},
        className="ag-theme-alpine-dark",
    )

    # Summary for all files (updated on selection change separately)
    total_conf = sum(r["n_confirmed"] for r in results)
    total_rej = sum(r["n_rejected"] for r in results)
    total_pend = sum(r["n_pending"] for r in results)

    summary = _build_summary(
        len(results), total_conf, total_rej, total_pend,
        conf_conv=sum(r["n_confirmed_conv"] for r in results),
        conf_nonconv=sum(r["n_confirmed_nonconv"] for r in results),
        rej_conv=sum(r["n_rejected_conv"] for r in results),
        rej_nonconv=sum(r["n_rejected_nonconv"] for r in results),
        is_seizure=is_seizure,
        bg=_bg_totals(rows),
    )

    # Animal ID warning
    id_warning = []
    if files_missing_ids:
        id_warning = [alert(
            f"Animal IDs not assigned for: {', '.join(files_missing_ids)}. "
            "Load each file and fill in the Animal ID column on the Load page "
            "before training. Animal IDs are needed for proper train/validation splitting.",
            "warning",
        )]

    content = html.Div([
        html.H6(f"Found {len(results)} annotated file{'s' if len(results) != 1 else ''}",
                style={"marginBottom": "12px"}),
        *id_warning,
        grid,
        html.Div(id="ml-summary", children=summary,
                 style={"marginTop": "16px"}),
    ])

    return content, rows, {"display": "block", "marginTop": "24px"}, {"display": "block", "marginTop": "8px"}


def _bg_totals(rows: list[dict] | None) -> dict:
    """Sum the background columns of grid rows (scan rows or selected rows)."""
    rows = rows or []
    return {
        "confirmed": sum(r.get("bg_confirmed", 0) or 0 for r in rows),
        "rejected": sum(r.get("bg_rejected", 0) or 0 for r in rows),
        "pending": sum(r.get("bg_pending", 0) or 0 for r in rows),
        "confirmed_sec": sum(r.get("bg_confirmed_sec", 0) or 0 for r in rows),
    }


def _bg_summary_line(bg: dict) -> html.Div:
    """One muted line with the background-stretch totals."""
    if not (bg["confirmed"] or bg["rejected"] or bg["pending"]):
        text = ("Background: no background stretches yet (propose them in "
                "Detection, review them in Training > Background).")
    else:
        text = (f"Background: {bg['confirmed']} confirmed stretch"
                f"{'es' if bg['confirmed'] != 1 else ''} "
                f"({_fmt_minutes(bg['confirmed_sec'])}), "
                f"{bg['rejected']} rejected, {bg['pending']} pending")
    return html.Div(text, style={"fontSize": "0.82rem", "marginTop": "8px",
                                 "color": "var(--ned-text-muted)"})


def _build_summary(n_files, n_confirmed, n_rejected, n_pending,
                   conf_conv=0, conf_nonconv=0, rej_conv=0, rej_nonconv=0,
                   is_seizure=True, bg=None):
    """Build metric cards summarising the selected dataset.

    For seizures the confirmed/rejected counts are split into separate
    convulsive vs non-convulsive cards; for spikes the plain cards are used.
    For seizures a line with the background-stretch totals (``bg``, from
    ``_bg_totals``) follows the cards.
    """
    if is_seizure:
        confirmed_cards = [
            metric_card("Confirmed · Convulsive", str(conf_conv), accent=True),
            metric_card("Confirmed · Non-conv", str(conf_nonconv), accent=True),
        ]
        rejected_cards = [
            metric_card("Rejected · Convulsive", str(rej_conv)),
            metric_card("Rejected · Non-conv", str(rej_nonconv)),
        ]
    else:
        confirmed_cards = [metric_card("Confirmed", str(n_confirmed), accent=True)]
        rejected_cards = [metric_card("Rejected", str(n_rejected))]

    cards = html.Div(
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        children=[
            metric_card("Files", str(n_files)),
            *confirmed_cards,
            *rejected_cards,
            metric_card("Pending", str(n_pending)),
            metric_card("Total Events",
                        str(n_confirmed + n_rejected + n_pending)),
        ],
    )
    if not is_seizure:
        return cards
    return html.Div([cards, _bg_summary_line(bg or _bg_totals([]))])


# ── Update summary on selection change ───────────────────────────────


@callback(
    Output("ml-summary", "children", allow_duplicate=True),
    Input("ml-file-grid", "selectedRows"),
    State("ml-type-radio", "value"),
    prevent_initial_call=True,
)
def update_summary(selected_rows, ann_type):
    """Update summary metrics when file selection changes."""
    if not selected_rows:
        # Keep the cards of the chosen annotation type (was always seizure)
        return _build_summary(0, 0, 0, 0, is_seizure=(ann_type != "spike"))

    n_files = len(selected_rows)
    total_conf = sum(r.get("confirmed", 0) for r in selected_rows)
    total_rej = sum(r.get("rejected", 0) for r in selected_rows)
    total_pend = sum(r.get("pending", 0) for r in selected_rows)

    return _build_summary(
        n_files, total_conf, total_rej, total_pend,
        conf_conv=sum(r.get("confirmed_conv", 0) for r in selected_rows),
        conf_nonconv=sum(r.get("confirmed_nonconv", 0) for r in selected_rows),
        rej_conv=sum(r.get("rejected_conv", 0) for r in selected_rows),
        rej_nonconv=sum(r.get("rejected_nonconv", 0) for r in selected_rows),
        is_seizure=bool(selected_rows[0].get("is_seizure", True)),
        bg=_bg_totals(selected_rows),
    )


# ── Save dataset ─────────────────────────────────────────────────────


@callback(
    Output("ml-save-status", "children"),
    Output("ml-load-dropdown", "options"),
    Input("ml-save-btn", "n_clicks"),
    State("ml-dataset-name", "value"),
    State("ml-file-grid", "selectedRows"),
    State("ml-folder-input", "value"),
    State("ml-type-radio", "value"),
    prevent_initial_call=True,
)
def save_ds(n_clicks, name, selected_rows, folder, ann_type):
    """Save the current selection as a named dataset."""
    if not n_clicks:
        return no_update, no_update
    if not name or not name.strip():
        return alert("Please enter a dataset name.", "warning"), no_update
    if not selected_rows:
        return alert("No files selected.", "warning"), no_update

    name = name.strip()

    definition = {
        "name": name,
        "folder": folder,
        "type": ann_type,
        "files": [
            {
                "edf_path": r["edf_path"],
                "included": True,
                "n_confirmed": r.get("confirmed", 0),
                "n_rejected": r.get("rejected", 0),
                "n_pending": r.get("pending", 0),
            }
            for r in selected_rows
        ],
    }

    path = save_dataset(definition)
    updated_options = [{"label": n, "value": n} for n in list_datasets()]
    return (
        alert(f"Dataset '{name}' saved to {path}", "success"),
        updated_options,
    )


# ── Load dataset ─────────────────────────────────────────────────────


@callback(
    Output("ml-folder-input", "value", allow_duplicate=True),
    Output("ml-type-radio", "value"),
    Output("ml-save-status", "children", allow_duplicate=True),
    Output("ml-dataset-name", "value"),
    Input("ml-load-btn", "n_clicks"),
    State("ml-load-dropdown", "value"),
    prevent_initial_call=True,
)
def load_ds(n_clicks, ds_name):
    """Load a saved dataset definition — populates folder + type, then user clicks Scan."""
    if not n_clicks or not ds_name:
        return no_update, no_update, no_update, no_update

    definition = load_dataset(ds_name)
    if definition is None:
        return (
            no_update, no_update,
            alert(f"Dataset '{ds_name}' not found.", "warning"),
            no_update,
        )

    folder = definition.get("folder", "")
    ann_type = definition.get("type", "seizure")

    return (
        folder,
        ann_type,
        alert(
            f"Loaded '{ds_name}' — folder and type restored. "
            "Click Scan to refresh file list.",
            "info",
        ),
        ds_name,
    )


# ── Delete dataset ───────────────────────────────────────────────────


@callback(
    Output("ml-save-status", "children", allow_duplicate=True),
    Output("ml-load-dropdown", "options", allow_duplicate=True),
    Output("ml-load-dropdown", "value"),
    Input("ml-delete-btn", "n_clicks"),
    State("ml-load-dropdown", "value"),
    prevent_initial_call=True,
)
def delete_ds(n_clicks, ds_name):
    """Delete a saved dataset definition."""
    if not n_clicks or not ds_name:
        return no_update, no_update, no_update

    ok = delete_dataset(ds_name)
    updated_options = [{"label": n, "value": n} for n in list_datasets()]
    if ok:
        return (
            alert(f"Dataset '{ds_name}' deleted.", "info"),
            updated_options,
            None,
        )
    return (
        alert(f"Dataset '{ds_name}' not found.", "warning"),
        updated_options,
        no_update,
    )


# ── Training ────────────────────────────────────────────────────────

import json as _json
import threading as _threading
from pathlib import Path as _Path

_TRAIN_PROGRESS_DIR = _Path.home() / ".eeg_seizure_analyzer" / "cache"


def _progress_path(sid: str) -> _Path:
    return _TRAIN_PROGRESS_DIR / f"train_progress_{sid}.json"


def _write_train_progress(sid, info):
    _TRAIN_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_progress_path(sid), "w") as f:
            _json.dump(info, f)
    except Exception:
        pass


def _read_train_progress(sid) -> dict | None:
    p = _progress_path(sid)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return _json.load(f)
    except Exception:
        return None


def _stop_path(sid: str) -> _Path:
    return _TRAIN_PROGRESS_DIR / f"train_stop_{sid}.flag"


def _request_stop(sid: str) -> None:
    _TRAIN_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _stop_path(sid).write_text("stop")
    except Exception:
        pass


def _stop_requested(sid: str) -> bool:
    return _stop_path(sid).exists()


def _clear_stop(sid: str) -> None:
    try:
        _stop_path(sid).unlink()
    except OSError:
        pass


def _to_float(v, default: float) -> float:
    """Parse a (possibly text-input) value to float, falling back on default."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _to_int(v, default: int) -> int:
    try:
        return int(round(float(str(v).strip())))
    except (TypeError, ValueError):
        return default


# Columns shown in the per-epoch table and copied as TSV.
# "Seiz" = all seizures (channel 0); "Conv" = convulsive subset (channel 1).
_EPOCH_COLS = [
    ("epoch", "Epoch"), ("train_loss", "Train loss"), ("val_loss", "Val loss"),
    ("event_f1", "Seiz F1@.5"), ("event_precision", "Seiz P"),
    ("event_recall", "Seiz R"), ("best_event_f1", "Seiz Best F1"),
    ("best_threshold", "Seiz @thr"),
    ("conv_event_f1", "Conv F1@.5"), ("conv_event_recall", "Conv R"),
    ("conv_best_event_f1", "Conv Best F1"), ("conv_best_threshold", "Conv @thr"),
    ("sample_f1", "Sample F1"), ("lr", "LR"), ("elapsed_sec", "Sec"),
]


def _epoch_row_values(h: dict) -> list:
    """Flatten one history entry into the _EPOCH_COLS order."""
    m = h.get("val_metrics", {}) or {}
    out = []
    for key, _ in _EPOCH_COLS:
        if key in ("epoch", "train_loss", "val_loss", "lr", "elapsed_sec"):
            out.append(h.get(key, ""))
        else:
            out.append(m.get(key, ""))
    return out


def _fmt(key: str, v) -> str:
    if v == "" or v is None:
        return ""
    if key == "lr":
        return f"{v:.1e}"
    if key in ("best_threshold", "conv_best_threshold"):
        return f"{float(v):.2f}"
    if key in ("epoch", "elapsed_sec"):
        return str(int(round(float(v))))
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return str(v)


def _render_epochs(history: list, best_epoch: int = 0):
    """Build the per-epoch table + a Copy (TSV) button from the history list."""
    if not history:
        return None
    headers = [label for _, label in _EPOCH_COLS]
    tsv_lines = ["\t".join(headers)]
    body = []
    for h in history:
        vals = _epoch_row_values(h)
        tsv_lines.append("\t".join(
            _fmt(key, v) for (key, _), v in zip(_EPOCH_COLS, vals)))
        is_best = best_epoch and h.get("epoch") == best_epoch
        body.append(html.Tr(
            [html.Td(_fmt(key, v),
                     style={"padding": "2px 10px", "fontSize": "0.78rem",
                            "whiteSpace": "nowrap"})
             for (key, _), v in zip(_EPOCH_COLS, vals)],
            style={"backgroundColor": "rgba(46,160,67,0.15)"} if is_best else {},
        ))
    tsv = "\n".join(tsv_lines)
    head = html.Thead(html.Tr([
        html.Th(h, style={"padding": "2px 10px", "fontSize": "0.78rem",
                          "textAlign": "left",
                          "color": "var(--ned-text-muted)"})
        for h in headers]))
    return html.Div([
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "margin": "10px 0 4px 0"},
            children=[
                html.Span("Per-epoch metrics",
                          style={"fontSize": "0.82rem", "fontWeight": "600",
                                 "color": "var(--ned-text-muted)"}),
                dcc.Clipboard(
                    content=tsv, title="Copy all epochs (TSV)",
                    style={"cursor": "pointer", "fontSize": "1rem"},
                ),
            ],
        ),
        html.Div(
            html.Table([head, html.Tbody(body)],
                       style={"borderCollapse": "collapse", "width": "100%"}),
            style={"maxHeight": "260px", "overflowY": "auto",
                   "border": "1px solid var(--ned-border)",
                   "borderRadius": "4px"},
        ),
    ])


def _import_train_fn(dataset_def, train_config):
    """Resolve the training entrypoint for the chosen dataset/architecture.

    Interictal-spike datasets train through their own pipeline; convulsive /
    seizure U-Net models share train.py; the re-ranker is a tabular sklearn fit.
    All honour the same progress-dict + return contract. Kept separate so the
    import (which can fail, e.g. the re-ranker needs scikit-learn/joblib) runs
    inside the worker's try/except and surfaces as a visible error.
    """
    if dataset_def.get("type") == "spike":
        from eeg_seizure_analyzer.ml.spike_train import train_spike_model as fn
    elif train_config.architecture == "convulsive_classifier":
        from eeg_seizure_analyzer.ml.train_convulsive import train_convulsive_model as fn
    elif train_config.architecture == "reranker":
        from eeg_seizure_analyzer.ml.train_reranker import train_reranker_model as fn
    else:
        from eeg_seizure_analyzer.ml.train import train_model as fn
    return fn


def _train_worker(sid, dataset_def, dataset_config, train_config, model_name):
    """Background thread: run training and write progress after each epoch."""
    history: list = []  # accumulate so the UI can show every epoch live

    def _on_epoch(info):
        # The re-ranker has no epochs: it emits build-stage dicts (no "epoch"
        # key) while reading EDFs, then one final epoch=1 dict. Surface the
        # build progress as a status line and skip the history append for it.
        if "epoch" not in info:
            _write_train_progress(sid, {
                "status": "building_dataset",
                "epoch": 0,
                "total_epochs": getattr(train_config, "epochs", 0),
                "files_done": info.get("files_done"),
                "n_files": info.get("n_files"),
                "events": info.get("events"),
            })
            return
        history.append(info)
        _write_train_progress(sid, {
            "status": "training",
            "epoch": info["epoch"],
            "total_epochs": train_config.epochs,
            "train_loss": info["train_loss"],
            "val_loss": info["val_loss"],
            "val_metrics": info.get("val_metrics", {}),
            "best_epoch": info["best_epoch"],
            "lr": info.get("lr", 0),
            "elapsed_sec": info.get("elapsed_sec", 0),
            "history": history,
        })

    try:
        _write_train_progress(sid, {
            "status": "building_dataset",
            "epoch": 0,
            "total_epochs": train_config.epochs,
        })

        # Import inside the guard: a missing optional dep (e.g. scikit-learn /
        # joblib for the re-ranker) raises here and becomes a visible "error"
        # status instead of silently killing this thread and freezing the UI.
        train_fn = _import_train_fn(dataset_def, train_config)

        result = train_fn(
            dataset_def=dataset_def,
            dataset_config=dataset_config,
            train_config=train_config,
            model_name=model_name,
            progress_callback=_on_epoch,
            stop_check_fn=lambda: _stop_requested(sid),
        )

        _write_train_progress(sid, {
            "status": "done",
            "epoch": result["best_epoch"],
            "total_epochs": len(result["history"]),
            "best_val_loss": result["best_val_loss"],
            "best_metrics": result["best_metrics"],
            "model_path": result["model_path"],
            "model_name": result["model_name"],
            "n_params": result["n_params"],
            "stopped": result.get("stopped_by_user", False),
            "history": result["history"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        _write_train_progress(sid, {
            "status": "error",
            "error": str(e),
        })


@callback(
    Output("ml-bendr-train-params", "style"),
    Output("ml-pretrained-weights", "options"),
    Output("ml-train-hyperparams", "style"),
    Output("ml-reranker-note", "children"),
    Input("ml-architecture", "value"),
    prevent_initial_call=True,
)
def toggle_arch_train_params(architecture):
    """Show/hide architecture-specific training controls.

    - BENDR: reveal the pre-trained-weights / encoder-LR / freeze panel.
    - Re-ranker: hide the gradient-training hyperparameters entirely (it's a
      tabular fit with no epochs/batch/LR/patience/pos-weight/neg-ratio) and
      explain what it does instead. Exclude-animals stays visible — it's honoured.
    """
    options = []
    if architecture == "bendr":
        from pathlib import Path
        pretrained_dir = Path.home() / ".eeg_seizure_analyzer" / "pretrained"
        if pretrained_dir.exists():
            for f in sorted(pretrained_dir.glob("*.pt")):
                options.append({"label": f.stem, "value": str(f)})
    bendr_style = {"display": "block"} if architecture == "bendr" else {"display": "none"}

    if architecture == "reranker":
        hp_style = {"display": "none"}
        note = ("Event Re-ranker has no epochs/batch/learning rate — it fits a "
                "tabular model on every confirmed & rejected seizure candidate "
                "(per-animal cross-validated). Use “Exclude animal IDs” to hold "
                "animals out of the fit.")
    else:
        hp_style = {"display": "block"}
        note = ""
    return bendr_style, options, hp_style, note


@callback(
    Output("ml-bg-source-wrap", "style"),
    Input("ml-architecture", "value"),
    Input("ml-type-radio", "value"),
)
def toggle_bg_source(architecture, ann_type):
    """Background source applies only to the seizure U-Net (DatasetConfig);
    the convulsive classifier and the spike pipeline don't use it."""
    show = (architecture or "unet") == "unet" and ann_type != "spike"
    return {"display": "block"} if show else {"display": "none"}


def _bg_source_state(rows, current):
    """Options, value and note for ml-bg-source from the selected grid rows."""
    bg = _bg_totals(rows)
    has_confirmed = bg["confirmed"] > 0
    has_proposed = (bg["confirmed"] + bg["pending"]) > 0
    options = _bg_source_options(has_confirmed, has_proposed)
    value = current or "random"
    if not has_confirmed and value in ("confirmed", "confirmed+random"):
        value = "random"
    if not has_proposed and value == "proposed":
        value = "random"
    if has_confirmed:
        note = (f"Selected files: {bg['confirmed']} confirmed background "
                f"stretch{'es' if bg['confirmed'] != 1 else ''} "
                f"({_fmt_minutes(bg['confirmed_sec'])}).")
    else:
        note = ("No confirmed background in the selected files, so the "
                "confirmed options are off. Review background stretches in "
                "Training > Background, then scan again.")
    return options, value, note


@callback(
    Output("ml-bg-source", "options"),
    Output("ml-bg-source", "value"),
    Output("ml-bg-source-note", "children"),
    Input("ml-scan-data", "data"),
    State("ml-bg-source", "value"),
    prevent_initial_call=True,
)
def update_bg_source_from_scan(rows, current):
    """After a scan (all files selected), enable the confirmed options if any."""
    if rows is None:
        return no_update, no_update, no_update
    return _bg_source_state(rows, current)


@callback(
    Output("ml-bg-source", "options", allow_duplicate=True),
    Output("ml-bg-source", "value", allow_duplicate=True),
    Output("ml-bg-source-note", "children", allow_duplicate=True),
    Input("ml-file-grid", "selectedRows"),
    State("ml-bg-source", "value"),
    prevent_initial_call=True,
)
def update_bg_source_from_selection(selected_rows, current):
    """Follow the file selection: confirmed options need confirmed background."""
    return _bg_source_state(selected_rows, current)


@callback(
    Output("ml-train-progress", "children"),
    Output("ml-train-poll", "disabled"),
    Output("ml-train-running", "data"),
    Output("ml-train-btn", "disabled"),
    Input("ml-train-btn", "n_clicks"),
    State("ml-dataset-name", "value"),
    State("ml-model-name", "value"),
    State("ml-file-grid", "selectedRows"),
    State("ml-folder-input", "value"),
    State("ml-type-radio", "value"),
    State("ml-epochs", "value"),
    State("ml-batch-size", "value"),
    State("ml-lr", "value"),
    State("ml-patience", "value"),
    State("ml-pos-weight", "value"),
    State("ml-neg-ratio", "value"),
    State("ml-architecture", "value"),
    State("ml-encoder-lr", "value"),
    State("ml-freeze-epochs", "value"),
    State("ml-pretrained-weights", "value"),
    State("ml-freeze-backbone", "value"),
    State("ml-exclude-animals", "value"),
    State("ml-bg-source", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_training(n_clicks, ds_name, model_name, selected_rows, folder,
                   ann_type, epochs, batch_size, lr, patience, pos_weight,
                   neg_ratio, architecture, encoder_lr, freeze_epochs,
                   pretrained_weights, freeze_backbone, exclude_animals,
                   bg_source, sid):
    """Start model training in a background thread."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    if not selected_rows:
        return alert("No files selected. Scan a folder first.", "warning"), True, False, False
    if not model_name or not model_name.strip():
        model_name = ds_name or "unnamed"

    # Build dataset definition from selected files
    dataset_def = {
        "name": ds_name or model_name,
        "folder": folder,
        "type": ann_type,
        "files": [
            {
                "edf_path": r["edf_path"],
                "included": True,
                "n_confirmed": r.get("confirmed", 0),
                "n_rejected": r.get("rejected", 0),
                "n_pending": r.get("pending", 0),
            }
            for r in selected_rows
        ],
    }

    from eeg_seizure_analyzer.ml.train import TrainConfig

    # Animal IDs to drop from the dataset (comma- or space-separated).
    excl = tuple(s for s in re.split(r"[,\s]+", (exclude_animals or "").strip()) if s)
    if ann_type == "spike":
        from eeg_seizure_analyzer.ml.spike_dataset import SpikeDatasetConfig
        dataset_config = SpikeDatasetConfig(
            neg_pos_ratio=_to_float(neg_ratio, 2.0),
            exclude_animals=excl,
        )
    else:
        from eeg_seizure_analyzer.ml.dataset import DatasetConfig
        dataset_config = DatasetConfig(
            neg_pos_ratio=_to_float(neg_ratio, 2.0),
            exclude_animals=excl,
        )
        # Background source: seizure U-Net only (the control is hidden for the
        # convulsive classifier, which doesn't use it). UI values only.
        if (architecture or "unet") == "unet":
            allowed = {v for v, _, _ in _BG_SOURCE_CHOICES}
            dataset_config.background_source = (
                bg_source if bg_source in allowed else "random")
    _arch = architecture or "unet"
    # The radio uses the short value "convulsive"; the saved/loaded architecture
    # tag is "convulsive_classifier" (matches metadata + load_trained_model).
    if _arch == "convulsive":
        _arch = "convulsive_classifier"
    train_config = TrainConfig(
        epochs=_to_int(epochs, 50),
        batch_size=_to_int(batch_size, 8),
        learning_rate=_to_float(lr, 1e-3),
        patience=_to_int(patience, 10),
        pos_weight=_to_float(pos_weight, 5.0),
        architecture=_arch,
    )
    if _arch == "bendr":
        train_config.encoder_lr = _to_float(encoder_lr, 1e-5)
        train_config.freeze_encoder_epochs = _to_int(freeze_epochs, 5)
        train_config.freeze_backbone = bool(
            freeze_backbone and "freeze" in freeze_backbone)
        if pretrained_weights:
            train_config.pretrained_path = pretrained_weights

    # Clear old progress + any stale stop request
    p = _progress_path(sid)
    if p.exists():
        p.unlink()
    _clear_stop(sid)

    # Launch training thread
    t = _threading.Thread(
        target=_train_worker,
        args=(sid, dataset_def, dataset_config, train_config, model_name.strip()),
        daemon=True,
    )
    t.start()

    # Echo the settings this run actually received, so it's obvious whether UI
    # edits (LR, pos_weight, freeze backbone…) took effect — no guessing.
    settings = (
        f"arch={train_config.architecture} · "
        f"LR={train_config.learning_rate:.1e} · "
        f"pos_weight={train_config.pos_weight:g} · "
        f"neg/pos={dataset_config.neg_pos_ratio:g} · "
        f"epochs={train_config.epochs} · patience={train_config.patience}"
    )
    if ann_type != "spike" and train_config.architecture == "unet":
        settings += f" · background={dataset_config.background_source}"
    if train_config.architecture == "bendr":
        settings += (
            f" · freeze_backbone={train_config.freeze_backbone} · "
            f"encoder_LR={train_config.encoder_lr:.1e} · "
            f"pretrained={'yes' if train_config.pretrained_path else 'NO'}"
        )

    progress_bar = html.Div([
        dbc.Progress(
            value=0, striped=True, animated=True,
            style={"height": "24px", "marginBottom": "8px"},
            id="ml-train-progress-bar",
        ),
        html.Div(
            "Building dataset...",
            id="ml-train-progress-text",
            style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                   "textAlign": "center"},
        ),
        html.Div(
            f"Settings in use — {settings}",
            style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)",
                   "textAlign": "center", "marginTop": "4px"},
        ),
    ])

    return progress_bar, False, True, True  # enable polling, disable button


@callback(
    Output("ml-stop-btn", "disabled"),
    Output("ml-stop-btn", "children"),
    Input("ml-train-running", "data"),
)
def toggle_stop_btn(running):
    """Enable Stop only while training is running; (re)set its label."""
    return (not running), "■ Stop"


@callback(
    Output("ml-stop-btn", "children", allow_duplicate=True),
    Input("ml-stop-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stop_training(n_clicks, sid):
    """Request cooperative cancellation; training stops after the current batch
    and keeps the best model saved so far."""
    if not n_clicks:
        return no_update
    _request_stop(sid)
    return "Stopping…"


@callback(
    Output("ml-train-progress", "children", allow_duplicate=True),
    Output("ml-train-poll", "disabled", allow_duplicate=True),
    Output("ml-train-running", "data", allow_duplicate=True),
    Output("ml-train-btn", "disabled", allow_duplicate=True),
    Output("ml-train-results", "children"),
    Output("ml-train-epochs", "children", allow_duplicate=True),
    Input("ml-train-poll", "n_intervals"),
    State("ml-train-running", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_training(n_intervals, is_running, sid):
    """Poll training progress and update UI."""
    if not is_running:
        return no_update, True, no_update, no_update, no_update, no_update

    info = _read_train_progress(sid)
    if info is None:
        return no_update, no_update, no_update, no_update, no_update, no_update

    status = info.get("status", "")

    if status == "building_dataset":
        # The re-ranker reads every EDF and extracts features per event before
        # any "training" — slow (minutes). Surface the per-file counter it emits
        # so the bar visibly advances instead of looking hung.
        done = info.get("files_done")
        total_f = info.get("n_files")
        events = info.get("events")
        if done is not None and total_f:
            pct = int(100 * done / total_f) if total_f > 0 else 0
            detail = (f"Reading recordings & extracting features — "
                      f"file {done}/{total_f}"
                      + (f" · {events} events so far" if events else ""))
            bar = html.Div([
                dbc.Progress(
                    value=pct, striped=True, animated=True, color="info",
                    label=f"{done}/{total_f}",
                    style={"height": "24px", "marginBottom": "8px"},
                ),
                html.Div(
                    detail,
                    style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                           "textAlign": "center"},
                ),
            ])
        else:
            bar = html.Div([
                dbc.Progress(
                    value=100, striped=True, animated=True,
                    color="info",
                    style={"height": "24px", "marginBottom": "8px"},
                ),
                html.Div(
                    "Building dataset (loading EDF files, extracting "
                    "features)...",
                    style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                           "textAlign": "center"},
                ),
            ])
        return bar, no_update, no_update, no_update, no_update, ""

    if status == "training":
        epoch = info.get("epoch", 0)
        total = info.get("total_epochs", 1)
        pct = int(100 * epoch / total) if total > 0 else 0
        train_loss = info.get("train_loss", 0)
        val_loss = info.get("val_loss", 0)
        metrics = info.get("val_metrics", {})
        event_f1 = metrics.get("event_f1", "—")
        best_ep = info.get("best_epoch", 0)
        lr_val = info.get("lr", 0)
        elapsed = info.get("elapsed_sec", 0)

        label = f"Epoch {epoch}/{total}"
        detail = (
            f"train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f} — "
            f"event F1: {event_f1 if isinstance(event_f1, str) else f'{event_f1:.3f}'} — "
            f"best: epoch {best_ep} — lr: {lr_val:.1e} — {elapsed:.0f}s/epoch"
        )

        bar = html.Div([
            dbc.Progress(
                value=pct, striped=True, animated=True,
                label=label,
                style={"height": "24px", "marginBottom": "8px"},
            ),
            html.Div(
                detail,
                style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)",
                       "textAlign": "center"},
            ),
        ])
        epochs = _render_epochs(info.get("history", []),
                                info.get("best_epoch", 0))
        return bar, no_update, no_update, no_update, no_update, epochs

    if status == "done":
        # Training complete
        best_metrics = info.get("best_metrics", {})
        history = info.get("history", [])
        model_path = info.get("model_path", "")
        n_params = info.get("n_params", 0)
        was_stopped = info.get("stopped", False)
        title = ("Training stopped (best model kept)" if was_stopped
                 else "Training complete")

        # Build results summary
        results = html.Div([
            html.Hr(style={"borderColor": "#2ea043", "margin": "16px 0"}),
            html.H5(title,
                     style={"color": "var(--ned-success)", "marginBottom": "12px"}),
            dbc.Row([
                dbc.Col(metric_card("Model", info.get("model_name", "")),
                        width=2),
                dbc.Col(metric_card("Best Epoch",
                                    str(info.get("epoch", ""))), width=2),
                dbc.Col(metric_card("Val Loss",
                                    f"{info.get('best_val_loss', 0):.4f}"),
                        width=2),
                dbc.Col(metric_card("Event F1",
                                    f"{best_metrics.get('event_f1', 0):.3f}",
                                    accent=True), width=2),
                dbc.Col(metric_card("Event Precision",
                                    f"{best_metrics.get('event_precision', 0):.3f}"),
                        width=2),
                dbc.Col(metric_card("Event Recall",
                                    f"{best_metrics.get('event_recall', 0):.3f}"),
                        width=2),
            ], className="g-2 mb-3"),
            dbc.Row([
                dbc.Col(metric_card("Parameters", f"{n_params:,}"), width=2),
                dbc.Col(metric_card("Sample F1",
                                    f"{best_metrics.get('sample_f1', 0):.3f}"),
                        width=2),
                dbc.Col(metric_card("Sample Precision",
                                    f"{best_metrics.get('sample_precision', 0):.3f}"),
                        width=2),
                dbc.Col(metric_card("Sample Recall",
                                    f"{best_metrics.get('sample_recall', 0):.3f}"),
                        width=2),
            ], className="g-2 mb-3"),
            html.Div(
                f"Model saved to: {model_path}",
                style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)",
                       "marginTop": "8px"},
            ),
        ])

        done_bar = html.Div([
            dbc.Progress(
                value=100, color="success",
                label="Complete",
                style={"height": "24px", "marginBottom": "8px"},
            ),
        ])

        epochs = _render_epochs(history, info.get("epoch", 0))

        # Clean up progress + stop flag
        try:
            _progress_path(sid).unlink()
        except Exception:
            pass
        _clear_stop(sid)

        return done_bar, True, False, False, results, epochs

    if status == "error":
        err = info.get("error", "Unknown error")
        error_bar = html.Div([
            dbc.Progress(
                value=100, color="danger",
                label="Error",
                style={"height": "24px", "marginBottom": "8px"},
            ),
            alert(f"Training failed: {err}", "danger"),
        ])

        try:
            _progress_path(sid).unlink()
        except Exception:
            pass
        _clear_stop(sid)

        return error_bar, True, False, False, no_update, no_update

    return no_update, no_update, no_update, no_update, no_update, no_update


@callback(
    Output("ml-bg-source-help", "children"),
    Input("ml-bg-source", "value"),
)
def show_bg_source_help(value):
    """Explain the selected background source."""
    return _BG_SOURCE_OPTION_HELP.get(value or "random", _BG_SOURCE_OPTION_HELP["random"])

