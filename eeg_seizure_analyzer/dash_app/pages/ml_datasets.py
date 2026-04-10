"""ML Dataset Builder — curate annotation datasets for model training."""

from __future__ import annotations

import os
import subprocess
import sys

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert, metric_card
from eeg_seizure_analyzer.io.dataset_store import (
    scan_annotation_files,
    save_dataset,
    load_dataset,
    list_datasets,
    delete_dataset,
)
from eeg_seizure_analyzer.io.channel_ids import load_channel_ids


# ── Layout ───────────────────────────────────────────────────────────


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
                "automatically from files saved by the Training tab.",
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
            html.Label("Annotation type",
                       style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
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
            dcc.Loading(
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

                    # Model name
                    html.Label("Model name",
                               style={"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}),
                    dbc.Input(
                        id="ml-model-name",
                        placeholder="e.g. study1_v1",
                        type="text",
                        style={"maxWidth": "300px", "marginBottom": "12px"},
                    ),

                    # Config row
                    dbc.Row([
                        dbc.Col([
                            html.Label("Epochs",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(id="ml-epochs", type="number",
                                      value=50, min=1, max=500, step=1,
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Batch size",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(id="ml-batch-size", type="number",
                                      value=8, min=1, max=128, step=1,
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Learning rate",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(id="ml-lr", type="number",
                                      value=0.001, min=0.00001, max=0.1,
                                      step=0.0001,
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Patience",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(id="ml-patience", type="number",
                                      value=10, min=1, max=100, step=1,
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Pos weight",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(id="ml-pos-weight", type="number",
                                      value=5.0, min=0.1, max=50.0, step=0.5,
                                      className="form-control", size="sm"),
                        ], width=2),
                        dbc.Col([
                            html.Label("Neg/Pos ratio",
                                       style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                            dbc.Input(id="ml-neg-ratio", type="number",
                                      value=2.0, min=0.5, max=10.0, step=0.5,
                                      className="form-control", size="sm"),
                        ], width=2),
                    ], className="g-2 mb-3"),

                    # Train button
                    dbc.Button(
                        "🚀 Start Training",
                        id="ml-train-btn",
                        style={
                            "backgroundColor": "#238636",
                            "border": "1px solid #2ea043",
                            "color": "#fff",
                            "fontWeight": "600",
                        },
                        size="lg",
                        className="mb-3",
                    ),

                    # Progress area
                    html.Div(id="ml-train-progress"),

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
                "Annotate recordings in the Training tab first.",
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
        })

    col_defs = [
        {
            "field": "filename",
            "headerName": "File",
            "checkboxSelection": True,
            "headerCheckboxSelection": True,
            "flex": 2,
            "minWidth": 250,
        },
        {"field": "confirmed", "headerName": "Confirmed", "width": 110,
         "type": "numericColumn"},
        {"field": "rejected", "headerName": "Rejected", "width": 110,
         "type": "numericColumn"},
        {"field": "pending", "headerName": "Pending", "width": 110,
         "type": "numericColumn"},
        {"field": "total", "headerName": "Total", "width": 100,
         "type": "numericColumn"},
    ]

    grid = dag.AgGrid(
        id="ml-file-grid",
        rowData=rows,
        columnDefs=col_defs,
        defaultColDef={"sortable": True, "resizable": True},
        dashGridOptions={
            "rowSelection": {"mode": "multiRow"},
            "suppressRowClickSelection": True,
        },
        selectedRows=rows,  # select all by default
        style={"height": f"{min(60 + len(rows) * 42, 500)}px"},
        className="ag-theme-alpine-dark",
    )

    # Summary for all files (updated on selection change separately)
    total_conf = sum(r["n_confirmed"] for r in results)
    total_rej = sum(r["n_rejected"] for r in results)
    total_pend = sum(r["n_pending"] for r in results)

    summary = _build_summary(len(results), total_conf, total_rej, total_pend)

    # Animal ID warning
    id_warning = []
    if files_missing_ids:
        id_warning = [alert(
            f"Animal IDs not assigned for: {', '.join(files_missing_ids)}. "
            "Load each file and fill in the Animal ID column on the Load tab "
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


def _build_summary(n_files, n_confirmed, n_rejected, n_pending):
    """Build metric cards summarising the selected dataset."""
    return html.Div(
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        children=[
            metric_card("Files", str(n_files)),
            metric_card("Confirmed", str(n_confirmed), accent=True),
            metric_card("Rejected", str(n_rejected)),
            metric_card("Pending", str(n_pending)),
            metric_card("Total Events",
                        str(n_confirmed + n_rejected + n_pending)),
        ],
    )


# ── Update summary on selection change ───────────────────────────────


@callback(
    Output("ml-summary", "children", allow_duplicate=True),
    Input("ml-file-grid", "selectedRows"),
    prevent_initial_call=True,
)
def update_summary(selected_rows):
    """Update summary metrics when file selection changes."""
    if not selected_rows:
        return _build_summary(0, 0, 0, 0)

    n_files = len(selected_rows)
    total_conf = sum(r.get("confirmed", 0) for r in selected_rows)
    total_rej = sum(r.get("rejected", 0) for r in selected_rows)
    total_pend = sum(r.get("pending", 0) for r in selected_rows)

    return _build_summary(n_files, total_conf, total_rej, total_pend)


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


def _train_worker(sid, dataset_def, dataset_config, train_config, model_name):
    """Background thread: run training and write progress after each epoch."""
    from eeg_seizure_analyzer.ml.train import train_model

    def _on_epoch(info):
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
        })

    try:
        _write_train_progress(sid, {
            "status": "building_dataset",
            "epoch": 0,
            "total_epochs": train_config.epochs,
        })

        result = train_model(
            dataset_def=dataset_def,
            dataset_config=dataset_config,
            train_config=train_config,
            model_name=model_name,
            progress_callback=_on_epoch,
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
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_training(n_clicks, ds_name, model_name, selected_rows, folder,
                   ann_type, epochs, batch_size, lr, patience, pos_weight,
                   neg_ratio, sid):
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

    from eeg_seizure_analyzer.ml.dataset import DatasetConfig
    from eeg_seizure_analyzer.ml.train import TrainConfig

    dataset_config = DatasetConfig(
        neg_pos_ratio=float(neg_ratio or 2.0),
    )
    train_config = TrainConfig(
        epochs=int(epochs or 50),
        batch_size=int(batch_size or 8),
        learning_rate=float(lr or 1e-3),
        patience=int(patience or 10),
        pos_weight=float(pos_weight or 5.0),
    )

    # Clear old progress
    p = _progress_path(sid)
    if p.exists():
        p.unlink()

    # Launch training thread
    t = _threading.Thread(
        target=_train_worker,
        args=(sid, dataset_def, dataset_config, train_config, model_name.strip()),
        daemon=True,
    )
    t.start()

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
    ])

    return progress_bar, False, True, True  # enable polling, disable button


@callback(
    Output("ml-train-progress", "children", allow_duplicate=True),
    Output("ml-train-poll", "disabled", allow_duplicate=True),
    Output("ml-train-running", "data", allow_duplicate=True),
    Output("ml-train-btn", "disabled", allow_duplicate=True),
    Output("ml-train-results", "children"),
    Input("ml-train-poll", "n_intervals"),
    State("ml-train-running", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_training(n_intervals, is_running, sid):
    """Poll training progress and update UI."""
    if not is_running:
        return no_update, True, no_update, no_update, no_update

    info = _read_train_progress(sid)
    if info is None:
        return no_update, no_update, no_update, no_update, no_update

    status = info.get("status", "")

    if status == "building_dataset":
        bar = html.Div([
            dbc.Progress(
                value=100, striped=True, animated=True,
                color="info",
                style={"height": "24px", "marginBottom": "8px"},
            ),
            html.Div(
                "📦 Building dataset (loading EDF files, extracting windows)...",
                style={"fontSize": "0.85rem", "color": "var(--ned-text-muted)",
                       "textAlign": "center"},
            ),
        ])
        return bar, no_update, no_update, no_update, no_update

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
        return bar, no_update, no_update, no_update, no_update

    if status == "done":
        # Training complete
        best_metrics = info.get("best_metrics", {})
        history = info.get("history", [])
        model_path = info.get("model_path", "")
        n_params = info.get("n_params", 0)

        # Build results summary
        results = html.Div([
            html.Hr(style={"borderColor": "#2ea043", "margin": "16px 0"}),
            html.H5("✅ Training Complete",
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

        # Clean up progress file
        try:
            _progress_path(sid).unlink()
        except Exception:
            pass

        return done_bar, True, False, False, results

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

        return error_bar, True, False, False, no_update

    return no_update, no_update, no_update, no_update, no_update
