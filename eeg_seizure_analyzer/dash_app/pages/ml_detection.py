"""ML Detection — run trained models on EEG recordings."""

from __future__ import annotations

import json as _json
import os
import threading as _threading
from pathlib import Path as _Path

from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import (
    apply_fig_theme,
    alert,
    metric_card,
    no_recording_placeholder,
)
from eeg_seizure_analyzer.ml.train import list_models, MODELS_DIR


# ── Progress helpers ────────────────────────────────────────────────

_PROGRESS_DIR = _Path.home() / ".eeg_seizure_analyzer" / "cache"


def _progress_path(sid: str) -> _Path:
    return _PROGRESS_DIR / f"ml_detect_progress_{sid}.json"


def _write_progress(sid, info):
    _PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_progress_path(sid), "w") as f:
            _json.dump(info, f)
    except Exception:
        pass


def _read_progress(sid) -> dict | None:
    p = _progress_path(sid)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return _json.load(f)
    except Exception:
        return None


# ── Layout ───────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Return the ML detection layout."""
    state = server_state.get_session(sid)

    # Get available models
    models = list_models()
    model_options = []
    for m in models:
        f1 = m.get("best_event_f1", 0)
        ds = m.get("dataset", "")
        label = f"{m['name']}  —  F1: {f1:.3f}  ({ds})" if ds else f"{m['name']}  —  F1: {f1:.3f}"
        model_options.append({"label": label, "value": m["name"]})

    # Restore previous settings
    prev_model = state.extra.get("ml_det_model", None)
    prev_threshold = state.extra.get("ml_det_threshold", 0.5)
    prev_min_dur = state.extra.get("ml_det_min_dur", 3.0)
    prev_merge_gap = state.extra.get("ml_det_merge_gap", 2.0)

    has_recording = state.recording is not None

    # Check for existing ML detections
    ml_events = state.extra.get("ml_detected_events", [])
    results_section = _build_results_section(state, ml_events) if ml_events else html.Div()

    return html.Div(
        style={"padding": "24px", "maxWidth": "1100px"},
        children=[
            html.H4("ML Seizure Detection", style={"marginBottom": "8px"}),
            html.P(
                "Run a trained model on the currently loaded recording. "
                "ML detections are stored separately from spike-train detections "
                "and can be reviewed in the Training tab.",
                style={"color": "#8b949e", "fontSize": "0.9rem",
                       "marginBottom": "24px"},
            ),

            # No recording warning
            *([] if has_recording else [
                alert(
                    "No recording loaded. Load an EDF file first on the Load tab.",
                    "warning",
                ),
            ]),

            # ── Model selector ──────────────────────────────────────
            html.Label("Trained model",
                       style={"fontSize": "0.82rem", "color": "#8b949e"}),
            dcc.Dropdown(
                id="ml-det-model",
                options=model_options,
                value=prev_model if prev_model and any(
                    o["value"] == prev_model for o in model_options) else None,
                placeholder="Select a trained model..." if model_options else "No models trained yet",
                clearable=True,
                disabled=not model_options,
                className="mb-3",
            ),

            # Model info card
            html.Div(id="ml-det-model-info", className="mb-3"),

            # ── Inference settings ──────────────────────────────────
            html.Label("Inference settings",
                       style={"fontSize": "0.82rem", "color": "#8b949e",
                              "marginBottom": "8px"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Threshold",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(
                        id="ml-det-threshold", type="number",
                        value=prev_threshold, min=0.05, max=0.95, step=0.05,
                        className="form-control", size="sm",
                    ),
                    html.Small(
                        "Higher = fewer detections, more confident",
                        style={"color": "#484f58", "fontSize": "0.72rem"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Min duration (s)",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(
                        id="ml-det-min-dur", type="number",
                        value=prev_min_dur, min=0.5, max=30, step=0.5,
                        className="form-control", size="sm",
                    ),
                    html.Small(
                        "Discard events shorter than this",
                        style={"color": "#484f58", "fontSize": "0.72rem"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Merge gap (s)",
                               style={"fontSize": "0.78rem", "color": "#8b949e"}),
                    dbc.Input(
                        id="ml-det-merge-gap", type="number",
                        value=prev_merge_gap, min=0.0, max=30, step=0.5,
                        className="form-control", size="sm",
                    ),
                    html.Small(
                        "Merge events closer than this",
                        style={"color": "#484f58", "fontSize": "0.72rem"},
                    ),
                ], width=3),
            ], className="g-2 mb-4"),

            # ── Run button ──────────────────────────────────────────
            dbc.Button(
                "🧠 Run ML Detection",
                id="ml-det-run-btn",
                style={
                    "backgroundColor": "#238636",
                    "border": "1px solid #2ea043",
                    "color": "#fff",
                    "fontWeight": "600",
                },
                size="lg",
                disabled=not has_recording or not model_options,
                className="mb-3",
            ),

            # Progress
            html.Div(id="ml-det-progress"),

            # Results
            html.Div(id="ml-det-results", children=results_section),

            # Stores
            dcc.Store(id="ml-det-running", data=False),
            dcc.Interval(id="ml-det-poll", interval=1000, disabled=True),
        ],
    )


def _build_results_section(state, ml_events):
    """Build the results summary from ML detections."""
    if not ml_events:
        return html.Div()

    rec = state.recording
    # Count by channel
    ch_counts = {}
    for ev in ml_events:
        ch = ev.get("channel", ev.channel if hasattr(ev, "channel") else 0)
        ch_name = ""
        if rec and hasattr(rec, "channel_names") and ch < len(rec.channel_names):
            ch_name = rec.channel_names[ch]
        else:
            ch_name = f"Ch {ch}"
        ch_counts[ch_name] = ch_counts.get(ch_name, 0) + 1

    total = sum(ch_counts.values())

    ch_badges = []
    for ch_name, count in sorted(ch_counts.items()):
        ch_badges.append(
            dbc.Badge(
                f"{ch_name}: {count}",
                color="info",
                className="me-1",
                style={"fontSize": "0.8rem"},
            )
        )

    return html.Div([
        html.Hr(style={"borderColor": "#2ea043", "margin": "16px 0"}),
        html.H5(f"ML Detection Results — {total} events",
                style={"color": "#58a6ff", "marginBottom": "12px"}),
        html.Div(ch_badges, style={"marginBottom": "12px"}),
        html.P(
            "These detections are stored separately from spike-train detections. "
            "Switch to the Training tab to review and annotate them.",
            style={"color": "#8b949e", "fontSize": "0.85rem"},
        ),
    ])


# ── Model info callback ─────────────────────────────────────────────


@callback(
    Output("ml-det-model-info", "children"),
    Input("ml-det-model", "value"),
    prevent_initial_call=True,
)
def show_model_info(model_name):
    """Show model details when selected."""
    if not model_name:
        return html.Div()

    meta_path = MODELS_DIR / model_name / "metadata.json"
    if not meta_path.exists():
        return html.Div()

    try:
        with open(meta_path) as f:
            meta = _json.load(f)
    except Exception:
        return html.Div()

    best_metrics = meta.get("best_metrics", {})
    tc = meta.get("train_config", {})
    dc = meta.get("dataset_config", {})

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(metric_card("Dataset", meta.get("dataset_name", "—")),
                        width=2),
                dbc.Col(metric_card("Event F1",
                                    f"{best_metrics.get('event_f1', 0):.3f}",
                                    accent=True), width=2),
                dbc.Col(metric_card("Precision",
                                    f"{best_metrics.get('event_precision', 0):.3f}"),
                        width=2),
                dbc.Col(metric_card("Recall",
                                    f"{best_metrics.get('event_recall', 0):.3f}"),
                        width=2),
                dbc.Col(metric_card("Parameters",
                                    f"{meta.get('n_params', 0):,}"),
                        width=2),
                dbc.Col(metric_card("Device", meta.get("device", "—")),
                        width=2),
            ], className="g-2"),
            html.Div(
                f"Trained: {meta.get('created', '—')[:10]} — "
                f"Best epoch: {meta.get('best_epoch', '—')} — "
                f"Window: {dc.get('window_sec', 60)}s @ {dc.get('target_fs', 250)}Hz",
                style={"fontSize": "0.78rem", "color": "#484f58",
                       "marginTop": "8px"},
            ),
        ]),
        style={"backgroundColor": "#161b22", "border": "1px solid #21262d"},
    )


# ── Run detection ────────────────────────────────────────────────────


def _detect_worker(sid, edf_path, model_name, threshold, min_dur, merge_gap,
                   channels):
    """Background thread: run ML inference and write progress."""
    from eeg_seizure_analyzer.ml.predict import predict_seizures

    def _on_progress(current, total):
        _write_progress(sid, {
            "status": "running",
            "current": current,
            "total": total,
        })

    try:
        _write_progress(sid, {
            "status": "loading_model",
            "current": 0,
            "total": 1,
        })

        events = predict_seizures(
            edf_path=edf_path,
            model_name=model_name,
            channels=channels,
            threshold=threshold,
            min_duration_sec=min_dur,
            merge_gap_sec=merge_gap,
            progress_callback=_on_progress,
        )

        # Enrich events with quality metrics (spectral, amplitude, etc.)
        _write_progress(sid, {
            "status": "computing_features",
            "current": 0,
            "total": len(events),
        })
        try:
            from eeg_seizure_analyzer.io.edf_reader import read_edf
            from eeg_seizure_analyzer.detection.confidence import (
                compute_event_quality, compute_local_baseline_ratio,
            )
            # Load recording for feature computation
            feat_rec = read_edf(edf_path, channels=channels)
            feat_rec.source_path = edf_path
            for fi, ev in enumerate(events):
                try:
                    qm = compute_event_quality(feat_rec, ev)
                    ev.quality_metrics.update(qm)
                    local_bl = compute_local_baseline_ratio(feat_rec, ev)
                    ev.quality_metrics["local_baseline_ratio"] = local_bl
                except Exception:
                    pass
                if fi % 10 == 0:
                    _write_progress(sid, {
                        "status": "computing_features",
                        "current": fi + 1,
                        "total": len(events),
                    })
        except Exception:
            import traceback
            traceback.print_exc()

        # Store results as dicts
        event_dicts = []
        for ev in events:
            event_dicts.append({
                "onset_sec": ev.onset_sec,
                "offset_sec": ev.offset_sec,
                "duration_sec": ev.duration_sec,
                "channel": ev.channel,
                "confidence": ev.confidence,
                "event_type": ev.event_type,
                "animal_id": ev.animal_id,
                "source": "ml_detector",
                "features": dict(ev.features) if ev.features else {},
                "quality_metrics": dict(ev.quality_metrics) if ev.quality_metrics else {},
                "event_id": ev.event_id,
            })

        # Save to server state
        state = server_state.get_session(sid)
        state.extra["ml_detected_events"] = event_dicts
        state.extra["ml_det_model"] = model_name

        # Also save to disk as sidecar
        if edf_path:
            _save_ml_detections(edf_path, model_name, event_dicts,
                                threshold, min_dur, merge_gap)

        _write_progress(sid, {
            "status": "done",
            "n_events": len(event_dicts),
            "by_channel": _count_by_channel(event_dicts),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        _write_progress(sid, {
            "status": "error",
            "error": str(e),
        })


def _save_ml_detections(edf_path, model_name, events, threshold, min_dur,
                        merge_gap):
    """Save ML detections as a sidecar JSON file."""
    from pathlib import Path
    from datetime import datetime, timezone

    stem = Path(edf_path).stem
    out_path = Path(edf_path).parent / f"{stem}_ned_ml_detections.json"

    data = {
        "model_name": model_name,
        "threshold": threshold,
        "min_duration_sec": min_dur,
        "merge_gap_sec": merge_gap,
        "created": datetime.now(timezone.utc).isoformat(),
        "n_events": len(events),
        "events": events,
    }

    try:
        with open(out_path, "w") as f:
            _json.dump(data, f, indent=2)
    except Exception:
        pass


def _count_by_channel(events):
    counts = {}
    for ev in events:
        ch = ev.get("channel", 0)
        counts[ch] = counts.get(ch, 0) + 1
    return counts


@callback(
    Output("ml-det-progress", "children"),
    Output("ml-det-poll", "disabled"),
    Output("ml-det-running", "data"),
    Output("ml-det-run-btn", "disabled"),
    Input("ml-det-run-btn", "n_clicks"),
    State("ml-det-model", "value"),
    State("ml-det-threshold", "value"),
    State("ml-det-min-dur", "value"),
    State("ml-det-merge-gap", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_detection(n_clicks, model_name, threshold, min_dur, merge_gap, sid):
    """Launch ML detection in a background thread."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    if not model_name:
        return alert("Select a model first.", "warning"), True, False, False

    state = server_state.get_session(sid)
    if state.recording is None:
        return alert("No recording loaded.", "warning"), True, False, False

    edf_path = state.recording.source_path
    if not edf_path:
        return alert("Recording has no source path.", "warning"), True, False, False

    # Persist settings
    state.extra["ml_det_model"] = model_name
    state.extra["ml_det_threshold"] = float(threshold or 0.5)
    state.extra["ml_det_min_dur"] = float(min_dur or 3.0)
    state.extra["ml_det_merge_gap"] = float(merge_gap or 2.0)

    # Get EEG channel indices
    from eeg_seizure_analyzer.io.edf_reader import scan_edf_channels, auto_pair_channels
    ch_info = scan_edf_channels(edf_path)
    eeg_idx, _, _ = auto_pair_channels(ch_info)

    # Clear old progress
    p = _progress_path(sid)
    if p.exists():
        p.unlink()

    # Launch thread
    t = _threading.Thread(
        target=_detect_worker,
        args=(sid, edf_path, model_name,
              float(threshold or 0.5),
              float(min_dur or 3.0),
              float(merge_gap or 2.0),
              eeg_idx),
        daemon=True,
    )
    t.start()

    progress_bar = html.Div([
        dbc.Progress(
            value=0, striped=True, animated=True,
            style={"height": "24px", "marginBottom": "8px"},
        ),
        html.Div(
            "Loading model...",
            style={"fontSize": "0.85rem", "color": "#8b949e",
                   "textAlign": "center"},
        ),
    ])

    return progress_bar, False, True, True


@callback(
    Output("ml-det-progress", "children", allow_duplicate=True),
    Output("ml-det-poll", "disabled", allow_duplicate=True),
    Output("ml-det-running", "data", allow_duplicate=True),
    Output("ml-det-run-btn", "disabled", allow_duplicate=True),
    Output("ml-det-results", "children"),
    Input("ml-det-poll", "n_intervals"),
    State("ml-det-running", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_detection(n_intervals, is_running, sid):
    """Poll detection progress."""
    if not is_running:
        return no_update, True, no_update, no_update, no_update

    info = _read_progress(sid)
    if info is None:
        return no_update, no_update, no_update, no_update, no_update

    status = info.get("status", "")

    if status == "loading_model":
        bar = html.Div([
            dbc.Progress(
                value=100, striped=True, animated=True, color="info",
                style={"height": "24px", "marginBottom": "8px"},
            ),
            html.Div(
                "🧠 Loading model...",
                style={"fontSize": "0.85rem", "color": "#8b949e",
                       "textAlign": "center"},
            ),
        ])
        return bar, no_update, no_update, no_update, no_update

    if status == "computing_features":
        current = info.get("current", 0)
        total = info.get("total", 1)
        pct = int(100 * current / total) if total > 0 else 0
        bar = html.Div([
            dbc.Progress(
                value=pct, striped=True, animated=True, color="info",
                label=f"{current}/{total}",
                style={"height": "24px", "marginBottom": "8px"},
            ),
            html.Div(
                f"📊 Computing features for {total} events...",
                style={"fontSize": "0.85rem", "color": "#8b949e",
                       "textAlign": "center"},
            ),
        ])
        return bar, no_update, no_update, no_update, no_update

    if status == "running":
        current = info.get("current", 0)
        total = info.get("total", 1)
        pct = int(100 * current / total) if total > 0 else 0
        label = f"{current}/{total} windows"

        bar = html.Div([
            dbc.Progress(
                value=pct, striped=True, animated=True,
                label=label,
                style={"height": "24px", "marginBottom": "8px"},
            ),
            html.Div(
                f"Processing EEG windows... ({pct}%)",
                style={"fontSize": "0.85rem", "color": "#8b949e",
                       "textAlign": "center"},
            ),
        ])
        return bar, no_update, no_update, no_update, no_update

    if status == "done":
        n_events = info.get("n_events", 0)
        by_channel = info.get("by_channel", {})

        ch_badges = []
        state = server_state.get_session(sid)
        rec = state.recording if state else None
        for ch_str, count in sorted(by_channel.items(), key=lambda x: int(x[0])):
            ch = int(ch_str)
            ch_name = ""
            if rec and hasattr(rec, "channel_names") and ch < len(rec.channel_names):
                ch_name = rec.channel_names[ch]
            else:
                ch_name = f"Ch {ch}"
            ch_badges.append(
                dbc.Badge(
                    f"{ch_name}: {count}",
                    color="info",
                    className="me-1",
                    style={"fontSize": "0.8rem"},
                )
            )

        done_bar = html.Div([
            dbc.Progress(
                value=100, color="success", label="Complete",
                style={"height": "24px", "marginBottom": "8px"},
            ),
        ])

        results = html.Div([
            html.Hr(style={"borderColor": "#2ea043", "margin": "16px 0"}),
            html.H5(f"ML Detection Results — {n_events} events",
                    style={"color": "#58a6ff", "marginBottom": "12px"}),
            html.Div(ch_badges, style={"marginBottom": "12px"}),
            html.P(
                "Detections saved to disk. Switch to the Training tab to "
                "review and annotate them.",
                style={"color": "#8b949e", "fontSize": "0.85rem"},
            ),
        ])

        try:
            _progress_path(sid).unlink()
        except Exception:
            pass

        return done_bar, True, False, False, results

    if status == "error":
        err = info.get("error", "Unknown error")
        error_bar = html.Div([
            dbc.Progress(
                value=100, color="danger", label="Error",
                style={"height": "24px", "marginBottom": "8px"},
            ),
            alert(f"Detection failed: {err}", "danger"),
        ])

        try:
            _progress_path(sid).unlink()
        except Exception:
            pass

        return error_bar, True, False, False, no_update

    return no_update, no_update, no_update, no_update, no_update
