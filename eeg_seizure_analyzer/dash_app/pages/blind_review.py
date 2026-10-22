"""Dataset / Model > Blind review: choose or build a review, then review it here.

No review open: the chooser (model, its earlier reviews with Open, options for a new
review, Build with progress). A review open: a banner (events, progress, Score,
Merge, Exit review) above Training > Seizure's reviewer in blind mode, on the review
folder's recordings. See docs/BLIND_REVIEW_PLAN.md.

Opening a review replaces the session's project with the review folder's recordings
(so the file bar and "Next pending" work as usual); the user's own project is
stashed first and Exit review brings it back.

Session keys (state.extra):
  blind_review_id  set while a review is open; Training > Seizure keys its blind
                   mode on it (together with blind_folder)
  blind_folder     the open review's folder (= project_folder while it is open)
  blind_stash      the user's project before the review: project keys + the path of
                   the loaded recording
  blind_model      the model last chosen in the chooser
  blind_msg        a one-off message for the chooser (e.g. a failed restore)
"""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert
from eeg_seizure_analyzer.ml import blind_review as br

_MUTED = {"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}
# Bootstrap tables otherwise keep the base (dark) theme's background
_TABLE = {"--bs-table-bg": "transparent", "--bs-table-color": "var(--ned-text)",
          "--bs-table-border-color": "var(--ned-border)", "color": "var(--ned-text)"}

# The user's project, put aside while a review is open (plus the loaded recording)
_STASH_KEYS = ("project_files", "project_folder", "project_active_idx",
               "show_upload_form", "show_batch_form", "channel_animal_ids")
# Per-recording page state that must not carry over between the two projects
# (as main.py's file_bar_switch clears it)
_PER_RECORDING_KEYS = ("tr_annotations", "tr_current_idx", "trb_annotations",
                       "trb_current_idx", "trb_view", "trb_browse_start", "trb_cache",
                       "trb_sz_cache", "trb_for", "sz_selected_event_key", "video_path")


# ---------------------------------------------------------------------------
# Opening and closing a review (session state)
# ---------------------------------------------------------------------------

def review_open(state) -> bool:
    """A review is open and its folder is still the loaded project."""
    return (state is not None and bool(state.extra.get("blind_review_id"))
            and state.extra.get("project_folder") == state.extra.get("blind_folder"))


def _clear_recording_state(state) -> None:
    from eeg_seizure_analyzer.dash_app.pages.upload import _reset_to_landing
    _reset_to_landing(state)
    for k in _PER_RECORDING_KEYS:
        state.extra.pop(k, None)


def open_review(state, review_id: str) -> str | None:
    """Load a review folder's recordings as this session's project, after stashing
    the user's own project. Returns an error message, or None on success."""
    review = br.load_review(review_id)
    if not review:
        return f"No blind review with id {review_id!r} was found."
    folder = Path(review["folder"])
    if not folder.is_dir():
        return f"The review folder {folder} no longer exists."
    edfs = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".edf")
    if not edfs:
        return ("This review has no events to review (no never-reviewed detections "
                "in its bands).")
    from eeg_seizure_analyzer.dash_app.pages.upload import _load_edf_into_state
    if not review_open(state):   # don't stash a review over the user's project
        rec = state.recording
        state.extra["blind_stash"] = {
            "recording": getattr(rec, "source_path", None) if rec is not None else None,
            "extra": {k: copy.deepcopy(state.extra[k]) for k in _STASH_KEYS
                      if k in state.extra},
        }
    _clear_recording_state(state)
    files = [{"edf_path": str(p), "filename": p.name, "has_detections": False,
              "has_annotations": True,
              "has_ids": p.with_name(p.stem + "_ned_channels.json").exists()}
             for p in edfs]
    pend = _pending_per_file(files)
    first = next((i for i, n in enumerate(pend) if n), 0)
    state.extra["blind_review_id"] = review_id
    state.extra["blind_folder"] = str(folder)
    state.extra["project_files"] = files
    state.extra["project_folder"] = str(folder)
    state.extra["project_active_idx"] = first
    try:
        _load_edf_into_state(state, files[first]["edf_path"])
    except Exception as e:
        traceback.print_exc()
        close_review(state)
        return f"Could not open the review's first recording: {e}"
    return None


def close_review(state) -> str | None:
    """End the review: restore the stashed project and reload the recording that was
    loaded before (none if none was). Returns a message if the reload failed."""
    stash = state.extra.pop("blind_stash", None) or {}
    for k in ("blind_review_id", "blind_folder"):
        state.extra.pop(k, None)
    _clear_recording_state(state)
    state.extra.update(stash.get("extra") or {})
    path = stash.get("recording")
    if not path:
        return None
    if not os.path.isfile(path):
        return f"The recording loaded before the review ({path}) was not found."
    from eeg_seizure_analyzer.dash_app.pages.upload import _load_edf_into_state
    try:
        _load_edf_into_state(state, path)
    except Exception as e:
        traceback.print_exc()
        return f"Could not reload {os.path.basename(path)}: {e}"
    return None


def _pending_per_file(files: list[dict]) -> list[int]:
    """Blind-review items still pending in each project file (read from disk)."""
    out = []
    for f in files:
        p = Path(f["edf_path"])
        d = br._read_json(p.with_name(p.stem + "_ned_annotations.json"), {}) or {}
        out.append(sum(1 for a in d.get("annotations", [])
                       if (a.get("features") or {}).get("review_item") is not None
                       and a.get("label", "pending") not in ("confirmed", "rejected")))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seizure_model_options() -> list[dict]:
    """Trained seizure U-Net models, newest first."""
    from eeg_seizure_analyzer.ml.train import MODELS_DIR
    rows = []
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            md = d / "metadata.json"
            if not md.is_file():
                continue
            try:
                meta = json.loads(md.read_text())
            except (OSError, ValueError):
                continue
            target = (meta.get("dataset_config") or {}).get("target", "seizure")
            if meta.get("architecture", "unet") != "unet" or target != "seizure":
                continue
            held = " · held-out list saved" if meta.get("val_pairs") else ""
            rows.append((meta.get("created", ""), {"label": d.name + held, "value": d.name}))
    return [o for _, o in sorted(rows, key=lambda r: r[0], reverse=True)]


def _band_rows(bands: list[dict]) -> list[dict]:
    """Back-end bands -> table rows (an upper limit above 1 is shown as 1)."""
    return [{"lo": b["lo"], "hi": min(float(b["hi"]), 1.0), "n": int(b["n"])} for b in bands]


def _held_out(model: str):
    """(existing pairs, n listed, n missing) or None if the model has no list."""
    pairs = br.held_out_recordings(model) if model else None
    if pairs is None:
        return None
    ok = [(a, p) for a, p in pairs if os.path.isfile(p)]
    return ok, len(pairs), len(pairs) - len(ok)


def _folder_recordings(folder: str) -> list[tuple[str, str]]:
    """(animal, EDF) for every animal in the channel maps of the folder's EDFs."""
    from eeg_seizure_analyzer.io.channel_ids import load_channel_ids
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() != ".edf":
            continue
        ids = load_channel_ids(str(p)) or {}
        for a in sorted({str(v).strip() for v in ids.values()}):
            if a and a.lower() not in ("x", "none", "nan", "-"):
                out.append((a, str(p)))
    return out


def _n_files_animals(pairs) -> tuple[int, int]:
    return len({p for _, p in pairs}), len({a for a, _ in pairs})


def _fmt_pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def _fmt_num(x, nd=1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _label(text, tip=None):
    return html.Div(text, title=tip, style={"fontSize": "0.85rem", "fontWeight": "600",
                                            "marginBottom": "4px"})


# ---------------------------------------------------------------------------
# The subtab
# ---------------------------------------------------------------------------

def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid) if sid else None
    if state is not None and state.extra.get("blind_review_id") and not review_open(state):
        # Other recordings were loaded (Load tab) while the review was open: they
        # replaced the review, and with it the stashed project.
        for k in ("blind_review_id", "blind_folder", "blind_stash"):
            state.extra.pop(k, None)
        state.extra["blind_msg"] = ("The blind review was closed because other "
                                    "recordings were loaded.")
    if review_open(state):
        job = _JOBS.get(sid)
        if job and job.get("status") != "running":
            _JOBS.pop(sid, None)
        return _review_view(sid, state)
    return _chooser(sid, state)


def _chooser(sid, state) -> html.Div:
    msg = state.extra.pop("blind_msg", None) if state is not None else None
    opts = _seizure_model_options()
    names = [o["value"] for o in opts]
    remembered = state.extra.get("blind_model") if state is not None else None
    model = remembered if remembered in names else (names[0] if names else None)
    info, folder_style = _recordings_panel(model)
    bands = br.default_bands(model) if model else [{"lo": 0.5, "hi": 1.01, "n": 40},
                                                    {"lo": 0.25, "hi": 0.5, "n": 40}]
    job = _JOBS.get(sid) if sid else None
    running = bool(job and job.get("status") == "running")
    return html.Div(style={"padding": "24px", "maxWidth": "980px"}, children=[
        html.H4("Blind review", style={"marginBottom": "6px"}),
        html.Div("Check what a model's detections really are. The model runs once over "
                 "recordings it never trained on, a random sample of its never-reviewed "
                 "detections is drawn per score band, and you confirm or reject each "
                 "one without seeing the model's confidence or band.",
                 style={**_MUTED, "marginBottom": "16px"}),
        alert(msg, "warning") if msg else None,
        dbc.Card(className="ned-card mb-3", children=dbc.CardBody([
            _label("Model"),
            dcc.Dropdown(id="br-model", options=opts, value=model, clearable=False,
                         placeholder="No trained seizure U-Net models",
                         style={"maxWidth": "480px"}),
        ])),
        dbc.Card(className="ned-card mb-3", children=dbc.CardBody([
            _label("Earlier reviews of this model"),
            html.Div("Open one to continue reviewing, score it or merge it. While a "
                     "review is open it replaces the loaded recordings; Exit review "
                     "brings them back.", style={**_MUTED, "marginBottom": "6px"}),
            html.Div(id="br-past", children=_earlier_reviews(model)),
            html.Div(id="br-open-msg"),
        ])),
        html.H5("New review", style={"marginTop": "8px", "marginBottom": "10px"}),
        dbc.Card(className="ned-card mb-3", children=dbc.CardBody([
            _label("Recordings"),
            html.Div(id="br-heldout-info", children=info),
            html.Div(id="br-folder-area", style=folder_style, children=[
                html.Div(style={"display": "flex", "gap": "8px", "marginTop": "6px"},
                         children=[
                    dbc.Input(id="br-rec-folder", type="text", debounce=True,
                              placeholder="Folder of recordings (EDF + channel maps)",
                              className="form-control", size="sm"),
                    dbc.Button("Browse", id="br-rec-browse", size="sm",
                               className="btn-ned-secondary"),
                ]),
                html.Div(id="br-rec-summary", style={**_MUTED, "marginTop": "6px"}),
            ]),
        ])),
        dbc.Card(className="ned-card mb-3", children=dbc.CardBody([
            _label("Score bands"),
            html.Div("Each band is a range of the model's peak score; the review draws "
                     "the given number of events from each. The upper band is what the "
                     "app reports at the model's threshold; the lower band is what the "
                     "threshold leaves out.", style={**_MUTED, "marginBottom": "8px"}),
            html.Div(id="br-band-rows", children=_band_inputs(_band_rows(bands))),
            dbc.Button("+ Add band", id="br-band-add", size="sm",
                       className="btn-ned-secondary", style={"marginTop": "6px"}),
        ])),
        dbc.Card(className="ned-card mb-3", children=dbc.CardBody([
            _label("Output folder"),
            html.Div("The review folder: links to the recordings and blinded annotation "
                     "files. Must be empty or new. The key (which band each event came "
                     "from) is kept in the app's data folder, not here.",
                     style={**_MUTED, "marginBottom": "6px"}),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                dbc.Input(id="br-out-folder", type="text", debounce=True,
                          placeholder="e.g. /Users/me/reviews/model_check_1",
                          className="form-control", size="sm"),
                dbc.Button("Browse", id="br-out-browse", size="sm",
                           className="btn-ned-secondary"),
            ]),
        ])),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"},
                 className="mb-3", children=[
            dbc.Button("Build review", id="br-build-btn", className="btn-ned-primary",
                       disabled=running),
            dbc.Button("■ Stop", id="br-stop-btn", color="danger", outline=True,
                       disabled=not running),
        ]),
        # A build started earlier keeps running when the user leaves the subtab;
        # coming back resumes its progress (or reports how it ended).
        html.Div(id="br-build-progress", children=_progress_bar(job) if running else None),
        dcc.Interval(id="br-build-poll", interval=1000, disabled=not job),
    ])


def _recordings_panel(model):
    if not model:
        return html.Div("No trained seizure U-Net models found.", style=_MUTED), \
            {"display": "none"}
    ho = _held_out(model)
    if ho is None:
        return html.Div([
            html.Div("This model predates saving its held-out recordings, so choose a "
                     "folder of recordings. Every animal in their channel maps is "
                     "reviewed.", style=_MUTED),
            alert("Use only recordings this model never trained on. If any were in "
                  "its training data, the review is meaningless.", "warning"),
        ]), {"display": "block"}
    ok, n_listed, n_missing = ho
    nf, na = _n_files_animals(ok)
    kids = [html.Div(f"{nf} held-out recordings ({na} animals) from this model's "
                     "training.", style={"fontSize": "0.88rem"})]
    if n_missing:
        kids.append(alert(f"{n_missing} of {n_listed} held-out (animal, recording) pairs "
                          "are not on this computer and will be skipped.", "warning"))
    return html.Div(kids), {"display": "none"}


def _band_inputs(rows: list[dict]):
    head = html.Div(style={"display": "grid", "gridTemplateColumns": "120px 120px 120px 40px",
                           "gap": "8px", **_MUTED}, children=[
        html.Div("From score"), html.Div("Up to"), html.Div("Events"), html.Div()])
    out = [head]
    for i, r in enumerate(rows):
        out.append(html.Div(style={"display": "grid",
                                   "gridTemplateColumns": "120px 120px 120px 40px",
                                   "gap": "8px", "marginTop": "4px"}, children=[
            dbc.Input(id={"type": "br-band", "field": "lo", "index": i}, type="number",
                      value=r["lo"], min=0, max=1, step=0.01, size="sm"),
            dbc.Input(id={"type": "br-band", "field": "hi", "index": i}, type="number",
                      value=r["hi"], min=0, max=1, step=0.01, size="sm"),
            dbc.Input(id={"type": "br-band", "field": "n", "index": i}, type="number",
                      value=r["n"], min=1, step=1, size="sm"),
            dbc.Button("✕", id={"type": "br-band-del", "index": i}, size="sm",
                       color="secondary", outline=True, title="Remove this band"),
        ]))
    return out


def _earlier_reviews(model):
    """The model's earlier reviews, newest first, each with Open."""
    try:
        reviews = [r for r in br.list_reviews() if model and r.get("model") == model]
    except Exception:
        traceback.print_exc()
        reviews = []
    if not reviews:
        return html.Div("None yet." if model else "", style=_MUTED)
    rows = []
    for r in reviews:
        try:
            n, k = br.review_progress(r["review_id"])
            prog = f"{k}/{n}"
        except Exception:
            prog = ""
        merged = r.get("merged")
        folder = str(r.get("folder", "")).rstrip("/")
        rows.append(html.Tr([
            html.Td(str(r.get("created", ""))[:16].replace("T", " ")),
            html.Td(prog),
            html.Td(str(merged.get("at", ""))[:10] if isinstance(merged, dict)
                    else ("yes" if merged else "")),
            html.Td(os.path.basename(folder), title=folder),
            html.Td(dbc.Button("Open", id={"type": "br-open", "index": r["review_id"]},
                               size="sm", className="btn-ned-secondary",
                               style={"padding": "1px 10px"})),
        ]))
    return html.Table(className="table table-sm mb-0",
                      style={**_TABLE, "fontSize": "0.82rem"}, children=[
        html.Thead(html.Tr([html.Th("Created (UTC)"), html.Th("Reviewed"),
                            html.Th("Merged"), html.Th("Folder"), html.Th("")])),
        html.Tbody(rows)])


@callback(
    Output("br-heldout-info", "children"),
    Output("br-folder-area", "style"),
    Output("br-band-rows", "children"),
    Output("br-past", "children"),
    Input("br-model", "value"),
    Input("br-band-add", "n_clicks"),
    Input({"type": "br-band-del", "index": ALL}, "n_clicks"),
    State({"type": "br-band", "field": ALL, "index": ALL}, "value"),
    State({"type": "br-band", "field": ALL, "index": ALL}, "id"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def edit_options(model, _add, _dels, values, ids, sid):
    trig = ctx.triggered_id
    nu = no_update
    if trig == "br-model":
        if sid and model:
            server_state.get_session(sid).extra["blind_model"] = model
        info, style = _recordings_panel(model)
        return (info, style, _band_inputs(_band_rows(br.default_bands(model))),
                _earlier_reviews(model))
    rows = _rows_from_inputs(values, ids)
    if trig == "br-band-add":
        lo = min((r["lo"] or 0 for r in rows), default=0.5)
        rows.append({"lo": round(lo / 2, 4), "hi": lo, "n": br.N_PER_BAND})
    elif isinstance(trig, dict) and trig.get("type") == "br-band-del":
        if not any(_dels or []):
            return nu, nu, nu, nu
        rows = [r for i, r in enumerate(rows) if i != trig["index"]]
    else:
        return nu, nu, nu, nu
    return nu, nu, _band_inputs(rows), nu


def _rows_from_inputs(values, ids) -> list[dict]:
    rows: dict[int, dict] = {}
    for v, i in zip(values or [], ids or []):
        rows.setdefault(i["index"], {})[i["field"]] = v
    return [rows[k] for k in sorted(rows)]


def _validate_bands(rows) -> tuple[list[dict] | None, str | None]:
    out = []
    for i, r in enumerate(rows, start=1):
        try:
            lo, hi, n = float(r.get("lo")), float(r.get("hi")), int(r.get("n"))
        except (TypeError, ValueError):
            return None, f"Band {i}: fill in all three numbers."
        if not (0 <= lo < hi <= 1) or n < 1:
            return None, f"Band {i}: need 0 ≤ from < up to ≤ 1 and at least 1 event."
        out.append({"lo": lo, "hi": 1.01 if hi >= 1 else hi, "n": n})
    if not out:
        return None, "Add at least one band."
    s = sorted(out, key=lambda b: b["lo"])
    if any(a["hi"] > b["lo"] for a, b in zip(s, s[1:])):
        return None, "Bands must not overlap."
    return out, None


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Output("br-open-msg", "children"),
    Input({"type": "br-open", "index": ALL}, "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def open_earlier(_clicks, sid, refresh):
    """Open an earlier review in this subtab."""
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not (ctx.triggered and ctx.triggered[0].get("value")):
        return no_update, no_update     # buttons (re)created, not clicked
    state = server_state.get_session(sid)
    job = _JOBS.get(sid)
    if job and job.get("status") == "running":
        return no_update, alert("A review is being built; wait for it or stop it.",
                                "warning")
    try:
        err = open_review(state, trig["index"])
    except Exception as e:
        traceback.print_exc()
        err = f"Could not open the review: {e}"
    if err:
        return no_update, alert(err, "warning")
    return (refresh or 0) + 1, no_update


@callback(
    Output("br-rec-folder", "value"),
    Input("br-rec-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_rec_folder(n):
    if not n:
        return no_update
    from eeg_seizure_analyzer.dash_app.pages.upload import _browse_folder
    return _browse_folder("Select a folder of recordings") or no_update


@callback(
    Output("br-out-folder", "value"),
    Input("br-out-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_out_folder(n):
    if not n:
        return no_update
    from eeg_seizure_analyzer.dash_app.pages.upload import _browse_folder
    return _browse_folder("Select an empty folder for the review") or no_update


@callback(
    Output("br-rec-summary", "children"),
    Input("br-rec-folder", "value"),
    prevent_initial_call=True,
)
def summarise_folder(folder):
    if not folder:
        return ""
    if not os.path.isdir(folder):
        return "Folder not found."
    pairs = _folder_recordings(folder)
    if not pairs:
        return "No recordings with channel maps (animal IDs) in this folder."
    nf, na = _n_files_animals(pairs)
    return f"{nf} recordings ({na} animals)."


# ---------------------------------------------------------------------------
# Build (background thread, polled)
# ---------------------------------------------------------------------------

_JOBS: dict[str, dict] = {}


def _run_build(sid, model, recs, out, bands):
    job = _JOBS[sid]

    def _cb(i, n, msg):
        job.update(i=i, n=n, msg=msg)

    try:
        rid = br.build_review(model, recs, out, bands=bands, progress_callback=_cb,
                              stop_flag=lambda: job["stop"])
        # Load the review here, not in the poll: reading the first recording takes
        # seconds, and a poll that slow is superseded by the next interval tick
        # (Dash drops its response).
        job.update(msg="Opening the review", review_id=rid)
        review = br.load_review(rid) or {}
        if not review.get("n_items"):
            job.update(status="empty")
            return
        err = open_review(server_state.get_session(sid), rid)
        job.update(status="error", error=err) if err else job.update(status="done")
    except RuntimeError as e:
        if str(e) == "stopped":
            job.update(status="stopped")
        else:
            traceback.print_exc()
            job.update(status="error", error=str(e))
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=str(e))


def _check_out_folder(out: str, recs) -> str | None:
    if not out:
        return "Choose an output folder."
    p = Path(os.path.expanduser(out))
    if not p.is_absolute():
        return "Give the output folder as a full path."
    if p.exists() and not p.is_dir():
        return "The output path is a file, not a folder."
    if p.exists() and any(p.iterdir()):
        return "The output folder is not empty: choose an empty or new folder."
    rec_dirs = {str(Path(e).parent.resolve()) for _, e in recs}
    if str(p.resolve()) in rec_dirs:
        return "The output folder cannot be a recordings folder."
    return None


@callback(
    Output("br-build-progress", "children"),
    Output("br-build-poll", "disabled"),
    Output("br-build-btn", "disabled"),
    Output("br-stop-btn", "disabled"),
    Input("br-build-btn", "n_clicks"),
    State("br-model", "value"),
    State("br-rec-folder", "value"),
    State({"type": "br-band", "field": ALL, "index": ALL}, "value"),
    State({"type": "br-band", "field": ALL, "index": ALL}, "id"),
    State("br-out-folder", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def start_build(n, model, rec_folder, band_vals, band_ids, out, sid):
    if not n:
        return no_update, no_update, no_update, no_update
    if not model:
        return alert("Choose a model.", "warning"), True, False, True
    job = _JOBS.get(sid)
    if job and job.get("status") == "running":
        return no_update, False, True, False
    ho = _held_out(model)
    if ho is not None:
        recs = ho[0]
        if not recs:
            return alert("None of this model's held-out recordings are on this "
                         "computer.", "danger"), True, False, True
    else:
        recs = _folder_recordings(rec_folder or "")
        if not recs:
            return alert("Choose a folder with recordings and channel maps.",
                         "warning"), True, False, True
    bands, err = _validate_bands(_rows_from_inputs(band_vals, band_ids))
    if err:
        return alert(err, "warning"), True, False, True
    err = _check_out_folder(out or "", recs)
    if err:
        return alert(err, "warning"), True, False, True
    out = str(Path(os.path.expanduser(out)))
    _JOBS[sid] = {"status": "running", "i": 0, "n": _n_files_animals(recs)[0],
                  "msg": "Starting", "stop": False, "t0": time.time()}
    threading.Thread(target=_run_build, args=(sid, model, recs, out, bands),
                     daemon=True, name=f"blind-review-{sid[:8]}").start()
    return _progress_bar(_JOBS[sid]), False, True, False


def _progress_bar(job):
    i, n = job.get("i", 0), max(job.get("n", 1), 1)
    el = int(time.time() - job.get("t0", time.time()))
    return html.Div([
        dbc.Progress(value=int(100 * i / n), striped=True, animated=True,
                     label=f"{i}/{n}", style={"height": "22px", "marginBottom": "6px"}),
        html.Div(f"{job.get('msg', '')} · {el // 60}:{el % 60:02d} elapsed",
                 style={**_MUTED, "textAlign": "center"}),
    ])


@callback(
    Output("br-stop-btn", "disabled", allow_duplicate=True),
    Input("br-stop-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stop_build(n, sid):
    job = _JOBS.get(sid)
    if n and job and job.get("status") == "running":
        job["stop"] = True
        job["msg"] = "Stopping after the current recording"
    return True


@callback(
    Output("br-build-progress", "children", allow_duplicate=True),
    Output("br-build-poll", "disabled", allow_duplicate=True),
    Output("br-build-btn", "disabled", allow_duplicate=True),
    Output("br-stop-btn", "disabled", allow_duplicate=True),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("br-build-poll", "n_intervals"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def poll_build(_n, sid, refresh):
    nu = no_update
    job = _JOBS.get(sid)
    if not job:
        return nu, True, nu, nu, nu
    st = job.get("status")
    if st == "running":
        return _progress_bar(job), nu, nu, nu, nu
    _JOBS.pop(sid, None)
    if st == "stopped":
        return alert("Stopped. Nothing was written.", "warning"), True, False, True, nu
    if st == "error":
        return alert(f"Build failed: {job.get('error')}", "danger"), True, False, True, nu
    if st == "empty":
        return alert(f"Review {job.get('review_id')} built, but no never-reviewed "
                     "detections fell in the bands, so there is nothing to review.",
                     "warning"), True, False, True, nu
    # done: the session now holds the review; re-render shows the review view
    return "", True, True, True, (refresh or 0) + 1


# ---------------------------------------------------------------------------
# Review view: banner (Score, Merge, Exit) above the blinded reviewer
# ---------------------------------------------------------------------------

def _review_view(sid, state) -> html.Div:
    from eeg_seizure_analyzer.dash_app.pages import training
    try:
        text = banner_text(state)
    except Exception:
        traceback.print_exc()
        text = "Blind review"
    return html.Div([banner(text), training.layout(sid)])


def banner(text: str = "") -> html.Div:
    btn = {"size": "sm", "className": "btn-ned-secondary"}
    return html.Div(
        id="blind-banner",
        children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px",
                            "padding": "10px 24px", "flexWrap": "wrap",
                            "background": "var(--ned-sidebar)",
                            "borderBottom": "1px solid var(--ned-border)"}, children=[
                html.Span(text, id="blind-banner-text",
                          style={"fontWeight": "600", "fontSize": "0.9rem"}),
                html.Div(style={"flex": "1"}),
                dbc.Button("Score", id="blind-score-btn", **btn),
                dbc.Button("Merge into annotations…", id="blind-merge-btn", **btn),
                dbc.Button("Exit review", id="blind-exit-btn", **btn,
                           title="Close the review and reload the recordings you had "
                                 "open before"),
            ]),
            dcc.Interval(id="blind-banner-poll", interval=4000),
            dbc.Modal(id="blind-score-modal", size="lg", is_open=False, children=[
                dbc.ModalHeader(dbc.ModalTitle("Blind review results")),
                dbc.ModalBody(id="blind-score-body"),
            ]),
            dbc.Modal(id="blind-merge-modal", size="lg", is_open=False, children=[
                dbc.ModalHeader(dbc.ModalTitle("Merge verdicts into the annotations")),
                dbc.ModalBody(id="blind-merge-body"),
                dbc.ModalFooter([
                    dbc.Input(id="blind-merge-annotator", type="text", size="sm",
                              placeholder="Annotator name", style={"maxWidth": "220px"}),
                    dbc.Button("Merge", id="blind-merge-confirm", color="danger",
                               disabled=True),
                ]),
            ]),
        ],
    )


def banner_text(state) -> str:
    rid = state.extra.get("blind_review_id")
    if not rid:
        return ""
    review = br.load_review(rid) or {}
    n, k = br.review_progress(rid)
    files = state.extra.get("project_files") or []
    m = len(files)
    model = review.get("model", "")
    return (f"Blind review{f' of {model}' if model else ''} — {n} events in {m} "
            f"recording{'s' if m != 1 else ''}, {k} reviewed")


@callback(
    Output("blind-banner-text", "children"),
    Input("blind-banner-poll", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def refresh_banner(_n, sid):
    state = server_state._sessions.get(sid) if sid else None
    if not review_open(state):
        return no_update
    try:
        return banner_text(state)
    except Exception:
        traceback.print_exc()
        return no_update


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("blind-exit-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def exit_review(n, sid, refresh):
    """Close the review, bring back the user's project, show the chooser."""
    if not n:
        return no_update
    state = server_state.get_session(sid)
    err = close_review(state)
    if err:
        state.extra["blind_msg"] = err
    return (refresh or 0) + 1


def _share_bar(share, lo, hi) -> html.Div:
    """Horizontal bar: the share as a fill, its 95% interval as a darker band."""
    def pct(x):
        return 0 if x is None or (isinstance(x, float) and math.isnan(x)) else max(0, min(100, 100 * x))
    return html.Div(style={"position": "relative", "height": "10px", "borderRadius": "5px",
                           "background": "var(--ned-border)", "margin": "6px 0 4px"},
                    children=[
        html.Div(style={"position": "absolute", "left": 0, "top": 0, "bottom": 0,
                        "width": f"{pct(share)}%", "borderRadius": "5px",
                        "background": "var(--ned-accent)", "opacity": 0.85}),
        html.Div(title="95% interval", style={
            "position": "absolute", "top": "-3px", "bottom": "-3px",
            "left": f"{pct(lo)}%", "width": f"{max(0.5, pct(hi) - pct(lo))}%",
            "border": "2px solid var(--ned-text)", "borderRadius": "4px", "opacity": 0.55}),
    ])


def _band_card(r: dict, thr: float) -> html.Div:
    reported = r["lo"] >= thr
    title = (f"Scores {r['band']}: what the app reports" if reported
             else f"Scores {r['band']}: below the threshold, not reported")
    n, k = r["reviewed"], r["sample_seizures"]
    kids = [html.Div(title, style={"fontWeight": "600", "marginBottom": "2px"})]
    if not r["detections"]:
        kids.append(html.Div("The model made no detections in this band.", style=_MUTED))
        return html.Div(kids, className="ned-card",
                        style={"padding": "12px 14px", "marginBottom": "10px"})
    if n:
        kids += [
            html.Div([html.Span(f"{k} of {n}", style={"fontWeight": "600"}),
                      f" reviewed events were seizures ({_fmt_pct(r['sample_share'])}; "
                      f"likely between {_fmt_pct(r['ci_low'])} and {_fmt_pct(r['ci_high'])})."]),
            _share_bar(r["sample_share"], r["ci_low"], r["ci_high"]),
        ]
    else:
        kids.append(html.Div("No events reviewed in this band yet.", style=_MUTED))
    est = r["est_seizures_never_reviewed"]
    kids.append(html.Div(
        f"The model made {r['detections']} detections here: {r['annotated_seizure']} were "
        f"already annotated as seizures and {r['annotated_not_seizure']} as not seizures. "
        f"Of the {r['never_reviewed']} nobody had annotated, "
        + (f"about {_fmt_num(est, 0)} are probably seizures." if n
           else "the share of seizures isn't known yet."),
        style={"fontSize": "0.85rem", "marginTop": "4px"}))
    if n and not reported:
        sh = r["sample_share"]
        msg = ("Most of what the threshold leaves out here are seizures: a lower threshold "
               "would find more of them." if sh >= 0.5 else
               "Few seizures here: the threshold mostly leaves out non-seizures." if sh <= 0.2
               else "Mixed: a lower threshold would add seizures and false alarms in "
                    "similar numbers.")
        kids.append(html.Div(msg, style={**_MUTED, "marginTop": "4px"}))
    if r["pending"]:
        kids.append(html.Div(f"{r['pending']} sampled event{'s' if r['pending'] != 1 else ''} "
                             f"in this band {'are' if r['pending'] != 1 else 'is'} still pending.",
                             style={**_MUTED, "marginTop": "2px"}))
    return html.Div(kids, className="ned-card",
                    style={"padding": "12px 14px", "marginBottom": "10px"})


def score_card(res: dict, folder: str = "") -> html.Div:
    thr = res["threshold"]
    at = res["at_threshold"]
    top = [r for r in res["bands"] if r["lo"] >= thr]
    annotated = sum(r["annotated_seizure"] for r in top)
    new = sum(r["est_seizures_never_reviewed"] for r in top
              if r["reviewed"]) if top else float("nan")
    pending = sum(r["pending"] for r in res["bands"])
    head = [html.Div([
        f"At its threshold ({thr:g}), {res['model']} made ",
        html.Strong(f"{at['detections']} detections"),
        " on these recordings. About ",
        html.Strong(f"{_fmt_num(at['est_seizures'], 0)} ({_fmt_pct(at['est_share'])})"),
        " are seizures."], style={"fontSize": "0.95rem"})]
    if top and any(r["reviewed"] for r in top):
        head.append(html.Div(
            f"{annotated} of them were already annotated as seizures; about "
            f"{_fmt_num(new, 0)} more are seizures nobody had annotated.",
            style={"fontSize": "0.85rem", "marginTop": "2px"}))
    if pending:
        head.append(html.Div(f"{pending} event{'s' if pending != 1 else ''} still pending: "
                             "the numbers update as you review.",
                             style={**_MUTED, "marginTop": "2px"}))
    kids = [html.Div(head, style={"marginBottom": "12px"})]
    kids += [_band_card(r, thr) for r in res["bands"]]
    kids.append(html.Div(
        "Each band's share comes from the random sample you reviewed; \"likely between\" is "
        "the 95% interval. Detections already annotated before the review count as they "
        "were labelled." + (f" Saved in {folder}/ned_review_results.json." if folder else ""),
        style={**_MUTED, "marginTop": "4px"}))
    return html.Div(kids)


@callback(
    Output("blind-score-modal", "is_open"),
    Output("blind-score-body", "children"),
    Input("blind-score-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def show_score(n, sid):
    state = server_state._sessions.get(sid) if sid else None
    rid = state.extra.get("blind_review_id") if state else None
    if not n or not rid:
        return no_update, no_update
    try:
        res = br.score_review(rid)
    except Exception as e:
        traceback.print_exc()
        return True, alert(f"Could not score: {e}", "danger")
    return True, score_card(res, (br.load_review(rid) or {}).get("folder", ""))


def merge_preview(plan: dict, review: dict) -> html.Div:
    rows = [html.Tr([html.Td(os.path.basename(f["edf"]), title=f["edf"]),
                     html.Td(f["added"]), html.Td(f["updated"]),
                     html.Td(f["skipped_reviewed"])]) for f in plan["files"]]
    kids = []
    if review.get("merged"):
        kids.append(alert(f"This review was already merged on "
                          f"{str(review['merged'].get('at', ''))[:16].replace('T', ' ')} "
                          "(UTC). Verdicts merged then are now skipped.", "info"))
    if not plan["files"]:
        kids.append(html.Div("No reviewed events yet: nothing to merge.", style=_MUTED))
        return html.Div(kids)
    kids += [
        html.Div(f"{plan['added']} added, {plan['updated']} updated (pending candidates "
                 f"given your verdict), {plan['skipped_reviewed']} skipped because they "
                 "were already confirmed or rejected.", style={"marginBottom": "8px"}),
        html.Table(className="table table-sm", style={**_TABLE, "fontSize": "0.82rem"}, children=[
            html.Thead(html.Tr([html.Th("Recording"), html.Th("Added"), html.Th("Updated"),
                                html.Th("Skipped (already reviewed)")])),
            html.Tbody(rows)]),
        alert(f"These recordings were held out from {review.get('model', 'this model')}. "
              "Once their verdicts are merged into the annotations, future comparisons "
              "of models should use other recordings.", "warning"),
        html.Div("Each annotation file is backed up before it is changed.", style=_MUTED),
    ]
    return html.Div(kids)


@callback(
    Output("blind-merge-modal", "is_open"),
    Output("blind-merge-body", "children"),
    Output("blind-merge-confirm", "disabled"),
    Output("blind-merge-annotator", "value"),
    Input("blind-merge-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def show_merge(n, sid):
    state = server_state._sessions.get(sid) if sid else None
    rid = state.extra.get("blind_review_id") if state else None
    if not n or not rid:
        return no_update, no_update, no_update, no_update
    try:
        plan = br.merge_review(rid, apply=False)
    except Exception as e:
        traceback.print_exc()
        return True, alert(f"Could not prepare the merge: {e}", "danger"), True, no_update
    can = bool(plan["added"] or plan["updated"])
    return (True, merge_preview(plan, br.load_review(rid) or {}), not can,
            state.extra.get("tr_annotator", "") or "")


@callback(
    Output("blind-merge-body", "children", allow_duplicate=True),
    Output("blind-merge-confirm", "disabled", allow_duplicate=True),
    Input("blind-merge-confirm", "n_clicks"),
    State("blind-merge-annotator", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_merge(n, annotator, sid):
    state = server_state._sessions.get(sid) if sid else None
    rid = state.extra.get("blind_review_id") if state else None
    if not n or not rid:
        return no_update, no_update
    try:
        plan = br.merge_review(rid, apply=True, annotator=(annotator or "").strip())
    except Exception as e:
        traceback.print_exc()
        return alert(f"Merge failed: {e}", "danger"), True
    backup = br.REVIEWS_DIR / rid / "merge_backup"
    return html.Div([
        alert(f"Merged: {plan['added']} added, {plan['updated']} updated, "
              f"{plan['skipped_reviewed']} skipped (already reviewed).", "success"),
        html.Div(f"Backups of the original annotation files: {backup}", style=_MUTED),
    ]), True
