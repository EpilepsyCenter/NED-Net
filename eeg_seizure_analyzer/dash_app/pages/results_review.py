"""Results > Review events: review a database's events in the Training reviewer.

The "Review events" button under the Results event table takes the seizure events
the filters show, finds their recordings, links the events into the recordings'
annotation files (``eeg_seizure_analyzer.results_review``) and opens the review: a
banner above Training > Seizure's reviewer, in the Results tab. Every verdict is
written to the annotation file and to the database at once.

As in a blind review, opening replaces the session's project with the review's
recordings (so the file bar and "Next pending" work); the user's own project is
stashed first and "Exit review" brings it back.

Session keys (state.extra):
  rr_review   the open review: project, folder (a label standing in for the project
              folder), ids (database events under review), files
  rr_stash    the user's project before the review + the path of the loaded recording
  rr_msg      a one-off message for the Results page (e.g. a failed restore)
"""
from __future__ import annotations

import copy
import json
import os
import traceback
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update

from eeg_seizure_analyzer import db
from eeg_seizure_analyzer import results_review as rr
from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app.components import alert
from eeg_seizure_analyzer.io.annotation_store import load_annotations

_FOLDERS_PATH = Path.home() / ".eeg_seizure_analyzer" / "results_review_folders.json"


def _saved_folder(project: str) -> str:
    try:
        return json.loads(_FOLDERS_PATH.read_text()).get(project, "")
    except Exception:
        return ""


def _save_folder(project: str, folder: str) -> None:
    try:
        try:
            data = json.loads(_FOLDERS_PATH.read_text())
        except Exception:
            data = {}
        data[project] = folder
        _FOLDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FOLDERS_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Opening and closing a review (session state)
# ---------------------------------------------------------------------------

def review_open(state) -> bool:
    return state is not None and rr.review_open(state.extra)


def _pending_per_file(review: dict) -> list[int]:
    scope = set(review["ids"])
    out = []
    for f in review["files"]:
        try:
            anns = load_annotations(f) or []
        except Exception:
            anns = []
        out.append(sum(a.label == "pending" and rr.in_scope(a, review["project"], scope)
                       for a in anns))
    return out


def open_review(state, review: dict) -> str | None:
    """Load the review's recordings as this session's project, after stashing the
    user's own. Returns an error message, or None on success."""
    from eeg_seizure_analyzer.dash_app.pages import blind_review as brp
    from eeg_seizure_analyzer.dash_app.pages.upload import _load_edf_into_state
    if brp.review_open(state):
        return "A blind review is open. Exit it first (Model > Blind Review)."
    if not review.get("files"):
        return "None of these events can be reviewed here."
    if not review_open(state):   # don't stash a review over the user's project
        rec = state.recording
        state.extra["rr_stash"] = {
            "recording": getattr(rec, "source_path", None) if rec is not None else None,
            "extra": {k: copy.deepcopy(state.extra[k]) for k in brp._STASH_KEYS
                      if k in state.extra},
        }
    brp._clear_recording_state(state)
    names = [os.path.basename(f) for f in review["files"]]
    files = []
    for f, name in zip(review["files"], names):
        p = Path(f)
        shown = name if names.count(name) == 1 else f"{p.parent.name}/{name}"
        files.append({"edf_path": f, "filename": shown, "has_detections": False,
                      "has_annotations": True,
                      "has_ids": p.with_name(p.stem + "_ned_channels.json").exists()})
    pend = _pending_per_file(review)
    first = next((i for i, n in enumerate(pend) if n), 0)
    state.extra["rr_review"] = {k: review[k] for k in ("project", "folder", "ids", "files")}
    state.extra["project_files"] = files
    state.extra["project_folder"] = review["folder"]
    state.extra["project_active_idx"] = first
    state.extra["show_upload_form"] = False
    state.extra["show_batch_form"] = False
    try:
        _load_edf_into_state(state, files[first]["edf_path"])
    except Exception as e:
        traceback.print_exc()
        close_review(state)
        return f"Could not open the first recording: {e}"
    return None


def close_review(state) -> str | None:
    """End the review: bring every verdict in the annotation files to the database,
    restore the stashed project and reload the recording loaded before. Returns a
    message if something needs the user's attention."""
    from eeg_seizure_analyzer.dash_app.pages import blind_review as brp
    from eeg_seizure_analyzer.dash_app.pages.upload import _load_edf_into_state
    review = state.extra.pop("rr_review", None) or {}
    stash = state.extra.pop("rr_stash", None) or {}
    msg = None
    if review:
        if review["project"] == db.get_active_project():
            try:
                rr.sync_files(review["files"], review["project"], set(review["ids"]))
            except Exception as e:
                traceback.print_exc()
                msg = f"The verdicts could not be written to the database: {e}"
        else:
            msg = (f"The project database was switched during the review. The verdicts "
                   f"are in the annotation files; open a review of project "
                   f"{review['project']} again to bring them into its database.")
    brp._clear_recording_state(state)
    for k in brp._STASH_KEYS:
        state.extra.pop(k, None)
    state.extra.update(stash.get("extra") or {})
    path = stash.get("recording")
    if not path:
        return msg
    if not os.path.isfile(path):
        return msg or f"The recording loaded before the review ({path}) was not found."
    try:
        _load_edf_into_state(state, path)
    except Exception as e:
        traceback.print_exc()
        return msg or f"Could not reload {os.path.basename(path)}: {e}"
    return msg


def take_view(sid, state):
    """The Results tab's content while a review is open, else None."""
    if state is None:
        return None
    if state.extra.get("rr_review") and not review_open(state):
        # Other recordings were loaded (Load tab) while the review was open: they
        # replaced the review, and with it the stashed project. The verdicts given
        # so far are already in the annotation files and the database.
        for k in ("rr_review", "rr_stash"):
            state.extra.pop(k, None)
        state.extra["rr_msg"] = ("The review was closed because other recordings were "
                                 "loaded. The verdicts given so far are saved.")
    if not review_open(state):
        return None
    from eeg_seizure_analyzer.dash_app.pages import training
    try:
        text = banner_text(state)
    except Exception:
        traceback.print_exc()
        text = "Review of database events"
    return html.Div([banner(text), training.layout(sid)])


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def banner_text(state) -> str:
    review = state.extra.get("rr_review") or {}
    p = rr.progress(review.get("ids") or [])
    m = len(review.get("files") or [])
    done = p["confirmed"] + p["rejected"]
    return (f"Reviewing the events of project {review.get('project', '')} — "
            f"{p['total']} events in {m} recording{'s' if m != 1 else ''}, {done} reviewed "
            f"({p['confirmed']} confirmed, {p['rejected']} rejected)")


def banner(text: str = "") -> html.Div:
    return html.Div(
        id="rr-banner",
        children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px",
                            "padding": "10px 24px", "flexWrap": "wrap",
                            "background": "var(--ned-sidebar)",
                            "borderBottom": "1px solid var(--ned-border)"}, children=[
                html.Span(text, id="rr-banner-text",
                          style={"fontWeight": "600", "fontSize": "0.9rem"}),
                html.Div(style={"flex": "1"}),
                html.Span("Verdicts are saved to the database and to the annotation "
                          "files as you go.",
                          style={"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}),
                dbc.Button("Exit review", id="rr-exit-btn", size="sm",
                           className="btn-ned-secondary",
                           title="Back to Results, with the recordings you had open "
                                 "before"),
            ]),
            dcc.Interval(id="rr-banner-poll", interval=4000),
        ],
    )


@callback(
    Output("rr-banner-text", "children"),
    Input("rr-banner-poll", "n_intervals"),
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
    Input("rr-exit-btn", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def exit_review(n, sid, refresh):
    """Close the review, bring back the user's project, show Results."""
    if not n:
        return no_update
    state = server_state.get_session(sid)
    err = close_review(state)
    if err:
        state.extra["rr_msg"] = err
    return (refresh or 0) + 1


# ---------------------------------------------------------------------------
# Starting a review from the Results page
# ---------------------------------------------------------------------------

def start_controls(sid) -> list:
    """The button, a one-off message and the dialog, for the Results layout."""
    state = server_state.get_session(sid) if sid else None
    msg = state.extra.pop("rr_msg", None) if state is not None else None
    lab = {"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}
    return [
        dbc.Button("Review events…", id="rr-start-btn", size="sm",
                   className="btn-ned-secondary ms-3",
                   title="Confirm or reject the seizure events shown by the filters, "
                         "one by one, on their recordings"),
        html.Div(alert(msg, "warning") if msg else None, id="rr-start-msg",
                 style={"marginTop": "8px"}),
        dbc.Modal(id="rr-modal", size="lg", is_open=False, children=[
            dbc.ModalHeader(dbc.ModalTitle("Review events")),
            dbc.ModalBody([
                html.Div(id="rr-modal-summary"),
                html.Label("Recordings are in this folder (only needed for recordings "
                           "not found where the database says)", style=lab,
                           className="mt-3"),
                dbc.InputGroup([
                    dbc.Input(id="rr-folder", type="text", size="sm", debounce=True,
                              value=_saved_folder(db.get_active_project()),
                              placeholder="Folder that holds the recordings "
                                          "(sub-folders are searched)"),
                    dbc.Button("Browse…", id="rr-folder-browse", size="sm",
                               className="btn-ned-secondary"),
                ], size="sm"),
                html.Div(id="rr-modal-error", style={"marginTop": "10px"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("Start review", id="rr-start-confirm", color="info",
                           disabled=True),
            ]),
        ]),
    ]


_FILTER_STATES = [
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
]


def _scope_events(filters) -> list[dict]:
    """The seizure events the Results filters show, minus those that cannot or need
    not be reviewed (see ``results_review.prepare_review``)."""
    from eeg_seizure_analyzer.dash_app.pages.results import _filtered_events
    return [e for e in _filtered_events(*filters)
            if (e.get("category") or "seizure") == "seizure"]


def _summary(events: list[dict], located: dict) -> tuple[html.Div, int]:
    """What a review of these events would cover, and how many events it has."""
    usable = [e for e in events if not (e.get("excluded") and not e.get("review_label"))]
    hand = len(events) - len(usable)
    no_ch = sum(e.get("channel") is None for e in usable)
    usable = [e for e in usable if e.get("channel") is not None]
    found = [e for e in usable if located.get(e.get("path") or "")]
    paths = {e.get("path") for e in usable}
    missing = sorted(p for p in paths if not located.get(p or ""))
    done = sum(bool(e.get("review_label")) for e in found)
    small = {"fontSize": "0.82rem", "color": "var(--ned-text-muted)"}
    rows = [html.Div(
        f"{len(found)} events in {len(paths) - len(missing)} recordings can be reviewed; "
        f"{done} of them already have a verdict.", style={"fontWeight": "600"})]
    if missing:
        rows.append(html.Div(
            f"{len(usable) - len(found)} events are left out because {len(missing)} "
            f"recording{'s were' if len(missing) != 1 else ' was'} not found on this "
            f"computer, for example {missing[0]}", style={**small, "color": "#d29922"}))
    if hand:
        rows.append(html.Div(f"{hand} events you excluded by hand are left out.", style=small))
    if no_ch:
        rows.append(html.Div(f"{no_ch} events from an older analysis have no channel "
                             f"recorded and cannot be shown.", style=small))
    rows.append(html.Div(
        "The events are added to the recordings' annotation files as candidates (each "
        "file is backed up first). Events that overlap a seizure you already confirmed "
        "or rejected take that verdict. Rejected events are excluded from the results; "
        "confirmed events keep the boundaries you set.", style={**small, "marginTop": "8px"}))
    return html.Div(rows), len(found)


@callback(
    Output("rr-modal", "is_open"),
    Output("rr-modal-summary", "children"),
    Output("rr-start-confirm", "disabled"),
    Output("rr-modal-error", "children"),
    Input("rr-start-btn", "n_clicks"),
    Input("rr-folder", "value"),
    State("rr-modal", "is_open"),
    *_FILTER_STATES,
    prevent_initial_call=True,
)
def show_dialog(n, folder, is_open, *filters):
    """Open the dialog (button) or refresh its summary (folder changed)."""
    from dash import ctx
    opening = ctx.triggered_id == "rr-start-btn"
    if opening and not n:
        return (no_update,) * 4
    if not opening and not is_open:
        return (no_update,) * 4
    if filters[0] == "spike_cnn":
        return (True, alert("Interictal spikes are not reviewed here: there are too many "
                            "to go through one by one. Switch to Seizures.", "warning"),
                True, None)
    try:
        events = _scope_events(filters)
        located = rr.locate_recordings([e.get("path") or "" for e in events], folder or None)
        body, n_ok = _summary(events, located)
    except Exception as e:
        traceback.print_exc()
        return True, alert(f"Could not read the events: {e}", "danger"), True, None
    return True, body, n_ok == 0, None


@callback(
    Output("rr-folder", "value", allow_duplicate=True),
    Input("rr-folder-browse", "n_clicks"),
    prevent_initial_call=True,
)
def browse_folder(n):
    if not n:
        return no_update
    from eeg_seizure_analyzer.dash_app.pages.upload import _browse_folder
    return _browse_folder("Select the folder that holds the recordings") or no_update


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Output("rr-modal-error", "children", allow_duplicate=True),
    Input("rr-start-confirm", "n_clicks"),
    State("rr-folder", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    *_FILTER_STATES,
    prevent_initial_call=True,
)
def start_review(n, folder, sid, refresh, *filters):
    """Link the events into the annotation files and open the review."""
    if not n:
        return no_update, no_update
    state = server_state.get_session(sid)
    project = db.get_active_project()
    try:
        events = _scope_events(filters)
        located = rr.locate_recordings([e.get("path") or "" for e in events], folder or None)
        review = rr.prepare_review(events, located, project)
    except Exception as e:
        traceback.print_exc()
        return no_update, alert(f"Could not prepare the review: {e}", "danger")
    if folder:
        _save_folder(project, folder)
    err = open_review(state, review)
    if err:
        return no_update, alert(err, "warning")
    return (refresh or 0) + 1, no_update
