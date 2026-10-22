"""Training tab, Background mode: review seizure-free background stretches.

Training needs "no seizure" examples, but random recording time can hide seizures
nobody annotated. The Detection tab proposes quiet stretches by a rule (pending); here
the user confirms or rejects them like events, or adds stretches by hand in Browse
mode. Background is its own annotation mode with its own file
(``<stem>_ned_background_annotations.json``) and colours (background_style), so a
rejected background never mixes with rejected seizure events. The recording's seizure
annotations are shown as read-only context, and a stretch that overlaps one is flagged.

The background file is the source of truth: every change is saved at once, and the page
re-reads the file when it changed on disk (e.g. after "Propose background" in the
Detection tab). State keys use the ``trb_`` prefix, component ids ``trb-``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, no_update, ctx, clientside_callback
import dash_bootstrap_components as dbc

from eeg_seizure_analyzer.dash_app import server_state
from eeg_seizure_analyzer.dash_app import inspector_view as iv
from eeg_seizure_analyzer.dash_app.background_style import BACKGROUND_LABEL, bg_colors
from eeg_seizure_analyzer.dash_app.components import (
    fit_y_range,
    loading,
    alert,
    apply_fig_theme,
    get_plotly_theme,
    icon,
    metric_card,
    no_recording_placeholder,
    page_header,
    split_page,
)
from eeg_seizure_analyzer.dash_app.pages.training import (
    _LABEL_COLORS as _SZ_COLORS,
    _minmax_downsample,
)
from eeg_seizure_analyzer.io.annotation_store import (
    BACKGROUND_EVENT_TYPE,
    AnnotatedEvent,
    annotation_json_path,
    background_annotation_json_path,
    load_annotations,
    load_background_annotations,
    save_background_annotations,
)
from eeg_seizure_analyzer.processing.preprocess import bandpass_filter

_MUTED = {"fontSize": "0.78rem", "color": "var(--ned-text-muted)"}
_LABELS = ("pending", "confirmed", "rejected")


# ── State and file access ─────────────────────────────────────────────


def _sync_recording(state) -> None:
    """Forget the review position when a different recording is loaded.

    The app has several load paths (upload, folder, file bar); this keeps the page
    right whichever one ran, without relying on each of them clearing ``trb_`` keys.
    """
    rec = state.recording
    path = rec.source_path if rec is not None else None
    if state.extra.get("trb_for") != path:
        for k in ("trb_current_idx", "trb_view", "trb_browse_start", "trb_cache",
                  "trb_sz_cache", "trb_annotations"):
            state.extra.pop(k, None)
        state.extra["trb_for"] = path


def _file_stamp(path) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


def _sort_key(a: AnnotatedEvent):
    # Original onset, so a boundary edit never reorders the review list
    on = a.original_onset_sec if a.original_onset_sec is not None else a.onset_sec
    return (a.channel, on, a.event_id)


def has_background_file(state) -> bool:
    rec = state.recording
    return bool(rec is not None and rec.source_path
                and background_annotation_json_path(rec.source_path).is_file())


def _get_annotations(state) -> list[AnnotatedEvent]:
    """Background stretches of the loaded recording, re-read when the file changed."""
    rec = state.recording
    if rec is None:
        return []
    if not rec.source_path:  # no file to save to: keep them in memory only
        return [AnnotatedEvent.from_dict(d) for d in state.extra.get("trb_annotations", [])]
    path = background_annotation_json_path(rec.source_path)
    stamp = _file_stamp(path)
    cache = state.extra.get("trb_cache")
    if stamp is None:
        return []
    if not cache or tuple(cache.get("stamp") or ()) != stamp:
        try:
            anns = load_background_annotations(rec.source_path) or []
        except Exception:
            import traceback
            traceback.print_exc()
            anns = []
        anns = sorted(anns, key=_sort_key)
        cache = {"stamp": list(stamp), "anns": [a.to_dict() for a in anns]}
        state.extra["trb_cache"] = cache
    return [AnnotatedEvent.from_dict(d) for d in cache["anns"]]


def _auto_save(state, annotations: list[AnnotatedEvent]) -> str | None:
    """Write the background file (atomic). Returns an error message, or None."""
    annotations = sorted(annotations, key=_sort_key)
    rec = state.recording
    if rec is None:
        return "No recording loaded"
    if not rec.source_path:
        state.extra["trb_annotations"] = [a.to_dict() for a in annotations]
        return None
    try:
        save_background_annotations(rec.source_path, annotations,
                                    annotator=_annotator(state))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Could not save the background file: {e}"
    path = background_annotation_json_path(rec.source_path)
    stamp = _file_stamp(path)
    state.extra["trb_cache"] = {"stamp": list(stamp) if stamp else None,
                                "anns": [a.to_dict() for a in annotations]}
    return None


def _seizure_annotations(state) -> list[AnnotatedEvent]:
    """The recording's seizure annotations (any label), read-only context.

    Read from ``_ned_annotations.json``, which the Seizure page saves on every change.
    This page never writes that file.
    """
    rec = state.recording
    if rec is None or not rec.source_path:
        return []
    stamp = _file_stamp(annotation_json_path(rec.source_path))
    if stamp is None:
        return []
    cache = state.extra.get("trb_sz_cache")
    if not cache or tuple(cache.get("stamp") or ()) != stamp:
        try:
            anns = load_annotations(rec.source_path) or []
        except Exception:
            anns = []
        cache = {"stamp": list(stamp),
                 "anns": [[a.channel, a.onset_sec, a.offset_sec, a.label, a.event_id]
                          for a in anns]}
        state.extra["trb_sz_cache"] = cache
    return [AnnotatedEvent(file_path="", animal_id="", annotator="", onset_sec=on,
                           offset_sec=off, channel=ch, label=lbl, source="",
                           event_id=eid)
            for ch, on, off, lbl, eid in cache["anns"]]


def _annotator(state) -> str:
    return state.extra.get("trb_annotator", state.extra.get("tr_annotator", "")) or ""


def background_counts(state) -> dict | None:
    """Pending / confirmed / rejected / manual counts, or None without a file (sidebar)."""
    _sync_recording(state)
    if not has_background_file(state):
        return None
    out = {"pending": 0, "confirmed": 0, "rejected": 0, "manual": 0, "total": 0}
    for a in _get_annotations(state):
        out["total"] += 1
        if a.label in out:
            out[a.label] += 1
        if a.source == "manual":
            out["manual"] += 1
    return out


# ── Lists and small helpers ───────────────────────────────────────────


def _review_list(state, annotations, ch_filter=None, label_filter=None):
    """The stretches review mode steps through (channel and label filters)."""
    ch_filter = state.extra.get("trb_channel_filter") if ch_filter is None else ch_filter
    label_filter = (state.extra.get("trb_label_filter", "")
                    if label_filter is None else label_filter)
    out = list(annotations)
    if ch_filter not in (None, ""):
        try:
            out = [a for a in out if a.channel == int(ch_filter)]
        except (TypeError, ValueError):
            pass
    if label_filter:
        out = [a for a in out if a.label == label_filter]
    return out


def _first_pending(lst) -> int:
    for i, a in enumerate(lst):
        if a.label == "pending":
            return i
    return 0


def _find_next_pending(lst, idx):
    for i in range(idx + 1, len(lst)):
        if lst[i].label == "pending":
            return i
    return None


def _view_key(a: AnnotatedEvent):
    """Identity of a stretch for the view state and the stale-press check."""
    on = a.original_onset_sec if a.original_onset_sec is not None else a.onset_sec
    return [int(a.channel), int(a.event_id), a.source, round(float(on), 4)]


def _same(a: AnnotatedEvent, b: AnnotatedEvent) -> bool:
    return _view_key(a) == _view_key(b)


def _review_window(rec, a: AnnotatedEvent, state):
    return iv.view_window(state, "trb", _view_key(a), a.onset_sec, a.offset_sec,
                          rec.duration_sec)


def _ch_name(rec, ch: int) -> str:
    return rec.channel_names[ch] if 0 <= ch < len(rec.channel_names) else f"Ch{ch}"


def _overlapping(a: AnnotatedEvent, seizures):
    """Seizure annotations overlapping stretch ``a``: (same channel, other channels)."""
    hits = [s for s in seizures if s.onset_sec < a.offset_sec and s.offset_sec > a.onset_sec]
    return ([s for s in hits if s.channel == a.channel],
            [s for s in hits if s.channel != a.channel])


def _bg_badge(a: AnnotatedEvent | None) -> html.Span:
    if a is None:
        return html.Span()
    c = bg_colors(a.label)
    text = f"{BACKGROUND_LABEL} · {a.label}"
    if a.source == "manual":
        text += " (manual)"
    return html.Span(text.upper(), style={
        "fontSize": "0.72rem", "fontWeight": "600", "letterSpacing": "0.5px",
        "padding": "2px 10px", "borderRadius": "12px", "whiteSpace": "nowrap",
        "color": c["text"], "background": c["fill"], "border": f"1px solid {c['line']}",
    })


def _origin(a: AnnotatedEvent) -> str:
    if a.source == "manual":
        return "Added by hand in Training → Background (Browse mode)"
    f = a.features or {}
    k = f.get("k")
    return ("Proposed by the background rule (Seizures page)"
            + (f" · k = {k}" if k is not None else ""))


def _properties(rec, a: AnnotatedEvent | None, n_same: int = 0):
    if a is None:
        return dbc.Row(className="g-2 mb-2")
    f = a.features or {}
    rms, ll = f.get("rms_ratio_max"), f.get("ll_ratio_max")
    return html.Div([dbc.Row([
        dbc.Col(metric_card("Channel", _ch_name(rec, a.channel)), width=2),
        dbc.Col(metric_card("Duration", f"{a.offset_sec - a.onset_sec:.1f}s"), width=2),
        dbc.Col(metric_card("Onset", f"{a.onset_sec:.1f}s"), width=2),
        dbc.Col(metric_card("Max RMS ×med", f"{rms:.2f}" if rms is not None else "—"),
                width=2),
        dbc.Col(metric_card("Max LL ×med", f"{ll:.2f}" if ll is not None else "—"),
                width=2),
        dbc.Col(metric_card("Seizure overlap", str(n_same), accent=n_same > 0), width=2),
    ], className="g-2 mb-1"), html.Div(
        f"From: {_origin(a)}" + (f" · animal {a.animal_id}" if a.animal_id else ""),
        style={"fontSize": "0.75rem", "color": "var(--ned-text-muted)",
               "textAlign": "right", "marginBottom": "8px"})])


def _overlap_warning(rec, a: AnnotatedEvent | None, seizures):
    if a is None:
        return None
    # Only this channel matters: each channel is a separate animal.
    same, _ = _overlapping(a, seizures)
    if not same:
        return None
    ids = ", ".join(f"#{s.event_id} ({s.label})" if s.event_id else f"({s.label})"
                    for s in same[:5])
    return alert(f"This stretch overlaps {len(same)} seizure annotation"
                 f"{'s' if len(same) != 1 else ''} on this channel: {ids}. "
                 "Confirming it would put possible seizure time into background; "
                 "reject it or move its boundaries.", "danger")


def _counts_block(rec, annotations):
    n = {lbl: sum(1 for a in annotations if a.label == lbl) for lbl in _LABELS}
    total = len(annotations)
    reviewed = n["confirmed"] + n["rejected"]
    pct = int(100 * reviewed / total) if total else 0
    per_ch: dict = {}
    for a in annotations:
        d = per_ch.setdefault(_ch_name(rec, a.channel), {lbl: 0 for lbl in _LABELS})
        if a.label in d:
            d[a.label] += 1
    n_manual = sum(1 for a in annotations if a.source == "manual")
    return html.Div(style={"marginBottom": "16px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "marginBottom": "4px"}, children=[
            html.Span([
                f"Background: {reviewed} of {total} reviewed: ",
                html.Span(f"{n['confirmed']} confirmed",
                          style={"color": bg_colors("confirmed")["text"]}), ", ",
                html.Span(f"{n['rejected']} rejected",
                          style={"color": bg_colors("rejected")["text"]}), ", ",
                html.Span(f"{n['pending']} pending",
                          style={"color": bg_colors("pending")["text"]}),
                f" ({n_manual} added by hand)" if n_manual else "",
            ], style=_MUTED),
            html.Span(f"{pct}%", style=_MUTED),
        ]),
        dbc.Progress(value=pct, style={"height": "6px", "backgroundColor": "var(--ned-surface-hover)"},
                     color="info"),
        html.Div(style={"display": "flex", "gap": "16px", "marginTop": "6px",
                        "flexWrap": "wrap"}, children=[
            html.Span(f"{name}: {c['confirmed']}✓ {c['rejected']}✗ {c['pending']}?",
                      style={"fontSize": "0.72rem", "color": "var(--ned-text-muted)",
                             "border": "1px solid var(--ned-border)",
                             "borderRadius": "8px", "padding": "1px 8px"})
            for name, c in per_ch.items()
        ]),
    ])


def _legend():
    def swatch(fill, line, text, dash=False):
        return html.Span(style={"display": "flex", "alignItems": "center", "gap": "4px"},
                         children=[
            html.Span(style={"width": "12px", "height": "12px", "borderRadius": "2px",
                             "background": fill, "display": "inline-block",
                             "border": f"1px {'dashed' if dash else 'solid'} {line}"}),
            html.Span(text, style={"fontSize": "0.72rem", "color": line}),
        ])
    items = [swatch(bg_colors(l)["fill"], bg_colors(l)["line"],
                    f"{BACKGROUND_LABEL} {l}") for l in _LABELS]
    items.append(html.Span("|", style={"color": "var(--ned-border)"}))
    items.append(html.Span(style={"display": "flex", "alignItems": "center", "gap": "4px"},
                           children=[
        html.Span(style={"width": "12px", "height": "12px", "borderRadius": "2px",
                         "display": "inline-block",
                         "border": f"1px dashed {_SZ_COLORS[l]['line']}"})
        for l in ("pending", "confirmed", "rejected")
    ] + [html.Span("Seizure annotations (context, read-only; outline = seizure label)",
                   style={"fontSize": "0.72rem", "color": "var(--ned-text-muted)"})]))
    return html.Div(style={"display": "flex", "gap": "14px", "marginBottom": "6px",
                           "flexWrap": "wrap", "alignItems": "center"},
                    children=[html.Span("Legend:", style={**_MUTED, "fontSize": "0.72rem",
                                                          "fontWeight": "600"})] + items)


def _no_file_message():
    return alert(
        "This recording has no background file yet. Background stretches are proposed "
        "on the Seizures page (“Propose background”), or add them here in "
        "Browse mode: select a stretch with Add Background.", "info")


# ── Figures ───────────────────────────────────────────────────────────


def _sz_line(label: str) -> str:
    return _SZ_COLORS.get(label, _SZ_COLORS["pending"])["line"]


def _empty_figure(text: str, height: int = 400) -> go.Figure:
    fig = go.Figure()
    apply_fig_theme(fig)
    fig.add_annotation(text=text, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=15))
    fig.update_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _default_yrange(state, rec) -> float:
    yr = state.extra.get("trb_yrange")
    if yr:
        return float(yr)
    yr = state.extra.get("tr_yrange") or state.extra.get("viewer_settings", {}).get("yrange")
    if yr:
        return float(yr)
    n_samp = min(int(10 * rec.fs), rec.n_samples)
    ptps = [float(np.ptp(rec.data[i, :n_samp])) for i in range(rec.n_channels)]
    return float(f"{(float(np.median(ptps)) * 1.5 if ptps else 1.0):.3g}")


def _build_review_figure(rec, a: AnnotatedEvent, state, annotations, seizures,
                         y_range=None) -> go.Figure:
    """One channel around the stretch. shapes[0] is the stretch (editable, drag edges)."""
    ch = a.channel
    w0, w1 = _review_window(rec, a, state)
    i0, i1 = int(w0 * rec.fs), min(int(w1 * rec.fs), rec.n_samples)
    data = bandpass_filter(rec.data[ch, i0:i1].astype(np.float64), rec.fs, 1.0, 50.0)
    t = np.linspace(w0, w1, len(data))
    ds_t, ds_y = _minmax_downsample(t, data, target_points=max(2400, int(40 * (w1 - w0))))

    fig = go.Figure(go.Scattergl(
        x=ds_t, y=ds_y, mode="lines", name=_ch_name(rec, ch),
        line=dict(width=0.8, color="#1b2a4a" if get_plotly_theme() == "light" else "#58a6ff")))

    c = bg_colors(a.label)
    fig.add_shape(type="rect", x0=a.onset_sec, x1=a.offset_sec, y0=0, y1=1, yref="paper",
                  fillcolor=c["fill"], line=dict(color=c["line"], width=1.5),
                  layer="below", editable=True, name="highlight")
    # Neighbouring background stretches on this channel
    for b in annotations:
        if b.channel != ch or _same(b, a) or b.offset_sec < w0 or b.onset_sec > w1:
            continue
        cb = bg_colors(b.label)
        fig.add_shape(type="rect", x0=b.onset_sec, x1=b.offset_sec, y0=0, y1=1,
                      yref="paper", fillcolor=cb["fill"], opacity=0.5,
                      line=dict(color=cb["line"], width=0.5), layer="below",
                      editable=False)
    # Seizure annotations on this channel as faint context (other channels are other
    # animals and don't matter here)
    for s in seizures:
        if s.channel != ch or s.offset_sec < w0 or s.onset_sec > w1:
            continue
        col = _sz_line(s.label)
        same = True
        fig.add_shape(type="rect", x0=s.onset_sec, x1=s.offset_sec,
                      y0=0, y1=1, yref="paper", fillcolor="rgba(0,0,0,0)", opacity=0.6,
                      line=dict(color=col, width=1, dash="dash"), layer="above",
                      editable=False)
        if same:
            fig.add_annotation(
                x=max(s.onset_sec, w0), y=1, yref="paper", xanchor="left", yanchor="top",
                text=f"seizure{' #' + str(s.event_id) if s.event_id else ''} ({s.label})",
                showarrow=False, font=dict(size=9, color=col), opacity=0.8)

    half = (float(y_range) / 2.0 if y_range else
            (float(np.ptp(data)) * 0.6 if len(data) else 1.0))
    yc = float(np.mean(data)) if len(data) else 0.0
    unit = rec.units[ch] if ch < len(rec.units) else ""
    fig.update_layout(
        xaxis=dict(title="Time (s)", fixedrange=False, range=[w0, w1],
                   uirevision=f"x_{ch}_{w0:.3f}_{w1:.3f}"),
        yaxis=dict(title=f"Amplitude ({unit})" if unit else "Amplitude", fixedrange=False,
                   range=[yc - half, yc + half]),
        height=400, showlegend=False, dragmode="zoom", uirevision="trb_review_stable",
    )
    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=10, b=40))
    return fig


def _build_overview(rec, a: AnnotatedEvent, state, annotations, seizures) -> go.Figure:
    evs = [(b.onset_sec, b.offset_sec, bg_colors(b.label)["line"])
           for b in annotations if b.channel == a.channel]
    fig = iv.overview_figure(state, "trb", rec, a.channel, _review_window(rec, a, state),
                             evs, (a.onset_sec, a.offset_sec))
    for s in seizures:
        if s.channel == a.channel:
            fig.add_vrect(x0=s.onset_sec, x1=max(s.offset_sec, s.onset_sec + 0.5),
                          fillcolor="rgba(0,0,0,0)", opacity=0.7, layer="above",
                          line=dict(color=_sz_line(s.label), width=1, dash="dot"))
    return fig


def _build_browse_figure(rec, annotations, seizures, start_sec=0.0, window_sec=120.0,
                         channels=None, add_active=False) -> go.Figure:
    """All selected channels; background shaded by label, seizures as dashed outlines.

    Traces are one per channel in ``channels`` order, so a click's curveNumber gives
    its channel (used by Remove).
    """
    channels = channels or list(range(rec.n_channels))
    end_sec = min(start_sec + window_sec, rec.duration_sec)
    i0, i1 = int(start_sec * rec.fs), min(int(end_sec * rec.fs), rec.n_samples)
    n_samp = min(int(10 * rec.fs), rec.n_samples)
    ptps = [float(np.ptp(rec.data[i, :n_samp])) for i in channels]
    spacing = float(np.median(ptps)) * 1.5 if ptps else 1.0
    if spacing <= 0:
        spacing = 1.0
    half = spacing / 2.0

    fig = go.Figure()
    offsets = {}
    for i, ch in enumerate(channels):
        y = bandpass_filter(rec.data[ch, i0:i1].astype(np.float64), rec.fs, 1.0, 50.0)
        offsets[ch] = -i * spacing
        t = np.linspace(start_sec, end_sec, len(y))
        ds_t, ds_y = _minmax_downsample(t, y + offsets[ch])
        fig.add_trace(go.Scattergl(x=ds_t, y=ds_y, mode="lines",
                                   name=_ch_name(rec, ch), line=dict(width=0.8)))

    for b in annotations:
        if b.channel not in offsets or b.offset_sec < start_sec or b.onset_sec > end_sec:
            continue
        c = bg_colors("manual" if b.source == "manual" else b.label)
        x0, x1 = max(b.onset_sec, start_sec), min(b.offset_sec, end_sec)
        o = offsets[b.channel]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=o - half, y1=o + half,
                      fillcolor=c["fill"], line=dict(color=c["line"], width=1),
                      layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=o + half * 0.75,
                           text=f"B#{b.event_id}" + (" +" if b.source == "manual" else ""),
                           showarrow=False, font=dict(size=9, color=c["text"]),
                           opacity=0.9)
    for s in seizures:
        if s.channel not in offsets or s.offset_sec < start_sec or s.onset_sec > end_sec:
            continue
        col = _sz_line(s.label)
        x0, x1 = max(s.onset_sec, start_sec), min(s.offset_sec, end_sec)
        o = offsets[s.channel]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=o - half * 0.95, y1=o + half * 0.95,
                      fillcolor="rgba(0,0,0,0)", opacity=0.6, layer="above",
                      line=dict(color=col, width=1, dash="dash"))
        fig.add_annotation(x=(x0 + x1) / 2, y=o - half * 0.75,
                           text=f"sz{' #' + str(s.event_id) if s.event_id else ''}",
                           showarrow=False, font=dict(size=9, color=col), opacity=0.7)

    n = len(channels)
    fig.update_layout(
        xaxis=dict(title="Time (s)", fixedrange=False, uirevision="x_stable"),
        yaxis=dict(tickvals=[offsets[c] for c in channels],
                   ticktext=[_ch_name(rec, c) for c in channels],
                   zeroline=False, showgrid=False, fixedrange=False,
                   range=[-(n - 1) * spacing - spacing * 1.5, spacing / 2],
                   uirevision="y_stable"),
        height=600, showlegend=False, dragmode="select" if add_active else "zoom",
        uirevision="trb_browse_stable",
    )
    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=80, r=20, t=10, b=40))
    return fig


def _nav_text(rec, a, idx, n):
    if a is None:
        return "No background stretches to review"
    sid = f" [B#{a.event_id}]" if a.event_id > 0 else ""
    animal = f" ({a.animal_id})" if a.animal_id else ""
    return f"Stretch {idx + 1} of {n}{sid} — {_ch_name(rec, a.channel)}{animal}"


# ── Layout ────────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    """Training → Background: review / browse background stretches of the loaded EDF."""
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_recording_placeholder()
    _sync_recording(state)
    rec = state.recording
    annotations = _get_annotations(state)
    seizures = _seizure_annotations(state)
    has_file = has_background_file(state)

    mode = state.extra.get("trb_mode", "review")
    ch_filter = state.extra.get("trb_channel_filter")
    label_filter = state.extra.get("trb_label_filter", "")
    lst = _review_list(state, annotations)
    if "trb_current_idx" not in state.extra:
        state.extra["trb_current_idx"] = _first_pending(lst)  # pending first
    idx = max(0, min(state.extra["trb_current_idx"], len(lst) - 1))
    state.extra["trb_current_idx"] = idx
    cur = lst[idx] if lst else None
    yr = _default_yrange(state, rec)
    browse_window = state.extra.get("trb_browse_window", 120)
    browse_start = state.extra.get("trb_browse_start", 0)
    ch_options = [{"label": rec.channel_names[i], "value": i} for i in range(rec.n_channels)]
    n_same = len(_overlapping(cur, seizures)[0]) if cur else 0

    if cur is not None:
        review_fig = _build_review_figure(rec, cur, state, annotations, seizures, yr)
        ov_fig = _build_overview(rec, cur, state, annotations, seizures)
    else:
        review_fig = _empty_figure("No background stretches to review."
                                   if not annotations else
                                   "No stretches match the channel / label filter.")
        ov_fig = None

    small = {"width": "100px"}

    mode_toggle = dbc.RadioItems(
        id="trb-mode-toggle",
        options=[{"label": "Review", "value": "review"},
                 {"label": "Browse", "value": "browse"}],
        value=mode, inline=True, className="btn-group segmented",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-secondary",
        labelCheckedClassName="btn-ned-primary")

    # ── Work area (left): the stretch under review, or the browse view ──
    main = [
        dcc.Store(id="trb-keyboard-store", data={"key": "", "ts": 0}),
        dcc.Store(id="trb-autoscale-key", data=0),
        html.Div(id="trb-keyboard-listener", style={"display": "none"}),

        page_header(
            "Background Annotation",
            badge=html.Span("Seizure-free stretches for training", className="page-badge",
                            style={"borderColor": bg_colors("pending")["line"],
                                   "color": bg_colors("pending")["text"]}),
            actions=[mode_toggle],
        ),

        html.Div(id="trb-file-message", children=None if has_file else _no_file_message()),

        # ── Review mode ──────────────────────────────────────────────
        html.Div(id="trb-review-mode",
                 style={"display": "block" if mode == "review" else "none"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px",
                            "marginBottom": "8px"}, children=[
                dbc.Button("◀ Prev (← ,)", id="trb-prev-btn", size="sm",
                           className="btn-ned-secondary"),
                html.Div(id="trb-event-nav-text", children=_nav_text(rec, cur, idx, len(lst)),
                         style={"flex": "1", "textAlign": "center", "fontWeight": "600",
                                "fontSize": "0.9rem"}),
                html.Label("Go to #", style={"fontSize": "0.72rem", "margin": "0",
                                             "color": "var(--ned-text-muted)"}),
                dcc.Input(id="trb-jump-to", type="number", min=1, step=1, debounce=True,
                          placeholder="#", className="form-control",
                          style={"width": "60px", "fontSize": "0.78rem",
                                 "textAlign": "center"}),
                dbc.Button("Next (. →) ▶", id="trb-next-btn", size="sm",
                           className="btn-ned-secondary"),
            ]),
            html.Div(id="trb-overlap-warning", children=_overlap_warning(rec, cur, seizures)),

            # Decision bar: boundaries on the left, the label on the right
            html.Div(className="decision-bar", children=[
                html.Div(className="decision-bounds", children=[
                    html.Label("Onset (s)", className="decision-label"),
                    # Identity of the stretch the inputs belong to (stale-press check)
                    dcc.Store(id="trb-shown-key", data=_view_key(cur) if cur else None),
                    dcc.Input(id="trb-onset-input", type="number", step=0.01, debounce=True,
                              value=round(cur.onset_sec, 3) if cur else 0,
                              className="form-control", style=small),
                    html.Label("Offset (s)", className="decision-label"),
                    dcc.Input(id="trb-offset-input", type="number", step=0.01, debounce=True,
                              value=round(cur.offset_sec, 3) if cur else 0,
                              className="form-control", style=small),
                    html.Span(id="trb-duration-display",
                              children=f"({cur.offset_sec - cur.onset_sec:.2f}s)" if cur else "",
                              className="decision-label"),
                ]),
                html.Div(className="decision-actions", children=[
                    html.Div(id="trb-event-status", children=_bg_badge(cur)),
                    dbc.Button("Skip (S)", id="trb-skip-btn", className="btn-ned-secondary"),
                    dbc.Button([html.Span("✗ "), "Reject (R)"], id="trb-reject-btn",
                               className="btn-ned-danger"),
                    dbc.Button([html.Span("✓ "), "Confirm background (C)"],
                               id="trb-confirm-btn", className="btn-ned-primary"),
                ]),
                html.Div("Typed boundaries apply on Confirm / Reject; dragging the "
                         "shaded edges applies at once.",
                         className="decision-label", style={"flexBasis": "100%"}),
            ]),

            html.Div(style={"display": "flex", "alignItems": "center",
                            "justifyContent": "space-between", "flexWrap": "wrap",
                            "gap": "8px 16px", "marginBottom": "4px", "padding": "6px 0",
                            "borderTop": "1px solid var(--ned-border)"}, children=[
                iv.view_controls("trb", iv.context_sec(state, "trb"),
                                 iv.ovspan_sec(state, "trb")),
                html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"},
                         children=[
                    html.Label("Y range", className="decision-label"),
                    dcc.Input(id="trb-yrange", type="number", min=0, step=0.01, value=yr,
                              debounce=True, className="form-control",
                              style={"width": "64px"}),
                    dbc.Button("Autoscale (A)", id="trb-autoscale-btn", size="sm",
                               className="btn-ned-secondary",
                               title="Fit the Y range to this stretch's window"),
                ]),
            ]),
            iv.overview_graph("trb-overview", ov_fig),
            loading(dcc.Graph(
                id="trb-review-graph", figure=review_fig,
                config={"editable": False, "edits": {"shapePosition": True},
                        "scrollZoom": True, "displayModeBar": True},
                style={"borderRadius": "8px"}), type="circle", color="#58a6ff"),
            html.Div(id="trb-review-status"),
        ]),

        # ── Browse mode ──────────────────────────────────────────────
        html.Div(id="trb-browse-mode",
                 style={"display": "block" if mode == "browse" else "none"}, children=[
            dbc.Row([
                dbc.Col([html.Label("Window (s)", style=_MUTED),
                         dcc.Input(id="trb-browse-window", type="number", min=1, max=1800,
                                   step=1, value=browse_window, debounce=True,
                                   className="form-control", style={"width": "100%"})],
                        width=2),
                dbc.Col([html.Label("Start (s)", style=_MUTED),
                         dcc.Input(id="trb-browse-start", type="number", min=0,
                                   max=rec.duration_sec, step=1, value=browse_start,
                                   debounce=True, className="form-control",
                                   style={"width": "100%"})], width=2),
                dbc.Col([html.Label("Channel (for manual add)", style=_MUTED),
                         dcc.Dropdown(id="trb-browse-annotate-channel", options=ch_options,
                                      value=0 if ch_options else None, clearable=False)],
                        width=3),
                dbc.Col([html.Label("​", style={"fontSize": "0.78rem",
                                                     "display": "block"}),
                         html.Div(style={"display": "flex", "gap": "6px"}, children=[
                             dbc.Button("Add Background", id="trb-add-btn", size="sm",
                                        className="btn-ned-secondary", active=False,
                                        title="Then drag across the trace to select a "
                                              "seizure-free stretch"),
                             dbc.Button("Remove Background", id="trb-remove-btn", size="sm",
                                        className="btn-ned-secondary", active=False,
                                        title="Then click a stretch you added by hand"),
                         ])], width=5),
            ], className="g-2 mb-3"),
            html.Div(style={"marginBottom": "8px", "display": "flex", "alignItems": "center",
                            "gap": "8px", "flexWrap": "wrap"}, children=[
                html.Label("Channels:", style={**_MUTED, "margin": "0", "fontWeight": "500"}),
                dbc.Checklist(id="trb-browse-channel-checks", options=ch_options,
                              value=list(range(rec.n_channels)), inline=True),
                html.A("All", id="trb-browse-ch-all", href="#",
                       style={"fontSize": "0.75rem", "color": "var(--ned-accent)",
                              "marginLeft": "4px"}),
                html.A("None", id="trb-browse-ch-none", href="#",
                       style={"fontSize": "0.75rem", "color": "var(--ned-accent)"}),
            ]),
            dcc.Store(id="trb-annotations-version", data=0),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px",
                            "marginBottom": "4px"}, children=[
                dbc.Button(icon("skip-start"), id="trb-nav-start", size="sm",
                           className="btn-ned-secondary btn-icon-only",
                           title="Go to start"),
                dbc.Button(icon("rewind"), id="trb-nav-back-big", size="sm",
                           className="btn-ned-secondary btn-icon-only",
                           title="Back one window"),
                dbc.Button(icon("step-back"), id="trb-nav-back", size="sm",
                           className="btn-ned-secondary btn-icon-only",
                           title="Back half a window"),
                html.Div(id="trb-nav-time-display",
                         children=f"{float(browse_start):.1f}s – "
                                  f"{float(browse_start) + float(browse_window):.1f}s",
                         style={"flex": "1", "textAlign": "center", "fontWeight": "600",
                                "fontSize": "0.9rem"}),
                dbc.Button(icon("step-forward"), id="trb-nav-fwd", size="sm",
                           className="btn-ned-secondary btn-icon-only",
                           title="Forward half a window"),
                dbc.Button(icon("fast-forward"), id="trb-nav-fwd-big", size="sm",
                           className="btn-ned-secondary btn-icon-only",
                           title="Forward one window"),
            ]),
            _legend(),
            loading(dcc.Graph(
                id="trb-browse-graph",
                figure=_build_browse_figure(rec, annotations, seizures,
                                            float(browse_start), float(browse_window))
                if mode == "browse" else go.Figure(),
                config={"scrollZoom": True, "displayModeBar": True},
                style={"borderRadius": "8px"}), type="circle", color="#58a6ff"),
            html.Div(id="trb-browse-status"),
        ]),
    ]

    # ── Panel (right): which stretches, this stretch, progress ──────────
    panel = [
        html.Div(className="panel-field", children=[
            html.Label("Annotator", className="panel-label"),
            dcc.Input(id="trb-annotator", type="text", value=_annotator(state),
                      debounce=True, placeholder="Your name",
                      className="form-control", style={"width": "100%"}),
        ]),
        html.Div(className="panel-field", children=[
            html.Label("Channel", className="panel-label"),
            dcc.Dropdown(id="trb-channel-filter", options=ch_options,
                         value=ch_filter, placeholder="All channels", clearable=True),
            html.Label("Label", className="panel-label", style={"marginTop": "10px"}),
            dcc.Dropdown(id="trb-label-filter",
                         options=[{"label": "All labels", "value": ""}] + [
                             {"label": l.capitalize(), "value": l} for l in _LABELS],
                         value=label_filter, clearable=False),
        ]),
        html.Div(className="panel-field", children=[
            html.Label("Stretch", className="panel-label"),
            html.Div(id="trb-event-properties", children=_properties(rec, cur, n_same)),
        ]),
        html.Div(className="panel-field", children=[
            html.Label("Progress", className="panel-label"),
            html.Div(id="trb-annotation-counts", children=_counts_block(rec, annotations)),
        ]),
    ]

    return split_page(main, panel, panel_title="Annotation")


# ── Autoscale (A) ─────────────────────────────────────────────────────


@callback(
    Output("trb-yrange", "value", allow_duplicate=True),
    Input("trb-autoscale-btn", "n_clicks"),
    Input("trb-autoscale-key", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def autoscale_background_review(_n, _key_ts, sid):
    """Fit the Y range to the shown window of the current stretch (1-50 Hz,
    as the plot)."""
    state = server_state.get_session(sid)
    rec = state.recording
    if rec is None:
        return no_update
    lst = _review_list(state, _get_annotations(state))
    if not lst:
        return no_update
    a = lst[max(0, min(state.extra.get("trb_current_idx", 0), len(lst) - 1))]
    w0, w1 = _review_window(rec, a, state)
    i0, i1 = int(w0 * rec.fs), min(int(w1 * rec.fs), rec.n_samples)
    yr = fit_y_range(bandpass_filter(rec.data[a.channel, i0:i1].astype(np.float64),
                                     rec.fs, 1.0, 50.0))
    if yr is None:
        return no_update
    state.extra["trb_yrange"] = yr
    return yr


# ── Keyboard ──────────────────────────────────────────────────────────

clientside_callback(
    """
    function(n) {
        if (!window._trbKeyListenerActive) {
            window._trbKeyListenerActive = true;
            document.addEventListener('keydown', function(e) {
                // Only while this page is shown (the listener stays after leaving it)
                if (!document.getElementById('trb-keyboard-listener')) return;
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
                // Only while this page is shown (the listener outlives the page)
                if (!document.getElementById('trb-keyboard-listener')) return;
                var key = e.key.toLowerCase();
                if (key === 'a' && !e.metaKey && !e.ctrlKey && !e.altKey) {
                    // Autoscale the Y range (own store: no full review update)
                    e.preventDefault();
                    if (window.dash_clientside && window.dash_clientside.set_props) {
                        window.dash_clientside.set_props('trb-autoscale-key', {data: Date.now()});
                    }
                    return;
                }
                if (['c', 'r', 's', 'arrowleft', 'arrowright', ',', '.', '[', ']', '\\\\'].includes(key)) {
                    e.preventDefault();
                    if (window.dash_clientside && window.dash_clientside.set_props) {
                        window.dash_clientside.set_props('trb-keyboard-store', {data: {key: key, ts: Date.now()}});
                    }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("trb-keyboard-listener", "children"),
    Input("trb-keyboard-store", "data"),
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("trb-review-mode", "style"),
    Output("trb-browse-mode", "style"),
    Input("trb-mode-toggle", "value"),
    State("session-id", "data"),
)
def toggle_mode(mode, sid):
    server_state.get_session(sid).extra["trb_mode"] = mode
    if mode == "review":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@callback(
    Output("trb-annotator", "value"),
    Input("trb-annotator", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_annotator(val, sid):
    server_state.get_session(sid).extra["trb_annotator"] = val or ""
    return val


_KEY_ACTIONS = {"c": "confirm", "r": "reject", "s": "skip", "arrowleft": "prev",
                ",": "prev", "arrowright": "next", ".": "next", "[": "pan_left",
                "]": "pan_right", "\\": "pan_centre"}
_BTN_ACTIONS = {"trb-confirm-btn": "confirm", "trb-reject-btn": "reject",
                "trb-skip-btn": "skip", "trb-prev-btn": "prev", "trb-next-btn": "next",
                "trb-jump-to": "jump", "trb-pan-left": "pan_left",
                "trb-pan-right": "pan_right", "trb-pan-centre": "pan_centre",
                "trb-context": "context", "trb-overview": "overview_jump",
                "trb-ovspan": "ovspan", "trb-channel-filter": "filter",
                "trb-label-filter": "filter", "trb-yrange": "yrange",
                "trb-mode-toggle": "refresh"}


def _apply_boundaries(ann: AnnotatedEvent, new_on, new_off, duration) -> bool:
    """Set new boundaries (keeping the originals once). Returns True if they changed."""
    if new_on is None and new_off is None:
        return False
    on = float(new_on) if new_on is not None else ann.onset_sec
    off = float(new_off) if new_off is not None else ann.offset_sec
    if on > off:
        on, off = off, on
    on, off = max(0.0, on), min(float(duration), off)
    if off - on < 0.1:
        return False
    if abs(on - ann.onset_sec) <= 0.001 and abs(off - ann.offset_sec) <= 0.001:
        return False
    if ann.original_onset_sec is None:
        ann.original_onset_sec = ann.onset_sec
    if ann.original_offset_sec is None:
        ann.original_offset_sec = ann.offset_sec
    ann.onset_sec, ann.offset_sec = on, off
    return True


@callback(
    Output("trb-review-graph", "figure"),
    Output("trb-event-nav-text", "children"),
    Output("trb-event-status", "children"),
    Output("trb-onset-input", "value"),
    Output("trb-offset-input", "value"),
    Output("trb-duration-display", "children"),
    Output("trb-event-properties", "children"),
    Output("trb-overlap-warning", "children"),
    Output("trb-annotation-counts", "children"),
    Output("trb-overview", "figure"),
    Output("trb-shown-key", "data"),
    Output("trb-review-status", "children"),
    Output("trb-file-message", "children"),
    Input("trb-mode-toggle", "value"),
    Input("trb-channel-filter", "value"),
    Input("trb-label-filter", "value"),
    Input("trb-prev-btn", "n_clicks"),
    Input("trb-next-btn", "n_clicks"),
    Input("trb-jump-to", "value"),
    Input("trb-confirm-btn", "n_clicks"),
    Input("trb-reject-btn", "n_clicks"),
    Input("trb-skip-btn", "n_clicks"),
    Input("trb-keyboard-store", "data"),
    Input("trb-yrange", "value"),
    Input("trb-pan-left", "n_clicks"),
    Input("trb-pan-right", "n_clicks"),
    Input("trb-pan-centre", "n_clicks"),
    Input("trb-context", "value"),
    Input("trb-overview", "clickData"),
    Input("trb-ovspan", "value"),
    State("trb-onset-input", "value"),
    State("trb-offset-input", "value"),
    State("trb-shown-key", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_review(mode, ch_filter, label_filter, _prev, _next, jump_to, _conf, _rej,
                  _skip, kb_data, yrange, _pan_l, _pan_r, _pan_c, context_val, ov_click,
                  ovspan_val, onset_input, offset_input, shown_key, sid):
    """Review mode: navigation, confirm / reject / skip (buttons and keys), view."""
    n_out = 13
    if mode != "review":
        return (no_update,) * n_out
    state = server_state.get_session(sid)
    if state.recording is None:
        return (no_update,) * n_out
    _sync_recording(state)
    rec = state.recording
    trigger = ctx.triggered_id

    action = None
    if trigger == "trb-keyboard-store":
        action = _KEY_ACTIONS.get((kb_data or {}).get("key", ""))
    else:
        action = _BTN_ACTIONS.get(trigger)

    if trigger == "trb-yrange" and yrange is not None and float(yrange) > 0:
        state.extra["trb_yrange"] = float(yrange)
    if action == "filter":
        state.extra["trb_channel_filter"] = ch_filter
        state.extra["trb_label_filter"] = label_filter or ""

    annotations = _get_annotations(state)
    seizures = _seizure_annotations(state)
    lst = _review_list(state, annotations)
    file_msg = None if has_background_file(state) else _no_file_message()
    if action == "filter":
        state.extra["trb_current_idx"] = _first_pending(lst)
    idx = state.extra.get("trb_current_idx", 0)
    status = no_update

    if not lst:
        state.extra["trb_current_idx"] = 0
        fig = _empty_figure("No background stretches to review." if not annotations
                            else "No stretches match the channel / label filter.")
        return (fig, _nav_text(rec, None, 0, 0), html.Span(), 0, 0, "",
                _properties(rec, None), None, _counts_block(rec, annotations),
                go.Figure(), None, no_update, file_msg)
    idx = max(0, min(idx, len(lst) - 1))

    # Stale press: the screen still shows an earlier stretch, so these boundaries
    # belong to that one. Ignore the label action; the re-render shows the current one.
    stale = (shown_key is not None and list(shown_key) != _view_key(lst[idx]))
    if stale and action in ("confirm", "reject"):
        action = None

    if action in ("confirm", "reject"):
        ann = lst[idx]  # same object as in ``annotations``
        ann.label = "confirmed" if action == "confirm" else "rejected"
        ann.annotated_at = datetime.now(timezone.utc).isoformat()
        ann.annotator = _annotator(state)
        ann.event_type = BACKGROUND_EVENT_TYPE
        _apply_boundaries(ann, onset_input, offset_input, rec.duration_sec)
        err = _auto_save(state, annotations)
        status = alert(err, "danger") if err else None
        acted = _view_key(ann)
        lst = _review_list(state, annotations)
        pos = next((i for i, a in enumerate(lst) if _view_key(a) == acted), None)
        if pos is None:  # dropped by the label filter: the next one moved into place
            pos = sum(1 for a in lst if _sort_key(a) < _sort_key(ann)) - 1
        nxt = _find_next_pending(lst, pos)
        if nxt is not None:
            idx = nxt
        else:
            idx = min(pos + 1, len(lst) - 1) if lst else 0
        file_msg = None if has_background_file(state) else _no_file_message()
        if not lst:
            state.extra["trb_current_idx"] = 0
            fig = _empty_figure("No stretches match the channel / label filter.")
            return (fig, _nav_text(rec, None, 0, 0), html.Span(), 0, 0, "",
                    _properties(rec, None), None, _counts_block(rec, annotations),
                    go.Figure(), None, status, file_msg)
    elif action in ("skip", "next"):
        idx = min(idx + 1, len(lst) - 1)
    elif action == "prev":
        idx = max(idx - 1, 0)
    elif action == "jump" and jump_to is not None:
        idx = max(0, min(int(jump_to) - 1, len(lst) - 1))

    idx = max(0, min(idx, len(lst) - 1))
    state.extra["trb_current_idx"] = idx
    cur = lst[idx]
    key = _view_key(cur)
    if action in ("pan_left", "pan_right", "pan_centre"):
        iv.apply_view_action(state, "trb", key, action[4:], cur.onset_sec, cur.offset_sec,
                             rec.duration_sec)
    elif action == "context":
        iv.set_context(state, "trb", key, context_val, cur.onset_sec, cur.offset_sec,
                       rec.duration_sec)
    elif action == "overview_jump" and ov_click and ov_click.get("points"):
        iv.jump_view(state, "trb", key, float(ov_click["points"][0]["x"]),
                     cur.onset_sec, cur.offset_sec, rec.duration_sec)
    elif action == "ovspan" and ovspan_val is not None:
        state.extra["trb_ovspan"] = int(ovspan_val)

    fig = _build_review_figure(rec, cur, state, annotations, seizures,
                               _default_yrange(state, rec))
    n_same = len(_overlapping(cur, seizures)[0])
    out = [fig, _nav_text(rec, cur, idx, len(lst)), _bg_badge(cur),
           round(cur.onset_sec, 3), round(cur.offset_sec, 3),
           f"({cur.offset_sec - cur.onset_sec:.2f}s)", _properties(rec, cur, n_same),
           _overlap_warning(rec, cur, seizures), _counts_block(rec, annotations),
           _build_overview(rec, cur, state, annotations, seizures), key, status, file_msg]
    if action in ("pan_left", "pan_right", "pan_centre", "context", "overview_jump",
                  "ovspan", "yrange"):
        # Only the view moved: keep typed-but-unapplied boundaries
        out[3] = out[4] = out[5] = no_update
        out[10] = no_update
    return tuple(out)


@callback(
    Output("trb-review-status", "children", allow_duplicate=True),
    Output("trb-onset-input", "value", allow_duplicate=True),
    Output("trb-offset-input", "value", allow_duplicate=True),
    Output("trb-duration-display", "children", allow_duplicate=True),
    Output("trb-event-properties", "children", allow_duplicate=True),
    Output("trb-overlap-warning", "children", allow_duplicate=True),
    Output("trb-shown-key", "data", allow_duplicate=True),
    Input("trb-review-graph", "relayoutData"),
    State("trb-shown-key", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_boundary_drag(relayout, shown_key, sid):
    """Dragging the stretch's edges (shapes[0]) sets its boundaries and saves."""
    n_out = 7
    if not relayout:
        return (no_update,) * n_out
    new_on = new_off = None
    for k, v in relayout.items():
        m = re.match(r"shapes\[0\]\.(x0|x1)$", k)
        if m:
            if m.group(1) == "x0":
                new_on = float(v)
            else:
                new_off = float(v)
    if new_on is None and new_off is None and "shapes" in relayout:
        for sh in relayout["shapes"]:
            if sh.get("name") == "highlight":
                new_on, new_off = float(sh["x0"]), float(sh["x1"])
                break
    if new_on is None and new_off is None:
        return (no_update,) * n_out

    state = server_state.get_session(sid)
    if state.recording is None:
        return (no_update,) * n_out
    _sync_recording(state)
    rec = state.recording
    annotations = _get_annotations(state)
    lst = _review_list(state, annotations)
    idx = state.extra.get("trb_current_idx", 0)
    if not lst or not 0 <= idx < len(lst):
        return (no_update,) * n_out
    ann = lst[idx]
    if shown_key is not None and list(shown_key) != _view_key(ann):
        return (no_update,) * n_out  # the drag was on a stretch no longer current
    if not _apply_boundaries(ann, new_on, new_off, rec.duration_sec):
        return (no_update,) * n_out
    ann.annotated_at = datetime.now(timezone.utc).isoformat()
    err = _auto_save(state, annotations)
    seizures = _seizure_annotations(state)
    n_same = len(_overlapping(ann, seizures)[0])
    return (alert(err, "danger") if err else alert("Boundaries updated", "success"),
            round(ann.onset_sec, 3), round(ann.offset_sec, 3),
            f"({ann.offset_sec - ann.onset_sec:.2f}s)", _properties(rec, ann, n_same),
            _overlap_warning(rec, ann, seizures), _view_key(ann))


# ── Browse mode ───────────────────────────────────────────────────────


@callback(
    Output("trb-browse-start", "value"),
    Output("trb-nav-time-display", "children"),
    Input("trb-nav-start", "n_clicks"),
    Input("trb-nav-back-big", "n_clicks"),
    Input("trb-nav-back", "n_clicks"),
    Input("trb-nav-fwd", "n_clicks"),
    Input("trb-nav-fwd-big", "n_clicks"),
    Input("trb-browse-start", "value"),
    State("trb-browse-window", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_browse_navigation(_a, _b, _c, _d, _e, start_val, window, sid):
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update, no_update
    rec = state.recording
    window = float(window or 120)
    cur = float(start_val or 0)
    step = {"trb-nav-back-big": -5 * window, "trb-nav-back": -window,
            "trb-nav-fwd": window, "trb-nav-fwd-big": 5 * window}
    if ctx.triggered_id == "trb-nav-start":
        cur = 0.0
    else:
        cur += step.get(ctx.triggered_id, 0.0)
    cur = max(0.0, min(max(0.0, rec.duration_sec - window), cur))
    state.extra["trb_browse_start"] = cur
    return round(cur, 2), f"{cur:.1f}s – {cur + window:.1f}s"


@callback(
    Output("trb-browse-channel-checks", "value"),
    Input("trb-browse-ch-all", "n_clicks"),
    Input("trb-browse-ch-none", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def browse_channel_all_none(_a, _n, sid):
    state = server_state.get_session(sid)
    if state.recording is None:
        return no_update
    if ctx.triggered_id == "trb-browse-ch-all":
        return list(range(state.recording.n_channels))
    if ctx.triggered_id == "trb-browse-ch-none":
        return []
    return no_update


def _browse_channels(rec, selected):
    chans = [c for c in (selected or []) if 0 <= c < rec.n_channels]
    return chans or list(range(rec.n_channels))


@callback(
    Output("trb-browse-graph", "figure"),
    Input("trb-mode-toggle", "value"),
    Input("trb-browse-start", "value"),
    Input("trb-browse-window", "value"),
    Input("trb-add-btn", "active"),
    Input("trb-browse-channel-checks", "value"),
    Input("trb-annotations-version", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_browse_graph(mode, start_val, window_val, add_active, selected, _ver, sid):
    if mode != "browse":
        return no_update
    state = server_state.get_session(sid)
    if state.recording is None:
        return _empty_figure("No recording loaded", 600)
    _sync_recording(state)
    rec = state.recording
    window = float(window_val or 120)
    state.extra["trb_browse_window"] = window
    return _build_browse_figure(rec, _get_annotations(state), _seizure_annotations(state),
                                float(start_val or 0), window,
                                _browse_channels(rec, selected), bool(add_active))


@callback(
    Output("trb-add-btn", "active"),
    Output("trb-remove-btn", "active"),
    Output("trb-browse-status", "children"),
    Output("trb-browse-graph", "selectedData"),
    Output("trb-annotations-version", "data"),
    Output("trb-annotation-counts", "children", allow_duplicate=True),
    Output("trb-file-message", "children", allow_duplicate=True),
    Input("trb-add-btn", "n_clicks"),
    Input("trb-remove-btn", "n_clicks"),
    Input("trb-browse-graph", "selectedData"),
    Input("trb-browse-graph", "clickData"),
    State("trb-add-btn", "active"),
    State("trb-remove-btn", "active"),
    State("trb-browse-annotate-channel", "value"),
    State("trb-browse-channel-checks", "value"),
    State("trb-annotations-version", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_add_remove(_add, _rem, selected_data, click_data, add_active, remove_active,
                      annotate_channel, selected_channels, version, sid):
    """Add a manual background stretch (select a region) or remove a manual one (click)."""
    nu = (no_update,) * 7
    version = version or 0
    trig = ctx.triggered_id
    props = [t["prop_id"] for t in ctx.triggered] if ctx.triggered else []
    if trig == "trb-add-btn":
        return not add_active, False, no_update, no_update, no_update, no_update, no_update
    if trig == "trb-remove-btn":
        return False, not remove_active, no_update, no_update, no_update, no_update, no_update

    state = server_state.get_session(sid)
    if state.recording is None:
        return nu
    _sync_recording(state)
    rec = state.recording

    if any("selectedData" in p for p in props) and selected_data is not None:
        if not add_active:
            return nu
        xr = (selected_data.get("range") or {}).get("x")
        if not xr or len(xr) < 2:
            xs = [p["x"] for p in selected_data.get("points", []) if "x" in p]
            if len(xs) < 2:
                return nu
            xr = [min(xs), max(xs)]
        on, off = sorted((float(xr[0]), float(xr[1])))
        on, off = max(0.0, on), min(rec.duration_sec, off)
        if off - on < 1.0:
            return (no_update, no_update, alert("Selection too short (< 1 s)", "warning"),
                    None, no_update, no_update, no_update)
        ch = int(annotate_channel) if annotate_channel is not None else 0
        annotations = _get_annotations(state)
        clash = [b for b in annotations
                 if b.channel == ch and b.onset_sec < off and b.offset_sec > on]
        if clash:
            b = clash[0]
            return (no_update, no_update, alert(
                f"Not added: overlaps background B#{b.event_id} ({b.label}, "
                f"{b.onset_sec:.1f}–{b.offset_sec:.1f}s) on {_ch_name(rec, ch)}. "
                "Edit or review that stretch instead.", "warning"),
                None, no_update, no_update, no_update)
        from eeg_seizure_analyzer.io.channel_ids import load_channel_ids
        ids = state.extra.get("channel_animal_ids") or (
            load_channel_ids(rec.source_path) if rec.source_path else None) or {}
        animal = str(ids.get(ch, ids.get(str(ch), "")) or "")
        new = AnnotatedEvent(
            file_path=rec.source_path or "", animal_id=animal, annotator=_annotator(state),
            onset_sec=on, offset_sec=off, channel=ch, label="confirmed", source="manual",
            event_type=BACKGROUND_EVENT_TYPE,
            annotated_at=datetime.now(timezone.utc).isoformat(),
            event_id=max((a.event_id for a in annotations), default=0) + 1,
            features={"source": "manual"},
        )
        annotations.append(new)
        err = _auto_save(state, annotations)
        if err:
            return (False, False, alert(err, "danger"), None, version + 1,
                    no_update, no_update)
        same, _ = _overlapping(new, _seizure_annotations(state))
        msg = (f"Added background B#{new.event_id}: {on:.1f}s – {off:.1f}s on "
               f"{_ch_name(rec, ch)} (confirmed)")
        if same:
            msg += (f". It overlaps {len(same)} seizure annotation"
                    f"{'s' if len(same) != 1 else ''} on this channel; check it.")
        return (False, False, alert(msg, "warning" if same else "success"), None,
                version + 1, _counts_block(rec, annotations), None)

    if any("clickData" in p for p in props) and click_data is not None:
        if not remove_active:
            return nu
        pts = click_data.get("points") or []
        if not pts:
            return nu
        x = float(pts[0].get("x", 0))
        chans = _browse_channels(rec, selected_channels)
        cn = pts[0].get("curveNumber")
        ch = chans[cn] if isinstance(cn, int) and 0 <= cn < len(chans) else None
        annotations = _get_annotations(state)
        hits = [b for b in annotations if b.onset_sec <= x <= b.offset_sec
                and (ch is None or b.channel == ch)]
        if not hits:
            return (no_update, no_update,
                    alert("No background stretch at the click position", "warning"),
                    no_update, no_update, no_update, no_update)
        best = min(hits, key=lambda b: b.offset_sec - b.onset_sec)
        if best.source != "manual":
            return (no_update, no_update, alert(
                f"B#{best.event_id} was proposed by the rule, not added by hand: "
                "review it in Review mode (Reject) instead of removing it.", "warning"),
                no_update, no_update, no_update, no_update)
        annotations = [a for a in annotations if not _same(a, best)]
        err = _auto_save(state, annotations)
        if err:
            return (False, False, alert(err, "danger"), no_update, version + 1,
                    no_update, no_update)
        return (False, False, alert(
            f"Removed background B#{best.event_id}: {best.onset_sec:.1f}s – "
            f"{best.offset_sec:.1f}s on {_ch_name(rec, best.channel)}", "info"),
            no_update, version + 1, _counts_block(rec, annotations), no_update)

    return nu
