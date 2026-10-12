"""View window for the event inspectors (Training review mode, Seizures inspector).

The inspectors used to show a fixed ``onset - 10 s … offset + 10 s`` window, rebuilt from
the event's boundaries. Here the view has its own state:

- **context**: seconds shown either side of the event (10/30/60/120), remembered across
  events;
- **pan**: shift the view by half its width without touching the event. Moving to a
  different event resets the pan; "Centre" resets it explicitly.

The recording is already in server memory, so panning only re-slices it. What reaches the
browser is bounded by min/max downsampling the trace and capping the spectrogram's columns.

State lives in ``state.extra`` under ``<prefix>_context`` and ``<prefix>_view``. The view
is ``{"key": <event key>, "start": <absolute seconds> | None}``, and ``None`` means centred
on the event.

**Overview strip**: a thin plot above the trace covering ±5 min, ±15 min or the whole
recording around the view. It shows:
- the channel's 1–50 Hz RMS amplitude per second, divided by its median, so seizures
  stand out. Raw line length was tried first; high-frequency noise dominates it, and
  seizures scored about 1× the median. Band-passed RMS gives about 5–10×;
- the other events on the channel;
- the current view, as a box.
Clicking it centres the view there. The profile is computed once per channel (about
0.2 s for 90 min at 2 kHz) and cached on the recording object. Span in ``<prefix>_ovspan``.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc

CONTEXT_OPTIONS = (10, 30, 60, 120)
DEFAULT_CONTEXT = 10.0
# Spectrogram columns above this are thinned by a larger hop. A ±10 s view of a
# typical event stays below it, so default views are unchanged.
MAX_SPEC_COLUMNS = 600
# Overview half-spans in seconds; 0 = whole recording.
OVERVIEW_SPANS = ((300, "±5 min"), (900, "±15 min"), (0, "Whole"))
DEFAULT_OVSPAN = 300
OVERVIEW_MAX_POINTS = 3000


def context_sec(state, prefix: str) -> float:
    return float(state.extra.get(f"{prefix}_context", DEFAULT_CONTEXT))


def _clamp(start: float, width: float, duration: float) -> tuple[float, float]:
    width = min(width, duration)
    start = max(0.0, min(start, duration - width))
    return start, start + width


def view_window(state, prefix: str, key, onset: float, offset: float,
                duration: float) -> tuple[float, float]:
    """The (start, end) seconds to show for this event."""
    ctx = context_sec(state, prefix)
    width = (offset - onset) + 2 * ctx
    v = state.extra.get(f"{prefix}_view") or {}
    start = v.get("start") if v.get("key") == key else None
    if start is None:
        start = onset - ctx
    return _clamp(start, width, duration)


def apply_view_action(state, prefix: str, key, action: str, onset: float,
                      offset: float, duration: float) -> None:
    """Pan ('left' / 'right') by half the view, or 'centre' on the event."""
    if action == "centre":
        state.extra[f"{prefix}_view"] = {"key": key, "start": None}
        return
    w0, w1 = view_window(state, prefix, key, onset, offset, duration)
    step = (w1 - w0) / 2
    start = w0 - step if action == "left" else w0 + step
    start, _ = _clamp(start, w1 - w0, duration)
    state.extra[f"{prefix}_view"] = {"key": key, "start": start}


def set_context(state, prefix: str, key, value, onset: float, offset: float,
                duration: float) -> None:
    """Change the context. A panned view keeps its centre; a centred one stays centred."""
    if not value:
        return
    v = state.extra.get(f"{prefix}_view") or {}
    panned = v.get("key") == key and v.get("start") is not None
    if panned:
        w0, w1 = view_window(state, prefix, key, onset, offset, duration)
        centre = (w0 + w1) / 2
    state.extra[f"{prefix}_context"] = float(value)
    if panned:
        width = (offset - onset) + 2 * float(value)
        start, _ = _clamp(centre - width / 2, width, duration)
        state.extra[f"{prefix}_view"] = {"key": key, "start": start}


def spectrogram_overlap(nperseg: int, n_samples: int) -> int:
    """90% overlap, unless that would give more than MAX_SPEC_COLUMNS columns."""
    hop = max(int(nperseg * 0.1), int(np.ceil(n_samples / MAX_SPEC_COLUMNS)), 1)
    return max(0, nperseg - hop)


def minmax_downsample(t: np.ndarray, y: np.ndarray, max_points: int):
    """Min/max envelope with at most ``max_points`` points; unchanged if already fewer."""
    n = len(y)
    if n <= max_points:
        return t, y
    n_b = max_points // 2
    size = -(-n // n_b)  # ceil, so the buckets cover every sample
    pad = n_b * size - n  # < size; repeat the last sample to fill the last bucket
    yb = np.pad(y, (0, pad), mode="edge").reshape(n_b, size)
    tb = np.pad(t, (0, pad), mode="edge").reshape(n_b, size)
    i_min, i_max = yb.argmin(1), yb.argmax(1)
    first = np.minimum(i_min, i_max)
    second = np.maximum(i_min, i_max)
    rows = np.arange(n_b)
    tt = np.column_stack([tb[rows, first], tb[rows, second]]).ravel()
    yy = np.column_stack([yb[rows, first], yb[rows, second]]).ravel()
    return tt, yy


def jump_view(state, prefix: str, key, centre: float, onset: float, offset: float,
              duration: float) -> None:
    """Centre the view on ``centre`` (a click on the overview), keeping its width."""
    w0, w1 = view_window(state, prefix, key, onset, offset, duration)
    start, _ = _clamp(centre - (w1 - w0) / 2, w1 - w0, duration)
    state.extra[f"{prefix}_view"] = {"key": key, "start": start}


def ovspan_sec(state, prefix: str) -> int:
    return int(state.extra.get(f"{prefix}_ovspan", DEFAULT_OVSPAN))


def amplitude_profile(rec, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """1–50 Hz RMS per 1 s bin divided by the channel median, cached on ``rec``."""
    cache = rec.__dict__.setdefault("_ned_amp_profile", {})
    if ch not in cache:
        from eeg_seizure_analyzer.processing.preprocess import bandpass_filter
        fs = int(round(rec.fs))
        n_bins = rec.n_samples // fs
        if n_bins == 0:
            cache[ch] = (np.zeros(0), np.zeros(0))
            return cache[ch]
        x = bandpass_filter(np.asarray(rec.data[ch], dtype=np.float64), rec.fs, 1.0,
                            min(50.0, 0.45 * rec.fs))[:n_bins * fs]
        rms = np.sqrt((x.reshape(n_bins, fs) ** 2).mean(1))
        med = float(np.median(rms[rms > 0])) if np.any(rms > 0) else 1.0
        cache[ch] = (np.arange(n_bins) + 0.5, rms / med)
    return cache[ch]


def overview_figure(state, prefix: str, rec, ch: int, view: tuple[float, float],
                    events: list[tuple[float, float, str]],
                    current: tuple[float, float]) -> go.Figure:
    """Thin strip: amplitude profile, event markers, the current view as a box.

    ``events`` are (onset, offset, colour) on this channel; ``current`` is the event
    being reviewed, drawn outlined on top of its marker.
    """
    from eeg_seizure_analyzer.dash_app.components import apply_fig_theme

    dur = rec.duration_sec
    half = ovspan_sec(state, prefix)
    if half <= 0 or 2 * half >= dur:
        s0, s1 = 0.0, dur
    else:
        c = (view[0] + view[1]) / 2
        s0, s1 = _clamp(c - half, 2 * half, dur)

    t, ll = amplitude_profile(rec, ch)
    i0, i1 = int(np.searchsorted(t, s0)), int(np.searchsorted(t, s1))
    t, ll = t[i0:i1], ll[i0:i1]
    if len(t) > OVERVIEW_MAX_POINTS:      # keep peaks: max per bucket
        k = -(-len(t) // OVERVIEW_MAX_POINTS)
        pad = k * (-(-len(t) // k)) - len(t)
        t = np.pad(t, (0, pad), mode="edge").reshape(-1, k)[:, 0]
        ll = np.pad(ll, (0, pad), mode="edge").reshape(-1, k).max(1)
    top = max(4.0, float(np.percentile(ll, 99.5))) if len(ll) else 4.0

    fig = go.Figure(go.Scatter(
        x=t, y=np.minimum(ll, top), mode="lines", fill="tozeroy",
        line=dict(width=0.8, color="#8b949e"), fillcolor="rgba(139,148,158,0.25)",
        hovertemplate="%{x:.0f} s · amplitude ×%{y:.1f} median<extra>click to view</extra>",
    ))
    for on, off, colour in events:
        if off >= s0 and on <= s1:
            fig.add_vrect(x0=on, x1=max(off, on + 0.5), fillcolor=colour, opacity=0.55,
                          line_width=0, layer="below")
    fig.add_vrect(x0=current[0], x1=current[1], fillcolor="rgba(0,0,0,0)",
                  line=dict(color="#f85149", width=1.5), layer="above")
    fig.add_vrect(x0=view[0], x1=view[1], fillcolor="rgba(88,166,255,0.15)",
                  line=dict(color="#58a6ff", width=1.5), layer="above")
    fig.update_layout(
        height=92, showlegend=False, hovermode="x", dragmode=False,
        xaxis=dict(range=[s0, s1], fixedrange=True, showgrid=False,
                   ticks="outside", ticklen=3, tickfont=dict(size=9)),
        yaxis=dict(range=[0, top * 1.05], fixedrange=True, visible=False),
        uirevision=f"ov_{ch}_{s0:.1f}_{s1:.1f}",
    )
    apply_fig_theme(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=4, b=24))
    return fig


def overview_graph(graph_id: str, figure=None) -> dcc.Graph:
    return dcc.Graph(id=graph_id, figure=figure if figure is not None else go.Figure(),
                     config={"displayModeBar": False, "scrollZoom": False},
                     style={"height": "92px", "cursor": "pointer"})


def view_controls(id_prefix: str, context: float,
                  ovspan: int = DEFAULT_OVSPAN) -> html.Div:
    """◀ / Centre / ▶ buttons, the context selector and the overview span."""
    btn = dict(size="sm", color="secondary", outline=True,
               style={"whiteSpace": "nowrap"})
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px",
               "flexWrap": "nowrap"},
        children=[
            dbc.ButtonGroup([
                dbc.Button("◀ [", id=f"{id_prefix}-pan-left", title="Pan back half a view ( [ )", **btn),
                dbc.Button("Centre \\", id=f"{id_prefix}-pan-centre",
                           title="Centre the view on the event ( \\ )", **btn),
                dbc.Button("] ▶", id=f"{id_prefix}-pan-right", title="Pan forward half a view ( ] )", **btn),
            ]),
            html.Label("Context:", style={"fontSize": "0.78rem", "margin": "0",
                                          "color": "var(--ned-text-muted)"}),
            dcc.Dropdown(
                id=f"{id_prefix}-context",
                options=[{"label": f"±{c} s", "value": c} for c in CONTEXT_OPTIONS],
                value=int(context) if int(context) in CONTEXT_OPTIONS else int(DEFAULT_CONTEXT),
                clearable=False, searchable=False,
                style={"width": "105px", "fontSize": "0.8rem"},
            ),
            html.Label("Overview:", style={"fontSize": "0.78rem", "margin": "0 0 0 4px",
                                           "color": "var(--ned-text-muted)"}),
            dcc.Dropdown(
                id=f"{id_prefix}-ovspan",
                options=[{"label": lbl, "value": v} for v, lbl in OVERVIEW_SPANS],
                value=ovspan if ovspan in [v for v, _ in OVERVIEW_SPANS] else DEFAULT_OVSPAN,
                clearable=False, searchable=False,
                style={"width": "105px", "fontSize": "0.8rem"},
            ),
        ],
    )
