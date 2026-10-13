"""Shared Dash components and utilities."""

from __future__ import annotations

import json
from pathlib import Path

from dash import html, dcc, callback, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc


# ── Save / load user defaults ─────────────────────────────────────────

_DEFAULTS_DIR = Path.home() / ".eeg_seizure_analyzer"
_DEFAULTS_FILE = _DEFAULTS_DIR / "defaults.json"


def save_user_defaults(params: dict) -> str:
    """Save parameters to a JSON file."""
    serialisable = {
        k: v for k, v in params.items()
        if isinstance(v, (str, int, float, bool, list))
    }
    _DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)
    return str(_DEFAULTS_FILE)


def load_user_defaults() -> dict | None:
    """Load user defaults from disk."""
    if not _DEFAULTS_FILE.exists():
        return None
    try:
        with open(_DEFAULTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


_SP_DEFAULTS_FILE = _DEFAULTS_DIR / "spike_defaults.json"


def save_spike_user_defaults(params: dict) -> str:
    """Save spike detection parameters to a JSON file."""
    serialisable = {
        k: v for k, v in params.items()
        if v is None or isinstance(v, (str, int, float, bool, list))
    }
    _DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SP_DEFAULTS_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)
    return str(_SP_DEFAULTS_FILE)


def load_spike_user_defaults() -> dict | None:
    """Load spike user defaults from disk."""
    if not _SP_DEFAULTS_FILE.exists():
        return None
    try:
        with open(_SP_DEFAULTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── Layout helpers ────────────────────────────────────────────────────


def metric_card(label: str, value: str, accent: bool = False) -> html.Div:
    """A single metric display card."""
    cls = "metric-value accent" if accent else "metric-value"
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(value, className=cls),
        ],
    )


def section_header(title: str) -> html.Div:
    """Sidebar section header."""
    return html.Div(
        className="sidebar-section",
        children=[html.Div(title, className="sidebar-section-label")],
    )


def sidebar_divider() -> html.Hr:
    """Sidebar horizontal rule."""
    return html.Hr(className="sidebar-divider")


def empty_state(icon: str, title: str, text: str) -> html.Div:
    """Placeholder for empty content areas."""
    return html.Div(
        className="empty-state",
        children=[
            html.Div(icon, className="empty-icon"),
            html.Div(title, className="empty-title"),
            html.Div(text, className="empty-text"),
        ],
    )


# ── Parameter presets menu (Detection: Seizure / Interictal Spikes) ──

def params_menu(prefix: str, has_results: bool) -> dbc.DropdownMenu:
    """The four parameter-preset actions in one menu.

    The item ids keep the old button ids (``<prefix>-recall-defaults-btn`` …), so the
    callbacks that read their ``n_clicks`` are unchanged.
    """
    return dbc.DropdownMenu(
        label="Parameters",
        size="sm", color="link", align_end=True,
        className="ned-menu",                    # themed in style.css
        toggleClassName="btn-ned-secondary",
        toggle_style={"whiteSpace": "nowrap"},
        children=[
            dbc.DropdownMenuItem("Load my saved defaults",
                                 id=f"{prefix}-recall-settings-btn"),
            dbc.DropdownMenuItem("Save current as my defaults",
                                 id=f"{prefix}-save-settings-btn"),
            dbc.DropdownMenuItem("Reset to factory defaults",
                                 id=f"{prefix}-recall-defaults-btn"),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem("Load the parameters used for this file's detections",
                                 id=f"{prefix}-recall-det-btn",
                                 disabled=not has_results),
        ],
    )


# ── Result filters (Detection, Training, Spikes) ─────────────────────
# One layout for every filter block: a header row (title, on/off switch, any
# dropdowns, a "N of M shown" summary) and a uniform grid of min–max pairs that
# fits seven pairs on one row at normal widths and wraps evenly below that.

_FILTER_LABEL_STYLE = {"fontSize": "0.75rem", "color": "var(--ned-text-muted)",
                       "marginBottom": "2px"}


def filter_range(label: str, fid_min: str, fid_max: str, min_val, max_val, step,
                 value_min, value_max=None) -> html.Div:
    """A compact min–max input pair; one cell of the filter grid."""
    inp = {"width": "100%", "height": "28px", "fontSize": "0.78rem"}

    def _num(fid, value, ph):
        return dcc.Input(id=fid, type="number", min=min_val, max=max_val, step=step,
                         value=value, placeholder=ph, debounce=True,
                         className="form-control", style=inp)

    return html.Div([
        html.Label(label, style=_FILTER_LABEL_STYLE),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "4px"},
                 children=[_num(fid_min, value_min, "min"),
                           html.Span("–", style={"color": "var(--ned-text-muted)",
                                                 "fontSize": "0.8rem"}),
                           _num(fid_max, value_max, "max")]),
    ])


def filter_dropdown(label: str, dropdown) -> html.Div:
    """A labelled dropdown for the filter header row."""
    return html.Div([html.Label(label, style=_FILTER_LABEL_STYLE), dropdown],
                    style={"minWidth": "170px"})


def filter_panel(switch_id: str, enabled: bool, ranges: list, *, title="Result filters",
                 extras=(), summary_id: str | None = None, summary="",
                 section_id: str | None = None, visible: bool = True,
                 hidden=()) -> html.Div:
    """Header row (title, switch, extras, summary) plus a uniform grid of ranges.

    ``hidden`` holds components that must stay in the layout for callbacks but
    are not shown.
    """
    header = [
        html.Span(title, style={"fontSize": "0.82rem", "fontWeight": "600",
                                "color": "var(--ned-text-muted)"}),
        dbc.Switch(id=switch_id, value=enabled, style={"fontSize": "0.78rem"}),
    ]
    if summary_id:
        header.append(html.Span(summary, id=summary_id,
                                style={"fontSize": "0.78rem",
                                       "color": "var(--ned-text-muted)"}))
    header.append(html.Div(style={"flex": "1"}))
    header.extend(extras)
    # The outer div carries the id and only display/margin: pages show/hide the
    # section by replacing its whole style, which must not wipe the panel's look.
    kw = {"id": section_id} if section_id else {}
    return html.Div(
        **kw,
        style={"display": "block" if visible else "none", "marginBottom": "16px"},
        children=html.Div(
            style={"padding": "10px 14px", "border": "1px solid var(--ned-border)",
                   "borderRadius": "8px"},
            children=[
                html.Div(header, style={"display": "flex", "alignItems": "flex-end",
                                        "gap": "14px", "flexWrap": "wrap",
                                        "marginBottom": "8px"}),
                html.Div(ranges, style={
                    "display": "grid", "gap": "8px 14px",
                    "gridTemplateColumns": "repeat(auto-fill, minmax(130px, 1fr))"}),
                *([html.Div(list(hidden), style={"display": "none"})] if hidden else []),
            ],
        ),
    )


def filter_summary(n_shown: int, n_total: int, enabled: bool, noun: str = "events") -> str:
    """'Filters on: 12 of 40 events shown' / 'Filters off: all 40 events shown'."""
    if not enabled:
        return f"Filters off: all {n_total} {noun} shown"
    if n_shown == n_total:
        return f"Filters on: all {n_total} {noun} pass"
    return f"Filters on: {n_shown} of {n_total} {noun} shown"


def hidden_by_filters(n_total: int, noun: str, button_id: str) -> html.Div:
    """Empty state for when results exist but the filters hide all of them."""
    return html.Div(
        className="empty-state",
        children=[
            html.Div("⧗", className="empty-icon"),
            html.Div(f"All {n_total} {noun} are hidden by the filters",
                     className="empty-title"),
            html.Div("Loosen the result filters above, or show everything.",
                     className="empty-text"),
            dbc.Button("Show all (turn filters off)", id=button_id, size="sm",
                       color="secondary", outline=True, className="mt-2"),
        ],
    )


def alert(message: str, variant: str = "info") -> html.Div:
    """Styled alert.  variant: info, warning, danger, success."""
    cls = f"ned-alert {variant}" if variant != "info" else "ned-alert"
    return html.Div(message, className=cls)


def param_control(
    label: str,
    id_key: str,
    min_val: float,
    max_val: float,
    step: float,
    value: float,
    tooltip: str | None = None,
) -> html.Div:
    """Paired slider + number input for a detection parameter.

    Uses pattern-matching IDs so a single clientside callback
    can sync all instances.
    """
    is_int = isinstance(min_val, int) and isinstance(max_val, int)
    marks = None  # clean slider look

    slider_id = {"type": "param-slider", "key": id_key}
    input_id = {"type": "param-input", "key": id_key}

    display_val = int(value) if is_int else value

    # Dash 4's slider has its own number field, but shows it only when the slider
    # is wide enough, so it appeared on some cards and not others. It is off;
    # this helper's own synced number box is shown on every card instead
    # (the MATCH clientside callback in main.py syncs slider, box and label).
    return html.Div(
        className="param-row",
        children=[
            html.Div(
                className="param-label",
                children=[
                    label,
                    # Synced value text; the box below shows the value now
                    html.Span(
                        f"  {display_val}",
                        id={"type": "param-display", "key": id_key},
                        style={"display": "none"},
                    ),
                    html.Span(
                        " (?)",
                        title=tooltip or "",
                        style={"cursor": "help", "opacity": "0.5"},
                    ) if tooltip else None,
                ],
            ),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px"},
                children=[
                    html.Div(
                        dcc.Slider(
                            id=slider_id,
                            min=min_val,
                            max=max_val,
                            step=step,
                            value=value,
                            marks=marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                            allow_direct_input=False,
                            className="param-slider",
                        ),
                        style={"flex": "1 1 auto", "minWidth": "0"},
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        min=min_val, max=max_val, step=step,
                        value=display_val,
                        debounce=True,
                        className="form-control",
                        style={"width": "64px", "height": "26px",
                               "fontSize": "0.78rem", "flex": "0 0 64px"},
                    ),
                ],
            ),
        ],
    )


def collapsible_section(
    title: str,
    section_id: str,
    children: list,
    default_open: bool = False,
) -> html.Div:
    """A collapsible parameter section with header + body."""
    return html.Div([
        html.Div(
            className="param-section-header",
            id=f"{section_id}-header",
            children=[
                html.Span(title, className="section-title"),
                html.Span("\u25BC" if default_open else "\u25B6",
                          className="section-chevron",
                          id=f"{section_id}-chevron"),
            ],
        ),
        dbc.Collapse(
            html.Div(children, className="param-section-body"),
            id=f"{section_id}-collapse",
            is_open=default_open,
        ),
    ])


def blinding_badge(is_on: bool) -> html.Span:
    """Small badge showing blinding state."""
    if is_on:
        return html.Span(
            [html.Span("\u25CF", style={"fontSize": "0.6rem"}), " BLINDED"],
            className="blinding-badge on",
        )
    return html.Span(
        [html.Span("\u25CF", style={"fontSize": "0.6rem"}), " UNBLINDED"],
        className="blinding-badge off",
    )


def no_recording_placeholder() -> html.Div:
    """Shown when no file is loaded."""
    return empty_state(
        icon="\u2191",
        title="No recording loaded",
        text="Upload an EDF file in the Upload tab to get started.",
    )


# ── Plotly figure defaults ────────────────────────────────────────────

_FONT = "IBM Plex Sans, sans-serif"

_PLOTLY_DARK = {
    "paper_bgcolor": "#1c2128",
    "plot_bgcolor": "#0f1117",
    "font_color": "#e6edf3",
    "gridcolor": "#2d333b",
    "colorway": [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#f778ba", "#79c0ff", "#56d364",
    ],
}

_PLOTLY_LIGHT = {
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#f6f8fa",
    "font_color": "#1f2328",
    "gridcolor": "#d0d7de",
    "colorway": [
        "#0969da", "#1a7f37", "#9a6700", "#cf222e",
        "#8250df", "#bf3989", "#0550ae", "#116329",
    ],
}

# Global mutable — toggled by the theme switch callback
_current_theme = "light"


def set_plotly_theme(theme: str):
    """Set the active Plotly colour scheme ('dark' or 'light')."""
    global _current_theme
    _current_theme = theme


def get_plotly_theme() -> str:
    return _current_theme


def _plotly_palette():
    return _PLOTLY_DARK if _current_theme == "dark" else _PLOTLY_LIGHT


PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#1c2128",
        "plot_bgcolor": "#0f1117",
        "font": {"color": "#e6edf3", "family": _FONT, "size": 12},
        "xaxis": {"gridcolor": "#2d333b", "zerolinecolor": "#2d333b"},
        "yaxis": {"gridcolor": "#2d333b", "zerolinecolor": "#2d333b"},
        "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
        "colorway": _PLOTLY_DARK["colorway"],
    },
}


def apply_fig_theme(fig):
    """Apply the current NED-Net theme (dark or light) to a Plotly figure."""
    p = _plotly_palette()
    fig.update_layout(
        paper_bgcolor=p["paper_bgcolor"],
        plot_bgcolor=p["plot_bgcolor"],
        font=dict(color=p["font_color"], family=_FONT, size=12),
        xaxis=dict(gridcolor=p["gridcolor"], zerolinecolor=p["gridcolor"]),
        yaxis=dict(gridcolor=p["gridcolor"], zerolinecolor=p["gridcolor"]),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig
