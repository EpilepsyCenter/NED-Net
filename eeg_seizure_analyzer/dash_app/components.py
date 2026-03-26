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

    return html.Div(
        className="param-row",
        children=[
            html.Div(
                className="param-label",
                children=[
                    label,
                    html.Span(
                        " (?)",
                        title=tooltip or "",
                        style={"cursor": "help", "opacity": "0.5"},
                    ) if tooltip else None,
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Slider(
                            id=slider_id,
                            min=min_val,
                            max=max_val,
                            step=step,
                            value=value,
                            marks=marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="param-slider",
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=input_id,
                            type="number",
                            min=min_val,
                            max=max_val,
                            step=step,
                            value=int(value) if is_int else value,
                            debounce=True,
                            className="form-control",
                            style={"padding": "4px 6px", "fontSize": "0.82rem"},
                        ),
                        width=3,
                    ),
                ],
                className="g-1 align-items-center",
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

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#1c2128",
        "plot_bgcolor": "#0f1117",
        "font": {"color": "#e6edf3", "family": "Inter, sans-serif", "size": 12},
        "xaxis": {
            "gridcolor": "#2d333b",
            "zerolinecolor": "#2d333b",
        },
        "yaxis": {
            "gridcolor": "#2d333b",
            "zerolinecolor": "#2d333b",
        },
        "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
        "colorway": [
            "#58a6ff", "#3fb950", "#d29922", "#f85149",
            "#bc8cff", "#f778ba", "#79c0ff", "#56d364",
        ],
    },
}


def apply_fig_theme(fig):
    """Apply the NED-Net dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#1c2128",
        plot_bgcolor="#0f1117",
        font=dict(color="#e6edf3", family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor="#2d333b", zerolinecolor="#2d333b"),
        yaxis=dict(gridcolor="#2d333b", zerolinecolor="#2d333b"),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig
