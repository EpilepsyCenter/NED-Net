"""Font & theme demo — run standalone to compare styling options.

Usage:  python font_theme_demo.py
Then open http://127.0.0.1:8051/
"""

import os
import tempfile
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# ── Font options ─────────────────────────────────────────────────────

FONTS = {
    "Inter (current)": "'Inter', sans-serif",
    "IBM Plex Sans": "'IBM Plex Sans', sans-serif",
    # ── Monospace / terminal vibes ──
    "JetBrains Mono": "'JetBrains Mono', monospace",
    "IBM Plex Mono": "'IBM Plex Mono', monospace",
    "Fira Code": "'Fira Code', monospace",
    "Source Code Pro": "'Source Code Pro', monospace",
    "Roboto Mono": "'Roboto Mono', monospace",
    "Space Mono": "'Space Mono', monospace",
    # ── Techy sans-serifs (terminal-adjacent) ──
    "Space Grotesk": "'Space Grotesk', sans-serif",
    "Share Tech": "'Share Tech', sans-serif",
}

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@300;400;500;600;700"
    "&family=IBM+Plex+Sans:wght@300;400;500;600;700"
    "&family=JetBrains+Mono:wght@300;400;500;600;700"
    "&family=IBM+Plex+Mono:wght@300;400;500;600;700"
    "&family=Fira+Code:wght@300;400;500;600;700"
    "&family=Source+Code+Pro:wght@300;400;500;600;700"
    "&family=Roboto+Mono:wght@300;400;500;600;700"
    "&family=Space+Mono:wght@400;700"
    "&family=Space+Grotesk:wght@300;400;500;600;700"
    "&family=Share+Tech"
    "&display=swap"
)

# ── Theme palettes ────────────────────────────────────────────────────

DARK_THEME = {
    "bg": "#0f1117",
    "sidebar": "#161b22",
    "surface": "#1c2128",
    "surface_hover": "#242a33",
    "border": "#2d333b",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "accent": "#58a6ff",
    "accent_hover": "#79c0ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
}

LIGHT_THEME = {
    "bg": "#f6f8fa",
    "sidebar": "#ffffff",
    "surface": "#ffffff",
    "surface_hover": "#f0f2f5",
    "border": "#d0d7de",
    "text": "#1f2328",
    "text_muted": "#656d76",
    "accent": "#0969da",
    "accent_hover": "#0550ae",
    "success": "#1a7f37",
    "warning": "#9a6700",
    "danger": "#cf222e",
}


# ── Card builder ──────────────────────────────────────────────────────

_counter = 0

def _make_sample_card(font_name, font_family, theme, scale=1.0):
    """Build a card with UI element samples in the given font + theme."""
    global _counter
    _counter += 1
    t = theme

    def sz(rem):
        """Scale a rem value."""
        return f"{rem * scale:.2f}rem"

    card_style = {
        "fontFamily": font_family,
        "backgroundColor": t["surface"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "10px",
        "padding": "20px",
        "color": t["text"],
    }
    header_style = {
        "fontSize": sz(1.1),
        "fontWeight": "600",
        "marginBottom": "4px",
        "color": t["accent"],
        "fontFamily": font_family,
    }
    muted_style = {"color": t["text_muted"], "fontSize": sz(0.82), "fontFamily": font_family}
    label_style = {"fontSize": sz(0.78), "fontWeight": "500", "marginBottom": "2px", "fontFamily": font_family}
    small_style = {"fontSize": sz(0.72), "color": t["text_muted"], "fontFamily": font_family}

    btn_primary = {
        "backgroundColor": t["accent"],
        "border": "none",
        "color": "#fff",
        "borderRadius": "6px",
        "padding": f"{4*scale:.0f}px {14*scale:.0f}px",
        "fontSize": sz(0.8),
        "fontWeight": "500",
        "fontFamily": font_family,
        "cursor": "pointer",
    }
    btn_outline = {
        "backgroundColor": "transparent",
        "border": f"1px solid {t['border']}",
        "color": t["text"],
        "borderRadius": "6px",
        "padding": f"{4*scale:.0f}px {14*scale:.0f}px",
        "fontSize": sz(0.8),
        "fontWeight": "500",
        "fontFamily": font_family,
        "cursor": "pointer",
    }
    btn_danger = {**btn_primary, "backgroundColor": t["danger"]}
    btn_success = {**btn_primary, "backgroundColor": t["success"]}

    badge_style = {
        "display": "inline-block",
        "padding": f"{2*scale:.0f}px {10*scale:.0f}px",
        "borderRadius": "12px",
        "fontSize": sz(0.7),
        "fontWeight": "600",
        "marginRight": "6px",
        "fontFamily": font_family,
    }

    input_style = {
        "backgroundColor": t["bg"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "6px",
        "color": t["text"],
        "padding": f"{4*scale:.0f}px {10*scale:.0f}px",
        "fontSize": sz(0.82),
        "fontFamily": font_family,
        "width": "100%",
    }

    dropdown_style = {
        "backgroundColor": t["bg"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "6px",
        "color": t["text"],
        "padding": f"{4*scale:.0f}px {10*scale:.0f}px",
        "fontSize": sz(0.82),
        "fontFamily": font_family,
        "width": "100%",
    }

    table_header_style = {
        "backgroundColor": t["bg"],
        "color": t["text_muted"],
        "fontSize": sz(0.72),
        "fontWeight": "600",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "padding": f"{6*scale:.0f}px {10*scale:.0f}px",
        "borderBottom": f"1px solid {t['border']}",
        "textAlign": "left",
        "fontFamily": font_family,
    }
    table_cell_style = {
        "padding": f"{6*scale:.0f}px {10*scale:.0f}px",
        "fontSize": sz(0.8),
        "borderBottom": f"1px solid {t['border']}",
        "fontFamily": font_family,
    }

    section_hdr = {
        "fontSize": sz(0.82), "fontWeight": "600",
        "borderBottom": f"2px solid {t['accent']}",
        "paddingBottom": "4px", "marginBottom": "10px",
        "fontFamily": font_family,
    }

    uid = f"c{_counter}"

    return html.Div(style=card_style, children=[
        # Header
        html.Div(font_name, style=header_style),
        html.Div(f"font-family: {font_family}", style=muted_style),
        html.Hr(style={"borderColor": t["border"], "margin": "12px 0"}),

        # Two-column layout
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            # ── Left column: form controls ──
            html.Div([
                html.Div("Baseline Parameters", style=section_hdr),

                # Slider 1
                html.Div(style={"marginBottom": "14px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
                        html.Span("Spike amplitude (x baseline)", style=label_style),
                        html.Span("3.0x", style={**label_style, "color": t["accent"]}),
                    ]),
                    dcc.Slider(
                        id=f"s1-{uid}", min=1, max=10, step=0.5, value=3,
                        marks=None, tooltip={"placement": "bottom"},
                    ),
                    html.Div("How many times above baseline a spike must be", style=small_style),
                ]),

                # Slider 2
                html.Div(style={"marginBottom": "14px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
                        html.Span("Bandpass low (Hz)", style=label_style),
                        html.Span("1.0 Hz", style={**label_style, "color": t["accent"]}),
                    ]),
                    dcc.Slider(
                        id=f"s2-{uid}", min=0.1, max=10, step=0.1, value=1,
                        marks=None, tooltip={"placement": "bottom"},
                    ),
                ]),

                # Dropdown
                html.Div(style={"marginBottom": "14px"}, children=[
                    html.Div("Baseline method", style=label_style),
                    html.Select(
                        [html.Option("Percentile"), html.Option("Rolling"), html.Option("First N min")],
                        style=dropdown_style,
                    ),
                ]),

                # Number input
                html.Div(style={"marginBottom": "14px"}, children=[
                    html.Div("Min duration (sec)", style=label_style),
                    dcc.Input(type="number", value=5.0, style=input_style),
                ]),
            ]),

            # ── Right column: buttons, badges, table, text ──
            html.Div([
                html.Div("Buttons", style=section_hdr),
                html.Div(style={"marginBottom": "14px", "display": "flex", "flexWrap": "wrap", "gap": "6px"}, children=[
                    html.Button("Detect", style=btn_primary),
                    html.Button("Save Params", style=btn_outline),
                    html.Button("Clear", style=btn_danger),
                    html.Button("Export", style=btn_success),
                ]),

                html.Div("Badges & Status", style={**section_hdr, "marginTop": "10px"}),
                html.Div(style={"marginBottom": "14px"}, children=[
                    html.Span("severe", style={**badge_style, "backgroundColor": t["danger"] + "22", "color": t["danger"]}),
                    html.Span("moderate", style={**badge_style, "backgroundColor": t["warning"] + "22", "color": t["warning"]}),
                    html.Span("mild", style={**badge_style, "backgroundColor": t["success"] + "22", "color": t["success"]}),
                    html.Span("spike_train", style={**badge_style, "backgroundColor": t["accent"] + "22", "color": t["accent"]}),
                ]),

                html.Div("Results Table", style={**section_hdr, "marginTop": "10px"}),
                html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
                    html.Thead(html.Tr([
                        html.Th("#", style=table_header_style),
                        html.Th("Onset", style=table_header_style),
                        html.Th("Duration", style=table_header_style),
                        html.Th("Conf", style=table_header_style),
                        html.Th("Severity", style=table_header_style),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td("1", style=table_cell_style),
                            html.Td("00:12:34.5", style=table_cell_style),
                            html.Td("18.2s", style=table_cell_style),
                            html.Td("0.87", style=table_cell_style),
                            html.Td(html.Span("moderate", style={
                                **badge_style, "margin": 0,
                                "backgroundColor": t["warning"] + "22", "color": t["warning"],
                            }), style=table_cell_style),
                        ]),
                        html.Tr([
                            html.Td("2", style=table_cell_style),
                            html.Td("01:45:02.1", style=table_cell_style),
                            html.Td("52.7s", style=table_cell_style),
                            html.Td("0.95", style=table_cell_style),
                            html.Td(html.Span("severe", style={
                                **badge_style, "margin": 0,
                                "backgroundColor": t["danger"] + "22", "color": t["danger"],
                            }), style=table_cell_style),
                        ]),
                    ]),
                ]),

                html.Div("Text Samples", style={**section_hdr, "marginTop": "14px"}),
                html.Div("Detected 14 seizures across 4 channels", style={"fontSize": sz(0.85), "marginBottom": "4px", "fontFamily": font_family}),
                html.Div("Recording: Rat_07_post-SE_day14.edf", style={"fontSize": sz(0.8), "color": t["text_muted"], "marginBottom": "4px", "fontFamily": font_family}),
                html.Div([
                    html.Span("0123456789 ", style={"fontVariantNumeric": "tabular-nums"}),
                    html.Span("Hz  sec  ms  mV  x", style={"color": t["text_muted"]}),
                ], style={"fontSize": sz(0.85), "marginBottom": "4px", "fontFamily": font_family}),
                html.Div("ABCDEFGHIJKLMNOPQRSTUVWXYZ", style={
                    "fontSize": sz(0.75), "letterSpacing": "0.05em",
                    "color": t["text_muted"], "fontFamily": font_family,
                }),
            ]),
        ]),
    ])


def _build_grid(theme, scale=1.0):
    global _counter
    _counter = 0
    cards = [_make_sample_card(name, family, theme, scale) for name, family in FONTS.items()]
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "20px",
        "padding": "10px 0",
    }, children=cards)


# ── App — use empty temp dir as assets_folder to avoid NED-Net CSS ───

_empty_assets = tempfile.mkdtemp()

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        GOOGLE_FONTS_URL,
    ],
    assets_folder=_empty_assets,
    suppress_callback_exceptions=True,
)

app.layout = html.Div(style={
    "backgroundColor": DARK_THEME["bg"],
    "minHeight": "100vh",
    "padding": "30px",
    "color": DARK_THEME["text"],
    "fontFamily": "'Inter', sans-serif",
}, children=[
    # Title bar
    html.Div(style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "24px",
    }, children=[
        html.H2("NED-Net Font & Theme Explorer", style={
            "margin": 0, "fontFamily": "'Inter', sans-serif",
        }),
        html.Div(style={"display": "flex", "gap": "24px", "alignItems": "center"}, children=[
            html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
                html.Span("Size:", style={"fontSize": "0.85rem"}),
                dbc.RadioItems(
                    id="size-toggle",
                    options=[
                        {"label": " S", "value": 0.85},
                        {"label": " M (current)", "value": 1.0},
                        {"label": " L", "value": 1.2},
                        {"label": " XL", "value": 1.4},
                    ],
                    value=1.0,
                    inline=True,
                    style={"fontSize": "0.85rem"},
                ),
            ]),
            html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
                html.Span("Theme:", style={"fontSize": "0.85rem"}),
                dbc.RadioItems(
                    id="theme-toggle",
                    options=[
                        {"label": " Dark", "value": "dark"},
                        {"label": " Light", "value": "light"},
                    ],
                    value="dark",
                    inline=True,
                    style={"fontSize": "0.85rem"},
                ),
            ]),
        ]),
    ]),

    # Cards — rendered inline (not via callback) so they show on first load
    html.Div(id="font-cards-container", children=_build_grid(DARK_THEME)),
])


@callback(
    Output("font-cards-container", "children"),
    Output("font-cards-container", "style"),
    Input("theme-toggle", "value"),
    Input("size-toggle", "value"),
)
def render_cards(theme_name, scale):
    theme = DARK_THEME if theme_name == "dark" else LIGHT_THEME
    s = float(scale) if scale else 1.0
    bg_style = {"backgroundColor": theme["bg"]}
    return _build_grid(theme, s), bg_style


if __name__ == "__main__":
    app.run(port=8051, debug=False)
