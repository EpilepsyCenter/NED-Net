"""Shared look of background stretches (seizure-free stretches for training).

Background is reviewed like seizure events but must never be confused with them,
so it has its own colour family (blues / grey), distinct from the seizure labels
(pending amber, confirmed green, rejected red, manual purple).
"""

BACKGROUND_LABEL = "Background"          # method label in tables and legends
BACKGROUND_METHOD = "background_rule"    # features.detection_method of proposals

BG_COLORS = {
    "pending":   {"fill": "rgba(56, 139, 253, 0.16)", "line": "#388bfd", "text": "#58a6ff"},
    "confirmed": {"fill": "rgba(57, 197, 207, 0.22)", "line": "#39c5cf", "text": "#39c5cf"},
    "rejected":  {"fill": "rgba(110, 118, 129, 0.28)", "line": "#6e7681", "text": "#8b949e"},
    "manual":    {"fill": "rgba(57, 197, 207, 0.22)", "line": "#39c5cf", "text": "#39c5cf"},
}

# Row tint in result tables (AG Grid getRowStyle)
BG_ROW_STYLE = {"backgroundColor": "rgba(56, 139, 253, 0.10)"}


def bg_colors(label: str) -> dict:
    return BG_COLORS.get(label, BG_COLORS["pending"])
