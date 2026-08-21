#!/usr/bin/env python
"""Build the Source Data workbook for the SV2A manuscript figures.

One sheet per figure panel, holding the values actually plotted, with animal
IDs and group labels attached — journals ask for source data a reader can map
onto the figure, not a dump of the analysis file's internal grids.

Panels are read from the Prism file so the numbers are exactly what is drawn.
Figure 3f and 3g are the exception: those two panels were added after the Prism
file was built, so they are recomputed from the same database and definitions.

    python scripts/paper_stats/make_source_data.py [--prism FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism_rewrite_all import Data                      # noqa: E402
from fig3_replacement_panels import cumulative          # noqa: E402

BOLD = Font(bold=True)
ITAL = Font(italic=True, size=9)
HDR = PatternFill("solid", fgColor="DDDDDD")
BASE = {1, 2, 3}

# panel -> (prism table title, caption, shape)
#   GG    2 rows (metric) x [EGFP block, SV2A block]
#   PP    2 rows (group)  x [baseline block, LEV block]
#   COL2  2 columns: EGFP, SV2A
#   WEEK  rows = week, [EGFP block, SV2A block]
#   DAY   rows = day,  [EGFP block, SV2A block]
#   RAW   emit the grid as-is (curve data)
PANELS = [
    ("Fig3C", "Seizures/day", "Seizures per 24 h of recording, baseline (weeks 1-3)", "GG"),
    ("Fig3D", "Convulsive free days baseline", "% of recorded days without a convulsive seizure, baseline", "COL2"),
    ("Fig3E", "Per week seizures", "All seizures per 24 h, by baseline week", "WEEK"),
    ("Fig3H", "Duration", "Mean seizure duration (s), whole recording", "GG"),
    ("Fig3I", "Average SE seizure grade", "Average behavioural seizure grade during status epilepticus", "COL2"),
    ("Fig3J", "Cumulative SE seizure grade", "Cumulative behavioural seizure grade during SE", "COL2"),
    ("Fig3K", "SE correlation", "Cumulative SE grade vs chronic seizure rate", "RAW"),
    ("Fig4C", "IS/hour", "Interictal spikes per animal-hour, baseline", "COL2"),
    ("Fig4D", "IS duration", "Mean interictal spike duration (s), baseline", "COL2"),
    ("Fig4E", "ISI frequency dist", "Inter-spike-interval probability density (pooled per group)", "RAW"),
    ("Fig4F", "IS ISI cumulative", "Inter-spike-interval cumulative distribution (pooled per group)", "RAW"),
    ("Fig5A", "Events/animal-hour ALL DAYS", "All seizures per animal-hour, by recording day", "DAY"),
    ("Fig5B", "Convulsive free days pre-post LEV", "% days convulsive-free, baseline vs levetiracetam", "PP"),
    ("Fig5C", "Convulsive pre-post LEV", "Convulsive seizures per 24 h, baseline vs levetiracetam", "PP"),
    ("Fig5D", "Non-Convulsive pre-post LEV", "Non-convulsive seizures per 24 h, baseline vs levetiracetam", "PP"),
    ("Fig5E", "Convulsive Duration pre-post LEV", "Mean convulsive seizure duration (s), baseline vs LEV", "PP"),
    ("Fig5F", "Non-Convulsive Duration pre-post LEV", "Mean non-convulsive seizure duration (s), baseline vs LEV", "PP"),
    ("Fig5G", "IS/hour pre-post LEV", "Interictal spikes per animal-hour, baseline vs levetiracetam", "PP"),
    ("Fig5H_EGFP", "IS ISI cumulative pre-post LEV EGFP", "ISI cumulative distribution, EGFP, baseline vs LEV", "RAW"),
    ("Fig5H_SV2A", "IS ISI cumulative pre-post LEV SV2A", "ISI cumulative distribution, SV2A, baseline vs LEV", "RAW"),
    ("Fig5I", "IS duration pre-post LEV", "Mean spike duration (s), baseline vs levetiracetam", "PP"),
]

TESTS = {
    "Fig3C": "Mann-Whitney U, two-tailed (per seizure type)",
    "Fig3D": "Mann-Whitney U, two-tailed",
    "Fig3E": "Two-way ANOVA (week x group)",
    "Fig3F": "Log-rank (Mantel-Cox)",
    "Fig3G": "Mann-Whitney U on the day-21 endpoint",
    "Fig3H": "Mann-Whitney U, two-tailed (per seizure type)",
    "Fig3I": "Mann-Whitney U, two-tailed",
    "Fig3J": "Mann-Whitney U, two-tailed",
    "Fig3K": "Linear regression per group",
    "Fig4C": "Mann-Whitney U, two-tailed",
    "Fig4D": "Mann-Whitney U, two-tailed",
    "Fig4E": "Descriptive (ISIs pooled within group)",
    "Fig4F": "Kolmogorov-Smirnov on pooled ISIs",
    "Fig5A": "Descriptive (group mean +/- SEM per day)",
    "Fig5B": "Wilcoxon matched-pairs (within animal); Mann-Whitney between groups",
    "Fig5C": "Wilcoxon matched-pairs (within animal); Mann-Whitney between groups",
    "Fig5D": "Wilcoxon matched-pairs (within animal); Mann-Whitney between groups",
    "Fig5E": "Wilcoxon matched-pairs (within animal)",
    "Fig5F": "Wilcoxon matched-pairs (within animal)",
    "Fig5G": "Wilcoxon matched-pairs (within animal)",
    "Fig5H_EGFP": "Kolmogorov-Smirnov on pooled ISIs",
    "Fig5H_SV2A": "Kolmogorov-Smirnov on pooled ISIs",
    "Fig5I": "Wilcoxon matched-pairs (within animal)",
}


def _num(c):
    c = (c or "").strip()
    if c in ("", "-"):
        return None
    try:
        return float(c.replace(",", "."))
    except ValueError:
        return None


def read_tables(prism):
    z = zipfile.ZipFile(prism)
    titles = {}
    for n in z.namelist():
        if n.endswith("sheet.json"):
            d = json.load(io.TextIOWrapper(z.open(n)))
            if (d.get("@class") == "DataSheet"
                    and "ANALYSIS_VIEW" not in (d.get("flags") or [])):
                tu = (d.get("table") or {}).get("uid")
                if tu:
                    titles.setdefault(d.get("title", ""), tu)
    out = {}
    for t, tu in titles.items():
        try:
            out[t] = list(csv.reader(io.TextIOWrapper(
                z.open(f"data/tables/{tu}/data.csv"))))
        except KeyError:
            pass
    return out


def write_header(ws, panel, caption, extra=()):
    ws["A1"] = f"{panel} — {caption}"
    ws["A1"].font = BOLD
    ws["A2"] = f"Statistical test: {TESTS.get(panel, '(see manuscript)')}"
    ws["A2"].font = ITAL
    r = 3
    for e in extra:
        ws.cell(row=r, column=1, value=e).font = ITAL
        r += 1
    return r + 1


def blocks(rows, nE, nS):
    width = max(len(r) for r in rows)
    b1 = 1
    b2 = 1 + (width - 1) // 2
    return b1, b2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prism", default="Seizure_data_conf0.5.prism")
    ap.add_argument("--out", default="SourceData_SV2A_Figures3-5.xlsx")
    ap.add_argument("--cut", type=float, default=0.5)
    args = ap.parse_args()

    tables = read_tables(args.prism)
    D = Data(args.cut)
    E, S = D.egfp, D.sv2a
    nE, nS = len(E), len(S)
    ids = [f"EGFP {a}" for a in E] + [f"SV2A {a}" for a in S]
    groups = ["EGFP"] * nE + ["SV2A"] * nS

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "README"
    lines = [
        ("Source Data — SV2A manuscript, Figures 3-5", BOLD),
        ("", ITAL),
        ("One sheet per figure panel. Values are those plotted in the figure.", ITAL),
        ("", ITAL),
        ("DATASET", BOLD),
        (f"  Detection database: lunarc_detect_wk1-6_final.db", ITAL),
        (f"  Detector-confidence threshold: >= {args.cut}", ITAL),
        ("  Baseline = recording weeks 1-3; levetiracetam = weeks 4-5.", ITAL),
        ("  Seizure rates are per 24 h of recording; spike rates per animal-hour.", ITAL),
        (f"  n = {nE} EGFP / {nS} SV2A. Animals 355676, 372837, 30 and the", ITAL),
        ("  unassigned channel 'x' are excluded throughout.", ITAL),
        ("", ITAL),
        ("ANIMALS, in the column order used across all sheets", BOLD),
        ("  EGFP: " + ", ".join(E), ITAL),
        ("  SV2A: " + ", ".join(S), ITAL),
        ("", ITAL),
        ("NOTES", BOLD),
        ("  Fig 3f and 3g were added after the analysis file was built and are", ITAL),
        ("  recomputed here from the same database and definitions.", ITAL),
        ("  Interictal-spike panels (Fig 4, Fig 5g-i) derive from the separate", ITAL),
        ("  spike-detection database and are unaffected by the seizure-detector", ITAL),
        ("  confidence threshold.", ITAL),
        ("  Behavioural seizure-grade panels (Fig 3i-k) are scored observations,", ITAL),
        ("  not detector output.", ITAL),
        ("", ITAL),
        ("CONTENTS", BOLD),
    ]
    r = 1
    for text, font in lines:
        ws.cell(row=r, column=1, value=text).font = font
        r += 1
    contents_row = r
    ws.column_dimensions["A"].width = 96

    made = []
    for panel, title, caption, shape in PANELS:
        rows = tables.get(title)
        if rows is None:
            print(f"  !! {panel}: table '{title}' not found — skipped")
            continue
        sh = wb.create_sheet(panel)
        rr = write_header(sh, panel, caption)

        if shape in ("GG", "PP"):
            b1, b2 = blocks(rows, nE, nS)
            if shape == "GG":
                labs = [(rows[i][0] or f"row{i}") for i in range(min(2, len(rows)))]
                head = ["animal", "group"] + labs
                for c, h in enumerate(head):
                    cell = sh.cell(row=rr, column=c + 1, value=h)
                    cell.font, cell.fill = BOLD, HDR
                for k in range(nE + nS):
                    base = b1 + k if k < nE else b2 + (k - nE)
                    sh.cell(row=rr + 1 + k, column=1, value=ids[k])
                    sh.cell(row=rr + 1 + k, column=2, value=groups[k])
                    for j in range(len(labs)):
                        v = _num(rows[j][base]) if base < len(rows[j]) else None
                        sh.cell(row=rr + 1 + k, column=3 + j, value=v)
            else:
                head = ["animal", "group", "baseline", "levetiracetam"]
                for c, h in enumerate(head):
                    cell = sh.cell(row=rr, column=c + 1, value=h)
                    cell.font, cell.fill = BOLD, HDR
                for k in range(nE + nS):
                    ri = 0 if k < nE else 1
                    off = k if k < nE else k - nE
                    sh.cell(row=rr + 1 + k, column=1, value=ids[k])
                    sh.cell(row=rr + 1 + k, column=2, value=groups[k])
                    for j, b in enumerate((b1, b2)):
                        c = b + off
                        v = _num(rows[ri][c]) if ri < len(rows) and c < len(rows[ri]) else None
                        sh.cell(row=rr + 1 + k, column=3 + j, value=v)

        elif shape == "COL2":
            for c, h in enumerate(["animal", "group", "value"]):
                cell = sh.cell(row=rr, column=c + 1, value=h)
                cell.font, cell.fill = BOLD, HDR
            colvals = []
            for cj in (0, 1):
                colvals.append([_num(r_[cj]) for r_ in rows
                                if cj < len(r_) and _num(r_[cj]) is not None])
            k = 0
            for gi, vals in enumerate(colvals):
                for v in vals:
                    sh.cell(row=rr + 1 + k, column=1,
                            value=ids[k] if k < len(ids) else "")
                    sh.cell(row=rr + 1 + k, column=2,
                            value="EGFP" if gi == 0 else "SV2A")
                    sh.cell(row=rr + 1 + k, column=3, value=v)
                    k += 1

        elif shape in ("WEEK", "DAY"):
            b1, b2 = blocks(rows, nE, nS)
            head = ["week" if shape == "WEEK" else "day"] + ids
            for c, h in enumerate(head):
                cell = sh.cell(row=rr, column=c + 1, value=h)
                cell.font, cell.fill = BOLD, HDR
            n = 0
            for r_ in rows:
                idx = _num(r_[0]) if r_ else None
                if idx is None:
                    continue
                sh.cell(row=rr + 1 + n, column=1, value=int(idx))
                for k in range(nE + nS):
                    c = b1 + k if k < nE else b2 + (k - nE)
                    v = _num(r_[c]) if c < len(r_) else None
                    sh.cell(row=rr + 1 + n, column=2 + k, value=v)
                n += 1

        else:   # RAW curve data
            sh.cell(row=rr - 1, column=1,
                    value="Curve data as plotted; columns as in the figure.").font = ITAL
            for i, r_ in enumerate(rows):
                for j, c in enumerate(r_):
                    v = _num(c)
                    sh.cell(row=rr + i, column=1 + j, value=v if v is not None else (c or None))

        sh.column_dimensions["A"].width = 14
        sh.column_dimensions["B"].width = 10
        for i in range(2, 26):
            sh.column_dimensions[get_column_letter(i + 1)].width = 13
        made.append((panel, caption))

    # ---- Fig 3f / 3g, recomputed ----
    first = {}
    for ci, a, t, _d in D.ev:
        if t == "convulsive" and D.wk.get(ci) in BASE:
            dn = D.day.get(ci)
            if dn and (a not in first or dn < first[a]):
                first[a] = dn
    sh = wb.create_sheet("Fig3F")
    rr = write_header(sh, "Fig3F", "Time to first convulsive seizure, baseline",
                      ("1 = convulsive seizure that day; 0 = censored (no convulsive "
                       "seizure during baseline, time = last recorded day).",))
    for c, h in enumerate(["animal", "group", "day", "event (1) / censored (0)"]):
        cell = sh.cell(row=rr, column=c + 1, value=h)
        cell.font, cell.fill = BOLD, HDR
    for k, a in enumerate(E + S):
        days = [D.day[ci] for ci, aa, _s in D.fa
                if aa == a and D.wk.get(ci) in BASE]
        t, e = (first[a], 1) if a in first else (max(days) if days else 21, 0)
        sh.cell(row=rr + 1 + k, column=1, value=ids[k])
        sh.cell(row=rr + 1 + k, column=2, value=groups[k])
        sh.cell(row=rr + 1 + k, column=3, value=t)
        sh.cell(row=rr + 1 + k, column=4, value=e)
    sh.column_dimensions["A"].width = 14
    sh.column_dimensions["D"].width = 26
    made.append(("Fig3F", "Time to first convulsive seizure, baseline"))

    cum = cumulative(D, E, S, "convulsive")
    sh = wb.create_sheet("Fig3G")
    rr = write_header(sh, "Fig3G",
                      "Cumulative convulsive seizure burden (min), baseline days 1-21",
                      ("Each animal is one column; the figure also plots the group medians.",))
    for c, h in enumerate(["day"] + ids + ["EGFP median", "SV2A median"]):
        cell = sh.cell(row=rr, column=c + 1, value=h)
        cell.font, cell.fill = BOLD, HDR
    for i, dd in enumerate(range(1, 22)):
        sh.cell(row=rr + 1 + i, column=1, value=dd)
        for k, a in enumerate(E + S):
            sh.cell(row=rr + 1 + i, column=2 + k, value=round(cum[a][i], 3))
        sh.cell(row=rr + 1 + i, column=2 + nE + nS,
                value=round(float(np.median([cum[a][i] for a in E])), 3))
        sh.cell(row=rr + 1 + i, column=3 + nE + nS,
                value=round(float(np.median([cum[a][i] for a in S])), 3))
    sh.column_dimensions["A"].width = 8
    for i in range(nE + nS + 2):
        sh.column_dimensions[get_column_letter(2 + i)].width = 13
    made.append(("Fig3G", "Cumulative convulsive seizure burden, baseline"))

    order = {p: i for i, p in enumerate(
        ["README"] + sorted({m[0] for m in made},
                            key=lambda x: (x[:4], x[4:])))}
    wb._sheets.sort(key=lambda s: order.get(s.title, 999))

    r = contents_row
    for panel, caption in sorted(made, key=lambda m: (m[0][:4], m[0][4:])):
        ws.cell(row=r, column=1, value=f"  {panel:12s} {caption}").font = ITAL
        r += 1

    wb.save(args.out)
    print(f"Wrote {args.out}: {len(made)} panel sheets + README")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
