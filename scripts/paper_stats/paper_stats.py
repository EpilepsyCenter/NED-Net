#!/usr/bin/env python
"""Reproduce the electrographic statistics reported in the SV2A manuscript.

Reads the per-animal values extracted by ``stats_dig.py`` (seizures) and
``spikes_dig.py`` (interictal spikes) and prints the three comparison tables
that appear in the appendix of ``SV2A_Methods_NEDNet_draft.md``.

Animal as the unit of analysis. Animals 355676 and 372837 are excluded from all
analyses, as is the unassigned channel 'x' — matching the ``excluded`` column of
``seizures_results_graph_data_ALLweeks.xlsx``, which gives n = 7 EGFP / 14 SV2A.
Baseline = recording weeks 1-3, levetiracetam = weeks 4-6.

No detector-confidence cut is applied. The exported workbook reproduces exactly
at min_conf = 0 (Control 7,574 events = 3,697 convulsive + 3,877
non-convulsive; SV2A 2,176 = 989 + 1,187), so none was used.

    python scripts/paper_stats/paper_stats.py

Regenerate the inputs first if the detection databases change::

    python scripts/paper_stats/stats_dig.py     # -> seizure_per_animal.json
    python scripts/paper_stats/spikes_dig.py    # -> spikes_per_animal.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXCLUDE = {"30", "355676", "372837", "x"}
HERE = Path(__file__).parent


def _load(name: str) -> list[dict]:
    rows = json.loads((HERE / name).read_text())
    return [r for r in rows if r["a"] not in EXCLUDE]


def _col(rows, group, phase, key) -> np.ndarray:
    v = np.array([r[key] for r in rows if r["g"] == group and r["ph"] == phase], float)
    return v[~np.isnan(v)]


def _between(rows, phase, measures, title):
    print(f"\n=== {title} (Mann-Whitney U, two-tailed) ===")
    print(f"{'measure':32}{'EGFP':>18}{'SV2A':>18}{'P':>12}")
    for key, label in measures:
        a = _col(rows, "control", phase, key)
        b = _col(rows, "sv2a", phase, key)
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        print(f"{label:32}{np.median(a):>10.2f} (n={len(a)}){np.median(b):>10.2f} (n={len(b)}){p:>12.4g}")


def _within(rows, measures, title):
    print(f"\n=== {title} (Wilcoxon signed-rank, within animal) ===")
    print(f"{'measure':28}{'group':10}{'baseline':>12}{'LEV':>12}{'P':>12}")
    for key, label in measures:
        for group in ("control", "sv2a"):
            base = {r["a"]: r[key] for r in rows if r["g"] == group and r["ph"] == "base"}
            lev = {r["a"]: r[key] for r in rows if r["g"] == group and r["ph"] == "lev"}
            ids = [a for a in base if a in lev]
            x = np.array([base[a] for a in ids])
            y = np.array([lev[a] for a in ids])
            p = stats.wilcoxon(x, y).pvalue
            print(f"{label:28}{group:10}{np.median(x):>12.2f}{np.median(y):>12.2f}{p:>12.4g}"
                  f"   (n={len(ids)})")


def main() -> None:
    sz = _load("seizure_per_animal.json")
    sp = _load("spikes_per_animal.json")

    sz_measures = [
        ("cpd", "convulsive seizures/day"),
        ("npd", "non-convulsive seizures/day"),
        ("csf", "convulsive seizure-free days %"),
        ("cd", "mean convulsive duration (s)"),
    ]
    sp_measures = [("rate", "interictal spikes/hour"), ("md", "spike duration (ms)")]

    _between(sz, "base", sz_measures, "SEIZURES - between groups, baseline")
    _between(sp, "base", sp_measures, "SPIKES - between groups, baseline")
    _between(sz, "lev", sz_measures[:3], "SEIZURES - between groups, during levetiracetam")
    _within(sz, sz_measures[:3], "SEIZURES - baseline vs levetiracetam")
    _within(sp, sp_measures[:1], "SPIKES - baseline vs levetiracetam")

    print("\n=== Shapiro-Wilk (baseline) ===")
    for rows, measures in ((sz, sz_measures[:3]), (sp, sp_measures)):
        for key, label in measures:
            a = _col(rows, "control", "base", key)
            b = _col(rows, "sv2a", "base", key)
            print(f"{label:32} EGFP P={stats.shapiro(a).pvalue:.4f}   "
                  f"SV2A P={stats.shapiro(b).pvalue:.4f}")
    print("\nSeizure-rate measures reject normality -> non-parametric tests throughout.")


if __name__ == "__main__":
    main()
