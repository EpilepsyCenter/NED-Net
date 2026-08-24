#!/usr/bin/env python
"""Extract per-animal, per-phase seizure rates from a NED-Net project database.

Baseline = recording weeks 1-3, levetiracetam = weeks 4-6 (parsed from the
``WeekN-DayNN`` folder in each chunk's path, so weeks are per-batch relative).

    python scripts/paper_stats/stats_dig.py [DB_NAME]

DB_NAME defaults to the database the manuscript's exported workbook was built
from (see below). Writes ``seizure_per_animal.json`` next to this script, which
``paper_stats.py`` consumes.

Which database?
---------------
``SV2A_UNet_wk1-6.db``          detection run of 17-22 Jun 2026; 10,918 events.
                                This is what ``seizures_results_graph_data_
                                ALLweeks.xlsx`` was exported from (exact match
                                on all four group x type counts, no confidence
                                filter), i.e. the current figures.
``lunarc_detect_wk1-6_final.db`` detection run of 26 Jun 2026; 24,839 events.
                                The final workflow described in the Methods:
                                threshold 0.5 + hysteresis boundary 0.1 +
                                cascade convulsive classifier.

These are different pipelines and give different counts. Pick one and use it
for the figures and the Methods alike.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROJECTS = Path(os.path.expanduser("~/.eeg_seizure_analyzer/projects"))
DEFAULT_DB = "lunarc_detect_wk1-6_final.db"

# Excluded everywhere: two noisy control animals + the unassigned channel.
# Matches the `excluded` column of the exported workbook (n = 7 EGFP / 14 SV2A).
EXCLUDE = {"30", "355676", "372837", "x"}


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    conn = sqlite3.connect(PROJECTS / db)
    conn.row_factory = sqlite3.Row
    print(f"database: {db}   excluding: {sorted(EXCLUDE)}\n")

    week = {}
    for r in conn.execute("SELECT id, path, date FROM chunks"):
        m = re.search(r"Week(\d+)-Day(\d+)", r["path"])
        week[r["id"]] = (int(m.group(1)) if m else None, r["date"])

    grp = {}
    for r in conn.execute("SELECT DISTINCT animal_id, group_id FROM file_animals"):
        g = (r["group_id"] or "").strip().lower()
        if g in ("control", "sv2a") and r["animal_id"] not in EXCLUDE:
            grp[r["animal_id"]] = g

    days = collections.defaultdict(set)
    for r in conn.execute("SELECT animal_id a, chunk_id ci FROM file_animals"):
        w, d = week.get(r["ci"], (None, None))
        if w is None or r["a"] not in grp:
            continue
        days[(r["a"], "base" if w <= 3 else "lev")].add(d)

    counts = collections.defaultdict(collections.Counter)
    durs = collections.defaultdict(list)
    conv_days = collections.defaultdict(set)
    for r in conn.execute("SELECT animal_id a, chunk_id ci, type, duration_sec "
                          "FROM events WHERE excluded=0"):
        w, d = week.get(r["ci"], (None, None))
        if w is None or r["a"] not in grp:
            continue
        ph = "base" if w <= 3 else "lev"
        counts[(r["a"], ph)][r["type"]] += 1
        durs[(r["a"], ph, r["type"])].append(r["duration_sec"])
        if r["type"] == "convulsive":
            conv_days[(r["a"], ph)].add(d)

    hdr = (f"{'animal':9}{'grp':8}{'ph':6}{'days':6}{'conv':7}{'nonconv':9}"
           f"{'conv/d':9}{'nonconv/d':11}{'%CSF':7}{'convDur':8}")
    print(hdr)
    rows = []
    for a in sorted(grp, key=lambda x: (grp[x], x)):
        for ph in ("base", "lev"):
            nd = len(days[(a, ph)])
            if nd == 0:
                continue
            nc = counts[(a, ph)]["convulsive"]
            nn = counts[(a, ph)]["non_convulsive"]
            csf = 100.0 * (nd - len(conv_days[(a, ph)])) / nd
            cdur = (statistics.mean(durs[(a, ph, "convulsive")])
                    if durs[(a, ph, "convulsive")] else float("nan"))
            rows.append(dict(a=a, g=grp[a], ph=ph, nd=nd, conv=nc, non=nn,
                             cpd=nc / nd, npd=nn / nd, csf=csf, cd=cdur))
            print(f"{a:9}{grp[a]:8}{ph:6}{nd:<6}{nc:<7}{nn:<9}"
                  f"{nc / nd:<9.2f}{nn / nd:<11.2f}{csf:<7.1f}{cdur:<8.1f}")

    (HERE / "seizure_per_animal.json").write_text(json.dumps(rows, indent=1))
    nC = len({r["a"] for r in rows if r["g"] == "control"})
    nS = len({r["a"] for r in rows if r["g"] == "sv2a"})
    print(f"\nn = {nC} EGFP / {nS} SV2A   ->  seizure_per_animal.json")


if __name__ == "__main__":
    main()
