#!/usr/bin/env python
"""Candidate per-animal series from the seizure DB, for matching Prism tables.

The Prism file holds the same quantities under several definitions — per-24h vs
per-recording-day, weeks 1-3 vs 4-5 vs 4-6 — and nothing in the file records
which. So instead of assuming, this generates every plausible definition and
lets the caller identify each table by exact numeric match against the values
already in it.

Series are ordered EGFP first then SV2A, animal ID sorted as text within group,
which is the order the file uses.
"""
from __future__ import annotations

import collections
import os
import re
import sqlite3
import statistics

PROJECTS = os.path.expanduser("~/.eeg_seizure_analyzer/projects")
DB = "lunarc_detect_wk1-6_final.db"
EXCLUDE = {"x", "355676", "372837", "30"}

WEEKSETS = {
    "base": {1, 2, 3}, "lev45": {4, 5}, "lev456": {4, 5, 6},
    "all": {1, 2, 3, 4, 5, 6},
    "w1": {1}, "w2": {2}, "w3": {3}, "w4": {4}, "w5": {5}, "w6": {6},
}


def _load(cut: float):
    conn = sqlite3.connect(os.path.join(PROJECTS, DB))
    conn.row_factory = sqlite3.Row
    wk, dt = {}, {}
    for r in conn.execute("SELECT id, path, date FROM chunks"):
        m = re.search(r"Week(\d+)-Day", r["path"])
        wk[r["id"]] = int(m.group(1)) if m else None
        dt[r["id"]] = r["date"]
    fa = [(r["chunk_id"], r["animal_id"], r["valid_sec"] or 0)
          for r in conn.execute("SELECT chunk_id, animal_id, valid_sec FROM file_animals")]
    ev = [(r["chunk_id"], r["animal_id"], r["type"], r["duration_sec"],
           r["cnn_confidence"] or 0)
          for r in conn.execute("SELECT chunk_id, animal_id, type, duration_sec, "
                                "cnn_confidence FROM events WHERE excluded=0")]
    grp = {r[0]: r[1] for r in conn.execute(
        "SELECT DISTINCT animal_id, group_id FROM file_animals")}
    conn.close()
    return wk, dt, fa, ev, grp


def build(cut: float) -> dict[str, list[float]]:
    """-> {series_name: [value per animal, EGFP then SV2A]}"""
    wk, dt, fa, ev, grp = _load(cut)
    animals = sorted({a for _c, a, _s in fa
                      if a not in EXCLUDE and a in grp and grp[a] in ("Control", "SV2A")})
    order = ([a for a in animals if grp[a] == "Control"]
             + [a for a in animals if grp[a] == "SV2A"])

    out: dict[str, list[float]] = {}
    for wname, weeks in WEEKSETS.items():
        hrs = collections.defaultdict(float)
        days = collections.defaultdict(set)
        for ci, a, s in fa:
            if wk.get(ci) in weeks:
                hrs[a] += s / 3600
                days[a].add(dt[ci])
        cnt = collections.defaultdict(collections.Counter)
        durs = collections.defaultdict(list)
        convdays = collections.defaultdict(set)
        anydays = collections.defaultdict(set)
        for ci, a, t, d, cf in ev:
            if cf < cut or wk.get(ci) not in weeks:
                continue
            cnt[a][t] += 1
            durs[(a, t)].append(d)
            durs[(a, "all")].append(d)
            anydays[a].add(dt[ci])
            if t == "convulsive":
                convdays[a].add(dt[ci])

        for dname, denom in (("h24", lambda a: hrs[a] / 24),
                             ("recday", lambda a: len(days[a])),
                             ("hour", lambda a: hrs[a])):
            for mname, fn in (
                ("conv", lambda a: cnt[a]["convulsive"]),
                ("non", lambda a: cnt[a]["non_convulsive"]),
                ("all", lambda a: cnt[a]["convulsive"] + cnt[a]["non_convulsive"]),
            ):
                vals = []
                for a in order:
                    d = denom(a)
                    vals.append(fn(a) / d if d else float("nan"))
                out[f"{wname}_{mname}_{dname}"] = vals

        for mname, key in (("convdur", "convulsive"), ("nondur", "non_convulsive"),
                           ("alldur", "all")):
            out[f"{wname}_{mname}"] = [
                statistics.mean(durs[(a, key)]) if durs[(a, key)] else float("nan")
                for a in order]

        out[f"{wname}_convfree"] = [
            100.0 * (len(days[a]) - len(convdays[a])) / len(days[a])
            if days[a] else float("nan") for a in order]
        out[f"{wname}_anyfree"] = [
            100.0 * (len(days[a]) - len(anydays[a])) / len(days[a])
            if days[a] else float("nan") for a in order]
        out[f"{wname}_convcount"] = [float(cnt[a]["convulsive"]) for a in order]
        out[f"{wname}_noncount"] = [float(cnt[a]["non_convulsive"]) for a in order]
        out[f"{wname}_allcount"] = [
            float(cnt[a]["convulsive"] + cnt[a]["non_convulsive"]) for a in order]

    out["_order"] = order
    out["_groups"] = [grp[a] for a in order]
    return out


def group_slices(series: dict):
    """-> (n_egfp, n_sv2a) for splitting a full-cohort series into group blocks."""
    g = series["_groups"]
    return g.count("Control"), g.count("SV2A")
