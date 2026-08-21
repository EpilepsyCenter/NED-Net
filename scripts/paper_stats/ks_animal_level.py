#!/usr/bin/env python
"""Distribution comparisons with the ANIMAL (or cell) as the unit of analysis.

The K-S tests in the manuscript pool every event within a group and treat them
as independent, so with 10^3-10^6 events they return vanishing P values for
slight differences and say nothing about whether animals differ. This runs the
same comparison at the right level:

  * build each animal's (or cell's) empirical cumulative distribution on a
    common grid;
  * take the largest gap between the two GROUP-MEAN curves as the statistic —
    the same quantity K-S measures, so it stays comparable to the published D;
  * generate the null by permuting the group labels of ANIMALS, not events.

The statistic is therefore interpretable exactly as before, while the P value
now reflects the number of animals rather than the number of events.

    python scripts/paper_stats/ks_animal_level.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import itertools
import json
import os
import re
import sqlite3
import sys
import zipfile

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism_rewrite_all import PROJECTS, EXCLUDE                 # noqa: E402

ANALYSIS = ("/Users/marcoledri/Dropbox/Work/Manuscripts and papers/"
            "SV2A paper/Data analysis")


def ecdf_matrix(units, grid):
    """-> array (n_units x len(grid)) of each unit's ECDF on a shared grid."""
    out = np.empty((len(units), len(grid)))
    for i, v in enumerate(units):
        s = np.sort(np.asarray(v, float))
        out[i] = np.searchsorted(s, grid, side="right") / len(s)
    return out


def perm_test(a_units, b_units, n_perm=20000, seed=0, log_grid=True):
    """Max gap between group-mean ECDFs, with animals permuted between groups."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([np.asarray(v, float) for v in a_units + b_units])
    pooled = pooled[pooled > 0] if log_grid else pooled
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    grid = (np.geomspace(max(lo, 1e-9), hi, 200) if log_grid
            else np.linspace(lo, hi, 200))
    M = ecdf_matrix(a_units + b_units, grid)
    na = len(a_units)

    def stat(idx_a, idx_b):
        return float(np.max(np.abs(M[idx_a].mean(0) - M[idx_b].mean(0))))

    obs = stat(np.arange(na), np.arange(na, len(M)))
    n = len(M)
    idx = np.arange(n)
    # exact enumeration when the number of splits is small, else sampling
    combos = list(itertools.combinations(range(n), na))
    if len(combos) <= n_perm:
        null = [stat(np.array(c), np.setdiff1d(idx, c)) for c in combos]
        mode = f"exact ({len(combos)} splits)"
    else:
        null = []
        for _ in range(n_perm):
            p = rng.permutation(n)
            null.append(stat(p[:na], p[na:]))
        mode = f"{n_perm} random permutations"
    null = np.asarray(null)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return obs, p, mode, len(a_units), len(b_units)


def spike_isis(cut=0.5, weeks=(1, 2, 3)):
    """-> {group: [per-animal ISI arrays]} for interictal spikes."""
    db = os.path.join(PROJECTS, "sv2a_spikes_wk1-6.db")
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    wk = {}
    for r in c.execute("SELECT id, path FROM chunks"):
        m = re.search(r"Week(\d+)-Day", r["path"])
        wk[r["id"]] = int(m.group(1)) if m else None
    grp = {r[0]: r[1] for r in c.execute(
        "SELECT DISTINCT animal_id, group_id FROM file_animals")}
    per = collections.defaultdict(list)
    for r in c.execute("SELECT chunk_id ci, animal_id a, start_sec s FROM events "
                       "WHERE type='interictal_spike' ORDER BY chunk_id, start_sec"):
        if wk.get(r["ci"]) in weeks and r["a"] not in EXCLUDE:
            per[(r["a"], r["ci"])].append(r["s"])
    c.close()
    out = collections.defaultdict(dict)
    for (a, _ci), v in per.items():
        if len(v) > 1 and grp.get(a) in ("Control", "SV2A"):
            g = "EGFP" if grp[a] == "Control" else "SV2A"
            out[g].setdefault(a, []).extend(np.diff(sorted(v)))
    return {g: [np.array(v) for v in d.values() if len(v) > 20]
            for g, d in out.items()}


def prism_columns(path, title):
    """-> list of per-column value arrays (one column per cell)."""
    z = zipfile.ZipFile(path)
    tit = {}
    for n in z.namelist():
        if n.endswith("sheet.json"):
            d = json.load(io.TextIOWrapper(z.open(n)))
            if d.get("@class") == "DataSheet":
                tu = (d.get("table") or {}).get("uid")
                if tu:
                    tit.setdefault(d.get("title", ""), tu)
    rows = list(csv.reader(io.TextIOWrapper(
        z.open(f"data/tables/{tit[title]}/data.csv"))))
    w = max(len(r) for r in rows)
    cols = []
    for j in range(w):
        v = []
        for r in rows:
            if j < len(r) and (r[j] or "").strip():
                try:
                    v.append(float(r[j].replace(",", ".")))
                except ValueError:
                    pass
        if len(v) > 20:
            cols.append(np.array(v))
    return cols


def report(name, a, b, published_D, **kw):
    obs, p, mode, na, nb = perm_test(a, b, **kw)
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    print(f"{name:34}{na:>4}v{nb:<4}{obs:>8.3f}{p:>10.4g}  {star:4}"
          f"  (published pooled D = {published_D})   [{mode}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", type=float, default=0.5)
    args = ap.parse_args()

    print("Unit of analysis = animal (spikes) or cell (ephys).")
    print("D is the largest gap between the GROUP-MEAN cumulative curves;")
    print("P is from permuting group labels of animals/cells.\n")
    print(f"{'comparison':34}{'n':>9}{'D':>8}{'P':>10}  {'':4}")

    isis = spike_isis(args.cut)
    if isis.get("EGFP") and isis.get("SV2A"):
        report("Fig 4f  spike ISI, EGFP vs SV2A",
               isis["EGFP"], isis["SV2A"], "0.125")

    eph = os.path.join(ANALYSIS, "Ephys")
    try:
        amp_e = prism_columns(os.path.join(eph, "Cont VC Amplitude Analysis.prism"),
                              "Cont VC EGFP Amplitude All")
        amp_s = prism_columns(os.path.join(eph, "Cont VC Amplitude Analysis.prism"),
                              "Cont VC SV2A Amplitude All")
        report("Fig 2g  sIPSC amplitude", amp_e, amp_s, "0.105", log_grid=False)
    except Exception as e:
        print(f"  sIPSC amplitude: {e!r}")

    try:
        f_e = prism_columns(os.path.join(eph, "Cont VC Frequency Analysis.prism"),
                            "Cont VC EGFP Frequency all")
        f_s = prism_columns(os.path.join(eph, "Cont VC Frequency Analysis.prism"),
                            "Cont VC SV2A Frequency all")
        report("Fig 2d  sIPSC inter-event interval", f_e, f_s, "0.225")
    except Exception as e:
        print(f"  sIPSC IEI: {e!r}")

    print("\nA large D with a non-significant P means the curves separate but")
    print("the animals/cells do not consistently — the pooled test was reading")
    print("event count, not a group difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
