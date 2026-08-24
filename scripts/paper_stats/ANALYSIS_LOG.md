# SV2A manuscript — analysis log

Record of analyses run, decisions taken, and work discarded, kept for the
revision. Every number here was computed from the databases and Prism files
named; the scripts are in the NED-Net repository under `scripts/paper_stats/`
and `scripts/local/`.

Last updated 24 Aug 2026.

---

## 1. Provenance of the figure numbers

**The figures come from `lunarc_detect_wk1-6_final.db`** (the final workflow:
U-Net `UNetv2_20260615` @ 0.5 + hysteresis boundary 0.1 + cascade convulsive
classifier `Convulsive_v4LUNARC_20260616` @ 0.45, no re-ranker).

Established by exact numeric match — all 29 non-zero per-animal values in the
figure reproduced from:

| setting | value |
|---|---|
| database | `lunarc_detect_wk1-6_final.db` |
| detector-confidence cut | see §2 |
| baseline | weeks 1–3 |
| levetiracetam | **weeks 4–5** (week 6 excluded) |
| rate | events ÷ (recording hours / 24) |
| exclusions | `x`, `355676`, `372837`, **`30`** → n = 7 EGFP / 13 SV2A |
| ordering | animal ID sorted as text; "EGFP" = `Control` in the DB |
| zeros | floored to 0.01 (log axis); 0.1 in the per-week table |

### Things this turned up

- **`seizures_results_graph_data_ALLweeks.xlsx` is stale.** It was exported
  from the *older* `SV2A_UNet_wk1-6.db`, over all six weeks, with only three
  exclusions (animal 30 included). It does not match the figures.
- **Animal 30 is excluded** from the figures and from `paper_stats.py`, but was
  included in that workbook. n is 7/13, not 7/14.
- **`stats_dig.py` used weeks 4–6 for the LEV phase while the figures use 4–5.**
  Week 6 has only 13 recording days against 18–21 for the others.
- **Per-animal exclusions live per-database** in the `animal_status` table, not
  in code. `SV2A_UNet_wk1-6.db` carries three rows; `lunarc_detect_wk1-6_final.db`
  carries none, so a fresh Results-tab export from it marks nothing excluded.
- The original Prism file was **internally inconsistent for the LEV panels**:
  `LEV Seizures/day` disagreed with the LEV block of `Convulsive pre-post LEV`
  for the same animal (15.744 vs 26.748), and `LEV Duration` with
  `Convulsive Duration pre-post LEV`. Both now use weeks 4–5 consistently.

---

## 2. Detector-confidence cut — adopted at ≥ 0.5

The manuscript figures originally used **no cut**. A cut of 0.5 was adopted
after review of the AD cohort showed subthreshold detections were false
positives (§7), and the same threshold now applies to both datasets.

### Sensitivity analysis (`conf_sensitivity.py`)

The cut is **not group-neutral**: it keeps 87–96% of convulsive events but only
43–61% of non-convulsive, and unevenly by group.

| | conv kept | non-conv kept |
|---|---|---|
| baseline EGFP | 87% | 61% |
| baseline SV2A | 96% | 48% |
| LEV EGFP | 88% | 60% |
| LEV SV2A | 90% | 43% |

**Baseline results all hold.** Three comparisons break, all non-convulsive:

| measure | P (no cut) | P (≥0.5) |
|---|---|---|
| LEV-period non-convulsive/day | 0.042 | 0.056 |
| LEV-period any-seizure-free days % | 0.032 | 0.094 |
| within-animal EGFP non-convulsive, baseline vs LEV | 0.016 | 0.109 |

Initially read as the cut introducing group bias; on reflection the opposite —
SV2A animals have few real seizures, so a larger share of their detections are
false, and a cut removes proportionally more from them because more were wrong.

### Group-level LEV effects at the figure's definitions (weeks 4–5, per 24 h, Wilcoxon, n = 7)

| measure | EGFP baseline → LEV | P |
|---|---|---|
| convulsive/24 h | 6.97 → 4.21 | 0.156 ns |
| non-convulsive/24 h | 10.11 → 1.64 | 0.156 ns |
| **all seizures/24 h** | 14.60 → 6.88 | **0.031** |
| **% convulsive-free days** | 10.53 → 38.46 | **0.031** |

⚠ **The convulsive rate alone is not significant.** Claims about LEV in
controls should rest on overall burden and convulsive-free days. Worth checking
what Figure 5c's asterisks refer to.

### Possible understatement in Figure 3c

At cut 0.5 with the per-24 h denominator the figure plots:
convulsive **P = 0.00106 (\*\*)**, non-convulsive **P = 0.0114 (\*)**.
The PDF showed a single asterisk on both. Figure 3d is P = 0.00265, consistent
with its `**`.

---

## 3. Figure 3f/3g — replacement of the increase/decrease panel

The old panel F (proportion of animals increasing vs decreasing, week 3 vs
week 1) **lost its result** once the counts were corrected:

| | EGFP | SV2A | Fisher P |
|---|---|---|---|
| old file | 5/7 | 2/13 | 0.0223 \* |
| corrected, no cut | 5/7 | 4/13 | 0.1597 |
| corrected, cut 0.5 | 5/7 | 5/13 | 0.3498 |

Most of the loss is already present at cut 0 — the counts changed, not the cut.
The metric is fragile by construction: four SV2A animals never seize in either
week, and one flips category between cuts on the strength of three events.

**Replaced by two panels** (`fig3_replacement_panels.py`):

- **3f — time to first convulsive seizure (Kaplan–Meier).** EGFP 7/7 seized,
  median day 2; SV2A 8/13 seized, 5 censored. **Log-rank χ² = 5.58, P = 0.018.**
  Treats the never-seizing animals as censored rather than discarding them.
- **3g — cumulative convulsive seizure burden.** Median at day 21: EGFP 80.2
  min vs SV2A 0.5 min, **Mann-Whitney P = 0.0018** on the endpoint.
  Plotted as individual animals plus group medians, **not** mean ± s.e.m.: the
  SV2A group is bimodal (5 animals at exactly 0, 2 non-responders as high as
  EGFP), so means overlap (225 ± 78 vs 45 ± 29 min) while medians separate
  cleanly. A linear y-axis is required — flat-at-zero animals cannot be drawn
  on a log axis.

Convulsive-only was chosen over all-seizures for 3g so that its 5/13
never-seizing animals are the *same* five censored in 3f.

---

## 4. Statistics conventions and what they rest on

### Kolmogorov–Smirnov tests are on POOLED events

Retained as field convention. D values are now in the legends because the
asterisks flatten a real gradient:

| panel | measure | D |
|---|---|---|
| 2d | sIPSC inter-event interval | 0.225 |
| 2e | mIPSC inter-event interval | 0.147 |
| 2g | sIPSC amplitude | 0.105 |
| 2h | mIPSC amplitude | 0.089 |
| 2j | sIPSC rise time | 0.237 |
| 2k | mIPSC rise time | 0.139 |
| 4f | spike inter-spike interval | 0.125 |
| 5h | spike ISI, EGFP before vs LEV | 0.104 |
| 5h | spike ISI, SV2A before vs LEV | 0.066 |

The cumulative panels plot the **first N events per cell**, pooled — N = 100 in
the control condition (6 × 100 = 600), and set by the cell with the fewest
events otherwise. The `1-100` Prism tables hold inter-event **intervals in ms**;
the `…all` tables hold sorted **instantaneous frequencies in Hz**. Different
quantity, different units — a trap.

### If a reviewer challenges the pooling (`ks_animal_level.py`)

Permutation tests on per-unit ECDFs, with animals or cells as the unit:

| comparison | n | D | P |
|---|---|---|---|
| 4f spike ISI, EGFP vs SV2A | 7 v 13 | 0.085 | 0.42 |
| 2g sIPSC amplitude | 6 v 6 | 0.140 | 0.31 |
| 2d sIPSC inter-event interval | 6 v 6 | 0.279 | **0.057** |

None reaches significance at the correct unit of analysis. 2d is underpowered
rather than absent (effect size *grows*, 0.225 → 0.279; the floor for 6 v 6 is
P = 0.001). **4f is the exposed one** — with 7 and 13 animals the result is
clearly null, and the Results text builds on it ("a shift toward shorter
intervals… points to a modest increase in instantaneous spike frequency").

### Amplitude distributions cross rather than shift

Figure 2g/2h differ by K-S but the curves cross:

| sIPSC amplitude (pA) | 5th | 25th | 50th | 75th | 95th |
|---|---|---|---|---|---|
| EGFP | 6.7 | 22.9 | 40.0 | 68.6 | 140.3 |
| SV2A | 16.2 | 25.8 | 40.3 | 62.5 | 126.5 |

Identical medians, narrower SV2A distribution at both tails — reduced spread,
not a shift. The Results text says so. Rise time (2j) moves in one direction at
every percentile, a genuine shift.

### Error bars and summary statistics

Each symbol in the electrophysiology panels is **one cell's median** of all its
events; the bar is the **mean across cells**, with **s.d.** error bars (not
s.e.m.). Legends and Methods now state this.

Frequency must stay on medians: the per-event values are *instantaneous*
frequency, right-skewed, so per-cell means run ~2× the medians and the Figure 2c
group difference is lost (P = 0.048 on medians, 0.116 on means).

### Spot-check precision by confidence band

From `spotcheck.json` (171 events, 30 files):

| band | n | precision |
|---|---|---|
| < 0.35 | 1 | 100% |
| 0.35–0.50 | 8 | 62% |
| 0.50–0.70 | 41 | 88% |
| 0.70–1.00 | 121 | 98% |

The spot-check median confidence is 0.794, so it validates the high-confidence
portion; 43% of the SV2A dataset sits below 0.5 and rests on 9 checked events.
A stratified re-check of ~40 sub-0.5 events would close this if challenged.

---

## 5. Analyses run and DISCARDED

Each was computed in full; none is in the paper. Kept here in case a reviewer
asks for them.

### 5.1 Circadian / diel distribution (`temporal_analysis.py`)

Lights on 08:00–20:00. Baseline, cut 0.5, animal as unit.

| | EGFP | SV2A | P |
|---|---|---|---|
| % seizures in dark phase | 52.7 | 45.5 | 0.038 |
| vs 50% (Wilcoxon) | 0.22 | 0.22 | |
| pooled 24 h Rayleigh | R = 0.044 | R = 0.074 | both "significant" |

**Discarded.** The pooled Rayleigh tests are significant only because 6,873
seizures are treated as independent — vector lengths of 0.04–0.07 mean no
rhythm. Per animal the results are inconsistent. Reportable as a one-line
negative if useful: *seizures showed no diel preference in either group.*

### 5.2 Seizure clustering

| metric | correlation with seizure rate |
|---|---|
| median inter-seizure interval | ρ = −0.91 |
| fraction within 6 h | ρ = +0.91 |
| CV of intervals | ρ = +0.86 |
| longest seizure-free run | ρ = −0.60 |
| CV2 | ρ = −0.68 |

**Discarded.** Three metrics are essentially 1/rate — they restate Figure 3c.
CV2 differed between groups (1.03 vs 1.23, P = 0.007) but the difference
**vanishes against a rate-matched Poisson null** (median z +0.5 vs +1.6,
P = 0.23). Individual EGFP animals do span z = −4.3 to +6.6 — real heterogeneity
in temporal organisation at similar rates, but not a group effect at n = 7.

### 5.3 LEV effect on clustering or rhythmicity

**Discarded — underpowered.** Only 6 EGFP and 3 SV2A animals have ≥10 seizures
in *both* phases; with n = 3 a Wilcoxon cannot return P < 0.25. Dark-phase
proportion 51.5% → 51.0% in EGFP. The drinking-water hypothesis (higher
overnight exposure shifting seizures to the light phase) is **not supported**,
but this is a weak negative, not a real one.

### 5.4 LEV response as a function of baseline burden

Floor effect quantified; **survives** the regression-to-the-mean correction:

| | ρ | P |
|---|---|---|
| change vs baseline (naive) | −0.73 | 0.0002 |
| change vs mean(pre, post) — Oldham | −0.68 | 0.0011 |
| **% change vs baseline** | +0.18 | 0.51 ns |

11 of 13 SV2A animals started below 1 seizure/24 h, against 0 of 7 controls.
Absolute reduction: −5.81/24 h in high-baseline animals vs 0.00 in low
(P = 0.0022). **The proportional reduction does not depend on baseline** — LEV
removes a similar fraction regardless; there is simply more to remove. Not
pursued, but this is a defensible figure if the floor-effect argument is
challenged. Plot pre vs post on log–log with the identity line, not change vs
baseline, to sidestep the Oldham objection visually.

### 5.5 Seizure burden vs NORT / OLT discrimination index

**Impossible with the available data.** The behaviour cohort is not the EEG
cohort: 13 EGFP / 11 SV2A for behaviour against 9 / 14 implanted. Four EGFP
behaviour animals have no usable EEG (channels too noisy to assign an ID), and
three SV2A animals were lost before behaviour. The behaviour tables carry no
animal identifiers, so the pairing cannot be recovered from row order. Ceiling
would be n = 9 and 11 within group, and the across-group correlation would
largely restate the group difference anyway.

### 5.6 Acute levetiracetam electrophysiology — WITHDRAWN

Built as an Extended Data figure, then withdrawn. **The "LEV" recordings are
mIPSCs**: the protocol ran baseline (sIPSC) → TTX (mIPSC) → LEV (mIPSC), so the
drug was only ever applied in TTX, and it was being compared against sIPSC
controls.

| | sIPSC | mIPSC (TTX) | mIPSC + LEV |
|---|---|---|---|
| EGFP frequency (Hz) | 3.39 | 2.53 | 2.20 |
| SV2A frequency (Hz) | 7.47 | 3.27 | 3.03 |
| SV2A rise time (ms) | 1.44 | 0.97 | 0.96 |

The apparent "reversal of the SV2A effect" is the **TTX** step. Figure 2 already
shows the SV2A effect is on the action-potential-dependent component, which TTX
removes. **There are no sIPSC recordings under LEV**, so whether the drug
reverses the SV2A-induced increase in sIPSC frequency cannot be answered.

The valid comparison the dataset supports is **TTX vs TTX + LEV, paired within
cell** — per-cell TTX values are in `Ephys_summary.prism` (`TTX Frequency`,
`TTX Amplitude`, `TTX Rise Time`), but pairing needs cell identities the tables
do not carry. Data retained and clearly annotated in
`Data analysis/Ephys/ExtendedData_LEV_ephys.xlsx`.

---

## 6. Source data

`SourceData_SV2A.xlsx` — one sheet per figure, 66 panels, values as plotted,
with animal IDs and group labels attached and the test named per panel.
Built by `scripts/paper_stats/make_source_data.py` from the Prism files.

Covers Figures 1–7 and Extended Data 1–3. **Not covered:** Supplementary
Figure 1 (NED-Net validation).

⚠ **The workbook's Extended Data numbering does not update itself.** It was
silently stale for a period after a renumbering. If figures are renumbered
again, re-run the builder and check.

---

## 7. AD (5xFAD) cohort — Extended Data Fig. 1

Four 5xFAD mice, three weeks continuous EEG, 1,847 animal-hours (462 h each).
Detection on LUNARC from `ad_edf_data`; databases `ad_spikes.db` and
`ad_seizures.db`. Montage: four animals on channels ch0/ch3/ch5/ch6, numbered
1–4, all 5xFAD.

### Seizures — confidence ≥ 0.5, every event visually confirmed

| | convulsive | non-convulsive | per 24 h |
|---|---|---|---|
| animal 1 | 0 | 0 | 0 |
| animal 2 | 0 | 3 | 0.16 |
| animal 3 | 0 | 0 | 0 |
| animal 4 | 2 | 11 | 0.68 |

16 seizures in 2 of 4 animals on 7 recorded days. **None in week 1**, 2 in
week 2, 14 in week 3 (cohort rate 0 → 0.08 → 0.50 per animal-day). Both
convulsive events are animal 4. Non-convulsive 8.1 ± 3.0 s; convulsive 16.0 and
20.1 s.

**Sub-0.5 detections were reviewed and are false positives** — this is what
justified the 0.5 threshold for both cohorts. At no cut there were 310 events;
the confidence profile is inverted relative to the validated SV2A run (81% at
0.2–0.35 vs 15%).

Coverage is 317/319 files in both runs; the two failures are the short
end-of-day recordings, too brief for baseline estimation.

### ⚠ The spike-rate trend is confounded — decided to report anyway

Spike rate appears to roughly double from week 1 to week 3 (8.2 → 16.4/h
cohort; 3 of 4 animals increase). But **background amplitude falls in all four
animals** over the same period:

| animal | baseline µV, W1 → W3 | change | Spearman vs day |
|---|---|---|---|
| 1 | 17.5 → 15.1 | −14% | −0.74, P < 0.001 |
| 2 | 36.1 → 16.1 | −55% | −0.99, P < 0.001 |
| 3 | 30.9 → 18.6 | −40% | −0.88, P < 0.001 |
| 4 | 23.8 → 21.3 | −11% | −0.70, P < 0.001 |

The detector threshold is `baseline + 4σ`, so it falls in lockstep (animal 2:
149 → 68 µV). A control run with an **absolute** amplitude criterion reverses
the direction — spike rate falls 81% instead of doubling — because signal
amplitude declines globally. **Decision: report the relative-threshold result**,
on the grounds that relative thresholding is standard practice.

Arguments against the confound, should it be raised: the per-animal magnitudes
do not track (animal 3 has the second-largest baseline drop and *no* rate
increase; animals 1 and 4 have the smallest drops and large increases), and
spike amplitude falls *faster* than baseline (animal 2: spike/baseline 27.8 →
14.5), so surviving spikes are moving *toward* the rejection filters, not away.
The absolute-threshold run is itself biased in the opposite direction and used
a much more permissive operating point (369,281 events vs 23,362).

### Seizures and spikes do not track each other

Animal 2 had the steepest spike increase (+476%) but 3 seizures; animal 4 had
13 seizures with +63%; animal 3's spike rate was flat with no seizures. If both
measures appear together, say so.

### Cross-check against the MATLAB IED analysis

Three files were analysed both ways. Raw candidate counts are the same order of
magnitude; the ~10× headline gap is entirely the post-filters, and **local SNR
does nearly all the culling** (confidence halves the raw count, `snr ≥ 10`
removes ~90% of the rest, `xbl ≥ 15` almost nothing). To change spike
sensitivity, move `--min-local-snr`.

---

## 8. Practical gotchas

- **KAHA/AD EDFs are in mV**, and `read_edf` returns physical units unchanged —
  multiply by 1000 for µV. `--min-amplitude-uv` is applied in the data's own
  units, so 150 µV must be written `0.15` for these files.
- **`detect_spikes_all.sh` / `detect_all.sh`**: blank-able variables use the
  no-colon `${VAR=default}` form; with `:=` an exported empty string is silently
  re-defaulted, in both phases via `--export=ALL`.
- **The seizure detector runs on all EEG channels**, not only animal-mapped
  ones, so unused channels can produce blank-animal events. The spike detector
  requires an animal ID and errors without one.
- macOS: bash 3.2 errors on empty arrays under `set -u`; `/usr/bin/rsync` is
  Apple's openrsync (no `--chmod`, no `--partial`) and shadows Homebrew's.
