# Phase 1 — frozen-model validation on RAM_GDNF_2025

**Date:** 2026-10-02. **Status:** complete for the U-Net arm; classical arms not yet run.
Reproduce with `scripts/local/consolidate_mir_annotations.py` then
`scripts/local/validate_frozen_unet_vs_mir.py`. Alignment controls in
`VERIFICATION_LOG_20261001.md`.

## Run

Job 3770118, LUNARC `lu48` node cn096, 2026-10-01 14:27–18:36 CEST (4 h 09 m).
1,377/1,377 cohort EDFs, 0 errors, 2,014 recording hours (~16,100 animal-hours).
Frozen operating point: `UNetv2_20260615` @ 0.5 + boundary 0.1 →
`Convulsive_v4LUNARC_20260616` @ 0.45, no re-ranker.
Output: `ram_gdnf_unet_v0.db`, **41,408 events** (78.1% non-convulsive, 21.9% convulsive).

Ground truth: Mir's 69 per-day workbooks consolidated to
`~/ground_truth/mir_ramgdnf_annotations.csv` — 14,740 reviewed rows over 1,074
recordings; 430 confirmed seizures (368 convulsive, 51 behaviour, 11 untyped),
11,653 `False`, 2,657 `Normal`. Kept as CSV only, never as sidecars, so it cannot
enter a training scan.

## Headline

**The frozen model does not transfer.** 41,408 detections against 430 known seizures;
overall recall **9.1%**.

| filter | detections | recall vs 368 convulsive | precision (adjudicated) |
|---|---|---|---|
| all events | 41,408 | 9.8% | 13.7% (51/372) |
| conf ≥ 0.5 | 11,135 | 7.1% | 17.3% (32/185) |
| convulsive only | 9,089 | 6.8% | 27.6% (35/127) |
| convulsive + conf ≥ 0.5 | 3,769 | 4.6% | 29.6% (21/71) |

Restricting to convulsive roughly doubles precision — the non-convulsive detections are
disproportionately junk — but confidence filtering does not rescue it: precision plateaus
near 30% while recall collapses. **No operating point on this curve is usable.**

Precision denominators are small (only detections overlapping a candidate Mir adjudicated
count; the rest are UNKNOWN, since his pool is not exhaustive). 35/127 carries a 95% CI of
roughly 20–36%.

## The real finding: failure is heterogeneous across batches

Like-for-like, restricted to Mir-annotated recordings:

| batch | files | det/file | seizures | recall |
|---|---|---|---|---|
| Batch 1 | 335 | 2.6 | 60 | **33.3%** |
| Batch 2 | 304 | 18.9 | 24 | **33.3%** |
| Batch 3 | 338 | 1.7 | 154 | **5.2%** |
| Batch 4 | 97 | 69.3 | 192 | **1.6%** |

Two distinct failure modes:

- **Batch 3 is quiet and blind** — 1.7 detections/file, misses 95% of 154 seizures. Pure
  sensitivity failure; the model does not respond to those seizures at all.
- **Batch 4 is loud and blind** — 69.3 detections/file and still misses 98% of 192
  seizures. It fires hard at things that are not the seizures.

Even the best batches sit at 33%, so no batch is usable. Batch 4 alone produces ~82% of all
41,408 detections; Batch 2 another ~15%; batches 1+3 together ~3.5%.

This is **one lab, one rig, one protocol, one species, one model**. The heterogeneity
removes the usual explanations for cross-lab failure (different electrodes, species,
seizure definitions) and makes the generalization claim sharper than a cross-lab comparison
would.

## Controls — why this is not a mapping bug

| control | result |
|---|---|
| Channel mapping | `k−1` confirmed: 9.1% vs 1.4% (`k`) and 0.2% (`k+1`) — 6× margin |
| Time base | median gap to nearest detection −2.2 s; no systematic shift |
| File coverage | all 1,074 ground-truth recordings processed, after normalising Batch 4's `.edf` suffix (which initially hid 192 of 430 seizures) |
| Per-file channel order | ruled out — every file is 8 channels / 8 animals (user-confirmed) |
| Channel-indexed rig effect | ruled out — channel 1 is silent in batches 1–3 and floods in Batch 4, so the effect is per-batch, not per-slot |

## Still open

- **Why Batch 4 floods and Batch 3 is silent.** Batches span ~9 months (B1 Sep 2025,
  B2 Oct 2025, B3 Jan 2026, B4 Jun 2026). Acquisition drift — sampling rate, filter
  settings, gain — is a plausible contributor and is readable from the EDF headers on
  LUNARC with no detection run. If acquisition changed across the series, that is itself a
  finding.
- **Visual confirmation** of what Batch-4 channel-1 detections actually are, and why
  Batch-3 seizures elicit no response. Review queue prepared (below).
- **Classical-detector arms** (White 2006 autocorrelation, Casillas-Espinosa 2019 spectral,
  Twele 2017 spike-train) not yet run. If published methods from other labs fail the same
  way on the same data, the generalization claim becomes a four-method benchmark rather
  than one model's anecdote.

## Consequences for the plan

- **Capture–recapture is no longer load-bearing.** The paper's claim moves from absolute
  sensitivity to a *comparative* one — frozen vs retrained, and across detectors, scored
  against a common reference. Shared source biases largely cancel in that comparison, so
  the dependence problem that dominated the v2 handoff (§4) matters much less.
- **The review burden becomes the measurement.** "How much annotation does a lab need to
  adapt the detector to its own data" is the learning curve, and it is the single most
  useful number for anyone adopting the method.
- **Phase 2 is now load-bearing instead.** If retraining does not recover performance, the
  paper says detectors do not generalize *and* the proposed fix does not work. Run one fold
  early as a feasibility probe before committing the framing to a draft.
