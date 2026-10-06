# Recording exclusion criteria

**Written 2026-10-06, before the metrics were computed** — so the rule is pre-registered
rather than fitted to the outcome.

## Why this document exists

Three animals (483552, 483553, 483555) produce 74% of all U-Net detections and have one
confirmed seizure between them. Review of a sampled file found the detections to be
**regular ~10 Hz spiking artefact**, not events. Such recordings should be excluded from
analysis — but *how* they are excluded decides whether the result stands up.

Excluding them **because a detector fired on them** is circular: performance-driven
exclusion, and a reviewer will say so. Excluding them because an independent measure of
signal quality, computed from the raw EEG and blind to any model output, exceeds a
threshold fixed in advance is defensible and generalises to future cohorts.

## The metric

`scripts/lunarc/signal_quality.py` computes, per (recording, EEG channel), from the raw
signal only:

| metric | definition |
|---|---|
| `rhythmic_duty_cycle` | fraction of sampled 10 s windows whose 4–20 Hz spectral peak exceeds `peak_ratio` (default 10) × the 1–45 Hz median power |
| `peak_freq_hz` | median frequency of that peak |
| `line_noise_ratio` | 50 Hz band power / 1–45 Hz median power |
| `rms`, `baseline_iqr` | amplitude and its spread |

**`rhythmic_duty_cycle` is the discriminator.** Seizures are rare: in this cohort the
worst animals carry ~4 seizures/day of tens of seconds, i.e. a duty cycle well under 1%.
Persistent rhythmic artefact occupies a large fraction of the recording. A channel that is
rhythmic for 30% of its duration is not a seizing animal, and no plausible seizure burden
reaches that.

Windows are sampled (60 per channel, evenly spaced) rather than exhaustive — a duty-cycle
estimate needs far less than the whole recording, and sampling keeps the sweep cheap.

## Procedure (in this order — the order is the point)

1. Run the sweep over all 1,377 recordings. It reads EDFs only; the output is one CSV in
   `$HOME`, so no project-storage inodes are consumed.
2. Plot the distribution of `rhythmic_duty_cycle` across all 32 animals.
3. **Set the threshold at a natural gap in that distribution**, and record it here with the
   date, before applying it. If there is no clear gap, say so and use a fixed a-priori
   value (duty cycle > 0.05) rather than tuning one.
4. Apply it to exclude whole (animal, recording) channels. Report how many are excluded,
   which animals, and the metric values — not just the count.
5. **Report the primary analysis both with and without the exclusion.** If the conclusion
   depends on it, that dependence is itself the finding and must be stated.

## v1 FAILED its own test (2026-10-06) — recorded, not quietly replaced

The first metric compared the raw 4-20 Hz spectral peak against the 1-45 Hz median power.
Sweep job 3798487 (1,373 files, 10,984 channel-rows) showed it measured the **1/f slope**,
not rhythmicity:

* `peak_freq_hz` was 5.0 Hz at the median **and both quartiles** — pinned to the band edge,
  because EEG power falls with frequency so the 4-20 Hz maximum is almost always at 4-5 Hz.
* `rhythmic_duty_cycle` median 0.92 cohort-wide — no discrimination.
* **The predictions failed in both directions.** Flood animals: 483552 **0.37**, 483553
  0.73, 483555 0.97. High-seizure animals: 483557 **0.775**, 483559 **0.067**.

Nothing was excluded on it. That is what the "revise before excluding" clause above is for.

Two incidental findings from v1 that stand on their own:
* **449382 is a broken channel** on two independent measures — rms 2.44 (40x the ~0.06
  typical) across 322 of its rows, and the highest line-noise ratio at 2.46. 449385 shows
  97 similar rows.
* The three flood animals rank 2nd-4th on line-noise ratio (1.33, 1.32, 1.09 against
  0.5-0.9 typical) — suggestive, not a clean separator.

## v2 — 1/f-detrended spectral prominence

Fit log10(power) ~ a*log10(f) + b over 1-45 Hz and measure how far the 4-20 Hz peak rises
above that background, in dB. `prominence_db` is the primary metric.

An intermediate version required prominence **AND** an autocorrelation peak at 50-250 ms
lag. That was wrong: the two run in opposite directions — flood animals have high
prominence (5.4-7.9 dB) with low acorr (0.05-0.10), seizing animals low prominence
(3.4-4.2 dB) with high acorr (~0.25). The conjunction was near-empty everywhere.
`acorr_peak` is still reported separately, because that contrast is itself informative.

Provisional check on 3 Batch-4 files (2026-10-06) — **predictions pass**:

| animal | | prominence_db | duty |
|---|---|---|---|
| 483555 | flood, 1 seizure | 7.88 | 0.70 |
| 483553 | flood, 0 seizures | 5.57 | 0.40 |
| 483552 | flood, 0 seizures | 5.45 | 0.30 |
| 483551 | 28 seizures | 4.24 | 0.17 |
| 483559 | 81 seizures | 3.97 | 0.13 |
| 483557 | 78 seizures | 3.40 | 0.07 |

All three flood animals rank above all three seizure-bearing animals on both measures, and
the two highest-burden animals score lowest — seizing, not noisy. `peak_freq_hz` now has a
spread (median 7.9 Hz) instead of being pinned at 5.0.

## v2 ALSO FAILED at cohort scale (2026-10-06) — and the search stops here

Sweep job 3799553 (v2, 10,984 channel-rows). Per-animal median `prominence_db`:

| animal | | prominence_db | rank of 32 |
|---|---|---|---|
| 450096 | 11 seizures | 12.91 | 1 |
| 483555 | flood, 1 seizure | 8.78 | 4 |
| **459663** | **51 seizures** | **8.58** | **5** |
| 483553 | flood, 0 seizures | 6.23 | 15 |
| 483552 | flood, 0 seizures | 5.31 | 22 |
| 459658 | 50 seizures | 5.05 | 26 |
| 483557 | 78 seizures | 4.23 | 30 |
| 483559 | 81 seizures | 3.65 | 31 |

The two highest-burden animals do score low as predicted, but **459663 with 51 seizures
ranks 5th** and two of the three flood animals sit mid-table. The distribution is a smooth
gradient from 12.9 to 2.2 with **no natural gap**. The provisional 3-file separation did
not survive the full cohort.

**The search stops here.** Two metrics have now been tried against the same five animals; a
third would be fitting a threshold to a known answer, which is precisely what this document
exists to prevent. Recorded as a negative result: **spectral rhythmicity does not separate
noisy from seizure-bearing animals in this cohort** — and §"Why" below explains why not.

## What survives: amplitude-based exclusion only

**449382 is a dead electrode.** After demeaning, `rms` = 0.000, `rhythmic_duty_cycle` =
0.000, lowest prominence at 2.18, and a spurious 11.25 Hz peak. In v1 (no demeaning) it
read rms 2.44 with the cohort's highest line noise, 2.46 — so it is a large DC offset with
no signal underneath. Unambiguous, measurable, and unrelated to model performance.

**Rule (set 2026-10-06):** exclude any (recording, channel) whose demeaned `rms` is below
0.001 or above 0.5 — dead or saturated. This caught 449382 and nothing seizure-bearing.
Narrower than hoped, but defensible.

## Why rhythmicity could not find "bad animals"

Because the problem is **cohort-wide, not a few outliers**. Running the identical metric on
SV2A (the training cohort) versus RAM_GDNF:

| metric | SV2A | RAM_GDNF | P(GDNF>SV2A) | p |
|---|---|---|---|---|
| `rhythmic_duty_cycle` | 0.20 | 0.53 | 0.694 | 1.7e-32 |
| `prominence_db` | 4.92 | 6.17 | 0.673 | 3.8e-26 |
| `acorr_peak` | 0.076 | 0.119 | 0.657 | 8.8e-22 |
| `rms` | 0.077 | 0.053 | 0.375 | 2.2e-14 |

52% of RAM_GDNF channel-rows exceed SV2A's own 75th percentile of rhythmicity, against 22%
of SV2A's. `duty_cycle`, `acorr_peak` and `prominence_db` are scale-free ratios, so this is
not a units or gain artefact. The shifts are consistent but **moderate** (AUC 0.66-0.69);
the distributions overlap heavily. There is no subset to exclude because the whole cohort
sits further along the same axis.

## The flood animals: transparency instead of a metric

483552, 483553 and 483555 produced 74% of all detections. They are handled by disclosure,
not by a metric we could not validate:

> A stratified random sample of 113 of their detections — drawn before review, spanning all
> three confidence terciles — was adjudicated by a human. **0 were genuine events.**

They were flagged **because** they produced many detections, which is performance-driven in
origin, so the protection is to **report the primary analysis both including and excluding
them** and state the origin plainly. A reviewer then has everything needed to accept or
reject the exclusion. This is weaker than an independent signal criterion would have been,
and that weakness is part of the report.

## Review evidence that motivates the metric (2026-10-06)

Two sampled files, from the lowest and highest confidence terciles, 81 events reviewed in
full: **0 real events**. The reviewer's description — regular ~10 Hz spiking, and **no
evolution of the signal** in any event — is what the metric must capture. Confidence
carried no information: the highest-confidence file was as wrong as the lowest.

## Predictions to check against (stated in advance)

- 483552, 483553 and 483555 should show a **high** `rhythmic_duty_cycle` with
  `peak_freq_hz` near 10 Hz, matching the reviewer's description.
- The high-seizure animals (483557 at 78 and 483559 at 81 confirmed seizures) should show a
  **low** duty cycle despite their seizure burden — they are seizing, not noisy.
- If that separation does not appear, the metric does not capture what the reviewer saw and
  must be revised **before** being used to exclude anything.

## Separate, non-negotiable exclusion

The 37 Batch-4 recordings whose ground-truth timestamps fall outside their EDF
(`review/groundtruth_out_of_range.csv`) are excluded from both training and evaluation on
data-integrity grounds, independent of signal quality. Decided 2026-10-06.
