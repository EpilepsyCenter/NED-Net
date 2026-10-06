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

## Threshold

_Not yet set — the sweep has not run. Record it here with the date and the distribution it
was read from._

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
