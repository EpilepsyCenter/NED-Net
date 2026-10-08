# Model registry

**Updated 2026-10-08.** Every model trained for this paper, what it saw, and what it
scored. The project DBs record no model name, so this plus `RUN_LOG.md` is the only link
from a result to the weights that produced it.

All live in `~/.eeg_seizure_analyzer/models/` on LUNARC.

## READ THIS FIRST: the convulsive classifier does NOT affect detection

`_apply_convulsive_classifier` (`ml/predict.py:242`) runs **after** detection and
**mutates events in place**, setting `features["convulsive"]` and
`features["convulsive_probability"]`. It never adds, removes or filters an event.

Consequences, which govern how every number in this project is read:

* **Which events exist depends only on the U-Net** — `MODEL`, `THRESHOLD`,
  `BOUNDARY_THRESHOLD`, `min_duration`, `merge_gap`.
* **`CONV_THRESHOLD` cannot change what was detected.** It is inert for recall,
  precision, detection counts and coverage.
* **Stage 2 cannot affect recall of detection.** If the U-Net misses a seizure, Stage 2
  never sees it.
* The **only** thing Stage 2 moves is the convulsive/non-convulsive label on events the
  U-Net already found.

**Errors this caused, now corrected:** job 3820427 was described as testing "arm A +
retrained Stage 2" against the frozen cascade on convulsive recall. That metric is purely
the U-Net's; the 5.2% -> 3.3% change came from the U-Net and the operating point, and
Stage 2 was never tested by it. Threshold-validation sweeps likewise do not need
`CONV_THRESHOLD` varied.

**What Stage 2 should be measured on:** of the detections that overlap a convulsive
ground-truth event, what fraction does Stage 2 label convulsive? Frozen: 4 of 8. Retrained:
3 of 5. Both near 50-60% on single-digit counts.

## The cascade has two stages, trained separately

| | Stage 1 — U-Net | Stage 2 — convulsive classifier |
|---|---|---|
| decides | seizure vs not | of a seizure: convulsive vs not |
| needs | seizures **and** non-seizures | **only** seizures, split by type |
| uses `rejected` annotations? | **yes**, as hard negatives | **no** — "rejection is the detector's job" (`ml/dataset.py:844`) |

That second row is why Mir's contaminated `False` rows matter for Stage 1 and are irrelevant
to Stage 2.

## Stage 1 — U-Net

**Read the metrics correctly:** the checkpoint is selected on **`val_f1` first**, with
`val_loss` only as a tiebreak (`ml/train.py:655`). So the log line "Best val loss" is the
val loss **at the best-f1 epoch**, not the minimum val loss across epochs. Compare arms on
`event_f1`; a lower val_loss at some other epoch is not the saved model.

| model | trained on | positives | negatives | val_loss @ best-f1 | **event_f1** |
|---|---|---|---|---|---|
| `UNetv2_20260615` | SV2A only | 867 | 1,325 hard | 0.719 | **0.782** (on SV2A) |
| `ramgdnf_armB_holdB3` | RAM_GDNF only | 184 | 6,231 hard | 0.805 | 0.056 |
| `ramgdnf_armB_holdB3_stable` | RAM_GDNF only, stable split | ~184 | hard | 0.818 | 0.012 |
| `ramgdnf_armA_holdB3` | SV2A + RAM_GDNF | 865 | 12,925 hard **incl. Mir's `False`** | 0.676 | 0.329 (0.512 @ thr 0.9) |
| `ramgdnf_armA2_holdB3_randneg` | SV2A + RAM_GDNF, background negatives only | 865 | random | 0.639 | **0.532** (ep 15) |
| `ramgdnf_armA3_holdB3` | symlink tree, **no Mir negatives** | 864 | 1,371 SV2A hard + ~10,800 background | 0.704 | 0.410 (ep 7, plateaued) |

**armA2 was rescued** (2026-10-08). It timed out at epoch 17 of 50 (1,220 s/epoch;
`--neg-source random` reads scattered background windows, 6x slower than reusing
rejected-event windows) and never wrote `metadata.json`, orphaning its epoch-14
`best_model.pt`. Since only five fields matter for inference —
`architecture`, `target_fs`, `window_sec`, `include_activity`, `n_classes` — plus
`train_config`'s `base_filters`/`depth`/`dropout` for reconstructing the network, and arm A2
used flags identical to `UNetv2_20260615`, a metadata file was written by hand from that
model's. It loads with **23,421,826 params**, matching exactly, so the weights are usable.
The file carries a `RECONSTRUCTED` field: **training metrics are absent and must not be
quoted from it** — the real numbers live only in the job log (event_f1 **0.5323** at epoch
15, the best of any arm, which is also the epoch whose weights were saved since selection is
on f1). Note that event_f1 swung 0.056-0.532 epoch to epoch across every arm, so it is a
noisy criterion and single-epoch values should be treated with care — but it is the
criterion the code uses, so it is the one to compare on.

**armA3 is the clean version of that experiment** — same positives and split as arm A, one
variable changed, at arm A's speed. Trained against `~/train_nomirneg`, a flat directory of
symlinked EDFs plus real sidecars, so the live tree (and the human reviews in it) is
untouched. Verified: symlinks resolve, 38 animal groups with real IDs, 1,214 positives after
the Batch-3 holdout.

## Stage 2 — convulsive classifier

| model | trained on | classes | best F1 | optimal threshold |
|---|---|---|---|---|
| `Convulsive_v4LUNARC_20260616` | SV2A only | — | 0.885 (on SV2A) | 0.45 |
| `conv_armA_holdB3` | SV2A + RAM_GDNF | 451 convulsive / 383 non-convulsive | **0.659** | **0.55** |

The frozen Stage 2 labelled only **7 of 153** of Mir's Batch-3 convulsive seizures as
convulsive (4.6%) — not because of bad negatives, but because it had never seen a RAM_GDNF
convulsive seizure. `conv_armA_holdB3` has now seen 368 of them.

## Operating points do NOT transfer between models

This is a result, not a footnote.

| model | threshold | boundary | note |
|---|---|---|---|
| `UNetv2_20260615` | 0.5 | 0.1 | the frozen production point |
| `ramgdnf_armA_holdB3` | **0.9** | 0.5 | its own validation picked 0.9 |
| `Convulsive_v4LUNARC` | 0.45 | — | |
| `conv_armA_holdB3` | **0.55** | — | |

Running arm A at the **frozen** point (0.5 / boundary 0.1) produced events with a median
duration of **1,713 s** that flagged **72% of all recorded time** as ictal — because arm A's
probabilities sit lower-scaled, so once a 0.5 crossing occurs the 0.1 hysteresis boundary
almost never closes the event. Its apparent 69.3% convulsive recall was **at chance** for
72% coverage. Anyone reusing published weights together with published thresholds would hit
this, and could easily mistake it for high sensitivity if they looked only at recall.

## Ranking on the selection criterion

| arm | negatives | **event_f1** |
|---|---|---|
| **A2** | random background only | **0.532** |
| A3 | SV2A hard + background | 0.410 |
| A | includes Mir's `False` rows | 0.329 |
| B | RAM_GDNF only, 184 positives | 0.012 |

Both arms that exclude Mir's rejections beat arm A, A2 by 62%. That is the contamination
result, and it is robust to which of the two clean recipes is used.

**A correction on record:** an earlier version of this table compared arms on "best val
loss", mixing per-epoch minima for A2/A3 with the summary value for arm A. Since the
checkpoint is chosen on f1, that comparison was both inconsistent and measuring the wrong
thing. The conclusion was unchanged.

## Comparisons the paper needs

Three rows, same held-out Batch 3, each model at **its own** validated operating point:

1. **frozen cascade** — `UNetv2_20260615` @ 0.5/0.1 + `Convulsive_v4LUNARC` @ 0.45
   -> convulsive recall **5.2%** (8/153)
2. **Stage 2 retrained only** — arm A U-Net @ 0.9/0.5 + `conv_armA_holdB3` @ 0.55 (job 3820427)
3. **both stages retrained** — arm A3 U-Net + `conv_armA_holdB3` (after job 3820377)
