# Model registry

**Updated 2026-10-08.** Every model trained for this paper, what it saw, and what it
scored. The project DBs record no model name, so this plus `RUN_LOG.md` is the only link
from a result to the weights that produced it.

All live in `~/.eeg_seizure_analyzer/models/` on LUNARC.

## The cascade has two stages, trained separately

| | Stage 1 — U-Net | Stage 2 — convulsive classifier |
|---|---|---|
| decides | seizure vs not | of a seizure: convulsive vs not |
| needs | seizures **and** non-seizures | **only** seizures, split by type |
| uses `rejected` annotations? | **yes**, as hard negatives | **no** — "rejection is the detector's job" (`ml/dataset.py:844`) |

That second row is why Mir's contaminated `False` rows matter for Stage 1 and are irrelevant
to Stage 2.

## Stage 1 — U-Net

| model | trained on | positives | negatives | best val | event_f1 |
|---|---|---|---|---|---|
| `UNetv2_20260615` | SV2A only | 867 | 1,325 hard | 0.719 | **0.782** (on SV2A) |
| `ramgdnf_armB_holdB3` | RAM_GDNF only | 184 | 6,231 hard | 0.805 | 0.056 |
| `ramgdnf_armB_holdB3_stable` | RAM_GDNF only, stable split | ~184 | hard | 0.818 | 0.012 |
| `ramgdnf_armA_holdB3` | SV2A + RAM_GDNF | 865 | 12,925 hard **incl. Mir's `False`** | 0.676 | 0.329 (0.512 @ thr 0.9) |
| `ramgdnf_armA2_holdB3_randneg` | SV2A + RAM_GDNF, background negatives only | 865 | random | 0.595 @ ep14 | 0.532 @ ep15 |
| `ramgdnf_armA3_holdB3` | symlink tree, **no Mir negatives** | 864 | 1,371 SV2A hard + ~10,800 background | _pending_ | _pending_ |

**armA2 is unusable** — it timed out at epoch 17 of 50 (1,220 s/epoch; `--neg-source random`
reads scattered background windows, 6x slower) and never wrote `metadata.json`, so detection
cannot load it. Its epoch-14 `best_model.pt` exists but is orphaned. It is kept only as
evidence that removing Mir's negatives lowers val loss (0.595 against arm A's 0.676).

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

## Comparisons the paper needs

Three rows, same held-out Batch 3, each model at **its own** validated operating point:

1. **frozen cascade** — `UNetv2_20260615` @ 0.5/0.1 + `Convulsive_v4LUNARC` @ 0.45
   -> convulsive recall **5.2%** (8/153)
2. **Stage 2 retrained only** — arm A U-Net @ 0.9/0.5 + `conv_armA_holdB3` @ 0.55 (job 3820427)
3. **both stages retrained** — arm A3 U-Net + `conv_armA_holdB3` (after job 3820377)
