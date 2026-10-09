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

**Amendment, 2026-10-08 — the label is usable as a filter, and what that measures.**
Everything above still holds: Stage 2 only labels. But because the label *can* be used to
filter when reporting, a CONV_THRESHOLD sweep on an existing DB is meaningful — and free,
since `events.convulsive_confidence` is stored per event, so no re-detection is needed
(`scripts/local/conv_threshold_sweep.py`). That sweep found, on retained B1+B2 against
Mir's convulsive reference:

| | frozen + `Convulsive_v4LUNARC` | A3 + `conv_armA_holdB3` |
|---|---|---|
| precision, Stage 1 only | 33% | 10% |
| precision @ 0.45 | **61%** | 11% |
| precision @ 0.80 | **81%** | 16% |
| `convulsive_confidence` p50 | 0.12 | **0.66** |

So Stage 2 is an **informative** type filter for the frozen model and **effectively
uninformative** for the retrained one — precision is flat across 0.10-0.45, and raising
0.30 -> 0.45 costs 6 of 32 correct detections for nothing.

**Why, and this is the actionable part:** Stage 2 is trained on *confirmed seizures only*
(`build_convulsive_specs` ignored rejected rows — "rejection is the detector's job") yet it
is **deployed on every detection**, most of which are false positives. It had therefore
never seen a non-seizure. For the frozen classifier that failed safely: RAM_GDNF noise was
unfamiliar, scored low, and precision rose. After retraining on 368 of Mir's RAM_GDNF
convulsive seizures, the same noise resembles them, so it scores high. **No threshold can
separate classes the classifier was never shown.**

Fixed by `--conv-neg-from-rejected` (2026-10-08), which emits rejected events as
non-convulsive windows, capped at `--conv-neg-pos-ratio` total negatives per convulsive
positive (default 5; non-convulsive *seizures* are kept in full and rejected events fill
the rest). Mir's rejected rows are the right source and the asymmetry is deliberate:
**video-adjudicated "not a convulsive/behavioural seizure" is wrong as a DETECTION negative
and exactly right as a Stage-2 negative** (`build_merged_sidecars.py:76`). They are excluded
from the detector by `--hard-neg-exclude-method mir_candidate` and trained on here — same
rows, opposite role, because the two stages answer different questions.

This changes Stage 2's question from "given a seizure, is it convulsive?" to "is this
detection a convulsive seizure at all?" — which is what deployment actually asks. It still
only labels; it never filters.

## The cascade has two stages, trained separately

| | Stage 1 — U-Net | Stage 2 — convulsive classifier |
|---|---|---|
| decides | seizure vs not | of a seizure: convulsive vs not |
| needs | seizures **and** non-seizures | **only** seizures, split by type |
| uses `rejected` annotations? | **yes**, as hard negatives | **no** — "rejection is the detector's job" (`ml/dataset.py:844`) |

That second row is why Mir's contaminated `False` rows matter for Stage 1 and are irrelevant
to Stage 2.

## Stage 1 — U-Net

**Read the metrics correctly — this has caused three errors already.** `ml/train.py:654`
selects the checkpoint on **`best_event_f1`**: the event f1 at the *optimal* threshold found
by `_best_threshold_metrics`, with `val_loss` only as a tiebreak. Three different numbers
appear in the logs and they are not interchangeable:

| in the log | what it is | is it the criterion? |
|---|---|---|
| per-epoch `event_f1:` | f1 at the **default** threshold | **no** |
| `Best val loss:` | val loss **at the selected epoch**, not the minimum | no |
| `Best event_f1:` | `event_f1` at the selected epoch, default threshold | no |
| `best_metrics['best_event_f1']` | f1 at the **optimal** threshold | **YES** |

Worked example: arm A3's saved checkpoint is **epoch 3** (per-epoch `event_f1` 0.262) and
not epoch 7 (0.410), because epoch 3's `best_event_f1` was 0.4847 and higher. The code is
right; the log is just easy to misread.

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
| `conv_rejneg_holdB3` | + rejected events as negatives, hold out B3 | 451 / 383 (flags ignored — ran pre-pull) | 0.5628 | 0.85 |
| **`conv_temporal`** | + rejected negatives, **temporal split**, no batch held out | **723 convulsive / 3,171 non-convulsive** (766 seizures + 2,849 rejected) | **0.6040** | **0.75** |

**The three Stage-2 F1 figures above are NOT comparable with each other.** Adding
rejected negatives changed the task from "given a seizure, is it convulsive?" to "is this a
convulsive seizure at all?", and `conv_temporal` also changed the split from by-animal to
temporal. Compare them only on a common test set, through the threshold sweep.

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

| arm | negatives | `event_f1` @ default | **`best_event_f1`** (the criterion) |
|---|---|---|---|
| A | includes Mir's `False` rows | 0.329 | **0.512** |
| A3 | SV2A hard + background | 0.262 | **0.485** |
| A2 | random background only | 0.532 | **unknown** — timed out, no summary written |
| B | RAM_GDNF only, 184 positives | 0.012 | 0.127 |

### The contamination claim is NOT supported on this metric

Arm A (0.512) and arm A3 (0.485) are **essentially tied, marginally favouring the
contaminated arm**. Two earlier versions of this document claimed that removing Mir's
`False` rows improved performance; both compared A2's *default-threshold* f1 (0.532) against
arm A's *default-threshold* f1 (0.329). That comparison is internally consistent but is on a
metric neither model was selected for, and A2's value on the actual criterion was never
written because the job timed out. **Claim withdrawn pending a comparable number for A2.**

What remains true independently of this: Mir's `False` rows **do** contain real
non-convulsive activity (visually confirmed, including a 180 s block of clear bursting), so
they are mislabelled as seizure-detection negatives whatever the training effect turns out
to be. The data defect is established; the performance consequence is not.

### Getting a comparable number for A2
Its checkpoint loads and works, but `best_event_f1` is only computed during training. Either
rerun A2 (~17 h at `--neg-source random` speeds, or faster with the symlink-tree recipe), or
accept that A2 can be compared on held-out Batch-3 detection performance but not on the
training criterion.

## Comparisons the paper needs

Three rows, same held-out Batch 3, each model at **its own** validated operating point:

1. **frozen cascade** — `UNetv2_20260615` @ 0.5/0.1 + `Convulsive_v4LUNARC` @ 0.45
   -> convulsive recall **5.2%** (8/153)
2. **Stage 2 retrained only** — arm A U-Net @ 0.9/0.5 + `conv_armA_holdB3` @ 0.55 (job 3820427)
3. **both stages retrained** — arm A3 U-Net + `conv_armA_holdB3` (after job 3820377)
