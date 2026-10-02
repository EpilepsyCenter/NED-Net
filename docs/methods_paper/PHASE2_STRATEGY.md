# Phase 2 — retraining strategy

**Written 2026-10-02**, after the Phase-1 failure (`PHASE1_RESULTS_20261002.md`).
Phase 2 is now the load-bearing half of the paper: Phase 1 shows the frozen model fails,
Phase 2 must show the adaptation pipeline fixes it. If it does not, the paper has no
positive claim.

## The asset nobody has used yet

Mir's adjudicated rows are a ready-made in-domain training set, already partitioned by
batch for LOCO:

| batch | positives | hard negatives (`False`) |
|---|---|---|
| Batch 1 | 60 | 3,676 |
| Batch 2 | 24 | 1,803 |
| Batch 3 | 154 | 5,422 |
| Batch 4 | 192 | 752 |
| **total** | **430** | **11,653** |

For comparison the SV2A set that trained `UNetv2_20260615` had 867 positives and 1,325
hard negatives — so this is **8.8× more hard negatives**, in-domain, at zero annotation
cost. Every LOCO fold retains 238–406 positives for training.

`Normal` rows (2,657) are dropped, per `scripts/import_mir_annotations.py` — background
negatives are auto-sampled by the trainer.

## Split design — an adaptation curve, not a single holdout

Pure batch holdout answers only the zero-shot question and yields one point. The paper's
claim is about adaptation, so the design is nested:

> Pick a **target batch**. Reserve a fixed **test set** from it, never trained on at any k.
> Train on the other batches **plus k annotated day-folders from the target batch**, for
> k = 0, 1, 2, ... Plot recall/precision on the test set against k.

- **k = 0** is pure cross-batch zero-shot — the frozen-model scenario.
- **k > 0** is the adaptation curve, and its x-axis is the number an adopting lab cares
  about: how much of their own data they must annotate.

**Increment by day-folders, not animals.** A lab annotates recording sessions covering all
its animals, not individual animals; days also give ~21 points per batch instead of 6, and
avoid the treatment-group confound (B4's two highest-seizure animals are both CAG_GDNF at
81 and 78, while the eGFP controls have 28 and 3 — any animal split also splits groups
unevenly).

**Do not split animals randomly across all batches.** The Phase-1 failure is
batch-structured (2.6 vs 69.3 detections/file). Training on animals from every batch lets
the model see each batch's acquisition characteristics, so it would measure generalization
to a new *animal* under known conditions — the easy question — not to a new *batch*, which
is the one that failed.

**Target batch: B3.** Confirmed seizures per animal:

| batch | seizures per animal | suitability |
|---|---|---|
| B1 | 26, 15, 12, 5, 1, 1 | training only (60 total) |
| B2 | 11, 8, 3, 1, 1 | training only (24 total — underpowered) |
| **B3** | **51, 50, 25, 12, 9, 7** | **primary target** — spread, top animal only 33% |
| B4 | 81, 78, 28, 3, 1, 1 | secondary target — top two animals hold 83%, noisy curve |

Report both a **day-split curve** (train and test share animals — realistic for a
longitudinal study, state it plainly) and a stricter **animal-disjoint** variant as a
secondary.

## Round 0 — free. Two arms, and the contrast is the result.

Train from scratch, LOCO by batch (4 folds), no new annotation:

- **Arm A — SV2A + Mir's three training batches.** Tests whether more data helps.
- **Arm B — Mir's three training batches only.** Tests whether *in-domain* data is what
  matters.

The contrast answers the question another lab actually has: *do we need your dataset, or
just our own annotations?* If B ≈ A, the answer is "just your own", which is a far more
useful and more distributable conclusion than "download our weights". If B ≪ A, the
pretrained base has value and that is worth stating too.

Caveat to watch: 430 positives total is half what SV2A had, and per fold 238–406. Arm B may
underperform for lack of positives rather than for lack of relevance — Arm A hedges that,
and the gap between them is interpretable either way.

## Round 1 — targeted annotation: the model's own false positives

Mir's hard negatives are *his detector's* false positives. They will not fix the Batch-4
over-detection (69 detections/file), because the U-Net is firing on something his detector
never proposed. To fix that, the model needs **its own** false positives as hard negatives.

Review a sample of Round-0 detections, concentrated where the model floods, confirm/reject,
add as hard negatives, retrain. This is exactly the two-round active-learning loop that
built `UNetv2_20260615` in the first place — round 1 autocorrelation-proposed (640 pos /
310 neg), round 2 reviewing the predecessor U-Net's proposals (227 pos / 1,015 neg). It is
a method already validated internally, which is worth saying in the paper.

Efficiency note: the flooding detections are likely highly redundant (one dominant artefact
type), so a few hundred reviewed events should move the needle far more than their count
suggests. That redundancy is itself measurable and worth reporting.

## Round 2+ — iterate until the curve flattens

Report performance against **cumulative human-reviewed events**. That curve is the paper's
headline figure and the practical currency for adoption: a lab reading it can estimate
their own annotation cost before committing.

## Controls that must run alongside

- **Retrain both stages.** Stage-2 also failed — precision ceilings near 30% even when
  restricted to convulsive predictions.
- **Catastrophic-forgetting check on held-out SV2A test data at every round.** Critical for
  the "adapt per lab" claim: if adapting to RAM_GDNF destroys SV2A performance, the
  pipeline produces disposable per-lab models rather than an improving one. Either result
  is publishable; not checking is not an option.
- **Report per batch, never pooled only.** Batch heterogeneity is the central Phase-1
  finding; a pooled number would hide exactly what the paper is about.
- **Fixed hyperparameters across folds and rounds**, multiple seeds, versioned
  datasets/models. Any tuning per fold invalidates the comparison.
- **Never train on the held-out batch's labels** — LOCO enforces this, and the staging step
  should assert it rather than relying on care.

## Power, per `NEDNet_validation_HANDOFF.md` §13

Batch 2 has only 24 test seizures — underpowered as a LOCO test fold; report it with its CI
and pool B1+B2 as a secondary estimate. Batch 4 has 192 seizures but only 97 annotated
files; metrics stay restricted to the annotated region, which is valid for both precision
and recall.

## How LOCO is enforced — `--exclude-animals`, not staged trees

Animals partition **cleanly by batch** (verified 2026-10-02, zero pairwise overlap):

| batch | animals |
|---|---|
| Batch 1 | 449381-449388 (8) |
| Batch 2 | 450093-450098, 450916, 450917 (8) |
| Batch 3 | 459657-459664 (8) |
| Batch 4 | 483550, 483551, 483552, 483553, 483554, 483555, 483557, 483559 (8) |

**32 animals, 8 per batch.** `RAM_GDNF_2025_cohort_key.csv` lists 12 rows for Batch 4, but
four are flagged `EXCLUDED` with `channel = '(none)'` (died after kainate, died after virus
injection, negative control, not implanted). **Use `RAM_GDNF_2025_batch_metadata.csv` as
the authoritative animal list** — it is derived from the files themselves.

So a LOCO fold is just `--exclude-animals <held-out batch's IDs>`. Both
`train_unet.py` and `train_convulsive.py` accept it and `ml/dataset.py:259` drops those
animals from the dataset entirely. **No symlinked fold trees are needed** -- an earlier
draft of this plan proposed them and they are unnecessary complexity.

This means the sidecars live where they naturally belong: **beside the EDFs in the real
LUNARC folders**, which is also the only place the trainer looks (`scan_annotation_files`
walks `--data-dir` for `*_ned_annotations.json` and derives each EDF path from the sidecar
name). SV2A's sidecars are already there; Mir's need to be written there too.

Leakage safety still holds: the held-out batch's sidecars may sit in the tree, but its
animals are excluded from training, and **evaluation always reads Mir's CSV**
(`~/ground_truth/mir_ramgdnf_annotations.csv`), never the sidecars. So the metric cannot be
contaminated by what is on disk.

## Sidecar merge — one file, two sources

On 578 recordings both Mir's labels and U-Net detections apply, and they share one
filename. The converter must **merge, not overwrite**:

- Mir's adjudicated events -> `confirmed` / `rejected` (these are training labels)
- U-Net detections -> `pending` (these are the review queue)
- a U-Net detection overlapping a Mir event is not duplicated; keep Mir's label and record
  in `features["detectors"]` that the U-Net also found it

This is good for review as well as for safety: opening a file shows what Mir already
decided plus only the new U-Net proposals, so Marco clicks the pending ones and nothing
else. Writing an unmerged sidecar would destroy Mir's labels on those 578 files.

## Order of work

1. EDF-header comparison across batches (cheap, no compute) — may explain the Phase-1
   heterogeneity before any retraining.
1b. Build the merged-sidecar converter (Mir CSV + U-Net DB -> one sidecar per EDF).
2. One Round-0 fold as a **feasibility probe** — hold out Batch 3 (154 test seizures, worst
   recall at 5.2%, so the most headroom). If recall there does not move substantially, stop
   and rethink before spending on the full grid.
3. Full Round-0 grid: 2 arms × 4 folds × seeds.
4. Round-1 annotation and retrain.
5. Learning curve, forgetting check, final per-batch table.
