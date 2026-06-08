# BENDR pre-training — collapse bug + fix plan (handoff 2026-06-08)

**Status: BENDR self-supervised pre-training does not learn.** Fix before any
more cluster time. This note is the handoff for resuming on another machine
(e.g. Mac/arm64). Code under discussion: `bendr_model.py` (`BENDRPretrainModel`).

## Evidence (Arrhenius, 2026-06-08)

The contrastive objective sits at **chance in every run**:
`val_loss ≈ ln(num_negatives+1) = ln(101) = 4.615`, `acc ≈ 0` every epoch.

| run | train_acc | val_acc | notes |
|-----|-----------|---------|-------|
| test_gpu epoch 1 (job 12102) | 0.0002 | 0.0 | 3491 batches, LR 5e-4 |
| test_gpu epoch 2 | 0.0 | 0.0 | |
| pretrain_short epoch 1 (job 12159) | 0.000 | 0.000 | full data, ~2430 s/epoch |
| pretrain_short epoch 2 | 0.000 | 0.000 | |

~7000 optimizer steps at a healthy LR never moved off chance → **not
under-training; it never learned the task.** The earlier smoke tests only ever
validated plumbing (GPU/container/data path/checkpointing) — their `acc=0` was
the missed tell. Job 12159 was cancelled to stop GPU billing.

## Diagnosis: representation collapse

Loss parks a hair *above* `ln(101)` with `acc` pinned at exactly 0 — the
positive is consistently just below the (collapsed, all-equal) negatives. The
encoder minimizes loss by making all `z` vectors equidistant instead of
learning to predict.

## Fix — 3 surgical changes in `bendr_model.py`

These restore the standard wav2vec 2.0 / BENDR objective this code deviates from:

1. **Stop-gradient the targets + negatives.** `unmasked_z` is used as the
   positive target in `_compute_logits` and for negatives in
   `_generate_negatives` **without `.detach()`**, so gradients flow into the
   targets and the model collapses `z`. `enc_feat_l2` (penalizes `z²`) helps it
   into that basin. → detach `z`/negatives where they serve as targets.
2. **Restrict the contrastive loss to MASKED positions only.** `compute_loss`
   uses `labels = zeros(logits.shape[0])` over *all* timesteps; the returned
   `mask` is ignored. wav2vec2/BENDR compute the loss only at masked positions.
3. **Fix the `neg_in_target` dedup.** It compares negatives against `c`
   (context) instead of the positive target `z`, so it never fires.

`temp=0.5` is fine (logits = cos_sim×2, not over-squashed) — not the cause.
Also reconsider `mask_rate=0.1` (low; BENDR uses heavier effective masking)
once the objective learns at all.

## Validation BEFORE re-running on the cluster

Fast local sanity run (CPU or Mac MPS): shrink the model (fewer context layers,
smaller `encoder_h`) + a few EDFs + a few hundred steps. A **correct** objective
should drop `val_loss` below 4.6 and lift `acc` off zero within minutes; the
current code provably won't. Only after that curve looks right do we re-run
`scripts/arrhenius/pretrain_short.sh`. **Do NOT launch `resume.sh`** until validated.

## Everything else is working (don't re-litigate)

Two-cohort staging (`SV2A_2024` + `RAM_GDNF_2025`, 2976 EDFs on Arrhenius),
cohort-aware `scripts/make_bad_channels.py` (2635 files / 6897 exclusions),
740 GB transfer intact, ~40 min/epoch throughput, math-SDPA backend (deliberate
anti-NaN choice, fine at ~150 tokens). The full 30-epoch run would fit one
`resume.sh` job (~17 h) once the objective is fixed.
