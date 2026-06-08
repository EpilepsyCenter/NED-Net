# BENDR pre-training — collapse diagnosis + resolution (data2vec)

**Status (2026-06-08): RESOLVED locally. Pre-training switched to a data2vec
EMA-teacher objective, which learns and generalises without collapse. Validated
on held-out whole files on Mac/MPS. Ready for a cluster `pretrain_short.sh`
verification run before the full 30-epoch campaign.**

## The problem (Arrhenius runs)

The original wav2vec-2.0-style **contrastive** objective never learned —
`val_loss ≈ ln(num_negatives+1) = ln(101) = 4.615`, `acc ≈ 0` in every run.

## What we found (in order)

1. **Catastrophic collapse from three bugs** in the contrastive code, now fixed
   in `bendr_model.py` (`BENDRPretrainModel`):
   - targets/negatives weren't stop-gradient → encoder collapsed `z` to be
     equidistant. → `.detach()` the targets + negatives.
   - the loss covered *all* timesteps, not just masked ones. → `compute_loss`
     now restricts to masked positions (`select_masked_logits`).
   - the `neg_in_target` dedup compared `c` (context) instead of the target
     `z`, so it never fired. → compare against `z`.
   These took held-out accuracy from *exactly 0* to a peak of ~0.29.

2. **A second, slower collapse remained.** With the bugs fixed, held-out
   accuracy still **peaked (~step 100) then decayed back to chance**. Confirmed
   on held-out *whole files* (unseen cohorts), so it was real, not a metric
   artifact. Diagnosis: continuous-target contrastive SSL is inherently
   collapse-prone — once targets are detached, nothing keeps them diverse, and
   the context `c` drifts to a near-constant. VICReg variance/covariance reg,
   temperature 0.1, lower LR, and warmup each helped but none cured it.

## The fix: data2vec EMA-teacher objective (`--method data2vec`, default)

`BENDRData2VecPretrainModel` in `bendr_model.py`. A **student** (the trainable
encoder + contextualizer) sees the *masked* sequence; at masked positions it
regresses (smooth-L1) to targets from a **teacher** — an EMA copy of the same
networks run on the *unmasked* sequence. Targets are the average of the top-K
teacher layers, instance-normalised over time then layer-normed over features.
The EMA lag + target normalisation structurally prevent collapse: no negatives,
no quantization. Build via `build_data2vec_pretrain_model`; checkpoints are
weight-compatible with `BENDRSeizureModel.load_pretrained_combined` for
fine-tuning (only encoder+contextualizer are saved; teacher/predictor dropped).

### Local validation (held-out whole files, Mac/MPS)

`scripts/local/sanity_pretrain.py` — shrunk model, whole-file holdout, tracks
held-out prediction–target cosine. data2vec rises **monotonically and holds**
(cosine 0.00 → 0.12 over 1200 steps, loss steadily down), vs the contrastive
objective which spiked then decayed to chance. Gate PASSES. Run it before any
cluster time:

    python scripts/local/sanity_pretrain.py --data /path/to/edf_dir --steps 1200

### Training-loop changes (`bendr_pretrain.py`)

- `--method {data2vec,contrastive}` (default data2vec); EMA flags
  `--ema-decay/--ema-end-decay/--ema-anneal-steps/--top-k-layers`.
- `update_ema()` after each `optimizer.step()`; loss is smooth-L1 regression;
  the reported quality metric is held-out cosine (`val_cos`).
- **Linear LR warmup** (`--warmup-steps 500`) + tighter grad clip (3.0): a
  constant high LR with no warmup caused a late divergence in testing; warmup +
  the existing cosine schedule fixes it. Peak LR lowered to `5e-4`.
- mask_rate default 0.10 → 0.15.

Cluster scripts (`arrhenius/`, `lunarc/` pretrain_short/pretrain/resume) updated
to `--lr 5e-4 --method data2vec --warmup-steps 500`.

## Next step on the cluster

Run `pretrain_short.sh` (5 epochs) and confirm `val_loss` decreases and
`val_cos` rises across epochs. If clean, launch the full 30-epoch `pretrain.sh`
(+`resume.sh` if walltime is exceeded), then fine-tune and compare to from-scratch.

## Everything else (don't re-litigate)

Two-cohort staging, cohort-aware bad-channels, 740 GB transfer, ~40 min/epoch,
math-SDPA backend, `--target-fs 250` alignment — all fine. Channels 0–7 are the
Biopot EEG (each a separate animal); 8–15 are activity and are never trained on.
