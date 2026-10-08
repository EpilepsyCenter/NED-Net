# NED-Net methods paper — working folder

> ## Channel numbering — read this first
> **The reviewer always refers to UI channels. UI channel N = code/DB channel N-1.**
> The UI, the EDF labels (`Ch1 Biopot`...`Ch8 Biopot`) and Mir's annotation `channel`
> column are all 1-based. The database `events.channel`, the sidecar `channel` field,
> `animal_ch0..animal_ch7` in the metadata CSV and `_ned_channels.json` keys are all
> 0-based. Mixing them has already produced one wrong conclusion (2026-10-07), so state
> which convention is in use every time.

This is where the validation-and-retraining project for the **NED-Net methods paper** is
tracked. Separate from the SV2A biology manuscript, whose record lives in
`scripts/paper_stats/ANALYSIS_LOG.md`.

| File | What it is |
|---|---|
| `LUNARC_CHEATSHEET.md` | **Commands: conda env, srun vs sbatch, paths, queue, the training loop.** |
| `PAPER_OUTLINE.md` | **The paper: framing, results, discussion, and what is still missing.** |
| `NEXT_STEPS.md` | **Current state and what to do next. Start here.** |
| `PHASE1_RESULTS.md` | **Phase-1 results.** Supersedes the 2026-10-02 version, now in `archive/`. |
| `MODELS.md` | Every model trained, what it saw, what it scored. |
| `NEDNet_validation_HANDOFF.md` | The full plan (v2) — background, data, architecture facts. |
| `PHASE1_RESULTS_20261002.md` | Frozen-model validation: it fails, 9.1% recall, heterogeneous by batch. |
| `PHASE2_STRATEGY.md` | Retraining plan — LOCO by `--exclude-animals`, two arms, learning curve. |
| `VERIFICATION_LOG_20261001.md` | Evidence for every correction made to v1, with the commands that produced it. Referenced from the handoff as V1–V8. |
| `archive/HANDOFF_v1_20261001_as_received.md` | The original v1 as received, kept for diffing. Superseded — do not work from it. |

Related version-controlled assets (moved in 2026-10-01 so figure provenance is traceable to
a commit, since the project DBs record no model name):

- `scripts/lunarc/detect_ramgdnf_unet.sbatch` — Phase-1 frozen-model detection job
- `scripts/lunarc/RAM_GDNF_2025_batch_metadata.csv` — filename → cohort/group/animal map
- `scripts/lunarc/RAM_GDNF_2025_cohort_key.csv` — per-channel channel→animal→treatment key

Data and backups referenced by the plan:

- Hand-annotation backup: `sv2a_annotations_20261001.tar.gz` in `~/Dropbox/NED-Net_backups/`
  and the matching iCloud Drive folder (21 MB, SHA256-verified, 222 sidecars)
- Production models: `~/.eeg_seizure_analyzer/models/` — `UNetv2_20260615`,
  `Convulsive_v4LUNARC_20260616`, plus `Re-rankerv2_20260625` kept as ablation evidence

## State

See `NEXT_STEPS.md` — it is the single source of truth and is updated as steps complete.

As of 2026-10-02: Phase 1 detection and scoring are done (the frozen model fails);
annotation is blocked on lab-PC access; round-0 retraining can start without it.
