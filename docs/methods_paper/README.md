# NED-Net methods paper — working folder

This is where the validation-and-retraining project for the **NED-Net methods paper** is
tracked. Separate from the SV2A biology manuscript, whose record lives in
`scripts/paper_stats/ANALYSIS_LOG.md`.

| File | What it is |
|---|---|
| `NEDNet_validation_HANDOFF.md` | **The live plan (v2).** Read this one. |
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

## State as of 2026-10-01

- Phase 0 complete.
- Phase 1 U-Net detection **running** on LUNARC (job 3770118, 1377/1377 EDFs matched).
- Two blocking questions open before further compute — see handoff §10 Q1 and Q2.
