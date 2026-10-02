# NED-Net validation & retraining — full handoff

**Purpose of this document.** A complete, self-contained rundown of the NED-Net
seizure-detector validation-and-retraining project for the methods paper: the
plan, every decision and its rationale, what has been done, every script that
exists (with how to run it), and the exact next steps. Written to be executed by
someone (or Claude Code) with access to the NED-Net repo and the LUNARC cluster
**without** needing the prior conversation.

Repo: `~/Software/NED-Net` (package `eeg_seizure_analyzer`).
Staging folder with all built scripts: `~/.eeg_seizure_analyzer/phase0_newcohort/`.
Cluster: LUNARC COSMOS (`cosmos.lunarc.lu.se`).

---

## 1. What we are doing and why

NED-Net (Neural Event Detection Network) is the lab's EEG seizure/interictal-spike
detector (U-Net seizure detector + a Stage-2 convulsive classifier cascade). The
**methods paper** validates the *frozen* NED-Net seizure detector on data it has
never seen — **4 new animal cohorts (RAM_GDNF_2025)** recorded on the same rig with
different viral vectors — and then **retrains it from scratch** on old + new data and
revalidates. The central claim is **out-of-the-box generalization of the frozen model
to unseen cohorts**, followed by a quantified improvement from incremental data.

This is distinct from the SV2A biology manuscript (a separate thread).

### Paper framing / claims
- **Convulsive seizures:** rigorous quantitative validation against an independent,
  near-complete ground-truth set (Mir's reviewed annotations).
- **Non-convulsive seizures:** absolute sensitivity from a **2-source capture–recapture**
  (U-Net × the model-independent autocorrelation detector) — novel, honest, bounded.
- **Improvement:** from-scratch retraining with leave-one-cohort-out (LOCO) evaluation.
- **Baseline comparison:** NED-Net (U-Net) vs the classical autocorrelation detector.
- The dropped event re-ranker is reported as an ablation (it dropped real seizures
  out-of-sample).

### Target figures
Precision–confidence + FROC per cohort; NED-Net-vs-classical head-to-head;
capture–recapture sensitivity (convulsive & non-convulsive); Stage-2 convulsive
confusion matrix; retraining learning curve; catastrophic-forgetting check on the
original SV2A test data; calibration; example detections.

---

## 2. Data

### New cohorts — `RAM_GDNF_2025` (the validation set)
- 4 batches × ~8 animals × 21 days continuous recording. ~16,000 animal-hours.
- **3 vectors:** `AAV9_CAG_eGFP` (control), `AAV9_RAM_GDNF`, `AAV9_CAG_GDNF`.
- **~1,385 EDFs total; 1,377 are cohort files.** The 8 excluded = a baseline animal
  (`483580_baseline data/`) + 2023 "Hamza" test files. They are excluded by the
  path filter `Batch_[1-4]_Recordings` (cohort EDFs live under `Batch_N_Recordings/`).
- Per day-folder: the EDF + 8 per-channel `.dat` (Biopot, 1000 Hz) + a human
  `annotations.csv` + `*_ned_annotations.json` sidecar + `*_v2_predictions.xlsx`
  (model output). **We use only the human annotations; Mir's own NED-Net outputs on
  the share are ignored.**
- Metadata: `RAM_GDNF_2025_batch_metadata.csv` (filename → cohort / group / per-channel
  animal IDs) and `RAM_GDNF_2025_cohort_key.csv` (full per-channel channel→animal→
  treatment map incl. Batch-4 excluded animals).

### Old training set — `SV2A_2024` (needed for from-scratch retraining)
- Local: `/Users/marcoledri/Software/edf/SV2A_2024` — **192 EDFs + 192 sidecars**
  (the training manifest `SV2A_20260615` used a 165-file subset).
- **Already on LUNARC** at `lu2026-2-60/edf_data` (this is the SV2A training set — the same
  set used for both the U-Net and the convulsive model). For retraining it must sit in the
  same scanned tree as the new cohorts — a **within-cluster** symlink/copy, no Mac→LUNARC
  transfer. **TO VERIFY on-cluster:** that the `*_ned_annotations.json` sidecars sit beside
  those SV2A EDFs.

### Frozen production models / operating point
- Seizure U-Net: **`UNetv2_20260615`**; convulsive cascade: **`Convulsive_v4LUNARC_20260616`**.
- Frozen operating point (decided 2026-06-26 after human spot-check):
  **U-Net @ 0.5 + hysteresis boundary 0.1 → convulsive @ 0.45, NO re-ranker.**
  (boundary 0.3 clipped events ~3 s; the re-ranker dropped real seizures out-of-sample.)

---

## 3. NED-Net architecture — facts you must know (code-verified)

1. **Production detection is CONTINUOUS, not trigger-gated.** `scripts/lunarc/detect_batch.py`
   runs the Analysis-tab pipeline `analysis.process_chunk` → `ml/predict.py:predict_seizures`
   (U-Net slides over the whole recording) + the convulsive cascade. Writes `source="seizure_cnn"`.
2. **A separate, model-independent classical detector family** lives in
   `eeg_seizure_analyzer/detection/`: `AutocorrelationDetector` (range-autocorrelation +
   spike frequency, White et al. 2006), plus spectral_band / spike_train / ensemble.
   Continuous, independent of the U-Net. Previously only invoked from the Dash UI.
3. **Project SQLite DB (`db.py`) = ANALYSIS RESULTS ONLY** (chunks/events/chunk_summary/
   file_animals/animal_status). It is NOT the confirm/reject substrate.
4. **Confirm/reject + the TRAINING dataset = per-EDF `*_ned_annotations.json` SIDECARS**
   (`io/annotation_store.py`: `save_annotations` / `load_annotations` /
   `detections_to_annotations` / `merge_annotations`). The UI converts detections →
   annotations, the human edits, it saves the sidecar.
5. **Retraining is FROM SCRATCH** (`ml/train_unet.py` docstring: "from-scratch U-Net";
   also `train_convulsive.py`). Each refinement builds a NEW model by scanning a
   data-dir recursively for **EDFs + sidecars**. So every LOCO fold needs ALL relevant
   EDFs + sidecars (old SV2A + the fold's RAM_GDNF cohorts) in the scanned tree.
   (`os.walk` does not follow symlinked *directories* by default — stage per-fold trees
   as symlinked **files** or copies.)
6. **Detector origin is tracked** two ways: the DB `events.source` column (`seizure_cnn`,
   `autocorrelation`, `spectral_band`, `spike_train`, `ensemble`, …) and, in the sidecar,
   `features["detection_method"]` (e.g. `"ml_unet"`).
7. **Sidecar annotation field contract** (from `ml/dataset.py` + `import_mir_annotations.py`):
   `{event_type:"seizure", label:"confirmed"|"rejected"|"pending", onset_sec, offset_sec,
   channel (0-based), features:{convulsive:bool, detection_method:str, …}, source,
   detector_confidence, event_id}`. **Trainers use ONLY `confirmed` (positive seizure
   intervals; convulsive subset = `features.convulsive==True`) and `rejected` (negatives);
   `pending` is ignored until reviewed.**
8. **`merge_annotations` is for RE-DETECTION, not for combining detectors.** On overlap
   (same channel, onset within tolerance) it KEEPS one annotation and DROPS the other.
   Using it to merge U-Net + autocorrelation would destroy the both-caught membership
   capture–recapture needs — so the combined converter uses a **custom overlap-merge**
   that records which detector(s) proposed each event.

---

## 4. The plan (phases)

**Phase 0 — ground-truth readiness. [DONE — see §5]**

**Phase 1 — frozen-model validation (centerpiece).** Freeze both stages + thresholds.
Run the U-Net cascade and the autocorrelation detector over the cohort. Then, from the
adjudicated (human confirm/reject) proposals:
- **Precision** (both seizure types) vs confidence; stratified by cohort/vector,
  duration, animal; bootstrap CIs clustered by animal.
- **Convulsive recall** vs Mir's independent confirmed set; NED-Net-vs-classical head-to-head.
- **Absolute sensitivity (both types)** via **U-Net × autocorrelation 2-source
  capture–recapture** on the adjudicated union; FROC, FP/h, latency.
- Stage-2 convulsive-vs-non-convulsive confusion on confirmed events; calibration.
- Report at **both operating points** (FROC/PR curves + the fixed max-sens & min-FP/h point).

**Phase 2 — from-scratch incremental retraining, LOCO by cohort (4 folds).** Train on
{SV2A + 3 RAM_GDNF cohorts}, test the held-out cohort. Retrain **both** stages from
scratch; fixed hyperparameters; multiple seeds. Learning curve (performance vs amount of
added data); catastrophic-forgetting check on the original SV2A test data; versioned
datasets/models. Re-measure convulsive recall (Mir) and capture–recapture sensitivity
after each step. Never evaluate on events used in training (LOCO enforces this).

**Phase 3 — final revalidation & lock.** Best model evaluated once per fold, identical
metrics → out-of-the-box vs improved; cross-vector generalization made explicit.

**Phase 4 — reproducibility / assets.** Preprocessing, thresholds, event-matching rule,
training configs; release the conversion + matching + capture–recapture code, a benchmark
subset, versioned detector defaults.

### Capture–recapture (the non-convulsive sensitivity method)
- **Simple 2-source Lincoln–Petersen only:** U-Net (under test) × NED-Net's
  **autocorrelation** detector. Both scan continuously and fail in different ways.
- It is **downstream arithmetic** on the two confirmed-event lists (how many confirmed
  seizures each detector caught, and how many both) — it asks nothing special of NED-Net.
- **Reported as an upper bound on sensitivity / lower bound on true seizure count**
  (the two detectors share some failure modes → positive dependence → undercounts misses).
- **Log-linear / ≥3-source multiple-systems was explicitly rejected as over-engineered.**
- It is **optional and droppable** — if dropped, the paper still has confirm/reject
  precision + Mir-anchored convulsive recall.

### Methodological guardrails (do not violate)
- **Reviewing a model's own detections yields PRECISION ONLY, never recall.** Missed
  events are structurally absent from the detection list. Recall always needs positives
  from a source independent of the model under evaluation.
- The within-NED-Net detect→review→retrain loop is the **engine** for precision + training
  labels + relative change across versions — NOT a source of absolute recall.
- **Convulsive recall** comes from Mir's independent set (held OUT of training from the
  start, scored at every iteration).
- **Never evaluate on events used in training.**
- Mir's RAM_GDNF candidates were **NOT generated with NED-Net** (user-stated), which is
  exactly why NED-Net's autocorrelation detector is a valid *independent* second source.

---

## 5. What is done — Phase 0 results

Run on the copied human annotations (no raw EEG needed):
- Human labels = **69 per-day `Day_N.xlsx`** files (candidate-review schema).
  **14,740 reviewed rows across 1,074 recordings.**
- **430 confirmed seizures = 368 convulsive + 51 non-convulsive (behavior) + 11 hand-added.**
  → NOT convulsive-only: an **independent non-convulsive positive set exists** (small but
  real; 2,012 behavior candidates were reviewed).
- Per-batch confirmed seizures: **B1=60, B2=24, B3=154, B4=192** — large spread confirms
  the need for LOCO-by-cohort. **Batch 4 is only partially annotated** (6 of ~21
  day-folders, 97 recordings) — complete it or treat it as supplementary before it is a
  LOCO test fold.
- **Leakage audit CLEAN:** 0 overlap between the 1,385 new-cohort EDF names and the SV2A
  training EDF names (`SV2A_20260615`, 165 EDFs). The U-Net and the convulsive model share
  the same SV2A training set, so this covers both frozen models → all 4 batches genuinely
  unseen, consistent with the user's confirmation that no model was trained on these cohorts.
- Remaining Phase-0 item: units/onset QC (mV vs µV, channel map) — happens naturally at
  conversion/detection time on LUNARC.

---

## 6. Scripts built (ready to run)

All are saved as artifacts and copied to `~/.eeg_seizure_analyzer/phase0_newcohort/`.
Compile-checked; **the two net-new Python scripts were NOT runtime-tested in the agent
sandbox** (they need the package + EDFs) — do a small test run first.

### 6a. `detect_ramgdnf_unet.sbatch` — Phase-1 U-Net validation detection
Runs the frozen cascade over the 1,377 cohort EDFs → project DB `ram_gdnf_unet_v0.db`.
- Operating point: U-Net `UNetv2_20260615` @ 0.5 + boundary 0.1 → convulsive
  `Convulsive_v4LUNARC_20260616` @ 0.45, no re-ranker. **Do not change for the validation.**
- `--path-include 'Batch_[1-4]_Recordings'` → cohort files only (excludes the 8).
- Needs `RAM_GDNF_2025_batch_metadata.csv` at `~/NED-Net/scripts/lunarc/` (or edit the path).
- Partition `lu48`, whole exclusive node, account `lu2026-2-60`.
- **Run:** `sbatch detect_ramgdnf_unet.sbatch` (prints the matched EDF count first; expect 1377).

### 6b. `detect_autocorr_batch.py` + `detect_ramgdnf_autocorr.sbatch` — autocorrelation sweep
The model-independent 2nd source for capture–recapture. Runs `AutocorrelationDetector`
over the same 1,377 EDFs → a **SEPARATE** DB `ram_gdnf_autocorr_v0.db`
(`source="autocorrelation"`, `category="seizure"`).
- Self-contained; **does not modify the package** (per-file function mirrors
  `analysis.process_spike_chunk_classical`; same parallel part-DB/merge harness as
  `detect_spikes_batch.py`). Skips `write_summary` (this DB is an intermediate for the
  converter, not a Results burden report).
- Operating point = SV2A-training defaults = `config.AutocorrelationParams`:
  `spike_amplitude_x_baseline=3.0` (**"local BL over 3"**), `acorr_threshold_z=3.0`,
  `min_spike_freq_hz=2.0`, `baseline_method="percentile"`. **If "local BL" meant a
  time-local baseline, set `BASELINE_METHOD=rolling` in the `.sbatch`.** (The exact SV2A
  settings could not be recovered — those sidecars were overwritten by a later U-Net run.)
- **Run:** copy `detect_autocorr_batch.py` → `~/NED-Net/scripts/lunarc/`, **test small first**:
  ```
  conda activate bendr && cd ~/NED-Net
  python scripts/lunarc/detect_autocorr_batch.py \
    --edf-dir /lunarc/nobackup/projects/lu2026-12-29/RAM_GDNF_2025 \
    --db-path /tmp/ac_test.db \
    --metadata-csv scripts/lunarc/RAM_GDNF_2025_batch_metadata.csv \
    --path-include 'Batch_1_Recordings/Week_1/' --workers 4
  ```
  then `sbatch detect_ramgdnf_autocorr.sbatch` for the full run.
- **Never** run it onto the U-Net DB — the merge is destructive per EDF path.

### 6c. `db_to_sidecars.py` — DB → pending sidecars (U-Net only)
Reads a project DB, writes per-EDF `*_ned_annotations.json` with `label="pending"` via the
library's `save_annotations`, so batch detections become a review queue.
**SUPERSEDED for the final workflow by the combined converter (§7, not yet built)** — keep
it only if you want a U-Net-only review. Must run ON LUNARC in the `bendr` env (sidecars
are written beside each EDF). Has `--dry-run` and `--overwrite` (skips existing by default).

### 6d. Metadata / planning artifacts
- `RAM_GDNF_2025_batch_metadata.csv`, `RAM_GDNF_2025_cohort_key.csv`
- `NEDNet_validation_retraining_plan.md` (the plan), `NEDNet_runbook.md` (stage-by-stage runbook)

---

## 7. What is NOT yet built — the combined converter (next agent task)

After **both** detection DBs exist (`ram_gdnf_unet_v0.db` + `ram_gdnf_autocorr_v0.db`):
a converter that reads **both** DBs, and **per EDF**:
- overlap-merges the two detectors' events per channel (custom merge — **not**
  `merge_annotations`), emitting **one pending annotation per seizure** (review once),
- records which detector(s) proposed it in `features["detectors"]` (e.g.
  `["seizure_cnn","autocorrelation"]`) so capture–recapture membership is readable,
- carries `features["convulsive"]` from the U-Net/Stage-2 where available (a pre-fill the
  reviewer corrects),
- writes the merged pending sidecar beside each EDF via `save_annotations`.

It will be shaped to the actual event tables once the DBs exist. Until then,
`db_to_sidecars.py` covers the U-Net-only case.

---

## 8. Exact next steps (in order)

1. **Place** `RAM_GDNF_2025_batch_metadata.csv` on LUNARC at `~/NED-Net/scripts/lunarc/`.
2. **U-Net detection:** `sbatch detect_ramgdnf_unet.sbatch` → `ram_gdnf_unet_v0.db`.
   Confirm the "cohort EDFs matched" line says **1377**.
3. **Autocorrelation sweep:** copy `detect_autocorr_batch.py` → `~/NED-Net/scripts/lunarc/`,
   run the small test (§6b), then `sbatch detect_ramgdnf_autocorr.sbatch` →
   `ram_gdnf_autocorr_v0.db`.
4. **Combined converter** (agent builds against the real DBs) → merged pending sidecars.
5. **Review in the NED-Net UI (ThinLinc):** confirm/reject each seizure once → the sidecars
   become the confirmed/rejected ground truth + training labels.
6. **Phase-2 staging:** within-cluster symlink/copy the SV2A EDFs + sidecars
   (`lu2026-2-60/edf_data`) into the per-fold training trees alongside the RAM_GDNF cohorts.
   **Verify the SV2A sidecars sit beside the EDFs on LUNARC first.**
7. **From-scratch retraining, LOCO:** `train_unet.sh` + `train_convulsive.sh` (GPU
   `gpua100`), one model per fold, trained on {SV2A + 3 cohorts}, tested on the held-out one.
8. **Hand the result DBs/CSVs back** for analysis (next section).

---

## 9. Handoff boundary — what comes back for analysis

Compute model is **prepare-and-hand-off**: LUNARC requires password + 2FA on every login,
so it cannot be driven as an automated SSH target. The user runs detection/retraining on
the cluster; **only small result artifacts come back** for local analysis in Claude Science:
- the project DBs (`ram_gdnf_unet_v0.db`, `ram_gdnf_autocorr_v0.db`, per-fold retrained-model
  detection DBs) — ~MB each,
- event CSVs / metrics tables,
- the confirmed sidecars (or a dump of them).

Raw EEG (~16,000 animal-hours) stays on the cluster. The data-analysis work Claude Science
will then do: event-matching U-Net-vs-ground-truth, precision/recall, the 2-source
capture–recapture arithmetic, FROC/FP-h/latency, Stage-2 confusion, calibration, learning
curves, forgetting check, and all paper figures.

---

## 10. Open items / decisions to confirm
- **Autocorrelation baseline:** defaults use `percentile`; if "local BL over 3" meant a
  time-local baseline, switch to `rolling` (one line in the `.sbatch`).
- **Batch 4 annotation coverage** (6 of ~21 day-folders) — complete it or treat it as a
  supplementary / non-LOCO fold.
- **Compute account** `lu2026-2-60` — confirm it is still an active allocation for jobs
  (data storage is under `lu2026-12-29`; these can legitimately differ).
- **Capture–recapture** — keep it (2-source Lincoln–Petersen, reported as a bound) or drop
  it and lean on precision + convulsive recall.

---

## 11. Cluster / environment reference
- Host `cosmos.lunarc.lu.se`; `module load Anaconda3/2024.06-1`; `conda activate bendr`.
- `source config_conda.sh` from `~/NED-Net` before activating (the existing scripts do this).
- CPU detection: partition `lu48` (whole exclusive node). GPU retraining: `gpua100`.
- Account `-A lu2026-2-60`. Data storage path `lu2026-12-29/RAM_GDNF_2025`.
- Per-worker part-DBs merged at the end (db.py sets no busy_timeout, so one shared DB would
  hit "database is locked").
