# NED-Net validation & retraining — handoff (v2)

**Provenance.** v2, 2026-10-01. Revision of
`archive/HANDOFF_v1_20261001_as_received.md` (built in a Claude Science session).
Every correction is evidenced in `VERIFICATION_LOG_20261001.md` (referenced below as
**V1**–**V8**). Where v1 and v2 disagree, v2 is the one checked against the code and data.

Repo: `~/Software/NED-Net` (package `eeg_seizure_analyzer`). Working docs: this folder.
Staging folder for built scripts: `~/.eeg_seizure_analyzer/phase0_newcohort/`.
Version-controlled copies of the job scripts + metadata: `scripts/lunarc/`.
Cluster: LUNARC COSMOS (`cosmos.lunarc.lu.se`).

---

## 1. What we are doing and why

NED-Net is the lab's EEG seizure/interictal-spike detector (U-Net seizure detector +
Stage-2 convulsive classifier cascade). The **methods paper** validates the *frozen*
detector on data it has never seen — **4 new animal cohorts (RAM_GDNF_2025)**, same rig,
different viral vectors — then **retrains from scratch** on old + new data and revalidates.
Central claim: **out-of-the-box generalization of the frozen model to unseen cohorts**,
followed by a quantified improvement from incremental data.

Distinct from the SV2A biology manuscript (separate thread; see
`scripts/paper_stats/ANALYSIS_LOG.md`).

### Paper framing / claims
- **Convulsive seizures:** quantitative validation against Mir's independent, video-anchored
  reviewed set. *Strength of this claim depends on §10 Q1.*
- **Non-convulsive seizures:** sensitivity from a **2-source capture–recapture**
  (U-Net × autocorrelation detector), reported as a bound. *See §4 for the dependence
  caveat and the cross-check that resolves it.*
- **Improvement:** from-scratch retraining with leave-one-cohort-out (LOCO) evaluation.
- **Baseline comparison:** NED-Net (U-Net) vs the classical autocorrelation detector.
- The dropped event re-ranker is reported as an ablation (it dropped real seizures
  out-of-sample). Weights retained at `~/.eeg_seizure_analyzer/models/Re-rankerv2_20260625`.

### Target figures
Precision–confidence + FROC per cohort; NED-Net-vs-classical head-to-head;
capture–recapture sensitivity (convulsive & non-convulsive); Stage-2 convulsive confusion
matrix; retraining learning curve; catastrophic-forgetting check on the original SV2A test
data; calibration; example detections.

---

## 2. Data

### New cohorts — `RAM_GDNF_2025` (the validation set)
- 4 batches × ~8 animals × 21 days continuous recording. ~16,000 animal-hours.
- **3 vectors:** `AAV9_CAG_eGFP` (control), `AAV9_RAM_GDNF`, `AAV9_CAG_GDNF`.
- **1,385 EDFs total; 1,377 cohort files** — verified on cluster (**V1**). The 8 excluded =
  `483580_baseline data/` + 2023 "Hamza" test files, removed by the path filter
  `Batch_[1-4]_Recordings`.
- **Path on LUNARC: `/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025`.**
  ⚠️ v1 §2/§6b/§10/§11 gave this as `lu2026-12-29` — **wrong, and it already cost one
  failed job** (3769954, exited in 9 s with 0 matched EDFs). `lu2026-2-60` is the storage
  project *and* the working SLURM account; `lu2026-12-29` is the compute allocation ID whose
  symlink exists **only in the LUNARC home dir**, so the absolute
  `/lunarc/nobackup/projects/lu2026-12-29/...` does not resolve. (**V1**)
- Per day-folder on the **research share**: EDF + 8 per-channel `.dat` (Biopot, 1000 Hz) +
  human `annotations.csv` + `*_ned_annotations.json` + `*_v2_predictions.xlsx`.
  **The LUNARC copies carry no sidecars** (user-confirmed 2026-10-01), so nothing can be
  overwritten there yet — but see §12, which is now the real hazard.
- Metadata: `RAM_GDNF_2025_batch_metadata.csv` (filename → cohort/group/per-channel animal
  IDs) and `RAM_GDNF_2025_cohort_key.csv`. Both version-controlled in `scripts/lunarc/`.

### Old training set — `SV2A_2024` (needed for from-scratch retraining)
- Local: `/Users/marcoledri/Software/edf/SV2A_2024` — 192 EDFs + 192 sidecars (plus 30 in
  `SV2A_spotcheck`); the training manifest `SV2A_20260615` used a 165-file subset.
- On LUNARC at `lu2026-2-60/edf_data`, sidecars included. Retraining needs them in the same
  scanned tree as the new cohorts — **within-cluster** symlink/copy, no Mac→LUNARC transfer.
  **TO VERIFY on-cluster:** that the `*_ned_annotations.json` sidecars sit beside those EDFs.
- Backed up 2026-10-01: `sv2a_annotations_20261001.tar.gz` (21 MB, SHA256-verified) in
  `~/Dropbox/NED-Net_backups/` + iCloud Drive. **Not** LUNARC-only any more (**V4**).

### Frozen production models / operating point
- Seizure U-Net **`UNetv2_20260615`**; convulsive cascade **`Convulsive_v4LUNARC_20260616`**.
- Frozen operating point (decided 2026-06-26 after human spot-check):
  **U-Net @ 0.5 + hysteresis boundary 0.1 → convulsive @ 0.45, NO re-ranker.**
  (boundary 0.3 clipped events ~3 s; the re-ranker dropped real seizures out-of-sample.)
- This is the same operating point that produced the SV2A figures — confirmed three ways
  (DB processing date, `detect_all.sh` defaults at commit `82caed0`, `ANALYSIS_LOG.md:16`).

---

## 3. NED-Net architecture — facts you must know (code-verified)

1. **Production detection is CONTINUOUS, not trigger-gated.** `scripts/lunarc/detect_batch.py`
   runs `analysis.process_chunk` → `ml/predict.py:predict_seizures` (U-Net slides over the
   whole recording) + the convulsive cascade.
2. **`events.source` is `seizure_unet`.** ⚠️ v1 §3.1 said `seizure_cnn`. The per-event
   `source` is set by `_arch_source(features["detection_method"], "seizure")` and overrides
   the `write_events(source="seizure_cnn")` default; all 24,839 rows in the production SV2A
   DB are `seizure_unet`, none are `seizure_cnn`. `seizure_cnn` survives only as the column
   default and a legacy UI label. **The §7 converter must key on `seizure_unet`.** (**V2**)
3. **A separate, model-independent classical detector family** lives in
   `eeg_seizure_analyzer/detection/`: `AutocorrelationDetector` (range-autocorrelation +
   spike frequency, White et al. 2006), plus spectral_band / spike_train / ensemble.
   Continuous; previously only invoked from the Dash UI.
4. **Project SQLite DB (`db.py`) = ANALYSIS RESULTS ONLY** (chunks/events/chunk_summary/
   file_animals/animal_status). NOT the confirm/reject substrate. **It records no model
   name or version** — provenance for a figure DB is the git commit of the launching script
   at run time (**V8**).
5. **Confirm/reject + the TRAINING dataset = per-EDF `*_ned_annotations.json` SIDECARS**
   (`io/annotation_store.py`). The UI converts detections → annotations, the human edits, it
   saves the sidecar.
6. **Retraining is FROM SCRATCH** (`ml/train_unet.py`, `train_convulsive.py`). Each
   refinement scans a data-dir recursively for **EDFs + sidecars**, so every LOCO fold needs
   all relevant EDFs + sidecars in the scanned tree. `os.walk` does not follow symlinked
   *directories* by default — stage per-fold trees as symlinked **files** or copies.
   `train_unet.py` accepts only `--data-dir`; there is **no** flag to load a saved dataset
   definition, so a retrain uses whatever is in the folder at that moment. Diff counts
   against the frozen definition before training.
7. **Sidecar annotation field contract** (`ml/dataset.py`, `import_mir_annotations.py`):
   `{event_type, label:"confirmed"|"rejected"|"pending", onset_sec, offset_sec,
   channel (0-based), features:{convulsive, detection_method, …}, source,
   detector_confidence, event_id}`, plus `original_onset_sec`/`original_offset_sec` holding
   the raw detector boundaries. **Trainers use ONLY `confirmed` and `rejected`; `pending` is
   ignored.**
8. **`merge_annotations` is for RE-DETECTION, not for combining detectors.** On overlap it
   KEEPS one annotation and DROPS the other, which would destroy capture–recapture
   membership. §7 uses a custom overlap-merge.
9. **Arbitrary `features` keys survive the sidecar round-trip** — `annotation_store`
   serialises `dict(self.features)` with no whitelist, so `features["detectors"]` persists.
   Still unverified: that the UI's edit path doesn't rebuild `features` when a reviewer
   adjusts an event. Check before relying on it. (**V7**)
10. **The autocorrelation operating point used for SV2A has been recovered** from the
    surviving sidecars — see §6b. v1's claim that it was unrecoverable is false (**V3**).

---

## 4. The plan (phases)

**Phase 0 — ground-truth readiness. [DONE — see §5]**

**Phase 1 — frozen-model validation (centerpiece).** Freeze both stages + thresholds. Run
the U-Net cascade and the autocorrelation detector over the cohort. Then, from the
adjudicated (human confirm/reject) proposals:
- **Precision** (both seizure types) vs confidence; stratified by cohort/vector, duration,
  animal; bootstrap CIs clustered by animal.
- **Convulsive recall** vs Mir's independent set; NED-Net-vs-classical head-to-head.
- **Sensitivity (both types)** via 2-source capture–recapture on the adjudicated union;
  FROC, FP/h, latency.
- Stage-2 convulsive-vs-non-convulsive confusion on confirmed events; calibration.
- **Operating points:** the frozen point *plus* curves. ⚠️ Curves cannot be swept below the
  frozen core threshold from the frozen run alone — `cnn_confidence` is the event's **mean**
  probability while the gate is on peak probability ≥ 0.5, so events whose peak never
  reached 0.5 are absent and unrecoverable. 43% of SV2A events sit below mean-0.5, so a PR
  curve *is* drawable, but its high-sensitivity end is hard-gated. **Add a second run at a
  low core threshold (0.1) for the FROC.** Note also that boundary hysteresis grows event
  extent, so mean-probability sweeps mix confidence with extent — relevant to calibration.
  (**V5**)

**Phase 2 — from-scratch incremental retraining, LOCO by cohort.** Train on
{SV2A + 3 RAM_GDNF cohorts}, test the held-out cohort. Retrain **both** stages from scratch;
fixed hyperparameters; multiple seeds. Learning curve; catastrophic-forgetting check on the
original SV2A test data; versioned datasets/models. Re-measure convulsive recall and
capture–recapture sensitivity after each step. See §13 for the fold decision.

**Phase 3 — final revalidation & lock.** Best model evaluated once per fold, identical
metrics → out-of-the-box vs improved; cross-vector generalization made explicit.

**Phase 4 — reproducibility / assets.** Preprocessing, thresholds, event-matching rule,
training configs; release the conversion + matching + capture–recapture code, a benchmark
subset, versioned detector defaults.

### Capture–recapture (the non-convulsive sensitivity method)
- **Simple 2-source Lincoln–Petersen only:** U-Net (under test) × the **autocorrelation**
  detector. Both scan continuously and fail in different ways.
- **Downstream arithmetic** on the two confirmed-event lists — it asks nothing special of
  NED-Net.
- **Reported as an upper bound on sensitivity / lower bound on true seizure count.**
  Positive dependence between sources inflates the both-caught count *m*, shrinking
  N̂ = n₁n₂/m and inflating sensitivity = n₁/N̂.
- **Log-linear / ≥3-source multiple-systems remains rejected as over-engineered.**
- **Optional and droppable** — without it the paper still has confirm/reject precision +
  Mir-anchored convulsive recall.

#### Residual dependence: what is and isn't settled
The cohorts are **different animals**, so there is no event-level overlap between the
U-Net's SV2A training set and the GDNF validation set — the leakage audit is clean (§5), and
that removes the memorization channel. Agreed.

What animal separation does *not* remove: **640 of the U-Net's 867 training positives (74%)
were autocorrelation-proposed** (**V4**). The U-Net learned its decision function largely
from events that detector selected, and a learned selection criterion travels to new
animals. Seizure morphologies the autocorrelation detector systematically misses were
correspondingly rare in the U-Net's positive training set, so both are likely to miss the
same *kinds* of events on GDNF. That is a shared blind spot by inheritance, not by shared
subjects, and Lincoln–Petersen is sensitive to exactly this.

**This is now testable rather than arguable.** Mir's set is a third source on GDNF that is
independent of both detectors (if §10 Q1 resolves favourably). So:

> Compute U-Net recall against Mir's video-anchored confirmed set, and separately the
> capture–recapture sensitivity estimate. If the two agree, source dependence is mild and
> the capture–recapture figure stands on its own. If capture–recapture reads substantially
> higher, the gap *is* the dependence, and it should be reported as such.

This is a validation of the 2-source estimate against an independent anchor — not
≥3-source log-linear modelling, which stays rejected. Either outcome is publishable, and it
converts the objection into a measured quantity.

**Scope limit to state in the paper:** video anchors *convulsive* events only
(non-convulsive seizures have no behavioural correlate by definition). So the cross-check
validates capture–recapture on the convulsive subset, and its extension to the
non-convulsive estimate is an assumption — worth stating explicitly rather than leaving
implicit.

### Methodological guardrails (do not violate)
- **Reviewing a model's own detections yields PRECISION ONLY, never recall.** Missed events
  are structurally absent. Recall always needs positives from a source independent of the
  model under evaluation.
- The within-NED-Net detect→review→retrain loop is the **engine** for precision + training
  labels + relative change across versions — NOT a source of absolute recall.
- **Convulsive recall** comes from Mir's independent set, held OUT of training from the
  start (see §12 — this is now an enforcement problem, not just an intention), scored at
  every iteration.
- **Never evaluate on events used in training.**
- **Phase 2 inherits a shared blind spot.** New-cohort training labels come from reviewing
  U-Net + autocorrelation proposals, and the held-out fold's labels come from the same
  process. LOCO prevents memorization but not absence: seizures neither detector proposed
  are missing from training and test alike. So the Phase-2 "improvement" is improvement in
  precision and in *relative* recall. State this; do not let a reviewer find it.

---

## 5. What is done — Phase 0 results

Run on the copied human annotations (no raw EEG needed):
- Human labels = **69 per-day `Day_N.xlsx`** files (candidate-review schema).
  **14,740 reviewed rows across 1,074 recordings.**
- **430 confirmed seizures = 368 convulsive + 51 non-convulsive (behavior) + 11 hand-added.**
  → an **independent non-convulsive positive set exists** (small but real; 2,012 behavior
  candidates reviewed).
- Per-batch confirmed: **B1=60, B2=24, B3=154, B4=192.** Batch 4 is annotated for 6 of ~21
  day-folders (97 recordings). See §13 — this is usable, contrary to v1's framing.
- **Leakage audit CLEAN:** 0 overlap between the 1,385 new-cohort EDF names and the 165
  SV2A training EDF names. The U-Net and the convulsive model share the SV2A training set,
  so this covers both frozen models → all 4 batches genuinely unseen.
- Coverage note for the paper: 1,074 of 1,377 cohort recordings carry human review (78%).
  "Near-complete ground truth" needs that qualifier, or a restriction to the annotated
  region (§13).
- Remaining Phase-0 item: units/onset QC (mV vs µV, channel map) at conversion/detection
  time. Related known trap: KAHA/AD EDFs are in mV and `read_edf` returns physical units
  as-is.

---

## 6. Scripts (ready to run)

Staged in `~/.eeg_seizure_analyzer/phase0_newcohort/`; the job script and metadata CSVs are
now version-controlled in `scripts/lunarc/`. The two net-new Python scripts were **not**
runtime-tested in the agent sandbox — small test run first.

### 6a. `detect_ramgdnf_unet.sbatch` — Phase-1 U-Net validation detection  [RUNNING]
Frozen cascade over the 1,377 cohort EDFs → `ram_gdnf_unet_v0.db`.
- Operating point: `UNetv2_20260615` @ 0.5 + boundary 0.1 → `Convulsive_v4LUNARC_20260616`
  @ 0.45, no re-ranker. **Do not change for the validation.**
- `EDF_DIR=/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025` (corrected — **V1**).
- `--path-include 'Batch_[1-4]_Recordings'` → 1,377 of 1,385.
- Needs `RAM_GDNF_2025_batch_metadata.csv` at `~/NED-Net/scripts/lunarc/`.
- Sanity check no longer swallows errors: it omits `2>/dev/null` and `exit 1`s on zero
  matches, so a misconfiguration triggers `--mail-type=FAIL` instead of reporting "done".
- **Submit from `~/NED-Net`** (the `-o logs/...` path is relative to the submission dir and
  Slurm rejects the job if it doesn't exist): `mkdir -p ~/NED-Net/logs && cd ~/NED-Net &&
  sbatch scripts/lunarc/detect_ramgdnf_unet.sbatch`.
- Partition `lu48`, account `lu2026-2-60`. **`--exclusive` costs queue time** — it waits for
  a wholly empty node; the proven SV2A scripts omit it. Consider `-c 24` instead (**V8**).
- **Status:** job 3770118, node cn096, started 2026-10-01 14:27 CEST, 1377/1377 matched,
  48 workers, 0 errors at 50 files, ~0.09 files/s → ETA ~19:00 CEST. (Job 3769954 was the
  failed run from the v1 path error.)

### 6b. `detect_autocorr_batch.py` + `detect_ramgdnf_autocorr.sbatch` — autocorrelation sweep
The second source for capture–recapture. Runs `AutocorrelationDetector` over the same 1,377
EDFs → a **SEPARATE** DB `ram_gdnf_autocorr_v0.db` (`source="autocorrelation"`,
`category="seizure"`).
- Self-contained; does not modify the package. Skips `write_summary`.
- **Operating point — recovered from the surviving SV2A sidecars, not guessed (V3):**

  | Parameter | Value | Evidence |
  |---|---|---|
  | `spike_amplitude_x_baseline` | **3.0** | `(threshold − baseline_mean)/baseline_std` = 3.0000 exactly across all 8,722 events. It is a **z-multiplier** (`autocorrelation_seizure.py:128`: `bl_mean + param × bl_std`), i.e. 3σ — not an amplitude ratio. |
  | `baseline_method` | **`percentile`** | `baseline_mean` constant within **663/663** (file, channel) groups → global per file+channel, not time-local. |
  | `min_spike_freq_hz` | **2.0** | observed minimum exactly 2.0. |
  | effective amplitude floor | **≈6.0× baseline** | `max_amplitude_x_baseline` has a hard floor at 6.04 across the whole SV2A candidate pool. |

- ⚠️ **Delete v1's `BASELINE_METHOD=rolling` escape hatch.** v1 §10 offered it in case
  "local BL over 3" meant a time-local baseline. It did not — the baseline is global, and
  flipping this would silently break reproduction of the SV2A operating point.
- ⚠️ **"Local BL over 3" is a secondary, post-detection UI filter** (user, 2026-10-01), not
  the detector's internal threshold. The training-page control
  (`training.py:154-158`, on `max_amplitude_x_baseline`) defaults to 0 and binds only when
  set by hand — and at 3 it would not bind at all, since the SV2A pool already sits ≥6.04.
  **Decide before the run** whether the GDNF candidate pool should reproduce SV2A's ≈6×
  floor; if the batch script applies no post-filter, GDNF will yield a larger and
  differently-composed candidate pool than SV2A did. That is acceptable for
  capture–recapture (which only needs both detectors on GDNF) but must be stated for the
  NED-Net-vs-classical head-to-head, where the classical arm should run at the operating
  point it was actually used at.
- **Run:** copy `detect_autocorr_batch.py` → `~/NED-Net/scripts/lunarc/`, test small first:
  ```
  conda activate bendr && cd ~/NED-Net
  python scripts/lunarc/detect_autocorr_batch.py \
    --edf-dir /lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025 \
    --db-path /tmp/ac_test.db \
    --metadata-csv scripts/lunarc/RAM_GDNF_2025_batch_metadata.csv \
    --path-include 'Batch_1_Recordings/Week_1/' --workers 4
  ```
  then `sbatch detect_ramgdnf_autocorr.sbatch`. (Note the corrected `--edf-dir`; v1's test
  command used the non-resolving `lu2026-12-29` path and would have failed.)
- **Never** run it onto the U-Net DB — the merge is destructive per EDF path.

### 6c. `db_to_sidecars.py` — DB → pending sidecars (U-Net only)
Reads a project DB, writes per-EDF `*_ned_annotations.json` with `label="pending"` via
`save_annotations`. **Superseded for the final workflow by §7**; keep for a U-Net-only
review. Run ON LUNARC in the `bendr` env. Has `--dry-run` and `--overwrite` (skips existing
by default — keep that).

### 6d. Metadata / planning artifacts
`RAM_GDNF_2025_batch_metadata.csv`, `RAM_GDNF_2025_cohort_key.csv` (both now in
`scripts/lunarc/`), `NEDNet_validation_retraining_plan.md`, `NEDNet_runbook.md`.

---

## 7. Not yet built — the combined converter (next agent task)

After **both** detection DBs exist (`ram_gdnf_unet_v0.db` + `ram_gdnf_autocorr_v0.db`), a
converter that reads both and, **per EDF**:
- overlap-merges the two detectors' events per channel (custom merge — **not**
  `merge_annotations`), emitting **one pending annotation per seizure** (review once),
- records which detector(s) proposed it in `features["detectors"]`, using
  **`["seizure_unet", "autocorrelation"]`** — note `seizure_unet`, not v1's `seizure_cnn`
  (**V2**),
- carries `features["convulsive"]` from Stage-2 where available (a pre-fill the reviewer
  corrects),
- writes the merged pending sidecar beside each EDF via `save_annotations`, **skipping any
  existing sidecar unless `--overwrite` is passed**.

**Pre-register the overlap tolerance.** The both-caught count *m* drives N̂ = n₁n₂/m
directly, so the matching tolerance silently sets the sensitivity estimate. Fix a value
before looking at results, record it here, and report a sensitivity analysis across a range
(e.g. ±1, 2, 5 s onset tolerance). v1 said "onset within tolerance" and never named one.

---

## 8. Exact next steps (in order)

1. ~~Place `RAM_GDNF_2025_batch_metadata.csv` on LUNARC at `~/NED-Net/scripts/lunarc/`.~~ **DONE**
2. ~~U-Net detection → `ram_gdnf_unet_v0.db`.~~ **RUNNING** (job 3770118; confirm the final
   log shows 1377 processed and 0 errors, then copy the DB back).
3. **Answer §10 Q1** (what generated Mir's candidates). Blocking for the recall claim; costs
   one email.
4. **Autocorrelation sweep:** copy `detect_autocorr_batch.py` → `~/NED-Net/scripts/lunarc/`,
   settle the amplitude-floor question in §6b, run the small test, then `sbatch`.
5. **Low-threshold U-Net run** (core threshold 0.1, same cohort, separate DB
   `ram_gdnf_unet_lowthr_v0.db`) so the FROC can reach the high-sensitivity regime (**V5**).
   Cheapest while the cluster work is already queued.
6. **Combined converter** (build against the real DBs) → merged pending sidecars.
7. **Set up the ground-truth separation in §12 BEFORE importing Mir's annotations.**
8. **Review in the NED-Net UI (ThinLinc)** on the LUNARC copies: confirm/reject each seizure
   once → sidecars become the confirmed/rejected ground truth + training labels.
9. **Phase-2 staging:** within-cluster symlink/copy the SV2A EDFs + sidecars into per-fold
   trees alongside the RAM_GDNF cohorts. Verify the SV2A sidecars sit beside the EDFs on
   LUNARC first. Diff sidecar counts against `SV2A_20260615.json` before training (§3.6).
10. **From-scratch retraining, LOCO:** `train_unet.sh` + `train_convulsive.sh` (`gpua100`),
    one model per fold per §13.
11. **Hand result DBs/CSVs back** for analysis (§9).

---

## 9. Handoff boundary — what comes back for analysis

Compute model is **prepare-and-hand-off**: LUNARC needs password + 2FA to *establish* a
session. Within a session it can be scripted — `~/.ssh/config` sets `ControlMaster auto`
with a `ControlPath`, so after one login further `ssh`/`scp` reuse the socket without
re-auth (v1 §9 overstated this as impossible). Marco runs detection/retraining; only small
artifacts come back:
- project DBs (`ram_gdnf_unet_v0.db`, `ram_gdnf_autocorr_v0.db`, per-fold detection DBs) — ~MB,
- event CSVs / metrics tables,
- the confirmed sidecars (or a dump).

Raw EEG (~16,000 animal-hours) stays on the cluster. Local analysis then covers:
event-matching, precision/recall, capture–recapture arithmetic, FROC/FP-h/latency, Stage-2
confusion, calibration, learning curves, forgetting check, figures.

---

## 10. Open items

### BLOCKING

**Q1 — What generated Mir's candidates?** The convulsive-recall claim rests entirely on
this. Mir says video scoring was used, "or at least video was used for confirmation of
candidates" — and the distinction decides the claim:
- candidates *generated* from video/behaviour → genuinely independent, absolute recall is
  defensible, and Mir's set can also serve as the independent anchor for the §4 cross-check;
- candidates generated by an **automated detector** and merely *confirmed* on video →
  the number is "recall relative to that detector", by the same §4 guardrail that forbids
  self-review.

**Specific thing to ask:** Mir's export schema carries `candidate_spike_freq` and
`candidate_peak_score` columns (`scripts/import_mir_annotations.py`). Those are shaped like
an autocorrelation-family detector's outputs — spike frequency plus a peak autocorrelation
score. If his candidates came from such a detector, his set is **not** independent of our
autocorrelation arm either, which would undermine both the recall claim and the §4
cross-check. Ask what produced those two columns. (**V6**)

**Q2 — Capture–recapture dependence.** Resolved in method (§4): run the Mir-anchored
cross-check and report the gap. Depends on Q1. Fallback if Q1 goes badly: report
capture–recapture with the 74%-shared-training-lineage caveat stated plainly, or drop it per
§4.

### Decided (previously open)
- ~~Autocorrelation baseline `percentile` vs `rolling`~~ → **`percentile`**, evidenced
  (**V3**). Remove the `rolling` option from the sbatch.
- ~~Compute account `lu2026-2-60` still active?~~ → **yes**, job 3770118 ran on it
  2026-10-01. And the storage/compute labels in v1 were inverted (**V1**).
- ~~Batch 4 coverage: complete it or treat as supplementary~~ → **usable as-is**, see §13.

### Still open (non-blocking)
- Whether the autocorrelation batch run should reproduce SV2A's ≈6× amplitude floor (§6b).
- Whether the UI edit path preserves custom `features` keys through a reviewer edit (**V7**).
- Units/onset QC on the RAM_GDNF EDFs (mV vs µV, channel map).

---

## 11. Cluster / environment reference

- Host `cosmos.lunarc.lu.se` (ssh alias `cosmos`, user `ma6384le`).
- `module load Anaconda3/2024.06-1`; `source config_conda.sh` from `~/NED-Net`;
  `conda activate bendr`.
- CPU detection: partition `lu48`. GPU retraining: `gpua100`. Account `-A lu2026-2-60`.
- **Data storage: `/lunarc/nobackup/projects/lu2026-2-60/`** — `RAM_GDNF_2025/` (new cohorts)
  and `edf_data/` (SV2A). ⚠️ not `lu2026-12-29` (**V1**).
- Submit from `~/NED-Net` with `~/NED-Net/logs` existing (relative `-o` path).
- Per-worker part-DBs merged at the end (`db.py` sets no `busy_timeout`, so one shared DB
  would hit "database is locked").
- `WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"` resolves correctly under `--exclusive` (48 on
  cn096); `scontrol` showing `NumCPUs=1` while PENDING is requested-minimum bookkeeping.

---

## 12. Ground-truth separation — leakage control (NEW)

**The hazard.** `scripts/import_mir_annotations.py` converts Mir's Excel export into
`<edf_stem>_ned_annotations.json`. Its docstring states the purpose as "for direct
training", and `--inplace` writes each JSON **next to the EDF**. `ml/train_unet.py` builds
its dataset by recursively scanning `--data-dir` for exactly those files. So importing Mir's
set `--inplace` into the LUNARC tree would:
1. turn the **evaluation ground truth into training data**, violating §4's "never evaluate
   on events used in training" — the one guardrail that protects the headline claim; and
2. **collide by filename** with the pending review sidecars the §7 converter writes, in
   either direction depending on run order.

Confirmed in code (**V6**). This replaces v1's concern about overwriting the research-share
sidecars, which is moot — the LUNARC copies have none.

**The rule.** Mir's annotations never enter a tree that any trainer scans.

- Import **offline**: `python scripts/import_mir_annotations.py annotationsMir.xlsx -o
  ~/ground_truth/mir_20261001/` — the default, outside any `--data-dir`. **Do not pass
  `--inplace`.**
- Keep Mir's set as the **scoring** set only. Phase-1/2/3 metrics read it from
  `~/ground_truth/`; nothing under it is ever passed as `--data-dir`.
- The review sidecars (from §7, written beside the EDFs) are the **training** labels. Two
  physically separate locations, one filename convention — the separation is the directory,
  so it must be enforced by habit and by a check.
- **Add a pre-training assertion:** before any `train_unet.sh` / `train_convulsive.sh` run,
  assert that no sidecar under the fold's `--data-dir` originates from the Mir import (tag
  the imported events with `features["provenance"]="mir_video"` at import time and fail the
  staging step if any appears). Cheap, and it makes the guardrail mechanical instead of
  procedural.
- If Mir's events are ever *deliberately* added to training (a legitimate later choice),
  that fold's results stop being a recall measurement and must be relabelled.

---

## 13. LOCO fold decision (NEW — decision made 2026-10-01)

v1 specified "LOCO by cohort (4 folds)" while flagging that Batch 4 is annotated for only
6 of ~21 day-folders, and did not note that Batch 2 has just 24 confirmed events. Decision:

**Keep all 4 folds, and define a per-fold "evaluable region".**

1. **Restrict every metric to annotated day-folders.** Precision and recall are both
   estimable within an annotated region — recall's denominator is "confirmed events in the
   annotated region", which is complete there. Partial annotation only invalidates
   *rate/burden* extrapolation, which Phase 1 does not claim. So **B4 is usable as a LOCO
   test fold as-is**; v1's "complete it or treat it as supplementary" was over-cautious.
   Define and record the evaluable region per fold (list of day-folders) as a versioned
   artifact.
2. **Primary recall/sensitivity folds: B3 (154) and B4 (192).** Adequately powered.
3. **B2 (24 events) is underpowered** — a recall estimate there carries roughly ±0.2. Report
   it per-fold with its CI, and additionally report **B1+B2 pooled (84 events)** as a
   secondary estimate. Never print a per-fold recall without the event count beside it.
4. **Precision is reported for all 4 folds** — it conditions on detections and needs no
   complete positive set, only a representative annotated region.
5. **Completing B4's annotation is optional, not blocking** — but it is the highest-yield
   annotation effort available (already the richest fold at 192 events from 6 folders), and
   it would make B4 the anchor fold. Worth asking Mir, without gating Phase 2 on it.
6. **Train on all 4 cohorts minus the held-out one** regardless of annotation completeness —
   more labelled data helps training even where it's too sparse to evaluate on.

Rationale: the fold structure is the right design and the spread (B1=60, B2=24, B3=154,
B4=192) is exactly why it's needed; the only real constraint is statistical power, which is
handled by reporting rather than by dropping folds.
