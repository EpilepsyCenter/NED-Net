# Run log — NED-Net methods paper

**Append-only.** One entry per compute run or analysis that produces a number we might
quote. Add the entry when you launch, complete it when it finishes. The project DBs record
no model name (only `source` and `mode`), so **this log plus the script's git commit is the
only provenance** linking a result to the weights and settings that produced it.

Template:

```
### <date> — <what> — job <id>
script:   <path> @ <git commit>
config:   <the settings that matter>
inputs:   <data / DB / ground truth>
outputs:  <DB / model / figure path>
result:   <the headline number, or FAILED + why>
```

---

### 2026-10-01 — Phase-1 frozen detection, FAILED RUN — job 3769954
script:   `scripts/lunarc/detect_ramgdnf_unet.sbatch` (pre-fix)
config:   frozen operating point; `EDF_DIR=.../lu2026-12-29/RAM_GDNF_2025`
result:   **FAILED** — 0 EDFs matched, exited after 9 s. Wrong storage path
          (`lu2026-12-29` is the compute allocation; the symlink exists only in the LUNARC
          home dir). The sanity check used `2>/dev/null` so a missing dir reported `0` and
          the job printed "done". Both fixed; the check now `exit 1`s on zero matches.

### 2026-10-01 — Phase-1 frozen detection — job 3770118
script:   `scripts/lunarc/detect_ramgdnf_unet.sbatch` @ `74e6e47`
config:   `UNetv2_20260615` @ 0.5 + boundary 0.1 -> `Convulsive_v4LUNARC_20260616` @ 0.45,
          no re-ranker, min_dur 5, merge_gap 2, `--path-include 'Batch_[1-4]_Recordings'`
inputs:   1,377 cohort EDFs, `/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025`
outputs:  `~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db` (10 MB)
run:      lu48 node cn096, 14:27-18:36 CEST (4 h 09 m), 48 workers, 0 errors
result:   **41,408 events** over 2,014 recording hours (~16,100 animal-hours);
          78.1% non-convulsive / 21.9% convulsive

### 2026-10-02 — ground-truth consolidation
script:   `scripts/local/consolidate_mir_annotations.py` @ `74e6e47`
inputs:   69 `Day_N.xlsx` workbooks on the LU research share
outputs:  `~/ground_truth/mir_ramgdnf_annotations.csv` (4.2 MB)
result:   14,740 reviewed rows over 1,074 recordings; **430 confirmed seizures**
          (368 convulsive, 51 behaviour, 11 untyped), 11,653 `False`, 2,657 `Normal`

### 2026-10-02 — Phase-1 scoring, frozen model vs ground truth
script:   `scripts/local/validate_frozen_unet_vs_mir.py` @ `74e6e47`
config:   match tolerance 5 s; Mir channel `k` -> NED-Net `k-1`
result:   **recall 9.1%** (39/430; convulsive 9.8%, behaviour 5.9%);
          **precision 13.7%** (51/372 adjudicated overlaps).
          Convulsive-only: 6.8% recall / 27.6% precision.
          conf>=0.5: 7.1% / 17.3%. Convulsive+conf>=0.5: 4.6% / 29.6%.
          Controls: channel `k-1` confirmed at 6x margin over `k`/`k+1`; time bases
          aligned (median gap -2.2 s); all 1,074 GT recordings processed.
          **74% of detections are unadjudicated** — Mir reviewed only 6/3/22 candidates on
          animals 483552/483553/483555, which produce 30,669 detections (74.1%) and have
          1 confirmed seizure between them. Excluding them leaves precision and recall
          unchanged (only the count falls, 41,408 -> 10,739).

### 2026-10-02 — merged sidecars built
script:   `scripts/local/build_merged_sidecars.py` @ `ee72f24`
config:   dedup tolerance 5 s; `Normal` rows dropped; `event_id` int (schema requirement)
outputs:  `~/staged_sidecars/` — 1,364 sidecars, 44 MB
result:   430 confirmed + 11,653 rejected + 41,045 pending; 282 events proposed by both
          Mir and the U-Net. All 5,977 Batch-3 events verified to round-trip through
          `AnnotatedEvent.from_dict`.

### 2026-10-06 — sidecars transferred to LUNARC
result:   1,364 sidecars in place beside the EDFs; project at 6,635/50,000 files after the
          quota increase (was 5,000, and directories count). Grace cleared.

### 2026-10-06 — Round-0 arm B, hold out Batch 3, CANCELLED — job 3795249
script:   `scripts/lunarc/train_unet.sh` @ `f7736d1`
result:   **CANCELLED before running.** Submitted with `exclude=none` despite
          `EXCLUDE_ANIMALS` being set: `POS_WEIGHT` and `EXCLUDE_ANIMALS` used a bare
          `read`, which sets the variable to empty on Enter and destroyed the preset.
          Would have trained the hold-out-B3 fold on its own test set. Fixed in `a3ce36f`
          (both fields now use the preserving `ask()` helper, which also displays the
          current value). **Always check the summary line reads `exclude=<ids>`.**

### 2026-10-06 — Round-0 arm B, hold out Batch 3 — job 3795251
script:   `scripts/lunarc/train_unet.sh` @ `a3ce36f`
config:   `EDF_DIR=.../RAM_GDNF_2025` (RAM_GDNF only, no SV2A); exclude 459657-459664;
          neg_pos_ratio 10, pos_weight auto (=10), epochs 50, batch 32, lr 3e-4,
          patience 10, base_filters 32, depth 4, dropout 0.2, neg-source hard
inputs:   Mir's labels as merged sidecars + `*_ned_channels.json` (32 animals).
          After exclusion: 24 animal groups, **276 positives**, 6,231 hard negatives;
          default split train 184 pos / val 92 pos
outputs:  `~/.eeg_seizure_analyzer/models/ramgdnf_armB_holdB3/`
result:   _pending_
notes:    Prediction on record — this should improve recall but NOT the over-detection,
          because `_balance_negatives` prefers rejected events and only tops up with
          random background on a shortfall; 6,231 rejected exceeds the 2,760 target, so
          no background is drawn and the unlabelled flood regions are never seen.

### 2026-10-06 — channel-ID files written and transferred
script:   `scripts/local/build_channel_ids.py` @ `f7736d1`
outputs:  1,377 `*_ned_channels.json` beside the EDFs (project ~8,012/50,000 files)
result:   `--analyze` now reports **32 animal groups** with real IDs (was 8, pooled by
          channel index). Batch 3's eight animals total exactly 154 positives, matching
          Mir's count. Without this, `--exclude-animals` matched nothing.

### 2026-10-06 — Flood review, file 1 of 9 (low tercile)
file:     `B4_W2_D3_30062026(14)` — 46 pending events (whole file queue cleared;
          42 belong to the three flood animals, 4 to others on the same recording)
result:   **all noise except possibly event 1.** ~1/46 = **~2% precision**, against the
          13.7% measured on the animals Mir did review.
morphology (reviewer, EEG only — no video on LUNARC):
          channels 2 and 6 carry **regular ~10 Hz spiking** — rhythmic artefact, not ictal.
          Other channels show less regular background noise. None are events.
interpretation:
          This is a mechanistic explanation for the precision collapse, not just a count.
          `UNetv2_20260615` drew 74% of its training positives from the autocorrelation
          detector, whose confirmed events centre near 10 Hz spike frequency (the SV2A
          sidecars record `mean_spike_frequency_hz` 2-14 Hz, first confirmed = 10.48 Hz).
          The model therefore learned "rhythmic ~10 Hz spiking" as its seizure signature,
          and fires on channels carrying regular 10 Hz artefact.
          It also explains why these events are unadjudicated: Mir's candidate generator
          applied an amplitude floor (~6x baseline in SV2A), so low-amplitude rhythmic
          noise never entered his review queue.

### 2026-10-06 — Ground-truth defect: timestamps outside their EDF
found:    Review of `B4_W1_D1_21062026(4)` crashed the Training tab:
          `ValueError: The length of the input vector x must be greater than padlen`
          from `sosfiltfilt` in `_build_review_figure` -> `bandpass_filter`. Cause: the
          first event on that file starts at 7,813.9 s in a 5,400 s recording, so the
          extracted segment was empty.
extent:   **242 of 12,083** adjudicated rows (2.0%) end past their file's duration,
          including **38 of 430 confirmed seizures (8.8%)**. Max `end_s` = 10,740 s
          against a 5,401 s maximum file duration. Listed in
          `review/groundtruth_out_of_range.csv`.
not longer files:
          The recordings are 90-minute files that **tile the protocol continuously** —
          per batch 340/339/341/357 files covering 20.7/20.7/20.8/21.7 days; 2,014
          recording hours x 8 channels = **16,114 animal-hours**, matching the ~16,000
          expected, with **0 files over 92 minutes**. `chunk_end_sec` is the true duration
          read from the EDF (`analysis.py:421`), and neither `detect_batch.py` nor
          `process_chunk` caps length. So detection covered 100% of the data and no re-run
          is needed; the overruns are a misalignment in the source export.
effect:   Those rows can never match a detection, so counting them as misses understated
          recall. Corrected denominator: **recall 9.8% -> 10.6%** (convulsive 340).
          Per-batch figures unchanged (33.3 / 33.3 / 5.2 / 1.6%). Precision unaffected.
fixes:    `build_merged_sidecars.py` drops them (and logs them); `validate_frozen_unet_vs_mir.py`
          excludes them from the denominator.
to ask Mir:
          whether `start_s` is file-relative or session-relative for these rows — 10,740 s
          is about two files, so a multi-file session time base is the likely explanation.

### 2026-10-06 — Sidecar bug: detector_confidence out of domain
found:    Mir's `candidate_peak_score` was written into `detector_confidence`. It is an
          autocorrelation peak score (hundreds to 710,660), not a probability.
          12,072 of 53,128 events exceeded 1.0. The UI's confidence filters are
          `dcc.Input(min=0, max=1)`.
fix:      `detector_confidence=0.0` for human-adjudicated rows; his candidate scores
          (peak, mean, spike freq/count, low/gamma/broad band powers) moved into
          `features`, where they remain available for analysis. Commit `acb6ea2`.
note:     This was NOT the cause of the Training-tab crash (that was the timestamp defect
          above), but it was a real defect found while investigating it.

### 2026-10-06 — Is the ground truth still usable? Yes. (sensitivity analysis)
question: if Mir's and the U-Net's time bases disagree, is all ground truth invalid?
test 1 — alignment vs chance:
          For each in-range confirmed seizure, count matches within 5 s, then repeat with
          the seizure shifted by a random offset within its own file (200 draws).
          Observed **39**; null **mean 3.3, sd 4.0, max 32**. **z = 9.0, p < 0.005.**
          Systematic misalignment would give ~3 matches, so the alignment is real.
test 2 — localisation:
          All 277 overrunning rows are in **Batch 4 only**, affecting **37 of 1,074
          recordings (3.4%)**, median 5 rows each; worst file 30 rows.
test 3 — sensitivity (rows inside a suspect file could be shifted yet still land in range,
          so drop those recordings wholesale):

| treatment | overall | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| in-range rows only | 39/392 = 9.9% | 33.3% | 33.3% | 5.2% | 1.9% |
| also drop the 37 suspect recordings | 38/356 = 10.7% | 33.3% | 33.3% | 5.2% | 1.7% |

conclusion:
          The ground truth is usable. The defect is localised to Batch 4 and immaterial:
          Batch 4's near-zero recall is **not** a ground-truth artefact — excluding every
          suspect recording makes it slightly worse (1.9% -> 1.7%), and B1-B3 are
          unaffected. Report the in-range figure with this sensitivity analysis beside it.
caveat:   This establishes alignment on average, not per event. Sub-second drift is
          absorbed by the 5 s matching tolerance (matched events: median gap -2.2 s).

### 2026-10-06 — Precision was pessimistic: Mir's set has no non-convulsive events
fact:     Mir annotated **convulsive or behavioural seizures only** — no short
          non-convulsive events (user, 2026-10-06). But **78% of the U-Net's output
          (32,319 of 41,408) is non-convulsive**.
consequence:
          A U-Net non-convulsive detection overlapping one of Mir's `False` rows was being
          counted as a false positive. `False` means "not a convulsive/behavioural
          seizure"; it does not rule out a non-convulsive seizure. Breaking the 372
          adjudicated overlaps apart:

| | n |
|---|---|
| confirmed TP (overlaps a `Seizure`) | 51 |
| **confirmed** FP (U-Net said convulsive, Mir said no) | 92 |
| **ambiguous** (U-Net said non-convulsive, Mir said `False`/`Normal`) | 229 |

corrected numbers:
          * precision as first reported: 51/372 = **13.7%** — pessimistic
          * precision on unambiguous rows: 51/143 = **35.7%**
          * **convulsive-only precision: 35/127 = 27.6%** — the defensible figure
          * **recall unaffected: ~10.6%**, since Mir's set *is* the convulsive reference
what can and cannot be claimed:
          Convulsive recall and convulsive precision are both reportable. The
          non-convulsive arm — 78% of output — has **no ground truth in this cohort** and
          cannot be scored in either direction. The flood review is currently the only
          evidence about it (46 events, all noise bar one), which is why that sampling
          still matters even though the 10 Hz mechanism will not be reported as a finding.
for Phase 2:
          Arm B (RAM_GDNF only) **structurally cannot learn non-convulsive seizures** — its
          labels contain none. The SV2A batches are the only source of that class, so arm A
          is required rather than a comparison arm.

### 2026-10-06 — Decisions
* **Batch 4 is kept**, minus the 37 recordings with out-of-range ground-truth timestamps.
* **Noisy recordings will be excluded**, but on a pre-registered signal-quality metric
  computed from the raw EEG and blind to model output — never because a detector fired on
  them. See `EXCLUSION_CRITERIA.md`; sweep not yet run.

### 2026-10-06 — Flood review, files 1-2 of 9: 0 real events in 81
files:    `B4_W2_D3_30062026(14)` (46 events, **low** confidence tercile) and
          `B4_W1_D1_21062026(4)` (35 events, **high** tercile) — both queues cleared in full.
result:   **0 confirmed of 81.** 95% upper bound on the true-event rate **3.6%**, so at
          most ~1,100 of the 30,669 flood detections could be real; point estimate 0.
morphology (reviewer, EEG only — no video):
          regular ~10 Hz spiking on some channels, less regular background noise on others,
          and crucially **no evolution of the signal** in any of them. Real ictal activity
          evolves in frequency and amplitude across an event; none of these do. That is a
          morphological criterion that does not require video.
why the two terciles matter:
          The sample was stratified by mean confidence. Files from the lowest and highest
          terciles both came back all-noise, so **confidence carries no information here** —
          the model is not merely uncertain about these, it is wrong about them with equal
          force at every confidence level.
consequence:
          Full-cohort precision, treating the flood as noise: **51/41,408 = 0.12%**.
          The 51 TP are against the convulsive reference only.
coverage of the output now characterised:
          flood sampled and found to be noise **30,669 (74%)**; adjudicated against Mir's
          candidates 372 (0.9%); still uncharacterised 10,367 (25%).
next:     one mid-tercile file (`B4_W1_D1_21062026(5)`, 14 events) completes the stratified
          design cheaply; beyond that the marginal value is low and the remaining effort is
          better spent on Batch 3, which addresses recall rather than precision.

### 2026-10-06 — Spot-check of Mir's positives, and recall vs event duration
calibration (matched, both sources agree):
          `B1_W3_D5_22092025(13)` ch0 3763.6-3810.8 s and ch7 1598.9-1636.7 s, both
          convulsive. Reviewer: **both are good.** So the ground truth is sound for typical
          events and the recall failure is not an artefact of bad labels.
missed, long events:
          `B1_W1_D1_04092025(1)` — four missed "convulsive" events of 172-239 s on ch7 plus
          one of 62 s on ch1. Reviewer: **real convulsive seizures are never minutes long**,
          so the long ones are almost certainly **multiple chained seizures annotated as one
          block**; the 62 s event is rhythmic and is a seizure (possibly several in
          sequence). So Mir's long annotations are blocks, not single events.
match rate vs duration (all 383 in-range confirmed seizures):

| duration | n | match rate |
|---|---|---|
| <10 s | 100 | **3.0%** |
| 10-20 s | 100 | 8.0% |
| 20-30 s | 59 | 11.9% |
| 30-60 s | 94 | 17.0% |
| 60-120 s | 26 | 19.2% |
| 2-5 min | 10 | **0.0%** |
| >5 min | 3 | **0.0%** |

reading:
          The reviewer's hypothesis (never trained on long or chained events; the model's
          `window_sec` is 60 with per-channel z-scoring, so an event filling the window
          leaves no contrast) **holds at the extreme — 0 of 13 events over 2 min were
          caught** — but accounts for only ~4% of the 344 misses.
          The dominant pattern is the opposite: the model is **worst on short events** (3%
          under 10 s) and improves monotonically with duration to 2 min. 200 of 383 events
          are under 20 s. Median convulsive duration is 22 s, a third of a 60 s window with
          ample contrast, and 90% are still missed. That shape is the signature of broadly
          low per-window sensitivity, with longer events caught more often through more
          opportunities and greater salience — not a duration-specific bug.
          **The core recall failure remains unexplained.** Best remaining leads: the
          per-channel asymmetry (ch7: 52 detections cohort-wide against 107 confirmed
          seizures; ch3: 15,291 against 8) and the signal-quality sweep.
ground-truth caveat (bounded):
          38 of 340 convulsive annotations (11%) exceed 60 s and 13 exceed 2 min; these are
          likely chains and should carry a duration caveat, or be reported separately.
          Excluding all 38 changes recall by about a point.

### 2026-10-06 — Signal-quality sweep — job 3798487
script:   `scripts/lunarc/signal_quality.sbatch` @ `58ab100`
config:   1,377 cohort EDFs, 60 windows/channel, peak_ratio 10, lu48 -c 24
outputs:  `~/signal_quality_ramgdnf.csv` (home, no project inodes)
purpose:  pre-registered exclusion metric (`EXCLUSION_CRITERIA.md`) AND an explanation for
          the per-channel asymmetry (ch7: 52 detections cohort-wide vs 107 confirmed
          seizures; ch3: 15,291 vs 8)
result:   _pending_

### 2026-10-06 — LUNARC repo consolidated
          Pulled 48 commits (38be406 -> bf08e7b). Removed a stale nested clone
          (`~/NED-Net/NED-Net`, 166 files from Jun 16, older commit, nothing unique) and
          two empty scp-accident directories (`scriptsarrhenius`, `scriptslocal`). The two
          files previously scp'd were md5-verified identical to the committed versions
          before being discarded for the pull.

### 2026-10-06 — Round-0 arm A, hold out Batch 3 — job 3800378
script:   `scripts/lunarc/train_unet.sh` @ `4786254`
config:   `EDF_DIR=/lunarc/nobackup/projects/lu2026-2-60` (**parent** -> SV2A `edf_data`
          + `RAM_GDNF_2025`); exclude 459657-459664; neg_pos_ratio 10, pos_weight auto,
          **--stable-val-split**, epochs 50, batch 32, lr 3e-4, patience 10
dataset:  1,216 annotated EDFs; **1,369 confirmed**, 12,925 rejected (more than the 867 in
          `SV2A_20260615` because LUNARC's `edf_data` holds more annotated files than the
          165-file subset that trained UNetv2). After exclusion: 45 animal groups,
          1,215 positive windows, 7,503 hard-negative windows.
          Split with the stable option: **train 865 positives / val 350** (without it:
          603/612 — half the positives in validation).
caveat:   available neg/pos ratio is only 6.2 here, so requesting 10 yields everything;
          arm B had 23 available and used 10. The arms therefore differ in *effective*
          ratio because the data differ. Same requested value is the honest choice but
          must be stated rather than presented as matched.
result:   _pending_

### 2026-10-06 — Round-0 arm B RERUN with stable split — job 3801062
script:   `scripts/lunarc/train_unet.sh` @ `4786254`
config:   identical to arm A except `EDF_DIR=.../RAM_GDNF_2025` (no SV2A)
why:      the first arm B (3795251) ran before `--stable-val-split` existed, so comparing
          it with arm A would confound **data** with **split**. This run isolates the data
          difference. 3795251 is retained as the unstable-split reference.
what it separates:
          arm A trains on SV2A **plus** RAM_GDNF, so success alone cannot distinguish
          "in-domain data" from "more positives". If stable arm B (276 in-domain positives)
          also recovers -> the recommendation to other labs is "annotate your own cohort".
          If only arm A recovers -> the pretrained base carries real value. Different
          conclusions, so both arms are needed.
result:   _pending_

### 2026-10-06 — Batch-3 review INVERTS the Phase-1 conclusion
files:    `B3_W3_D7_06022026(6)` (17 U-Net events) and `B3_W3_D8_07022026(9)` (15).
          Both had **zero** of Mir's confirmed seizures, so anything real here is an event
          his reference does not contain.
result:   **23 of 32 real (72% precision)** — and the split by confidence is clean:

| confidence | n | real | precision |
|---|---|---|---|
| < 0.4 | 10 | 1 | **10%** |
| >= 0.5 | 22 | 22 | **100%** |

stage 2 is ANTI-PREDICTIVE:

| Stage-2 says | n | actually real |
|---|---|---|
| convulsive | 9 | 1 (**11%**) |
| non-convulsive | 23 | 22 (**96%**) |

          Trained on SV2A convulsive events, it fires on RAM_GDNF's rhythmic artefact (the
          flood was 26% convulsive-labelled) and correctly declines the real RAM_GDNF
          seizures, which are non-convulsive. Reviewer: the detections look like genuine
          seizures but were **misclassified as convulsive** — no signal evolution.
          So the flag is not merely broken, it is informative in reverse.
what this does to the earlier numbers:
          * "full-cohort precision 0.12%" pooled 41,408 detections of which 26,504 are
            below mean-confidence 0.5, plus the flood. **It is not the precision of a
            sensibly-thresholded detector.**
          * precision measured against Mir's candidates (13.7% / 27.6%) is an
            **UNDERestimate**: his 15-80 Hz, >=15 s detector cannot propose the ~10 Hz
            non-convulsive events this model is best at, so real detections were scored as
            false positives.
          * **recall ~10.6% stands** — the model does miss most of Mir's episodes. The two
            instruments are sensitive to different event populations with low overlap in
            both directions; neither is a superset.
          * At conf >= 0.5 excluding the flood animals: **4,336 detections = 6.5 per
            animal-day** against Mir's 430 total. If precision holds, the cohort's seizure
            burden is far higher than the reference captured.
SELECTION BIAS — the reason this is not yet a result:
          those two files were chosen **because** they had the highest confidence, so the
          sample is biased toward the regime where precision is good. 22 events, 2 files,
          1 batch.
next:     `review/precision_at_conf05_sample.csv` — 16 files / 84 events, files drawn at
          RANDOM stratified by batch with no confidence selection, flood animals excluded,
          seed 7. That measures precision at conf >= 0.5 properly.

### 2026-10-06 — Mir's candidate generator (answer to Q1)
          Peaks in the **15-80 Hz** envelope above median + 4 x MAD; peaks grouped at
          <=0.5 s, padded and merged; episodes retained at **>=15 s** with **>=2.5
          spikes/s**; scored on spike density plus low/gamma/broadband power; labelled
          convulsive when the peak score exceeds baseline, else behaviour. Then adjudicated
          **manually with video** as Seizure / False / Normal.
implications:
          * Candidates are **automated, not video-generated** — video entered only at
            adjudication. So recall against this set is "recall relative to a 15-80 Hz
            episode detector", NOT absolute recall. The convulsive/behaviour labels are
            video-anchored and sound.
          * The **>=15 s floor applies to the candidate episode**, not the final label:
            `False` rows have median 68.9 s (the episodes) while `Seizure` rows are median
            18.5 s, minimum 3.4 s — the human refined boundaries inside each window, so the
            Seizure timings are human-determined.
          * A brief isolated seizure with no surrounding >=15 s high-density episode
            **could never be a candidate** and cannot appear in the ground truth.
          * **Band mismatch is the crux**: his 15-80 Hz envelope versus the U-Net's ~2-14 Hz
            rhythmicity inherited from the autocorrelation detector. Different instruments,
            different event populations — which is exactly what the Batch-3 review shows.

### 2026-10-07 — Unbiased sample WITHDRAWS the "works at conf >= 0.5" claim
sample:   `review/precision_at_conf05_sample.csv` — 16 files drawn at random stratified by
          batch, no confidence selection, seed 7. Batch 4 not reviewed (now excluded).
          12 files / 112 adjudicated U-Net events returned.
headline: **precision at conf >= 0.5 is 18/53 = 34% (95% CI 22-48%)**, not the 100% seen in
          the two hand-picked Batch-3 files. **The 2026-10-06 claim is withdrawn.**
          Confidence remains monotonically informative: 10% (<0.5), 24%, 35%, 42%, 67% (>0.8).
by batch (conf >= 0.5): B1 6/7 = 86%, B2 6/40 = 15%, B3 6/6 = 100%.
the channel exclusion does NOT rescue B2:
          B2 is poor on every reviewed channel — ch1 5%, ch2 13%, ch3 9%, ch5 40% — and
          excluding code-ch2/ch3 gives 12% against 10% for those channels alone. So B2's
          problem is not confined to the channels flagged as noisy.
Stage-2 inversion also does not survive:
          unbiased sample gives 3/8 real among "convulsive" and 21/104 among
          "non-convulsive". The dramatic 11%-vs-96% inversion of 2026-10-06 was a
          small-sample artefact of the two hand-picked files. **Also withdrawn.**
WHY PRECISION ON THE RETAINED SET STILL CANNOT BE ESTIMATED:
          after excluding Batch 4, B2 code-ch2/ch3 and 449382, the retained output at
          conf >= 0.5 is 934 detections — B1 160 (17%), **B2 437 (47%)**, B3 337 (36%).
          The random sample spent 56 of its 71 B2 events on code-ch3, a channel now
          excluded, leaving **2 reviewed events on B2's retained channels**. So nearly half
          the output is unmeasured. A naive mean over retained reviewed events gives 87%,
          which is an artefact of that weighting, not a result.
next:     `review/b2_retained_sample.csv` — 10 files / 37 events on B2 channels 0,1,4,5,6 at
          conf >= 0.5, seed 11. That is the binding measurement.
lesson for the methods paper:
          the picture has now moved three times — pessimistic (pooled, unthresholded),
          optimistic (hand-picked high-confidence files), and back to uncertain (random
          sample, then re-weighted by output volume). Each correction came from sampling
          discipline rather than from new data. Worth reporting as such: it is the
          strongest available argument for pre-registered sampling in detector validation.

### 2026-10-07 — Round-0 RESULTS: arm A works, arm B does not
| | arm A (SV2A + RAM_GDNF), job 3800378 | arm B (RAM_GDNF only), job 3801062 |
|---|---|---|
| best val loss | **0.676** | 0.818 |
| event_f1 @ default thr | **0.329** | 0.012 |
| **event_f1 @ best thr (0.9)** | **0.512** | 0.127 |
| precision @ 0.9 | **0.690** | 0.080 |
| recall @ 0.9 | 0.407 | 0.308 |
| predicted vs true events | 1,485 for 371 | **12,070 for 52** |
| training positives | 865 | 184 |

reading:
          Arm B is unusable — 12,070 predictions for 52 true events, 98% recall at 0.6%
          precision, i.e. it learned to fire almost everywhere. **276 in-domain positives
          are not enough to train a U-Net from scratch; ~1,200 are.**
          Arm A is usable: at threshold 0.9, precision 0.69 / recall 0.41 / f1 0.51.
the transferable answer (and it is the inconvenient one):
          **Annotating your own cohort alone is not sufficient at this scale.** A lab cannot
          be handed the pipeline and told to label ~300 events; the requirement is nearer a
          thousand, or they need the base dataset. That is a concrete annotation-cost
          statement, and it is the opposite of the convenient conclusion.
also:     both models select `best_threshold = 0.9`, consistent with confidence being
          informative. Convulsive-specific f1 stays weak for arm A (conv_event_f1 0.154,
          conv_best 0.389).
not comparable to the frozen model:
          arm A's val set spans SV2A + RAM_GDNF animals, so 0.512 cannot be set against
          UNetv2's 0.78 on SV2A alone. Only a common test set can.

### 2026-10-07 — Arm A evaluated on Batch 3 — job 3809197
script:   `scripts/lunarc/detect_ramgdnf_unet.sbatch` @ `770fe29` (env overrides)
config:   `MODEL=ramgdnf_armA_holdB3`, `PATH_INCLUDE=Batch_3_Recordings`, frozen operating
          point otherwise (0.5 / boundary 0.1 / conv 0.45)
outputs:  `~/.eeg_seizure_analyzer/projects/armA_holdB3_on_B3.db`
purpose:  **the comparable number.** Frozen UNetv2 managed **5.2% recall on Batch 3**;
          scoring arm A the same way on the same held-out batch is the paper's positive
          result if it lands near its val recall of 41%.
result:   _pending_

### 2026-10-07 — Mir's convulsive labels verified; the U-Net misses 85% of them
verified: 8 of 8 of Mir's confirmed convulsive seizures inspected by the reviewer are
          **real seizures** (2 calibration events on 2026-10-06, 6 more today). His
          convulsive ground truth is sound, so the miss rate is not a labelling artefact.
scale:    of **148** confirmed convulsive seizures of typical duration (15-60 s) in
          retained recordings, the U-Net caught **22 — 15%**. It is not a convulsive
          detector in this cohort.
morphology (CORRECTED):
          The caught examples are on **UI ch8** (code ch7) and look like textbook
          convulsive seizures; the missed ones are on **UI ch7** (code ch6) and "look a
          little different... but still clear activity". So the model catches the typical
          morphology and misses the atypical — the sensible failure direction.
          An earlier note in this log claimed the reverse; that was a code/UI channel
          mix-up and is withdrawn. **The reviewer always means UI channels (= code + 1).**
Mir's `False` rows contain real activity:
          a 180.29 s block on UI Ch6 was shown labelled `REJECTED` with the convulsive flag
          set, containing several distinct bursts of clear high-amplitude spiking. His
          `False` means "not a convulsive/behavioural seizure" and does not rule out
          non-convulsive events. Arm A used 12,925 such windows as hard negatives.
          Addressed by `NEG_SOURCE=random` (commit `dc49cca`), which ignores rejected
          labels without touching any sidecar — so completed reviews stay intact.
B2 channel exclusion CONFIRMED correct:
          the reviewer's "B2 ch3 and ch4" are UI channels, i.e. code ch2 and ch3 (animals
          450093 and 450094) — exactly what was excluded. Those two carry 5,095 of B2's
          6,168 detections (83%), consistent with them being the noisy ones.

### 2026-10-08 — Precision at conf >= 0.5 on the retained set: ~68%
samples used (unbiased only — the two hand-picked Batch-3 files are EXCLUDED from every
estimate below, since they were chosen for high confidence and read 22/22):

| sample | n | real | precision |
|---|---|---|---|
| random stratified by batch (seed 7) | 53 | 18 | 34% |
| random within B2 retained channels (seed 11) | 34 | 12 | 35% |
| hand-picked B3 (**biased, not used**) | 22 | 22 | 100% |

retained set = all batches except 4, minus B2 UI ch3/ch4 (code ch2/ch3) and animal 449382:

| batch | detections @0.5 | share | reviewed | real | precision | 95% CI |
|---|---|---|---|---|---|---|
| B1 | 160 | 17% | 7 | 6 | 86% | 42–100% |
| B2 | 437 | 47% | 36 | 13 | 36% | 21–54% |
| B3 | 337 | 36% | 6 | 6 | 100% | 54–100% |

**VOLUME-WEIGHTED PRECISION: 68%** over 934 detections. Unweighted over reviewed events
51% (25/49) — the weighted figure is the right one, since it accounts for how much output
each batch actually produces.
**Excluding Batch 2 as well: 95%** over 497 detections. That is the trade: 934 detections
at 68%, or 497 at 95%.

B2 per UI channel (conf >= 0.5):

| UI ch | animal | reviewed | real | precision |
|---|---|---|---|---|
| ch2 | 450917 | 9 | 2 | 22% |
| **ch6** | **450096** | 14 | 9 | **64%** |
| ch7 | 450097 | 11 | 1 | 9% |

**The counterintuitive result worth keeping:** ch6 is animal **450096, the highest
`prominence_db` of all 32 animals (12.9)** — and it has the *best* precision of the three.
So high background rhythmicity lowers precision but does **not** make a recording unusable.
That refutes the premise of the rhythmicity exclusion metric directly, and vindicates
reviewing rather than excluding on the metric. "Noisy" should be reported as a precision
statement per channel, not a binary verdict.

binding uncertainty: B3 is 36% of the retained output with only **6** reviewed events.
caveat on method: an earlier version of this calculation double-counted by pooling samples
with different selection rules and by de-duplicating on non-unique keys. Both fixed; every
figure above uses per-event keys and unbiased samples only.

### 2026-10-08 — Operating-point validation on held-in batches — jobs 3822590-93
why:      every Phase-2 number so far was measured through a **hand-chosen** operating
          point. Arm A's threshold (0.9) came from its own validation but the hysteresis
          **boundary (0.5) was guessed**, and boundary choice has swung coverage from 0.01%
          to 72% of recorded time. So the comparisons were measuring threshold guesses as
          much as models.
design:   `ramgdnf_armA2_holdB3_randneg` (the best arm, event_f1 0.532) over
          **Batch 1 + Batch 2 only** — held-in, never the test fold — at four
          (threshold / boundary) points: **0.5/0.1** (the frozen point), 0.7/0.3, 0.8/0.5,
          0.9/0.5 (what was guessed). Winner applied to Batch 3 **once**.
          `CONV_THRESHOLD` is not varied: Stage 2 only labels and cannot affect detection
          (see `MODELS.md`), so it is inert for every metric here.
scored on: convulsive recall vs Mir's labels, precision vs his adjudicated candidates, and
          **fraction of recordings fired on** — coverage matters as much as recall, because
          the defect is per-recording silence.
caveats:  B1 and B2 were **in training** for every arm, so Mir's labels there were seen and
          absolute performance will be optimistic. That is acceptable for *selecting* an
          operating point — the purpose of validation data — but the chosen point may be
          mis-tuned for unseen data, and that must be stated.
          B1+B2 hold only **69 retained convulsive seizures** (56 + 13), so the ranking is
          coarse: enough to separate 60% coverage from 20%, not to split adjacent
          thresholds.
result:   _pending_

### 2026-10-09 — Stage 2 was training on its own predecessor's labels (26%)
how it surfaced: `conv_rejneg_holdB3` (job 3831585) reported **571** convulsive positives
          where `conv_armA_holdB3` had 451. The refreshed tree explains the rise — but the
          added events include confirmed **U-Net detections**, and
          `build_merged_sidecars.py:187` sets their convulsive flag from `events.type`,
          i.e. **the previous Stage 2's own prediction**.
provenance of every confirmed seizure in `~/train_nomirneg` (2026-10-09):

| `detection_method` | non-conv | convulsive | flag set by |
|---|---|---|---|
| `autocorrelation` (SV2A) | 457 | 204 | human review |
| `mir_candidate` | 70 | **340** | video adjudication — sound |
| **`ml_unet`** | **218** | **194** | **Stage 2 itself — circular** |
| `None` (reviewer's manual) | 52 | 0 | human; all non-convulsive, as stated |

          **CORRECTION (same day, before acting on it): the 26% figure below was WRONG.**
          Splitting the `ml_unet` rows by cohort and date shows most of them are SV2A's
          deliberate active-learning round 2, **hand-reviewed by the reviewer on
          2026-06-15** — legitimate training data, and how `UNetv2` was refined:

| cohort | conv | non-conv | annotated | flag set by |
|---|---|---|---|---|
| SV2A | **189** | 147 | 2026-06 | the reviewer — sound |
| RAM_GDNF | **5** | 71 | 2026-10 | Stage 2 — circular |

          So the genuinely self-labelled convulsive positives are **5 of 738 = 0.7%**, not
          194. **Immaterial.** `conv_temporal` (job 3831663), which ran without the filter,
          is unaffected and needs no re-run on these grounds.
          `convulsive_probability` is NOT a provenance marker — Stage 2 writes it for every
          U-Net detection in any cohort. Only the cohort/date distinguishes them.
          The 71 RAM_GDNF non-convulsive flags remain Stage-2-derived, but they agree with
          the reviewer's own position that this cohort's real events are non-convulsive, so
          they do not pull the classifier anywhere it should not go.
          A blanket `ml_unet` exclusion would have **discarded SV2A's 336 hand-reviewed
          round-2 events** — the lesson is that `detection_method` records who *proposed*
          an event, never who *labelled* it.
why it matters: those events ARE real seizures — a human confirmed them — but the
          *convulsive attribute* was never human-judged. Training on it teaches the
          classifier to reproduce its predecessor's errors, and the reviewer reported on
          2026-10-06 that exactly these detections were **"misclassified as convulsive"**.
          So the error being propagated is a known one.
fix:      `--conv-label-method` / `CONV_LABEL_METHODS` restricts labelled samples to
          chosen `detection_method` values. **Implemented and verified, but left OFF by
          default and NOT recommended**, now that the affected count is 5 events. Kept
          because the mechanism is sound and may matter once RAM_GDNF U-Net detections are
          reviewed in bulk.
          `manual` names rows with no `detection_method` field. Applied to BOTH classes.
          Verified on the SV2A tree: 430 -> 204 convulsive with the filter on (that tree
          holds `autocorrelation` + `ml_unet` only, so the drop is exactly the ml_unet
          share). On the full tree it should give ~544 rather than 738.
consequence: **none — `conv_rejneg_holdB3` (3831585) is NOT materially contaminated** after
          the correction above. Its `Best convulsive F1 0.5628 @ 0.85` is still not
          comparable to `conv_armA_holdB3`'s 0.659, but for a different reason: adding
          rejected negatives changed the task from "given a seizure, convulsive?" to "is
          this a convulsive seizure at all?".
general principle for the paper (the finding survives even though the count did not):
          in an active-learning loop, **model output written back as annotations becomes
          training data for the next generation**, and any attribute the human did not
          actually adjudicate can be laundered into ground truth. Here seizure/not-seizure
          was reviewed and is sound; the convulsive flag on RAM_GDNF U-Net rows was not.
          **A sidecar format that mixes human and model fields needs per-FIELD provenance,
          not per-event** — `detection_method` records who *proposed* an event and is
          routinely mistaken (by me, here) for who *labelled* it. The scale happened to be
          negligible this time only because the RAM_GDNF review queue is barely started;
          at 1,377 files reviewed it would not be.

### 2026-10-09 — Jobs 3831658/3831659 CANCELLED: pre-pull scripts, VAL_MODE ignored
          Submitted with `VAL_MODE=temporal` but the summary lines carried no `val_mode=`
          and no "Split mode" prompt appeared — the checked-out scripts predated
          `3b68d7d`/`dcc7b46`, so the variable was silently ignored and both would have
          trained with the **animal** split. 3831658 was therefore an exact duplicate of
          3831584.
          **Fourth occurrence of this failure mode** (3825994, 3831658, 3831659, and
          3795249's lost `EXCLUDE_ANIMALS`).
guard added — and the first attempt was wrong:
          an "unknown variable" check cannot work: a script cannot recognise a name it has
          never heard of, and filtering env vars to a known prefix list excludes exactly
          the unknown ones. Tested and it let `SOME_FUTURE_MODE=1` straight through.
          Replaced with the invariant that actually covers every case: **both launchers now
          refuse to submit when `$HOME/NED-Net` is behind `origin/main`**
          (`git rev-list --count HEAD..origin/main > 0`), overridable with `ALLOW_STALE=1`
          to reproduce an older run. Verified in both directions.
          This cannot fix a stale checkout retroactively — the old copy has no guard — but
          from the moment it lands, every future setting is protected.

### 2026-10-09 — BATCH IS NOT THE UNIT: per-animal variability dominates
prompted by: reviewer's observation that batches are an artefact of how many animals can be
          recorded at once, not a scientific grouping — another lab might record all animals
          together — so **animal comparisons are the more meaningful unit**.
test:     frozen model (which trained on **none** of these animals), retained B1-B3,
          convulsive ground truth, animals with >=5 events. `review/frozen_per_animal_B1B2B3.csv`

| batch | animal | gt | caught | recall | det |
|---|---|---|---|---|---|
| B1 | 449381 | 14 | 4 | 29% | 34 |
| B1 | 449385 | 10 | 1 | 10% | 792 |
| B1 | 449387 | 5 | 0 | **0%** | 0 |
| B1 | 449388 | 26 | 15 | **58%** | 31 |
| B2 | 450096 | 11 | 4 | 36% | 295 |
| B3 | 459657 | 12 | 1 | 8% | 148 |
| B3 | 459658 | 50 | 0 | **0%** | **82** |
| B3 | 459659 | 25 | 0 | **0%** | 177 |
| B3 | 459661 | 7 | 0 | **0%** | 6 |
| B3 | 459662 | 8 | 2 | 25% | 18 |
| B3 | 459663 | 51 | 5 | 10% | 111 |

RESULT — **within-batch spread exceeds between-batch spread**:
          * within B1: **0% -> 58%**, a **58-point** range across 4 animals
          * between batches (pooled): B1 36.4%, B2 36.4%, B3 5.2% — a **31-point** range
          * per-animal: median **10%**, IQR 0-27%, range 0-58%, **4 of 11 animals at exactly 0%**
consequence — **the "7.3x batch difficulty" framing is WITHDRAWN as a primary finding.**
          B3's apparent difficulty is largely two animals: 459658 and 459663 carry **101 of
          its 153 events** at 0% and 10%. Pooling by batch hides this.
          The defensible finding is **per-animal variability**, which also supports the
          paper's thesis better: performance depends on the individual implant — electrode
          position, local signal character — and batches matter only because that is how
          animals arrive. A lab recording all animals at once would see the same spread.
a failure mode distinct from silence:
          **459658 has 50 ground-truth seizures, fires 82 times, and catches none.** Same for
          459659 (25 events, 177 detections, 0 hits). The detector is *active* on these
          channels and systematically wrong about **where**. That is different from the
          silent channels (449387: 5 events, 0 detections) and should be reported as a
          separate category — "firing but mislocalised" vs "mute".
implications for the experimental design:
          1. **Report per-animal throughout.** Pooled batch numbers are misleading.
          2. **A leave-one-BATCH-out fold confounds held-out-ness with which animals happen
             to be in that batch.** The B1 fold proposed earlier inherits this flaw.
             Prefer a **difficulty-stratified animal holdout** — one zero-recall, one mid,
             one high — so the test set spans the range instead of sampling one end of it.
          3. `split_by_animal` already splits by animal, so an animal-level fold is
             consistent with how internal validation already works.
          4. Budget is not the constraint: **1,778 of 25,000 GPU-h/month used**, so several
             folds are affordable.
caveat:   B2 contributes only one animal with >=5 events, so its pooled 36.4% rests on a
          single implant. The between-batch comparison is weak on that side.

### 2026-10-09 — DECISION: retrain every U-Net arm at the production recipe
why:      see "OPTIMISER ≠ PRODUCTION" under 3831662 below. Frozen-vs-retrained has to
          change the data alone, so the arms are retrained to match `UNetv2_20260615`.
recipe:   **lr 1e-3, batch 8, pos_weight 5, fp32 (no bf16 autocast, TF32 off)**. Weight
          decay 1e-4, dropout 0.2, base_filters 32, depth 4, patience 10 and 50 epochs
          were already identical. Each arm keeps its own neg/pos ratio, because that is
          part of its data design. fp32 needed a new `--fp32` flag (`FP32=1`): on Ampere,
          cuDNN uses TF32 for convolutions by default, so that is switched off as well.
          These are now the defaults in `train_unet.sh`.
arms:     temporal, A3 (hold out B3), all_prod, and Round-0 A, A2 and B (stable split).
          **None of these is an exact repeat.** The training tree was refreshed on
          2026-10-08, so every rerun also picks up the manual and adjudicated labels.
          Only `ramgdnf_temporal` (3831662, old recipe) against its rerun differs in
          the recipe alone, which is why 3831662 was left to finish.
Stage 2 needs NO rerun for the recipe: `Convulsive_v4LUNARC_20260616` was trained on
          LUNARC with the same launcher defaults it has now (lr 3e-4, batch 16, dropout
          0.3, wd 1e-4, 30 epochs, patience 10, CUDA bf16, since AMP landed in 88b1c6f
          an hour before the convulsive trainer did). Its metadata's `pos_weight` 5.0 is an
          unused default: `train_convulsive.py` always uses n_nonconv/n_conv.
consequence: every operating-point sweep and detection on the old-recipe arms has to be
          redone for the new models. Operating points do not transfer between models.

### 2026-10-09 — U-Net, temporal split — job 3831662 (RUNNING)
started:  2026-10-09 14:22:59 on cg12, ~a day ahead of the 2026-10-10 14:57 estimate.
config verified from the log header (the post-3825994 check):
          `val_mode: temporal`, neg/pos 6, pos_weight 6.0, `max_positive_sec: 100`,
          `bg_avoid_rejected: 1`, `hard_neg_exclude: mir_candidate`, data dir
          `train_nomirneg`, exclusions = 449382 + 450093/450094 + all eight B4 animals
          (the retained set). 1,219 EDFs; 42 animals on both sides, no
          single-recording animals.
dataset:  **6,272 train / 3,150 val = 33% in validation**, 9,422 windows total (the ~9,400
          predicted for ratio 6). The 33% is not a config error. `split_by_recording`
          moves each animal's latest recordings into val until >= 20% of its
          *positive* windows are reached, so every animal overshoots by up to one
          whole recording, and that recording's negatives follow it. It is lower than
          the by-animal split's 47%. **Do not compare** it with `conv_temporal`'s 24%:
          that split balances on `center_convulsive` over a different window set.
speed:    **~531 s/epoch** (cache 97 s; bf16), against the 740 s estimate. 50 epochs ≈ 7.4 h,
          well inside the 14 h limit.
early:    event_f1 0.143 / 0.118 / 0.186 / 0.334 / 0.221 over epochs 1-5, a normal
          early swing. **Not comparable** with A3's 0.485, which was scored on a
          different val set (by-animal split).
OPTIMISER ≠ PRODUCTION (checked against `UNetv2_20260615/metadata.json`):
          production was trained on the Mac (MPS, fp32) at **lr 1e-3, batch 8**, which
          are `train_unet.py`'s CLI defaults. Every LUNARC run, including arms A, A2, A3 and
          B and this job, uses **lr 3e-4, batch 32**, the `train_unet.sh` defaults
          introduced in `85ce836` (2026-06-16, the day after production was trained). That
          was a throughput choice and was never compared. The result is ~4x fewer
          optimiser steps per epoch, each at a ~3x lower lr. Architecture, weight decay
          1e-4, dropout 0.2, augment, window 60 s and patience 10 are identical. The
          ratio and pos_weight differ deliberately (production: neg/pos 4, pos_weight 5).
          The arms are internally consistent with each other, but **frozen-vs-retrained
          confounds data with optimiser settings**. Methods must state both recipes.

### 2026-10-09 — Stage 2, temporal split + rejected negatives — job 3831663 (DONE)
script:   `scripts/lunarc/train_convulsive.sh` @ `dcc7b46`
config:   `EDF_DIR=~/train_nomirneg`, `MODEL_NAME=conv_temporal`, **`VAL_MODE=temporal`**,
          `CONV_NEG_FROM_REJECTED=1`, `CONV_NEG_POS_RATIO=5`, `MAX_POSITIVE_SEC=100`,
          `CONV_LABEL_METHODS=` (blank — the provenance concern was corrected to 0.7%,
          immaterial), exclude 355676 only (no batch held out: the temporal split holds out
          each animal's later recordings instead)
dataset:  `convulsive negatives: 723 convulsive positives, 766 non-convulsive seizures,
          2849 rejected events kept (cap 5.0:1)`; train 550 convulsive / 2,405
          non-convulsive; **pos_weight 4.37** (was 0.85 without rejected negatives);
          val convulsive windows 173 = **24%** of 723, matching the designed temporal share
result:   **Best convulsive F1 0.6040 @ threshold 0.75, best epoch 4 of 30.**
          723 positives vs `conv_rejneg_holdB3`'s 571 because Batch 3 is no longer excluded.
not comparable to earlier Stage-2 F1 figures:
          `conv_armA_holdB3` scored 0.659 and `conv_rejneg_holdB3` 0.5628, but all three
          answer different questions — adding rejected negatives changed the task from
          "given a seizure, is it convulsive?" to "is this a convulsive seizure at all?",
          and the split changed from by-animal to temporal. **Do not read 0.604 as a
          regression.**
flag:     **best epoch 4 of 30** — it converged almost immediately on 2,955 training
          windows, which suggests an easy decision boundary. Whether it is the right one is
          exactly what the threshold sweep on real output tests.
THE TEST THIS EXISTS FOR (still pending):
          re-detect B1+B2 with `CONV_MODEL=conv_temporal`, then `conv_threshold_sweep.py`.
          The symptom was precision **flat at 11%** across 0.10-0.45 with the old
          classifier. A **rising** curve means the missing negatives were the cause; still
          flat means they were not, and Stage 2 is written up as a limitation.

### 2026-10-09 — Job 3831584 cancelled; queue notes
          `ramgdnf_all_prod` (animal split) **cancelled while pending**, to free its earlier
          queue slot for `3831662` (`ramgdnf_temporal`) — which moved from an estimated
          2026-10-12 17:01 start to 2026-10-10 14:57. `all_prod` produces no reportable
          number under the settled in-sample scope (it is the final no-holdout production
          model) and can be retrained at any time.
queue diagnosis, so it is not repeated:
          `sprio` on gpua100 showed QOS 60000 + FAIRSHARE 5534 dominating, with only AGE
          differing between our jobs; cutting two pending jobs from 24 h to 14 h moved
          **neither** start estimate. **Wall clock is not a priority lever here.** The
          partition has 6 nodes (1 down), and the five higher-priority jobs ahead were
          blocked on `AssocGrpGRES` — those users at their concurrent-GPU limit, one with a
          4-day job 16 h in. Waiting is node contention. Budget is not the constraint:
          1,778 of 25,000 GPU-h used this month.
training-restart facts (checked in code, 2026-10-09):
          `best_model.pt` **is** written during training whenever validation improves
          (`train.py:666`), so a wall-clock kill leaves a usable model — but **there is no
          resume**: no optimizer or scheduler state is saved and nothing loads a checkpoint
          to continue, so a resubmitted job restarts from epoch 1. Chaining short jobs to
          fit backfill windows would require adding that.

### 2026-10-09 — OUT-OF-SAMPLE RESULT: A3 on held-out Batch 3 — job 3825993
run:      341 files, **1,850 events**, 0 errors, 3,619 s on lu48.
scored:   retained channels, 153 convulsive ground-truth events in Batch 3.
          Frozen is at its production point (0.5 / 0.1), A3 at its validated point
          (0.9 / 0.5) — **each at its own operating point, not a matched one**; state that.

| | det | recall | fired on | med IoU | med dur |
|---|---|---|---|---|---|
| frozen, Stage 1 | 559 | **5.2%** | 58% | 0.40 | 9 s |
| **A3, Stage 1** | 1,850 | **8.5%** | **90%** | 0.39 | 11 s |
| frozen, cascade | 253 | **2.6%** | 31% | 0.49 | 8 s |
| **A3, cascade** | 1,726 | **7.8%** | **88%** | 0.39 | 10 s |

retraining DOES beat frozen out-of-sample, modestly:
          **1.6x** on Stage 1 (5.2 -> 8.5%), **3x** on the cascade (2.6 -> 7.8%), and
          coverage 58 -> 90%. But **absolute recall stays under 10%**.
the boundary improvement does NOT transfer:
          median IoU 0.36 -> 0.60 in-sample; **0.40 -> 0.39** here. So the better event
          delineation seen on B1+B2 was fitted, not learned. Report it as an in-sample
          property only.
**CORRECTED FRAMING — the first version of this entry was wrong.**
It read the 38.2% -> 5.2% drop as the transfer penalty. It is mostly **not** transfer:

| | frozen (saw NEITHER) | A3 | A3 / frozen |
|---|---|---|---|
| B1+B2, n=68 (A3 trained on it) | 38.2% | 50.0% | **1.31x** |
| B3, n=153 (held out) | 5.2% | 8.5% | **1.63x** |

          The frozen model trained on **neither** batch, yet scores **7.3x** better on
          B1+B2 than on B3. That spread is therefore **pure batch difficulty**, and it
          accounts for most of the apparent in-sample/out-of-sample gap.
          Retraining's **relative** gain — the part not confounded by difficulty — is
          **1.31x in-sample and 1.63x out-of-sample**, i.e. *larger* out-of-sample. So the
          in-sample figure is **not** inflated by memorisation the way it first appeared.
WHAT CAN BE CLAIMED, in order of how well supported it is:
          1. **Batch difficulty dominates.** A **7.3x** spread in detectability between
             batches of one cohort, one lab, one rig, measured by a model that trained on
             none of them. Unconfounded, and the most direct evidence for the paper's
             thesis that electrode position, noise and rig drift change what a detector
             sees. **This is the strongest finding in the project.**
          2. **Retraining gives a consistent ~1.3-1.6x relative recall gain** plus a large
             coverage gain (16 -> 73% in-sample, 58 -> 90% out-of-sample), at both
             difficulty levels. Modest, real, reproducible.
          3. **The transfer penalty is NOT measured.** B3 is both held-out and hard, so the
             two cannot be separated from this fold. **A LOCO fold holding out B1 or B2 —
             an easy batch — is required**, not optional, for the central claim. Until then
             no number should be quoted as the cost of not annotating a batch.
Stage-2 sweep on B3 (free, post-hoc): precision **flat at 4%** across 0.10-0.45, same
          pathology as B1+B2 — consistent with the missing-negatives diagnosis.

### 2026-10-09 — Job 3825989 TIMED OUT; 3825994 ran stale code
3825989 (`ramgdnf_all_prod`):
          **TIMEOUT at 06:00:21**, killed at epoch 18 of 50 with `event_f1` **still rising**
          (0.425 ep7 -> 0.428 ep13 -> **0.468** ep18; A3's best was 0.485). At
          **1,165 s/epoch** 50 epochs is ~16 h.
          Wall clock raised to **24 h** and made overridable via `WALL_TIME` (the self-submit
          now passes `-t`, which beats the `#SBATCH` directive). Same for
          `train_convulsive.sh` (2 h -> 6 h).
          `background sampling: dropped 1 of 11999 draws` — Mir's rejected regions barely
          constrain the sampling, so the effective ratio held.
why it was so slow — a split pathology worth knowing:
          the split came out **7,870 train / 6,935 val = 47% in validation**, not 20%.
          `split_by_animal` balances the convulsive stratum by **convulsive-window count**
          (`_n_conv`, `dataset.py:838`) while `stable_convulsive_val` fills it
          smallest-first — so an animal with 1 convulsive window but hundreds of
          *background* windows counts as "small", is picked early, and drags all its
          background into val. 18 animals carried 6,935 windows.
          **Deliberately NOT fixed**: arms A/A2/A3 used this behaviour, and changing val
          composition would invalidate the `best_event_f1` 0.485 reference. Fix the cost
          instead, via the ratio (below).
`NEG_POS_RATIO` lowered 10 -> 6 for the rerun:
          10 made sense when the hard-negative pool capped the effective ratio at 6.2 (arm
          A). With background top-up, 10 means literally 10 — 13,460 negatives, 14,805
          windows. **6 matches arm A's effective ratio and roughly halves the compute**
          (~9,400 windows, ~740 s/epoch, ~10 h for 50 epochs). Better comparability and
          cheaper.
3825994 (`conv_rejneg_holdB3`): **ran the pre-pull script.** `Train: 451 convulsive / 383
          non-convulsive` is identical to `conv_armA_holdB3`, there is no
          `convulsive negatives:` line, and `MAX_POSITIVE_SEC=100` did not drop B1's 11
          over-100 s events (451 would have become 440). The old script accepts the new env
          vars silently and trains the old way, so it produced a plausible model
          (`Best convulsive F1 0.6745 @ 0.60`) rather than an error. **Discard it and
          resubmit after `git pull`.**
LESSON (third silent-config failure in two days, after the stale tree and the bare `read`):
          **`git pull` on LUNARC is part of the submission, not a precondition to assume.**
          The scripts take env vars they may not understand yet. Always confirm the
          submission summary line shows the new settings — here it would have shown
          `neg_from_rejected=` only after the pull.

### 2026-10-08 — Stage 2 retrained WITH rejected negatives — job 3825994
script:   `scripts/lunarc/train_convulsive.sh` @ `d04001a`
config:   `EDF_DIR=~/train_nomirneg`, `MODEL_NAME=conv_rejneg_holdB3`,
          **`CONV_NEG_FROM_REJECTED=1`**, `CONV_NEG_POS_RATIO=5`, `CONV_NEG_METHODS=`
          (blank = every rejected event), `MAX_POSITIVE_SEC=100`;
          exclude 355676 + Batch 3 (459657-459664) so it is comparable with A3
why:      Stage 2 was trained on *confirmed seizures only* yet is deployed on every
          detection, most of which are false positives — so it had never seen a
          non-seizure. For the frozen classifier that failed safely (RAM_GDNF noise was
          unfamiliar, scored low, precision 33% -> 61%); after retraining on 368 of Mir's
          RAM_GDNF convulsive seizures the same noise resembles them and **precision is
          flat at 11% at every threshold**. Negatives fix that; tuning cannot.
          Source: Mir's rejected rows — video-adjudicated "not a convulsive/behavioural
          seizure", **contaminated as DETECTION negatives and exactly right here**
          (`build_merged_sidecars.py:76`, written a week earlier and not acted on).
expected dataset (verify in the log):
          `convulsive negatives: ~700 convulsive positives, ~740 non-convulsive seizures,
          ~2800 rejected events kept (cap 5.0:1)`. **0 rejected = the flag did not take;
          ~12,800 = the cap did not.**
THE TEST, and it is a clean one:
          re-detect B1+B2 with `CONV_MODEL=conv_rejneg_holdB3` (cheap, lu48) then run
          `conv_threshold_sweep.py`. The symptom was precision **flat at 11%** across
          0.10-0.45. The fix is a curve that **rises**. If it is still flat, the missing
          negatives were not the cause and the Stage-2 line should be written up as a
          limitation rather than pursued.
result:   _pending_

### 2026-10-08 — A3 on held-out Batch 3, validated operating point — job 3825993
script:   `scripts/lunarc/detect_ramgdnf_unet.sbatch` @ `466c4d3`
config:   `MODEL=ramgdnf_armA3_holdB3`, `CONV_MODEL=conv_armA_holdB3`,
          **THRESHOLD=0.9 / BOUNDARY=0.5** (selected on held-in B1+B2, see the sweep entry),
          `CONV_THRESHOLD=0.3`, `PATH_INCLUDE=Batch_3_Recordings`
outputs:  `~/.eeg_seizure_analyzer/projects/armA3_on_B3_t0.9_b0.5.db`
purpose:  **the out-of-sample number.** Batch 3 was held out of A3's training. Frozen
          managed **5.2%** convulsive recall there. In-sample A3 reaches 47.1% recall /
          73% coverage at a tuned Stage 2, so B3 gives the gap between "annotate this
          cohort" and "transfer to the next batch" — the quantity the paper's thesis rests on.
note:     `CONV_THRESHOLD` does not bind — `convulsive_confidence` is stored per event, so
          any Stage-2 threshold can be re-scored post-hoc with `conv_threshold_sweep.py`.
          Only `THRESHOLD` and `BOUNDARY_THRESHOLD` are fixed at detection time.
result:   _pending_

### 2026-10-08 — PLAN: retrain Stage 2 with Mir's rejections as negatives
the finding that motivates it:
          `build_merged_sidecars.py:76` already records the asymmetry and we never acted on
          it — Mir's `False` rows are **"Valid as convulsive-CLASSIFIER negatives,
          contaminated as DETECTION negatives."** They are video-adjudicated "not a
          convulsive/behavioural seizure" labels: wrong for the detector, *exactly right*
          for Stage 2. **11,415 of them, in the deployment domain.**
          Note the symmetry with the Stage-1 work: the rows just excluded from the
          detector's negatives via `--hard-neg-exclude-method mir_candidate` are the rows
          Stage 2 should train on. Same data, opposite role, because the stages answer
          different questions.
what it fixes:
          1. **train/inference mismatch** — `build_convulsive_window_specs` emits one window
             per *confirmed* seizure and ignores rejected rows, so Stage 2 has never seen a
             non-seizure, yet it is applied to every detection (mostly false positives).
          2. **domain gap** — its only negatives today are SV2A non-convulsive seizures.
          3. **label quality** — the convulsive question can only be answered on video, and
             these were.
          Together these explain why precision is flat at 11% for A3 at every Stage-2
          threshold: no threshold can separate classes the classifier was never shown.
design decisions (TO CONFIRM):
          * **Stage 2's question changes** from "given a seizure, is it convulsive?" to
            "is this detection a convulsive seizure at all?", since the negative class would
            then hold noise as well as non-convulsive seizures. That is what deployment asks
            and what would restore the precision contribution. Stage 2 would still only
            label, never filter — but the label becomes usable as a filter, which
            `MODELS.md`'s "READ THIS FIRST" section will need updating to reflect.
          * **negative cap**: positives ~783 convulsive after the 100 s cap (430 SV2A +
            353 Mir). Available negatives 768 non-convulsive confirmed + ~11,415 Mir
            rejections + ~1,382 SV2A rejections = ~13,565, i.e. 1:17. `pos_weight`
            auto-compensates (`train_convulsive.py:204`) but recommend capping at ~5:1 —
            keep every non-convulsive *seizure* and sample rejections to fill — so the
            convulsive-vs-non-convulsive-seizure boundary is not swamped by easy noise.
implementation:
          `build_convulsive_window_specs` gains a rejected-negative option (with a
          `detection_method` filter and a ratio cap), plumbed through `train_convulsive.py`
          and `scripts/lunarc/train_convulsive.sh`. `max_positive_sec` already applies there.
validation:
          hold out Batch 3 again so the Stage-2 result is comparable with A3's, and ensure
          the val split contains both RAM_GDNF convulsive events and RAM_GDNF rejections —
          otherwise the metric cannot see the thing being fixed.

### 2026-10-08 — Stage-2 threshold was mis-set after retraining (no compute needed)
script:   `scripts/local/conv_threshold_sweep.py` (new)
method:   `events.convulsive_confidence` is stored per event and Stage 2 only **labels**,
          never filters (`_apply_convulsive_classifier`, `predict.py:242`). So the whole
          CONV_THRESHOLD sweep is arithmetic on an existing DB — **no SLURM job, no
          re-detection.** Re-deriving `type` from the stored probability reproduces exactly
          what a re-run at that threshold would give.
B1+B2 retained, convulsive ground truth (n=68):

**A3 @ thr 0.9 / bnd 0.5 — Stage 2 is pure loss at the inherited threshold**

| conv_thr | det | recall | precision | fired on |
|---|---|---|---|---|
| none (Stage 1) | 3,692 | **50.0%** | 10% | 77% |
| 0.10 | 3,593 | **50.0%** | 11% | 75% |
| **0.30 (recommended)** | 3,224 | **47.1%** | 11% | 73% |
| 0.45 (as deployed) | 2,868 | 38.2% | 11% | 71% |
| 0.60 | 2,233 | 35.3% | 13% | 67% |
| 0.80 | 1,044 | 29.4% | 16% | 54% |
| 0.95 | 672 | 20.6% | 17% | 36% |

**frozen UNetv2 + `Convulsive_v4LUNARC` — Stage 2 earns its place**

| conv_thr | det | recall | precision | fired on |
|---|---|---|---|---|
| none (Stage 1) | 1,844 | 38.2% | 33% | 51% |
| 0.20 | 623 | 35.3% | 52% | 35% |
| 0.45 | 189 | 27.9% | **61%** | 16% |
| 0.80 | 64 | 20.6% | **81%** | 5% |

headline: going 0.30 -> 0.45 on A3 costs **6 of 32 correct detections and buys zero
          precision** (11% -> 11%). The earlier "Stage 2 destroys ~25% of correct
          detections" finding is real but is a **mis-set threshold, not an intrinsic
          defect** — most of it is recoverable for free at 0.30.
mechanism — CORRECTED (the first explanation written here was wrong):
          `convulsive_confidence` p50 is **0.12** for the frozen classifier and **0.66**
          for the retrained one. That is **not** a class-imbalance artefact:
          `train_convulsive.py:204` sets `pos_weight = n_nonconv / n_conv` automatically,
          so the loss was balanced in both runs (SV2A alone 430 conv / 706 non-conv ->
          pos_weight 1.64; `conv_armA_holdB3` 451 / 383 -> 0.85). The composition flipped
          from 38% to 54% convulsive, but the weighting compensates for exactly that.
          The real cause is **domain familiarity**, as `MODELS.md:102-104` already stated:
          the frozen Stage 2 had never seen a RAM_GDNF convulsive seizure; `conv_armA_holdB3`
          has seen 368 of them. So RAM_GDNF activity now scores high — including RAM_GDNF
          activity that is noise.
          **The deeper defect is a train/inference mismatch.**
          `build_convulsive_window_specs` emits one window per *confirmed* seizure and
          explicitly ignores rejected rows ("rejection is the detector's job"), so Stage 2
          **has never seen a non-seizure** — yet it is applied to every U-Net detection,
          most of which are false positives. For the frozen classifier this failed safely:
          RAM_GDNF noise was unfamiliar, scored low, and precision rose 33% -> 61%. For the
          retrained classifier the same noise resembles the RAM_GDNF convulsive seizures it
          just learned, so it scores high and **precision is flat at 11% at every
          threshold**. No threshold can fix that; it needs negatives.
note:     the documented operating point for `conv_armA_holdB3` is **0.55** (`MODELS.md:115`),
          not the 0.45 inherited from the frozen pipeline. At 0.60 the sweep gives recall
          35.3% / precision 13% — still worse than 0.30 on both axes for this classifier.
RULE: **re-tune CONV_THRESHOLD after every Stage-2 retrain; never inherit it.** It is free
          to sweep post-hoc, so detection runs can use any value and be re-scored later.
this settles the reporting scope, on evidence rather than preference:
          Stage 2 is an **informative** type filter for the frozen model and **not** for A3
          (precision flat at 11% across 0.10-0.45). Filtering A3 by it therefore does not
          achieve type-matching against Mir's convulsive reference — it deletes detections
          near-randomly. **Report Stage 1 as detection capability for both models, and
          Stage 2 separately with this asymmetry stated.** The cascade tables above remain
          valid as "the pipeline as deployed", which is a different and also reportable thing.
also fixes the headline comparison:
          at a properly tuned Stage 2 (A3 @ 0.30 vs frozen @ 0.45), recall is
          **47.1% vs 27.9%** and coverage **73% vs 16%** — a larger gap than the 38.2% vs
          27.9% reported from the inherited threshold.

### 2026-10-08 — RESULT: retraining beats the frozen model IN-SAMPLE — jobs 3823858-61
script:   `scripts/local/operating_point_table.py` (new), `scripts/lunarc/detect_ramgdnf_unet.sbatch`
inputs:   `ramgdnf_armA3_holdB3` over **Batch 1 + Batch 2** (held IN, never the test fold)
          at four (threshold / boundary) points; frozen `UNetv2_20260615` through the
          **identical** code path for the paired baseline.
          Retained channels only (no B4, no B2 UI ch3/ch4, no 449382). Convulsive ground
          truth, n = **68** for every row, so the comparison is properly paired.
outputs:  `~/.eeg_seizure_analyzer/projects/armA3_tune_t{0.5,0.7,0.8,0.9}_b{0.1,0.3,0.5,0.5}.db`

Ground truth is **convulsive only** (Mir's reference). Two detection scopes, because
Stage 2 only labels and never filters, so they answer different questions.

**PRIMARY — full cascade** (detection must also be labelled convulsive by Stage 2). This is
the deployed pipeline and the type-matched comparison against a convulsive reference:

| | det | recall | precision | fired on | med IoU | med det dur |
|---|---|---|---|---|---|---|
| **frozen, full cascade** | 189 | **27.9%** | **61%** | **16%** | 0.39 | 14 s |
| A3 thr 0.5 / bnd 0.1 | 26,042 | 50.0% | 2% | 100% | 0.78 | 27 s |
| A3 thr 0.7 / bnd 0.3 | 8,652 | 42.6% | 4% | 96% | 0.71 | 30 s |
| A3 thr 0.8 / bnd 0.5 | 4,553 | 39.7% | 8% | 83% | 0.67 | 24 s |
| **A3 thr 0.9 / bnd 0.5 (SELECTED)** | 2,865 | **38.2%** | 11% | **71%** | 0.66 | 26 s |

**SECONDARY — Stage 1 alone** (all detections, whatever Stage 2 called them), which shows
where the loss sits:

| | det | recall | precision | fired on | med IoU | med det dur |
|---|---|---|---|---|---|---|
| **frozen UNetv2** | 1,844 | **38.2%** | 33% | **51%** | **0.36** | **10 s** |
| A3 thr 0.5 / bnd 0.1 | 32,713 | 72.1% | 2% | 100% | 0.66 | 13 s |
| A3 thr 0.7 / bnd 0.3 | 11,448 | 60.3% | 4% | 97% | 0.58 | 13 s |
| A3 thr 0.8 / bnd 0.5 | 5,911 | 52.9% | 7% | 87% | 0.60 | 17 s |
| **A3 thr 0.9 / bnd 0.5 (SELECTED)** | 3,692 | **50.0%** | 10% | **77%** | **0.60** | **19 s** |

THE FROZEN MODEL'S SILENCE IS WORSE IN THE CASCADE VIEW:
          with Stage 2 applied it fires on just **16% of recordings** — 189 convulsive
          detections across all of retained B1+B2. Precise when it speaks (61%) and almost
          mute. A3 at the strict point covers **71%**, a **4.4x** increase. This is the
          strongest single statement in the comparison and it survives both scopes.
STAGE 2 BEHAVES DIFFERENTLY BETWEEN THE TWO MODELS:
          it keeps **10%** of the frozen model's detections as convulsive (189/1,844) but
          **78%** of A3's (2,865/3,692) — `conv_armA_holdB3` was retrained on Mir's
          convulsive events and now recognises this cohort's morphology. Retraining Stage 2
          changed the cascade's character as much as retraining Stage 1, which is a
          reportable finding in itself (and the reason the Stage-2 retrain was not optional).
          Consistent with the earlier 22/34 = 65% figure; here 19/26 = 73%.
NOT COMPARABLE TO THE 15.3% FIGURE:
          "frozen 15.3% convulsive recall" in `PHASE1_RESULTS.md` uses a different
          denominator — 148 events of 15-60 s across all retained batches — against the 68
          here (retained B1+B2, any duration). Both correct; distinguish them in the text.

operating point selected: **thr 0.9 / bnd 0.5.** The trade is monotone with no knee, so take
          the conservative end: it still beats frozen on recall and coverage, keeps the
          output reviewable, and has the best precision and boundaries.
coverage is the unconfounded win:
          per-recording **silence** was the defect — the frozen model fired on only 51% of
          reviewed recordings, so half the data had no detector at all. A3 fires on 77% at
          the strictest point. This does not depend on the precision metric.
truncation resolves, and it was NOT a threshold artefact:
          median IoU 0.36 -> 0.60, matched duration 10 s -> 19 s against a ~37 s
          ground-truth median. Within A3 it barely moves across boundary 0.1/0.3/0.5
          (0.66/0.58/0.60), so the hysteresis boundary was **not** causing the 9 s
          fragments — the retraining fixed them. Answers the open question from the
          match-rule entry above.
precision is CONFOUNDED, do not report 33% -> 10% as a regression:
          it is measured against Mir's convulsive candidates, which structurally cannot
          credit a non-convulsive detection. A3 inherited SV2A's non-convulsive class and
          so produces more of exactly what this metric cannot score. Unresolvable without
          review sampling.

**PER-ANIMAL PAIRED COMPARISON** — `review/frozen_vs_A3_per_animal_B1B2.csv`

| batch | animal | UI ch | gt | frozen | % | det | A3 | % | det |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 449381 | 1 | 14 | 4 | 29 | 34 | 6 | 43 | 54 |
| B1 | 449385 | 5 | 10 | 1 | 10 | 792 | 4 | 40 | 729 |
| B1 | **449387** | 7 | 5 | 0 | **0** | **0** | 1 | 20 | 118 |
| B1 | 449388 | 8 | 26 | 15 | 58 | 31 | 18 | 69 | 69 |
| B2 | 450095 | 5 | 1 | 1 | 100 | 2 | 1 | 100 | 117 |
| B2 | 450096 | 6 | 11 | 4 | 36 | 295 | 4 | 36 | 685 |
| B2 | 450097 | 7 | 1 | 1 | 100 | 376 | 0 | 0 | 138 |
| | **total** | | **68** | **26** | **38.2** | 1,530 | **34** | **50.0** | 1,910 |

          4 improved, 2 unchanged, 1 worse. Three specifics carry more weight than the total:
          * **449387 was wholly silent — 0 detections over 5 seizures — and now fires.**
            The starkest single failure in Phase 1. Not fixed (1/5), but a detector now
            exists on that channel.
          * **449385 improved while firing LESS**: 792 -> 729 detections, 1 -> 4 caught. The
            frozen model's 792 were essentially noise. This is the cleanest evidence the
            gain is real and not a volume effect.
          * **efficiency is better than the batch table implies**: on the 7 channels that
            actually contain convulsive seizures, +25% detections buys +31% recall
            (1,530 -> 1,910 det, 26 -> 34 hits). The "2x detections" reading came from
            counting all retained channels; A3's extra output sits on channels with **no**
            convulsive ground truth, which is either non-convulsive events or false
            positives and Mir's reference cannot tell them apart.
          The single regression (450097, 100% -> 0%) is one seizure — noise.
**PER-ANIMAL, FULL CASCADE** (primary scope) — `review/frozen_vs_A3_per_animal_B1B2_cascade.csv`

| batch | animal | UI ch | gt | frozen | % | det | A3 | % | det |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 449381 | 1 | 14 | 1 | 7 | 6 | 3 | 21 | 23 |
| B1 | 449385 | 5 | 10 | 0 | 0 | 25 | **0** | **0** | **569** |
| B1 | 449387 | 7 | 5 | 0 | 0 | 0 | **0** | **0** | 117 |
| B1 | 449388 | 8 | 26 | 15 | 58 | 28 | 18 | 69 | 38 |
| B2 | 450095 | 5 | 1 | 1 | 100 | 1 | 1 | 100 | 71 |
| B2 | 450096 | 6 | 11 | 2 | 18 | 52 | 4 | 36 | 465 |
| B2 | 450097 | 7 | 1 | 0 | 0 | 34 | 0 | 0 | 105 |
| | **total** | | **68** | **19** | **27.9** | **146** | **26** | **38.2** | **1,388** |

          3 improved, 4 unchanged, **0 worse** — the single Stage-1 regression disappears.
          But two findings cut the other way:
STAGE 2 DESTROYS CORRECT DETECTIONS IN BOTH MODELS:
          A3 caught 4/10 on 449385 at Stage 1 and **0/10** after Stage 2; 1/5 -> 0/5 on
          449387. Overall Stage 2 takes A3 from 34 hits to 26 and the frozen model from 26
          to 19 — **both lose ~25% of their correct detections.** Consistent with the
          earlier 22/34 = 65%. This argues for reporting **Stage 1 as the detector's
          performance** and treating Stage 2 as a separate classifier with its own measured
          cost, rather than collapsing them into one number.
THE EFFICIENCY CLAIM DOES NOT SURVIVE THE CASCADE VIEW — **RETRACTED**:
          on seizure-bearing channels frozen uses **146** convulsive detections and A3 uses
          **1,388** — **9.5x**, not the "+25% detections for +31% recall" computed from
          Stage 1 alone. Stage 2 keeps 10% of frozen's output but 78% of A3's. So as
          deployed, A3 buys +10 points of recall for roughly an order of magnitude more
          output to review. Whether that is a good trade depends on true precision, which
          Mir's convulsive-only reference cannot measure — and **569 detections catching 0
          of 10 seizures on 449385** is not reassuring.
          The Stage-1 efficiency figure is correct for Stage 1 and is retained above as
          such; it must not be quoted as a property of the pipeline.

CAVEATS that must travel with this result:
          * **IN-SAMPLE.** B1 and B2 were in A3's training, so Mir's labels there were seen.
            This is the "what you get after annotating the cohort" number — the direct
            analogue of SV2A's 93.6% — **not** a generalisation claim. The out-of-sample
            claim stays with A3-on-held-out-B3.
          * n = 7 channels / 68 events. Enough to separate 0% from 20%, not to rank
            adjacent animals.
          * A3 trained on the **stale tree** (see the defect entry below), so it has none of
            the 52 manual additions or 344 adjudicated detections. `ramgdnf_all_prod`
            should do better still, and its operating point must be re-swept rather than
            inheriting 0.9/0.5 blindly.
what this does for the paper:
          **This is the first retrained arm to beat the frozen model on anything.** It makes
          the paper's central claim concrete and quantified: a model that fails on a new
          cohort (15.3% convulsive recall, firing on half the recordings) recovers to 50%
          recall and 77% coverage once that cohort is annotated and folded back in — using
          the same UI and pipeline, with no code changes.

### 2026-10-08 — Two training-set corrections (reviewer-requested)
request:  (1) exclude Mir's very long events — "everything above 100 s" — from the new
          training; (2) sample random background only from areas not already annotated
          as rejected.

**(1) Positive-duration cap** — `--max-positive-sec` / `MAX_POSITIVE_SEC`.
rationale:
          Mir's long "convulsive" rows are chained seizures annotated as one block
          (reviewer: "real convulsive seizures are never minutes long"); a 150-240 s block
          centred in a 60 s window fills it edge to edge, so per-channel z-scoring leaves
          no event-vs-baseline contrast. Measured: **0 of 13 events over 120 s were ever
          detected**.
cost, at 100 s:
          drops **15 of Mir's 430** confirmed seizures (3.5%) — all convulsive, 11 in B1
          and 4 in B3. Kept convulsive events run p50 21 s / p75 37 s / p95 66 s.
          **Drops 0 of SV2A's 1,136** — their longest human-reviewed event is **84 s**.
          That SV2A distribution is independent corroboration that Mir's >100 s rows are
          chains, and is the justification to cite rather than convenience.
          Capped events are NOT erased: they stay in the interval lists, so a window
          overlapping one is still labelled seizure and background is never drawn from it.
          Only the "centre a positive window on it" step is skipped. Applied to the
          convulsive classifier path too, for the same reason.

**(2) Background must avoid annotated regions** — `--bg-avoid-rejected` / `BG_AVOID_REJECTED`,
plus `--hard-neg-exclude-method` / `HARD_NEG_EXCLUDE_METHODS`.
the blocker this exposed:
          `refresh_training_tree.py` **deleted** Mir's rejected rows, so the tree held no
          record of those regions and background sampling could not avoid them — it drew
          background from exactly the regions known to contain real non-convulsive
          activity (2026-10-07, the 180 s block of clear bursting on UI Ch6).
          Those rows have **two** uses and deletion served only the first:
            1. as hard negatives they are WRONG (`rejected` = "not convulsive", not "not a
               seizure") and must be excluded — deletion achieves this;
            2. as regions to keep background out of they are RIGHT — deletion destroys this.
fix:      the exclusion moved into the dataset builder, keyed on `detection_method`, so
          both uses can be honoured at once. `refresh_training_tree.py` gains
          `--keep-mir-rejections` (now the preferred mode) and the rows stay in the tree.
verified (25 SV2A files, rejections tagged `mir_candidate` in memory only):

| configuration | negatives landing on a rejected region |
|---|---|
| baseline — rejections as hard negatives | 34 |
| **deletion alone (what the tree did)** | **23** |
| exclude as negatives + block background | **0** |

          The 34 comprises 11 hard negatives legitimately centred on rejected events plus
          23 background windows that landed there by chance. Deletion removes the 11 and
          leaves the 23 — the silent failure. The combination removes the 23.
also:     when 50 sampling attempts cannot place a clean window on a channel, the draw is
          now **dropped** rather than emitted anyway (it would have labelled a known event
          as background), and the count is printed.
cap verification: positives 1,136 -> 749 at a 20 s cap on the SV2A tree, hard negatives
          unchanged — so the cap bites and bites only the intended pool.
effect on dataset size: keeping Mir's rejections restores the tree's annotated-file count
          from 404 to ~1,364. The hard-negative pool is unchanged (~1,461, still SV2A plus
          our adjudicated U-Net rows), but many more channels now contribute background
          context, so the ~12,149 background draws come from a more diverse pool.

### 2026-10-08 — Match-rule sensitivity: recall is robust, boundaries are not
question (reviewer): does the scoring actually check that two events are the same event —
          overlapping boundaries — or only compare counts?
answer:   overlap, keyed on `(file_stem, channel)` after the Mir-`k` -> code-`k-1` mapping:
          `det_start < gt_end + 5` and `det_end > gt_start - 5`
          (`validate_frozen_unet_vs_mir.py:67` for recall, `:84` for precision). Never counts.
script:   `scripts/local/matching_sensitivity.py` (new) — recall under five rules.
frozen model, 340 convulsive ground-truth events on reviewed recordings:

| rule | recall |
|---|---|
| any overlap, +-5 s (as reported) | 36/340 = **10.6%** |
| any overlap, no slack | 36/340 = 10.6% |
| IoU >= 0.20 | 27/340 = 7.9% |
| detection covers >=50% of the seizure | 10/340 = 2.9% |
| any overlap, **one-to-one** (greedy by overlap) | 36/340 = 10.6% |

two worries tested and DISMISSED:
          * the **5 s slack inflates nothing** — identical with and without it.
          * the **flood does not buy cheap hits** — one-to-one matching gives the same 36,
            so no ground-truth event was claimed by multiple detections. Inflation 1.00x.
            (This had been a live concern: animal 449385 fired 792 times for 1 of 10.)
          So **recall at 10.6% / 15.3% is robust** and event-level any-overlap is the right
          primary criterion, as in the seizure-detection literature.
by ground-truth duration (not a block artefact):

| gt duration | n | loose | IoU >= 0.2 | covers >=50% |
|---|---|---|---|---|
| < 15 s | 118 | 5.9% | 5.9% | 4.2% |
| **15-60 s (typical)** | 184 | **13.0%** | **9.2%** | 2.7% |
| 60-120 s | 25 | 20.0% | 12.0% | 0.0% |
| > 120 s (chains) | 13 | 0.0% | 0.0% | 0.0% |

          Median IoU of a hit on typical events is **0.32**, so the weak overlap is real and
          not an artefact of Mir's long block annotations.
WHAT THE DETECTIONS LOOK LIKE (36 matched pairs):

| | median | IQR |
|---|---|---|
| ground-truth duration | 37.4 s | 20.5-52.8 |
| **detection duration** | **8.9 s** | 7.1-14.3 |
| onset error (det - gt) | **+11.9 s** | 1.3-21.2 |
| offset error (det - gt) | -4.7 s | -13.4 to -1.4 |

          **97%** of matched detections are shorter than the seizure; median **1** detection
          per event (max 3), so this is truncation, not fragmentation. When the frozen model
          fires it marks a ~9 s fragment beginning ~12 s into a ~37 s seizure — barely above
          the 5 s `min_duration` floor.
consequences:
          1. Report event-level recall as primary, with the IoU row as sensitivity. Do not
             report boundary agreement as if it were measured by the loose rule.
          2. **Any duration or burden metric derived from detected extents is short by ~4x.**
             Not quoted anywhere yet, but the SV2A manuscript's duration figures come from
             the same pipeline *in-domain* — check whether that cohort truncates too before
             relying on them. See [[sv2a-figure-numbers-provenance]].
          3. The running operating-point sweep (3823858-61) varies the hysteresis **boundary**
             (0.1/0.3/0.5), which is precisely what controls event extension. Add IoU and
             matched-duration columns to the selection table: if truncation is a threshold
             artefact the low-boundary arm gives longer events and higher IoU; if detections
             stay ~9 s regardless, it is a sensitivity limit of the model.

### 2026-10-08 — DEFECT: every retraining arm trained on a stale label tree
found:    before launching the all-batches run, compared `~/train_nomirneg` against the
          live sidecars for Batch 3:

| | confirmed | from Mir | manual | U-Net confirmed |
|---|---|---|---|---|
| LIVE B3 | 279 | 171 | **52** | **41** |
| TREE B3 | 154 | 154 | **0** | **0** |

          Only **240 of 338** B3 sidecars were present in the tree at all, and across the
          whole cohort the tree held 1,024 real sidecars where the live tree has 1,364.
          `refresh_training_tree.py` had been run once, before the review sessions, and
          never again.
consequence — this revises the Phase-2 results:
          **arms A, A2, A3 and B all trained on Mir's positives only.** None of them saw
          the 52 manual additions or the 344 adjudicated U-Net detections. So the
          retraining campaign never included the label class Mir's reference structurally
          lacks — RAM_GDNF **non-convulsive** events — which is precisely the class the
          frozen model was best at finding (2026-10-06 Batch-3 review; 2026-10-08
          precision analysis).
          That is a plausible mechanical reason the arms barely moved against the frozen
          model, and it is a *fixable* one. It does not invalidate the arm results: they
          remain valid measurements of "retraining on an external convulsive reference".
          They simply do not test "retraining on your own review work", which is the
          pipeline claim the paper makes.
          Related asymmetry already on record: Mir's positives were 276/1,215 = 23% of the
          U-Net's positive windows but 215/451 = 48% of the convulsive classifier's.
action:   job 3825957 **CANCELLED** before it ran. Tree refreshed (1,364 sidecars, 11,415
          of Mir's rejected rows dropped, 52 manual + 344 adjudicated U-Net kept), then
          resubmitted.
lesson:   `train_unet.py` takes only `--data-dir` and re-scans the folder, so it cannot be
          pointed at a frozen dataset definition and silently trains on whatever is in the
          tree. **Run `refresh_training_tree.py --dry-run` and check the manual/adjudicated
          counts immediately before every submission**, and record those counts in the log
          entry for that job. The refresh is the `review -> retrain` link in the loop and
          is the step that fails silently.

### 2026-10-08 — All-batches production retrain (no hold-out) — job 3825989
script:   `scripts/lunarc/train_unet.sh` @ `c6ac87d`
config:   `EDF_DIR=~/train_nomirneg` (symlink tree, Mir's rejections dropped);
          `NEG_SOURCE=hard`, neg_pos_ratio 10, pos_weight auto, stable-val-split on,
          epochs 50, batch 32, lr 3e-4, patience 10, base_filters 32, depth 4, dropout 0.2
exclude:  449382 (dead electrode), 450093 450094 (B2 UI ch3/ch4), and all eight Batch-4
          animals (483550 483551 483552 483553 483554 483555 483557 483559)
          -> trains on SV2A + B1 + B2-retained + **Batch 3**
purpose:  **the SV2A-equivalent number.** UNetv2's 93.6% precision was measured on the
          cohort it trained on; this model is the same construction for RAM_GDNF, so the
          per-animal comparison against the frozen pipeline can cover every retained
          animal — including Batch 3's eight, where frozen recall is worst (5.2%).
          This is NOT a generalisation measurement: there is no held-out fold. It answers
          "what do you get after annotating this cohort", which is the claim the paper
          actually needs for the retraining loop. The out-of-sample claim stays with the
          A3-on-B3 fold.
why no held-out fold is acceptable here:
          the three rows of the results table are deliberately different questions —
          frozen-on-new-cohort (before annotating), this model (after annotating), and
          A3-on-held-out-B3 (transfer to the *next* batch). Row 2 must be in-sample or it
          is not comparable to SV2A's 93.6%.
exclusion caveat:
          450093/450094 were dropped for consistency with the retraining arms, but the
          2026-10-08 precision analysis argues against binary noise exclusion (450096 has
          the highest `prominence_db` of all 32 animals and the *best* precision of B2's
          three retained channels). A production model would normally train on them. Note
          it; do not quietly re-run without recording why.
history: job **3825957 CANCELLED before running** — submitted against the stale label
          tree (see the defect entry above). Resubmitted as **3825989** after refreshing
          with `--keep-mir-rejections` and adding the two training-set corrections.
settings (all five confirmed on the submission summary line):
          `neg/pos=10`, `pos_weight=auto` (=10), `max_positive_sec=100`,
          `bg_avoid_rejected=1`, `hard_neg_exclude=mir_candidate`
dataset (verified by `--analyze`, job 3825985, before submitting):
          annotated EDFs **1,219** (was 404 — Mir's rejected rows are back in the tree);
          confirmed 1,535; raw rejected 13,074 but **hard-negative windows 1,461**, i.e.
          unchanged, which is the proof that the `mir_candidate` exclusion works;
          **positive windows 1,346** = 1,361 - 15.
          The cap's arithmetic reconciles exactly against Mir's CSV: 449381 -1,
          449385 -2, 449388 -8 = **11 in B1**, and 459659 -4 = **4 in B3**, totalling the
          15 events over 100 s. Nothing else moved.
          39 animal groups (B3's 459657-459664 present). Split: train 982 pos / val 364.
          Effective negatives at train time: 1,461 hard + ~12,000 background to reach the
          10:1 target, the background now avoiding annotated regions.
result:   _pending_
watch for: the new log line `background sampling: dropped N of ~12000 draws that could not
          avoid an annotated region`. A handful is fine; **over ~1,000 means Mir's rejected
          regions blanket those channels densely enough that clean background is scarce**,
          the effective neg/pos ratio falls below 10, and the result must be read with that
          in mind.

---

## Entries to add as you go

- Flood review: one mid-tercile file to complete the strata (files 1-2 done)
- Batch-3 review
- Round-0 remaining folds (hold out B1, B2; B4 now excluded)
- Arm A evaluated on Batch 3 (job 3809197)
- Post-training detection + scoring per fold

- Classical-detector sweeps
