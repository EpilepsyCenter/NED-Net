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

---

## Entries to add as you go

- Flood review: one mid-tercile file to complete the strata (files 1-2 done)
- Batch-3 review
- Round-0 remaining folds (hold out B1, B2; B4 now excluded)
- Arm A evaluated on Batch 3 (job 3809197)
- Post-training detection + scoring per fold

- Classical-detector sweeps
