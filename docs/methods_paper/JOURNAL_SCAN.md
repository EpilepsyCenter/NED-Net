# Prior art and journal choice

**Literature retrieved from PubMed, 2026-10-09.** Searches: rodent/mouse/rat + EEG +
seizure detection + deep learning/neural network (28 hits); toolbox/pipeline/GUI + EEG +
epilepsy + preclinical (7); seizure detection + generalization/transfer + across subjects
(2). The field is small: there are **very few rodent-EEG detector *tool* papers**, and
almost none that test transfer.

## The three papers we must position against

### 1. Fumeaux et al. 2020, *Epilepsia* — the transfer precedent
["Accurate detection of spontaneous seizures using a generalized linear model with external
validation"](https://doi.org/10.1111/epi.16628)
GLM over 141 EEG features. Trained on 16 rats with focal epilepsy (1,012 spontaneous
seizures), then validated on an independently constructed set of 96 rats with multifocal
epilepsy (2,883 seizures). Pooled AUROC **0.995**, leave-one-out **0.962**, and
**cross-dataset 0.890**.

**This is the most important paper for us.** It means *"transfer is never measured"* is
**false** and must not be claimed. They measured it and documented degradation.
**How we differ:** they report window-level AUROC pooled over animals, which hides exactly
what we found — per-animal collapse, with a third of implants at zero recall. AUROC 0.890
sounds survivable; "4 of 11 animals return nothing" does not, and it is the same phenomenon
seen at the unit a researcher actually cares about. They also stop at characterising the
drop; we supply the remedy (annotate the new cohort, retrain) and the software for it.

### 2. Kotloski 2023, *Scientific Reports* — independent support for our per-animal finding
["A machine learning approach to seizure detection in a rat model of post-traumatic
epilepsy"](https://doi.org/10.1038/s41598-023-40628-1)
GoogLeNet DCNN on multiplexed scalogram/kurtosis/entropy images, >2,200 h of rat EEG,
95.6% sensitivity at 1.52 false positives/h. Critically:
**"a DCNN trained specifically for the individual animal was superior to using DCNNs across
animals (intra-animal 0.960 ± 0.009 vs inter-animal 0.811 ± 0.015, p < 0.01)"**.

**This independently corroborates the paper's core claim** — the animal, not the cohort, is
the unit of variability — from a different lab, model, species and architecture. Cite it in
the Introduction as motivation rather than treating our result as novel in kind.
**How we differ:** we quantify the distribution rather than the mean (0-58% per animal, 4 of
11 at zero), separate the two failure modes (mute vs firing-but-mislocalised), and act on
it with a retraining loop instead of training one model per animal — which does not scale.

### 3. Kamintsky et al. 2025, *Epilepsia Open* — the closest tool paper
["An algorithm for seizure detection in rodents"](https://doi.org/10.1002/epi4.70070)
ANN over 22 engineered features, three mouse models (pilocarpine, albumin/BBB, synapsin
TKO), **with a GUI**, assessed on >2,800 h from 15 animals: sensitivity and PPV **>98%**.
In use since 2010.

**The direct competitor for the "software" claim.**
**How we differ:** (a) their >98% is within their own models and animals — no cross-cohort
test, which is precisely the gap we fill; (b) feature-engineered ANN versus end-to-end
learning from human annotations, so their model cannot be extended by a user's own labels;
(c) **no annotation/retraining loop** — the GUI executes detection, it does not close the
loop from review back into training; (d) no live/during-acquisition mode; (e) seizure-
specific by construction, where ours learns whatever is annotated.

## Also relevant, lower priority

* Vergara-Chozas et al., [SWD detection in GAERS and patients](https://doi.org/10.1016/j.eplepsyres.2023.107151)
  (*Epilepsy Research* 2023) — cross-species on one seizure type.
* Leguia et al., [Learning to generalize seizure forecasts](https://doi.org/10.1111/epi.17406)
  (*Epilepsia* 2022) — human forecasting, pretrain-then-transfer across patients; useful as
  the clinical analogue of our argument (generalisation works when calibration is
  maintained per subject).
* [Chronic multi-site mouse EEG protocol](https://doi.org/10.3791/69314) (*JoVE* 2025) —
  acquisition, not analysis; cite for the recording context only.

## Where the gap is

Nobody has published a **tool that closes the loop**: annotate in a GUI -> train -> detect
(batch and live) -> review -> retrain, with transfer failure measured per animal and the
remedy demonstrated. Kamintsky has the GUI without the loop; Fumeaux has the transfer
measurement without a tool; Kotloski has the per-animal observation without either.

## Journal recommendation

| journal | approx IF | fit | verdict |
|---|---|---|---|
| **eNeuro** (SfN) | ~3 | explicit *Methods/New Tools* and *Negative Results* tracks; open access | **first choice** |
| **Journal of Neuroscience Methods** | ~2.5-3 | the classic venue for exactly this | **strong second** |
| **Epilepsia Open** | ~3-4 | best domain fit; published Kamintsky | third — may see us as incremental to a paper they just ran |
| Journal of Neural Engineering | ~4 | engineering slant; would want more architecture novelty | possible |
| Scientific Reports | ~4 | broad; published Kotloski | fallback, low selectivity |
| Frontiers in Neuroinformatics | ~3 | software papers | fallback |
| SoftwareX | ~2.5 | pure software, no biology | too narrow — loses the validation story |

**Recommend eNeuro, Methods/New Tools.** It is the one venue whose remit explicitly covers
both halves of this paper: a new tool *and* a result that is partly negative (an inherited
model fails per-animal). That framing is a liability at a conventional journal and an asset
there. Mid-IF, SfN-backed, open access, and read by the people who would use the software.

**On high IF — not realistic as the work stands.** *Nature Methods* / *Nature
Communications* would require methodological novelty we do not claim (the architecture is a
standard U-Net), and *Epilepsia* (~6-7) would compare us directly with Fumeaux, who has
**96 rats and 2,883 seizures with external validation**. Our n is far smaller and precision
of the retrained model is still unmeasured.

**What would move it up a tier:** a second laboratory's recordings run through the loop,
showing the recovery reproduces outside our rig. That converts "our cohort needed
retraining" into "retraining is how this is done", and makes *Epilepsia* or *Journal of
Neural Engineering* defensible. Worth considering as a collaboration before submission,
since it is the one addition that changes the venue rather than the paper.
