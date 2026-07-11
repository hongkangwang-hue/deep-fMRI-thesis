# Methods

*Draft — integrates M0–M5. Written against verified real results (M4/M5 ran on
UTS03, commit `fdf0c1c`). Terminology unified across stages per M6 requirement.*

## 1. Design overview and preregistered scope deviation

We compare four pretrained language models — a Transformer (Pythia-160M), a
state-space model (Mamba-130M), a linear-attention RNN (RWKV-4-169M), and a
historical LSTM reference (AWD-LSTM, WT103) — as encoding models of BOLD fMRI
responses to naturalistic spoken stories, across three context lengths
(H ∈ {8, 32, 128} preceding words) and two regions of interest (left IFG,
bilateral PT), under a 3-fold story-level cross-validation.

**⚠️ Preregistered deviation (confirmed 2026-07-01):** the design was frozen
for 3 subjects (UTS01/UTS02/UTS03); the actual dataset available to us only
included UTS03. All results below are **single-subject (UTS03)**. This
constrains conclusions to within-subject, cross-architecture / cross-context
comparisons; it does not support claims of generalization across individuals
(see Conclusion Boundaries in `results.md`).

## 2. Stimuli and fMRI data

Data are from the LeBel et al. naturalistic-listening fMRI dataset
(OpenNeuro `ds003020`, version v3.1.1, commit `cb7c536d471d48ff95a90f0531d5e2bdb910a77a`),
subject UTS03, TR = 2.0 s. 84 stories are indexed; a single project-wide table
(`word_index.parquet`) links story, word, onset/offset time, and TR index, so
all downstream alignment and scoring share one canonical index rather than
relying on positional coincidence between separately-computed arrays.

## 3. Context-length definition

H denotes the number of preceding words available to the model, **not
including** the target word itself; a window therefore contains H+1 words.
Under 0-based indexing, the first word with a full H=128 history is target
index i=128. Each context window is processed independently (state reset per
window; no external KV-cache reuse across windows), so representations at
different H are not confounded by carry-over context.

## 4. Language models and feature extraction

| Model | HF/source id | Primary layer | Robustness (final) layer |
|---|---|---|---|
| Pythia | `EleutherAI/pythia-160m` | 8 | 11 |
| Mamba | `state-spaces/mamba-130m-hf` | 16 | 23 |
| RWKV | `RWKV/rwkv-4-169m-pile` | 8 | 11 |
| AWD-LSTM | fastai `WT103_FWD` | 1 | 2 |

(Verified against `results/m4_full_matrix/UTS03/cells/*.json` meta at commit
`fdf0c1c` — see `figures/UTS03/tables/table1_checkpoint_audit.md`.)

Per-word hidden representations are extracted in batches with right-side
padding (causal models, so batching does not change per-token output);
batched-vs.-single-window extraction is verified to agree to
max|Δ| < 3e-4 (production runs) / 6.4e-6 (AWD-LSTM smoke test) before any
window's features are trusted. Features are cached per (model, story, H) and
reused across ROIs and CV folds — the language models are never re-run inside
the CV loop.

## 5. Temporal alignment

Word-level features are downsampled to the TR grid by Lanczos interpolation
over each word's (onset+offset)/2 time, restricted to the subset of target
words that have a valid feature (Plan A: no zero-padding for words without
enough left context). A finite-impulse-response (FIR) expansion applies 4
hemodynamic delays (2/4/6/8 s → 1–4 TR shifts) independently within each
story (no cross-story bleed); the first 10 / last 5 TRs of each story are
trimmed to match the preprocessed BOLD response. The scoring mask for a given
story is the intersection of (>100 s into the story) ∩ (FIR-valid) ∩
(shift-valid, for the negative control — see §10).

## 6. ROI definition and voxel inclusion

A single BOLD-only voxel mask (pycortex `UTS03_auto` thick mask, 95,556
voxels) is built on the union of all 84 stories; **all 95,556 voxels are
retained** (0 excluded for NaN, 0 for zero variance), so every model/H/layer
condition scores the same column space. ROI membership uses the
`aparc.a2009s` parcellation with a **dominant-vertex** assignment rule: a
voxel is assigned to an ROI only if the cortical-surface vertex that
contributes most to that voxel (via pycortex's `line_nearest` mapper)
carries that ROI's label.

| ROI | Label(s) | Columns | Connectivity |
|---|---|---|---|
| Left IFG (primary) | `G_front_inf-Opercular` + `Triangul` (L) | 768 | single component, 100% |
| Bilateral PT (reference) | `G_temp_sup-Plan_tempo` (L+R) | 341 | two components, largest 58% |

## 7. Encoding model

For each outer CV fold, `StandardScaler` and `PCA` (k=100, full SVD) are fit
**only on the training-fold stories'** pre-FIR features, then applied to
both training and held-out stories before FIR expansion — no scaler/PCA/λ
selection ever sees held-out data. Ridge regression uses `himalaya`'s
`RidgeCV` with **per-voxel** λ selection via inner 2-fold CV on the training
data (validation Pearson r as the selection metric; ties broken toward the
larger λ). The λ grid is `logspace(-2, 7, 19)` = [0.01, 1e7].

*Note on the λ grid:* the originally frozen grid was `logspace(-2,4,13)` =
[0.01, 1e4]. A blind parameter check (M3a; inner-CV only, **no held-out r
touched**) found that voxels with real signal (inner-CV score > 0.10)
saturated this upper bound 100% of the time — traced to unwhitened PCA
component variances requiring much larger regularization than native
unit-variance features. The grid was expanded and **re-frozen before any
held-out evaluation** (tag `m3a-lambda-refreeze`, commit `10204f1`), after
which the signal-voxel boundary-hit rate fell to 0.000.

Fitting runs in float32 (validated bit-identical to float64 to 4 decimal
places for the ROI/voxel summary statistics in a controlled comparison),
except the Phase-1 fidelity gate against the original LeBel pipeline, which
retains float64 for numerical parity.

## 8. Cross-validation

Three-fold story-level CV (`frozen/fold_split.json`, split seed `20260629`):

| Fold | Train stories | Test stories |
|---|---|---|
| fold_0 | 55 | 28 |
| fold_1 | 55 | 28 |
| fold_2 | 56 | 27 |

## 9. Scoring

For each held-out story, voxelwise Pearson r is computed on that story's
scoring-mask TRs only (never pooling TRs across stories before scoring, to
avoid a Simpson's-paradox-like bias from between-story mean/variance
differences). ROI scores are obtained by Fisher-z-transforming the in-ROI
voxel r values and averaging in z-space. Both fold-level and cross-fold
aggregation use an **effective-TR-weighted mean in Fisher-z space** (weights
= each story's valid-TR count), converted back to r (tanh) only for
reporting — never re-transformed mid-pipeline.

## 10. Negative control (40-second time shift)

Feature streams are shifted 40 s forward within each story (no circular
wrap); TRs invalidated by the shift are dropped from **both** the normal and
shifted conditions' scoring (a shared common mask), so the two conditions are
scored on identical TR counts per story and are directly comparable.

## 11. Statistical analysis

A paired story bootstrap (`src/stats/bootstrap.py`) resamples stories with
replacement **within each outer fold**; the same resampled story indices are
reused across every model × H × layer × condition × ROI within one draw,
which is what makes difference estimates (architecture contrasts, Context
Gain) properly paired. 1000 draws, seed `20260701`. The point estimate uses
no resampling and applies the identical Fisher-z-weighted aggregation rule as
§9, so it exactly reproduces the standard cross-fold r.

The **confirmatory family** (the only contrasts permitted a confirmatory
conclusion) consists of two IFG-main-layer Δr_total (r_H128 − r_H8)
architecture differences: RWKV−Pythia and Mamba−Pythia. Significance uses a
two-sided percentile bootstrap p-value (p = 2·min(P(θ*≤0), P(θ*≥0))) with
Holm step-down family-wise error control at α = 0.05. All other estimates —
RQ1 H-specific differences, Δr_local/Δr_long, PT, the AWD-LSTM context curve,
final-layer robustness, and the shifted negative control — are exploratory,
descriptive, or robustness analyses reported with **unadjusted** 95%
percentile CIs, and are not to be read as confirmatory findings on their own
(see `results.md`).

## 12. Software and hardware

Feature extraction and Ridge fitting ran on an AutoDL GPU instance (RTX
4090); `himalaya` (torch_cuda backend) provided `RidgeCV`; HuggingFace
`transformers` (offline cache) served Pythia/RWKV, `mamba-ssm` served Mamba,
`fastai` served AWD-LSTM. Config version `v4.9-uts03-pilot`. See
`README.md` (this folder) for the exact git tags/commits and commands behind
every stage.

## 13. Pre-registered known biases

Three biases were identified and pre-registered before any new-subject
results existed (frozen at the UTS03 baseline, tag `uts03-graduation-
baseline`). None threaten either confirmatory finding, and none require
rerunning UTS03. They carry over unchanged to the three-subject extension —
they are properties of the pipeline (inner-CV fitting, FIR/training-mask
definitions, repeated-story derivatives), not of any one subject's data, so
they need no re-derivation for UTS01/UTS02.

1. **Mild inner-loop preprocessing optimism.** The inner 2-fold λ selection
   uses a Scaler and PCA fit on the *entire* outer-training set, so the
   inner validation fold participates in the PCA estimation it is later
   scored against. This does not touch the outer held-out fold — it only
   affects per-voxel λ selection — and is expected to cancel to a large
   degree in the four-model difference estimates that make up the
   confirmatory family, since all four models share the same inner-CV
   procedure.
2. **Training-side degenerate H=128 rows push Context Gain conservative,
   not inflated.** The training mask (`fir_valid`) is H-independent and
   does not additionally enforce the 100-second rule, so early in each
   story the H=128 window has near-degenerate (near-zero-support) training
   rows. This biases Context Gain estimates conservative (understates
   rather than overstates gain). The *scoring* side is unaffected: all 84
   stories' H=128 first-word midpoints were verified to fall after 100
   seconds, so held-out evaluation is fair across H.
3. **Repeatability may be unavailable.** If a repeated story
   (`wheretheressmoke`) is only distributed as a cross-repeat average with
   no per-repeat response, repeatability cannot be computed. When this
   happens, PT is reported conservatively as "no reliable effect detected,"
   not as region-specific support — this is what was in fact observed for
   UTS03 (§ Known gaps in `results.md`) and is expected to hold for
   UTS01/UTS02 as well, since it reflects a dataset-level derivatives
   limitation (confirmed across all 9 ds003020 subjects), not a per-subject
   one.

A fourth, related caveat from the negative-control analysis (§10) — that
single-model raw Δr_total values carry limited evidentiary weight about
genuine word-order-specific integration, even though the two *between-
architecture* confirmatory contrasts are unaffected — is not a pre-
registered bias (it was discovered post hoc from the shifted-condition
paired comparison) but must be attached to any single-model Context Gain
number quoted for any subject, present or future (see `results.md`).
