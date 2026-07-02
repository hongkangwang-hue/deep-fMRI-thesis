# Results

*Draft — all numbers below are read from the real M5 run on UTS03
(`results/m5_stats/UTS03/m5_results.json`, 1000 bootstrap draws, seed
`20260701`, commit `df33018`) and cross-checked against the M6 tables
(`figures/UTS03/tables/table2_full_numbers.md`, generated at commit
`fdf0c1c`). Confirmatory vs. exploratory/descriptive status follows
`frozen/contrast_registry.yaml` and is preserved verbatim from that
registry — it is not re-derived post hoc.*

## RQ1 — Do the core models differ in brain alignment at matched context length? (exploratory)

Direct paired differences (architecture − Pythia), left IFG, main layer,
normal condition, unadjusted 95% CI:

| Contrast | H=8 | H=32 | H=128 |
|---|---|---|---|
| RWKV − Pythia | −0.0013 [−0.0028, +0.0002] | **−0.0024 [−0.0040, −0.0006]** | **−0.0042 [−0.0058, −0.0028]** |
| Mamba − Pythia | **+0.0028 [+0.0014, +0.0042]** | **+0.0046 [+0.0029, +0.0062]** | **+0.0056 [+0.0040, +0.0072]** |

Mamba's advantage over Pythia is present and CI-excludes-zero at all three
context lengths, growing with H. RWKV's disadvantage relative to Pythia is
CI-excludes-zero at H=32 and H=128 but not distinguishable from zero at
H=8. **These are RQ1 exploratory estimates**; a CI excluding zero at an
individual H does not by itself constitute a confirmatory finding — see
the confirmatory family under RQ2.

## RQ2 — Do models differ in Context Gain? (confirmatory family + descriptive)

### Confirmatory family (the only contrasts in this project permitted a confirmatory conclusion)

Δr_total = r(H=128) − r(H=8), left IFG, main layer, Holm α = 0.05:

| Contrast | Δr_total diff | 95% CI | two-sided bootstrap p | Holm threshold | Decision |
|---|---|---|---|---|---|
| RWKV − Pythia | **−0.0029** | [−0.0043, −0.0015] | 0.0000 | 0.0250 | **reject H0** |
| Mamba − Pythia | **+0.0028** | [+0.0011, +0.0043] | 0.0040 | 0.0500 | **reject H0** |

Both contrasts survive Holm correction. **Confirmatory conclusion:** in
left IFG (main layer), Mamba's gain in brain alignment from short (H=8) to
long (H=128) context is significantly larger than Pythia's, and RWKV's gain
is significantly smaller than Pythia's.

### Descriptive Context Gain per model (left IFG, main layer)

| Model | Δr_total (128−8) | 95% CI |
|---|---|---|
| Pythia | +0.0035 | [+0.0020, +0.0051] |
| Mamba | **+0.0064** (largest) | [+0.0047, +0.0080] |
| RWKV | +0.0006 | [−0.0008, +0.0022] (CI crosses 0) |
| AWD-LSTM | ≈0.0000 | [−0.0001, +0.0001] |

RWKV's own gain is not distinguishable from zero on its own CI (only the
paired contrast against Pythia, above, is confirmatory). AWD-LSTM shows an
essentially flat, context-length-invariant curve (r values change by <0.0001
across H=8/32/128 in both ROIs; `figures/UTS03/tables/table2_full_numbers.md`),
consistent with its role as a shallow (1–2 layer) historical LSTM reference
rather than a core comparison model — **AWD-LSTM's absolute r is not used
in any core architecture ranking** (see Conclusion Boundaries).

## RQ3 — Do IFG and PT show different descriptive context patterns?

Δr_total (left IFG vs. bilateral PT, main layer) and the dedicated
IFG-minus-PT contrast (`ifg_minus_pt_delta_total_{model}`, secondary
exploratory per `contrast_registry.yaml`), unadjusted 95% CI:

| Model | Δr_total, IFG | Δr_total, PT | IFG − PT diff | 95% CI |
|---|---|---|---|---|
| Pythia | +0.0035 | +0.0027 | +0.0008 | [−0.0007, +0.0024] (crosses 0) |
| Mamba | +0.0064 | +0.0044 | **+0.0020** | **[+0.0004, +0.0035]** (excludes 0) |
| RWKV | +0.0006 | +0.0003 | +0.0004 | [−0.0012, +0.0019] (crosses 0) |
| AWD-LSTM | ≈0.0000 | ≈+0.0001 | ≈−0.0001 | [−0.0002, +0.00004] (crosses 0) |

Only for **Mamba** does the IFG-vs-PT gain difference have a CI that
excludes zero — Mamba's context gain is larger in IFG than in PT with
some confidence. For Pythia, RWKV, and AWD-LSTM the difference is not
distinguishable from zero. This is a **secondary exploratory** estimate
(not part of the confirmatory family); it should be read as "for one of
the four checkpoints, we have some evidence of a larger IFG than PT
context gain," not as a general claim that IFG is more context-sensitive
than PT across models.

**Repeatability caveat (per milestone acceptance criterion 7):** IFG/PT
repeatability could not be computed — `wheretheressmoke` only has
cross-repeat-averaged responses in the available `.hf5` files, not
per-repeat responses (`m5_results.json::known_gaps::ifg_pt_repeatability`).
Per the preregistered interpretation rule, a near-zero PT effect may only be
described as **region-specific** if repeatability is acceptable, base
alignment is detectable, the effect drops under shift, and the CI is
sufficiently narrow. Without repeatability, **we do not claim PT is
region-specifically non-contextual** — we report only that PT's descriptive
gain is numerically smaller than IFG's, with no reliable region-specificity
claim attached.

## RQ4 — Does the main conclusion depend on the primary layer rather than the final layer? (robustness)

Layer-flip judgment applies **only** when both layers' CIs are individually
non-zero **and** have opposite signs (not a point-estimate ranking):

| Contrast | Main-layer CI side | Final-layer CI side | Substantive flip? |
|---|---|---|---|
| RWKV − Pythia | − (significant) | crosses 0 (not significant) | **No** (not opposite-sign; but see caveat) |
| Mamba − Pythia | + (significant) | + (significant) | No — consistent |

Mamba's confirmatory-family finding is robust to layer choice. **RWKV's is
not**: it is significant in the main layer but loses significance in the
final (robustness) layer. This does not meet the strict substantive-flip
criterion (which requires opposite non-zero CIs), but the significance being
layer-dependent is a real robustness caveat that must be stated alongside the
RWKV finding — it should not be presented with the same evidentiary weight
as the Mamba result.

## Negative control (40-second time shift)

The 40 s within-story feature shift (no circular wrap; out-of-bounds TRs
dropped; normal and shifted scored on a shared valid-TR mask; scaler/PCA/
Ridge re-fit on training folds within each condition) is only a **partial**
pass, and must be reported at three distinct levels rather than as a blanket
"shift removes the effect":

1. **Absolute brain alignment — destroyed (clean pass).** IFG r collapses
   from ~0.13 (normal) to ~0.007–0.011 (shifted) across all models/H
   (`m5_results.json::estimands r_*_left_IFG_main_shift`). The main
   brain-alignment signal is genuinely word-order-locked.

2. **Between-architecture Context-Gain difference (the confirmatory family)
   — collapses under shift (supports the confirmatory claim).** The shifted
   architecture contrasts shrink to ~+0.0001…+0.0005 and no longer exclude
   zero (`shifted_diagnostic.shifted_reproduces_architecture_effect =
   false`). So the *difference* in Context Gain between architectures
   (Mamba > Pythia, RWKV < Pythia) is word-order-specific — the two
   confirmatory findings survive this control.

3. **Each model's own raw Context Gain — only partially word-order-specific
   (important caveat).** The paired normal − shifted Δr_total difference
   (`delta_total_normal_minus_shift_{model}_ifg_main`, computed within each
   bootstrap draw) shows that shifting significantly reduces the Context
   Gain for **Mamba only** (Δ ≈ +0.0023, CI excludes 0); for **Pythia** the
   reduction is not significant (CI crosses 0), and for **RWKV** the shifted
   Context Gain is actually *larger* than normal (Δ ≈ −0.0039, CI excludes 0
   on the negative side). In other words a substantial part of the raw
   per-model Δr_total survives temporal scrambling, so raw Context Gain must
   **not** be interpreted as purely genuine long-range linguistic
   integration — part of it plausibly reflects a low-level statistical
   property of longer-context representations (greater temporal
   smoothness/autocorrelation aligning with slow BOLD structure even when
   misaligned). This does not undermine the confirmatory *architecture
   difference* (level 2), but it bounds how the *magnitude* of any single
   model's Context Gain can be interpreted.

*(Exact point estimates + 95% CIs for level 3 are in `m5_results.json`
after re-running M5 with the paired estimand added; Figure 4c plots them.)*

## Known gaps

- **IFG/PT repeatability**: not computed (data unavailable); limits RQ3 to a
  descriptive, non-region-specific claim (see above).

---

## Conclusion boundaries

These boundaries are required framing for every claim above and must not be
dropped when this draft is condensed into the final manuscript.

1. **Checkpoint-level evidence, not architectural causation.** Results
   compare specific pretrained checkpoints (particular parameter counts,
   tokenizers, training corpora — see Methods §4/Table 1), approximately
   matched in scale. They support statements like "this Mamba checkpoint
   showed a larger context gain than this Pythia checkpoint," not the
   general architectural claim that "state-space models integrate context
   better than Transformers."
2. **"Layer depth" is an operational proxy.** Primary/robustness layer
   indices (Methods §4) are per-model choices, not a validated claim that
   equivalent depths correspond to equivalent processing stages across
   architectures with very different total depths and parameter counts.
3. **Three H values are coarse samples, not a fitted curve.** H ∈
   {8, 32, 128} describes trends across three sampled points; it does not
   estimate a true saturation point or imply a particular functional form
   between them.
4. **Single subject; within-subject patterns only.** All results are from
   UTS03 alone — a preregistered deviation from the original 3-subject
   design (data availability constraint, confirmed 2026-07-01). Findings
   describe within-subject cross-architecture / cross-context patterns and
   do not support population-level generalization claims.
5. **The voxelwise Ridge model is a shared readout, not a fifth model.**
   The same encoding pipeline (scaler/PCA/FIR/Ridge) is applied identically
   to every language model's features; it is a fixed measurement instrument
   for comparing the four language models, not itself a competing model of
   brain alignment.
