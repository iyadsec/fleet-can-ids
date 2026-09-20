# COORDINATION_STRENGTH_SELECTION_PLAN.md

**Status:** NEW / PROSPECTIVE — design only; **NOT EXECUTED**  
**Companion:** `PROSPECTIVE_CAMPAIGN_PROTOCOL_V1.md`  
**Candidates:** \(s \in \{0.75,\ 1.0\}\)

Historical P7/P8 numeric coordination strength: **INSUFFICIENT_EVIDENCE**.  
These candidates are from the recovered **allowed range** / peer implementations. Neither is the historical value.

---

## 1. Purpose of \(s\)

\(s\) defines a **controlled coordination condition** for descriptor-level campaign construction.

\(s\) is **not** a model hyperparameter to maximize detection performance.

Therefore selection must **not**:

- use final TEST data  
- choose the value with highest Campaign F1 / precision / recall  
- look at final detection F1 when defining or freezing \(s\)  
- weaken S4’s \(s\) relative to S3 because local evidence is weak  

S3 and S4 share the **same** frozen \(s\) once selected.

---

## 2. When selection may occur (future; not now)

Only after:

1. Prospective campaign protocol V1 is accepted, and  
2. Manifest schema is implemented, and  
3. A **validation-only** diagnostic harness exists that measures construction diagnostics **without** using detection F1 as the objective,

then this plan may be executed on **validation** constructions only.

**This document does not authorize execution.**

η remains unselected (`{2,3,5,10}`) and must stay frozen-out until **after** \(s\) is frozen.

---

## 3. Diagnostics to compute per candidate \(s\) (validation only)

For each candidate \(s \in \{0.75, 1.0\}\), build S3 and S4 validation campaigns under the prospective protocol (same seeds, sizes, budget) and record:

| Diagnostic | Definition (prospective) | Intent |
|------------|--------------------------|--------|
| `within_campaign_cosine_mean` | Mean pairwise cosine among coordinated-role behavioural vectors \(C\) (or recovered `measure_mean_pairwise_similarity` view, **predeclared**) | Coordination measurability |
| `within_campaign_cosine_p10` / `p50` / `p90` | Distribution of pairwise cosines inside campaign | Non-degeneracy / collapse check |
| `between_campaign_background_cosine_mean` | Mean cosine between campaign coordinated vectors and benign-background vectors | Separability from background |
| `cross_vehicle_edge_density` | Fraction of cross-vehicle edges among coordinated members under the **frozen** graph rule (τ=0.95, \(k\) settings) — diagnostic only | Graph-level coordination signal |
| `descriptor_variance_after` | Mean per-feature variance of coordinated \(C\) after transform | Collapse / diversity |
| `fraction_identical_vectors` | Fraction of coordinated pairs with exact identical \(C\) (within float tie) | Degeneracy at \(s=1\) |
| `noise_sigma` | \(0.02(1-s)\) | Recorded constant per candidate |
| `anomaly_score_unchanged` | Assert scores equal pre/post transform | Integrity check |

Optional supporting (still non-F1):

- mean \(\|d'_i - \tilde{p}\|_2\) after transform  
- fraction of coordinated pairs with behavioural cosine ≥ τ (τ is graph threshold; used here only as a **construction** diagnostic, not as a detection score)

---

## 4. Degenerate campaign construction (reject / prefer-other)

A candidate \(s\) yields a **degenerate** construction if **any** of the following hold on the predeclared validation seed×size grid:

1. **Total collapse without residual structure:** `fraction_identical_vectors = 1.0` for all coordinated members **and** the scientific goal requires retaining residual within-campaign diversity — flag as `DEGENERATE_FULL_COLLAPSE`.  
   - Note: \(s=1.0\) with \(\sigma=0\) **recovers** exact identity on \(C\) by design; that is not automatically invalid, but must be **explicitly accepted** as “full prototype replacement,” not treated as a silent accident.  
2. **Non-measurable coordination:** `within_campaign_cosine_mean` does not exceed `between_campaign_background_cosine_mean` by a predeclared margin \(\delta_{\mathrm{coord}}\) on S3 **and** S4 — flag `DEGENERATE_NO_COORDINATION_SIGNAL`.  
3. **Background entanglement:** coordinated-vs-background cosine ≥ within-campaign cosine — flag `DEGENERATE_BACKGROUND_COLLISION`.  
4. **Numerical / clip pathology:** post-transform variance ≈ 0 on all features while \(s < 1\) unexpectedly, or transform integrity asserts fail — flag `DEGENERATE_NUMERICAL`.  
5. **Asymmetric S3/S4 coordination (forbidden outcome of selection):** if the selection procedure would pick different \(s\) for S3 vs S4 — invalid; V1 requires one shared \(s\).

Exact numeric thresholds \(\delta_{\mathrm{coord}}\), edge-density floors, etc. are **PROSPECTIVE_CHOICE_REQUIRED** and must be frozen in writing **before** looking at diagnostic numbers.

---

## 5. Proposed selection rule (non-performance-based)

**Rule name:** `prospective_s_selection_v1_coordination_characterisation`

**Inputs:** diagnostic table for \(s\in\{0.75,1.0\}\) on validation S3/S4 only.  
**Forbidden inputs:** Campaign F1, cluster F1, gate pass rate used as optimization objective, TEST metrics, η sweeps.

**Procedure (to be executed later):**

1. Run integrity asserts (`anomaly_score` untouched; only \(C\) rewritten; budget 200 nodes).  
2. Drop any candidate marked `DEGENERATE_NUMERICAL` or `DEGENERATE_BACKGROUND_COLLISION`.  
3. Among remaining candidates, require **measurable but non-accidental** coordination:  
   - `within_campaign_cosine_mean - between_campaign_background_cosine_mean ≥ \delta_{\mathrm{coord}}` on both S3 and S4.  
4. Prefer the candidate that is **less degenerate on residual diversity**, subject to still meeting (3):  
   - Primary: lower `fraction_identical_vectors` (prefer partial blend when both pass).  
   - Tie-break: closer adherence to a predeclared target band for within-campaign cosine (e.g. “high but not forced-identical”), **not** maximization.  
5. If **only** \(s=1.0\) produces measurable cross-vehicle coordination under the frozen graph diagnostic, select \(s=1.0\) and document acceptance of full prototype replacement (`FULL_REPLACEMENT_ACCEPTED`).  
6. If **neither** candidate is non-degenerate, **do not** pick by F1; declare `COORDINATION_CONDITION_UNSATISFIED` and revise protocol — do not proceed to η or TEST.  
7. Freeze the chosen \(s\) as `FROZEN_PROSPECTIVE` in the run header; never relabel as historically recovered.

### Explicit non-rule

```text
argmax_s Campaign_F1(s)   # FORBIDDEN
argmax_s any_detection_metric(s)  # FORBIDDEN
```

---

## 6. Relationship to η

- η candidates remain `{2, 3, 5, 10}`.  
- η selection (future) uses validation only under `metric_protocol=revised_jaccard_0.5_v3`.  
- η selection starts only after \(s\) is frozen.  
- Provisional historical/recon η=2 remains **PROVISIONAL ONLY** and is not an input to \(s\) selection.

---

## 7. Execution checkbox

| Step | Status |
|------|--------|
| Design selection rule | Done (this file) |
| Freeze \(\delta_{\mathrm{coord}}\) and cosine target band | **PROSPECTIVE_CHOICE_REQUIRED** — not done |
| Compute diagnostics for 0.75 vs 1.0 | **Not executed** |
| Freeze \(s\) | **Not executed** |
| Evaluate η | **Not executed** |
| TEST / CTT | **Not executed** |
