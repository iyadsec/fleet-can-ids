# FINAL_VERDICT.md

# **STOP**

The exact P7/P8 metric matcher, membership emitter, fragmentation emitter, and campaign-gate **application code** have **not** been recovered as executable source connected to the authoritative publication run.

Frozen **configuration values** for DBSCAN (`eps=0.5`, `min_samples=2`) and the shared campaign-rule keys (`minimum_distinct_vehicles=2`, `minimum_cross_vehicle_support=1`, `minimum_campaign_cohesion=0.5`, fragment centroid `0.85`) **are** recovered from publication freeze artifacts. Artifact-level **algebraic identities** for many P7/P8 columns are verified. That is not enough for GO under the recovery rules.

## What remains UNRECOVERABLE

1. **`extract_run_metrics` / `safety_row`** (`final_shared_configuration.metrics`) — never committed  
2. **`run_refinement_fcgnn`** (`coordinated_campaign_refinement.refinement_pipeline`) — never committed  
3. **`_run_single_test`** and base **`generate_tables`** — never committed  
4. **Exact predicted↔GT cluster matcher** (Jaccard / IoU / Hungarian / overlap threshold) — not proven; ablation Jaccard≥0.5 **not** linked to P7/P8  
5. **η = minimum candidate campaign cluster size `|C_k|`** — no freeze key; **must not** use `dbscan_min_samples`  
6. **Which cohesion formula** (centroid vs pairwise) the missing refinement applied  
7. **How `minimum_cross_vehicle_support` was enforced** in code  
8. **End-to-end reproduction** from descriptors to P7/P8 (missing modules + missing intermediates)

## What is recovered (insufficient for GO)

| Item | Status |
|------|--------|
| P7/P8 = mean(`campaign_metrics`) by campaign_size | VERIFIED |
| Strong/Weak F1 headline numbers | VERIFIED in tables |
| Count-based campaign TP/FP/FN identities on saved rows | VERIFIED (artifact); emitter UNRECOVERABLE |
| Vehicle-level membership identities + benign-as-FP | VERIFIED (artifact) |
| `membership_purity ≡ membership_precision`; `completeness ≡ campaign_recall` | VERIFIED (artifact) |
| Fragmentation identities (`frag_per=n_pred`, `frag_rate=1{n_pred>n_true}`) | VERIFIED (artifact) |
| Incorrect merging mean 0.4 on unrelated | VERIFIED (artifact) |
| DBSCAN freeze 0.5/2 vs peer default 1.2/10 | VERIFIED provenance |
| Gate freeze keys 2 / 1 / 0.5 | VERIFIED freeze |

## Required next step (outside this repo state)

Recover the four missing packages (or agent-session / author-machine archives that still contain them), then re-run verification against frozen `campaign_metrics.csv` without overwriting `01_primary_ocslab_balanced/`.

**Do not modify CTT based on inferred metric identities.**
