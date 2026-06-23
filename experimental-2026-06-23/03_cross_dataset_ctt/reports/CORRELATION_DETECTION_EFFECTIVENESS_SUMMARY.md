# Correlation Detection Effectiveness Summary

## Output root

`new_experiments/correlation_detection_effectiveness_comparison`

**Validation:** PASS

## Figure purposes

| Figure | Purpose |
|--------|---------|
| `figure_CORR_EFF1_ocslab_vs_ctt_fleet_correlation` | Main paper: OCSLab vs CTT fleet-correlation safety and campaign metrics |
| `figure_CORR_EFF2_fleet_correlation_gain` | Local/incident detection vs fleet campaign decision (CTT) |
| `figure_CORR_EFF3_consistency_rule_ablation` | Consistency rule: unrelated merge 1.0→0.0 without hurting campaign F1 |
| `figure_CORR_EFF4_campaign_detection_only` | Simple campaign F1 comparison for strong/weak scenarios |

## Key fleet-correlation findings (CTT corrected)

- Strong campaign F1 = **1.0**
- Weak campaign F1 = **1.0**
- Benign false campaign = **0**
- Isolated false campaign = **0**
- Unrelated incorrect merge after rule = **0**

## OCSLab vs corrected CTT

Descriptive comparison on 200-node scenario graphs (no temporal edges). OCSLab strong/weak campaign F1 ≈ 0.69/0.67; CTT corrected achieves 1.0/1.0 with consistency rule. Not a strict benchmark — datasets differ in vehicles and attack taxonomy.

## Local IDS vs fleet correlation

Local IDS detects individual anomaly windows. Campaign identification is evaluated at the fleet-correlation layer only. Local-only campaign decisions are **N/A** unless explicitly supported by source data.

## Consistency rule ablation

- Without rule: unrelated merge = 1.0
- With rule: unrelated merge = 0.0
- Strong/weak F1 remain 1.0 with rule

## Recommended main-paper artifacts

- **Figure:** `figure_CORR_EFF4_campaign_detection_only` (simple) or `figure_CORR_EFF1_ocslab_vs_ctt_fleet_correlation` (full)
- **Table:** `CORR_EFF1_ocslab_vs_ctt_campaign_correlation`

## Limitations

- Descriptive cross-dataset comparison, not benchmark-equivalent
- OCSLab curated exports only (balanced run excluded)
- Local IDS campaign metrics not directly comparable across datasets

## Writing can start?

**Yes**
