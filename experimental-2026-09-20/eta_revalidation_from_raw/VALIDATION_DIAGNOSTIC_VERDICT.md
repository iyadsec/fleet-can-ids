# VALIDATION_DIAGNOSTIC_VERDICT.md

**Primary verdict:** `MULTIPLE_CAUSES_FOUND`

Generated: diagnostic audit after η validation (η=2 provisionally frozen).

Constraints honored: no TEST, no CTT, no manuscript, no η / metric / parameter changes.

Selected η=2 remains provisionally frozen. **ETA_GATE_FAILURE = 0**.

## Top 3 evidence-backed causes of low S3/S4 Campaign F1

1. **BETA_COHESION_FAILURE on fleet-scale mega-clusters** — On 19/20 missed runs, GT vehicles co-cluster (often r_k=20, \|C_k\|≳100) with cohesion c_k≈0.2–0.4 < β=0.5. γ and η pass; the campaign gate rejects at β. Only 2/20 runs produce any gate-passing campaign; only 1/20 is a Jaccard TP.

2. **SCENARIO_CONSTRUCTION_MISMATCH / UNRECOVERABLE historical builder** — `build_mixed_validation_suite` is missing (`MISSING_MODULES.md`). Reconstructed S3/S4 use a stand-in with prototype feature blending, campaign_size=5 only, and seeds {131…197} instead of publication {11…101}. Historical freeze selection_report V3/V4 Campaign F1 ≈0.65/0.42 vs reconstructed 0.10/0.00.

3. **Weak same-GT similarity at publication τ=0.95** — Mean cross-vehicle cosine among same-GT descriptors = **0.1692 ≪ 0.95**, so reconstructed “coordination” does not create GT↔GT neighbors in the edge feature space. Cross edges exist (~253 mean) but are not campaign-selective; DBSCAN then over-merges.

## Not the dominant cause

- **η gate:** 0 ETA_GATE_FAILURE; η=2 rejects 0/159 clusters suite-wide.
- **revised_jaccard_0.5_v3 alone:** only 2 gate-passing campaigns exist; matcher cannot explain 18/20 misses that never pass the gate.

## Pipeline hyperparameter check

Frozen graph / GraphSAGE / DBSCAN / γ / β values **MATCH** `final_shared_fleet_configuration.yaml` (hash `72dbfc…`). See `PUBLICATION_VS_RECONSTRUCTED_PIPELINE.csv`.

## Key measured facts

- S3/S4 runs: 20; TP: 1; gate-passing: 2
- GT co-clustered (≥2 vehicles): 20/20
- Mean same-GT cosine: 0.1692
- Mean emb within-GT / GT↔unrelated: 6.5022 / 14.3086
- Primary failure: BETA_COHESION_FAILURE (19)

```text
MULTIPLE_CAUSES_FOUND
```
