# Provenance — real_ocslab_corrected

| Item | Value |
|------|-------|
| Parent diagnostic | PR #22 `experimental-reviewer-ablation/real_ocslab/` (preserved; paper-ineligible) |
| Reviewer comment | Sec 4.5.2 component ablation |
| Campaign GT semantics | Recovered from `coordination_strength.py` + `campaign_analysis_corrected.py` |
| Similarity scaler | `recovered_publication_pipeline/.../scalers/fleet_benign_scaler.json` |
| Graph params | τ=0.95, same-k=2, cross-k=5 (frozen) |
| Clustering | StandardScaler → PCA(8) → DBSCAN(eps=0.5, min_samples=2) |
| Campaign gate | β=0.5 cohesion; fragment merge 0.85 |
| Seeds | 11, 23, 37, 41, 53, 67, 71, 83, 97, 101 |
| Dataset required | OCSLab challenge car_track Sonata / Soul / Spark |
| Dataset status | **MISSING locally** — experiment STOPPED |
| Synthetic descriptors | Forbidden |

Absolute metrics are **not** claimed identical to frozen historical Section VII tables even after a successful corrected run.
