# OLD vs aligned CTT results

## Old CTT (pre-alignment)

### A. Legacy GraphSAGE pilot archive
Source: `legacy_archive/old_pipeline_results/scenario_evaluation/run_level_metrics.csv`

| Scenario | campaign_f1 (seed 11 only) |
|----------|----------------------------|
| benign_fleet_control | 0.0 |
| isolated_attack | 0.0 |
| strong_campaign | 0.0 |

Method: 32-D features, τ=0.85, MSE reconstruction GraphSAGE (lr=1e-3), cosine DBSCAN eps=0.8, attack-type cohesion scoring.

### B. “Corrected” publication tables (consistency-rule path — not GraphSAGE freeze)
Source: `legacy_archive/corrected_publication_tables/tables_csv/FLEET_CORR1_corrected_ctt_fleet_summary.csv`

| Scenario | campaign_f1 |
|----------|-------------|
| Benign-Fleet Control | 0.0 |
| Isolated Single-Vehicle Attack | 0.0 |
| Unrelated Multi-Vehicle Incidents | 0.0 (after consistency rule; merge was 1.0 before) |
| Strong Behaviourally Related Campaign | **1.0** |
| Weak Behaviourally Related Campaign | **1.0** |

These F1=1.0 figures came from a **different** corrected-correlation / consistency-rule methodology, not the authoritative OCSLab FLEET-GUARD GraphSAGE freeze.

## Aligned CTT (this branch)

Method: shared `publication_fleet_core` with freeze τ=0.95, k_same=2, k_cross=5, structure GraphSAGE (lr=0.01, wd=5e-4, λ=0.25, 30 epochs), StandardScaler→PCA(8)→euclidean DBSCAN (0.5/2), centroid cohesion gate (β=0.5).

### Synthetic fixture run (dataset missing)

| Scenario | mean campaign_f1 (10 seeds) |
|----------|-----------------------------|
| benign_fleet_control | 0.0 |
| isolated_attack | 0.0 |
| unrelated_incidents | 0.0 (incorrect_merging mean 1.0) |
| strong_campaign | 1.0 |
| weak_campaign | 1.0 |

**These synthetic numbers are not scientific cross-dataset results.** The fixture injects strongly correlated same-family features across vehicles, so F1=1.0 is expected and does **not** validate real CTT performance.

## Why results differ / may differ on real CTT

1. **Graph threshold** 0.85→0.95 sparsifies edges; benign/unrelated connectivity drops.
2. **Input features** 32-D→9-D publication view changes embedding geometry.
3. **Training** MSE reconstruction → structure link+score objective.
4. **Clustering** cosine-raw → Euclidean PCA; eps 0.8→0.5.
5. **Gate** attack-family cohesion → centroid behavioural cohesion without labels.
6. Old F1=1.0 “corrected” tables used a consistency rule outside the GraphSAGE freeze path.

On real CTT data, aligned F1 is expected to be **lower** than the corrected F1=1.0 tables; that is scientifically preferable to methodological mismatch.

## Re-run requirement

Place DTU can-train-and-test under `Dataset/can-train-and-test/` (or `CTT_DATASET_ROOT`), run local IDS + descriptor stages, then:

```bash
python scripts/run_ctt_aligned_publication.py --mode full
```

Do not tune fleet hyperparameters on CTT outcomes.
