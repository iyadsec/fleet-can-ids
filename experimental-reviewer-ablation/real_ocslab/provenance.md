# Provenance — Real OCSLab Reviewer Ablation

## A. Real OCSLab artifact / pipeline

| Item | Value |
|------|-------|
| Classic traces | `/workspace/Dataset/ocslab_classic_only/{normal_run_data,DoS_dataset,Fuzzy_dataset,gear_dataset,RPM_dataset}` |
| Challenge traces | `/workspace/Dataset/ocslab/0_Preliminary/0_Training/Pre_train_{D,S}_{0,1,2}.csv` |
| Feature code | `src/features/feature_extractor.py` (`BEHAVIOURAL_FEATURE_COLUMNS`, 24-D) |
| Windowing | `src/features/window_generator.py` (100 frames, stride 50) |
| Isolation Forest | `src/models/vehicle_ids.py` (`fit_self_supervised_isolation_forest` / `score_self_supervised_isolation_forest`) |
| Ablation methods | Parent `experimental-reviewer-ablation/methods.py` (M1–M4) |
| Metrics | Parent `experimental-reviewer-ablation/metrics.py` |
| Pool builder | `experimental-reviewer-ablation/real_ocslab/build_scored_pool.py` |
| Scored TEST pool | `artifacts/test_window_pool.csv` |

**Synthetic descriptors: none.**

## B. Test-data provenance

- Temporal protocol: interleaved contiguous blocks, 70/15/15 train/val/test by block count.
- Each block windowed independently (no cross-gap inter-arrivals).
- IF fit on train-benign windows only, per physical vehicle (`OCSLab_Car`, `Challenge_D`, `Challenge_S`).
- Scenario membership drawn exclusively from the TEST pool.

## C. Strong / weak sampling

- Strong malicious: `label=1` and `anomaly_score >= 0.80`
- Weak malicious: `label=1` and `0.55 <= anomaly_score < 0.80`
- Suspicious benign background: `label=0` and `anomaly_score >= 0.55`
- Thresholds taken from the runnable publication pipeline; **not** retuned after seeing ablation metrics.

## D. Final common configuration

See `config.yaml`. Graph τ=0.95, same-k=2, cross-k=5; GNN 9→64→32, 30 epochs, Adam 0.01, wd 5e-4, λ=0.25; Scaler→PCA(8)→DBSCAN eps=0.5 min_samples=2.

## E. β and fragment-merge thresholds

| Parameter | Value | Role |
|-----------|-------|------|
| `cohesion_threshold` (β) | **0.5** | Minimum intra-campaign cohesion for accepting a DBSCAN cluster |
| `fragment_merge_threshold` | **0.85** | Centroid cosine merge of accepted campaign fragments |

These are **distinct**. PR #21 synthetic config incorrectly set both to 0.85; this real run uses the publication pair (0.5 / 0.85).

## F. M1–M4 definitions

Unchanged from the controlled harness:

- M1 local IF only
- M2 descriptor clustering on real 9-D features
- M3 GCN on the shared real graph
- M4 GraphSAGE on the same graph

## G–J

Filled after the 10-seed run in `FINAL_REPORT.md` / PR body (table mean±std, verification, metric defs, interpretation).

## Validation checklist (pre-paper)

1. All M2/M3/M4 inputs from actual OCSLab held-out TEST observations? **YES** (after pool rebuild)
2. Zero synthetic descriptor values? **YES** (`synthetic=0` asserted per scenario)
3. Scenarios identical across methods per seed? **YES** (single frozen scenario → all methods)
4. GCN and GraphSAGE graphs identical? **YES** (`edge_sig` assert)
5. β distinguished from fragment merge? **YES** (0.5 vs 0.85)
6. Hyperparameters changed after seeing results? **NO**
7. Exactly 10 seeds? **YES** (11,23,37,41,53,67,71,83,97,101)
