# PUBLICATION_CLUSTER_ARTIFACT_AUDIT.md

## Source of “strong mean largest_cluster_size ≈ 34”

| Item | Value |
|------|-------|
| Exact mean | **33.6333…** (= 1009/30) |
| Primary artifact | `recovered_publication_pipeline/frozen_results_reference/campaign_metrics.csv` |
| Identical archive copy | `experimental-2026-06-23/01_primary_ocslab_balanced/results/campaign_metrics.csv` |
| Also present in | `tables/table_P7_strong_campaign_results.csv` (per–campaign_size means) |
| Filter | `test_condition == "Strong Coordinated Campaign"` (**S3 / strong_campaign**), n=30 |
| Aggregation | mean over **10 seeds × 3 campaign sizes** (2, 5, 10) |

### By campaign size (same artifact)

| campaign_size | mean `largest_cluster_size` | median | min | max | n |
|---------------|----------------------------:|-------:|----:|----:|--:|
| 2 | 14.1 | 16.0 | 9 | 18 | 10 |
| 5 | 29.6 | 31.5 | 6 | 43 | 10 |
| 10 | 57.2 | 67.0 | 15 | 83 | 10 |

### Weak equivalent

| Filter | mean `largest_cluster_size` |
|--------|----------------------------:|
| `Weak Coordinated Campaign` (n=30) | **11.1** |

Present in `campaign_metrics.csv` and `table_P8_weak_campaign_results.csv`.

---

## What the column is (and is not)

| Question | Answer | Confidence |
|----------|--------|------------|
| Scenario | Strong coordinated TEST (not validation stand-in) | HIGH |
| `graph_nodes` / `total_nodes` | **200** for these rows | HIGH (artifact) |
| Seed aggregation | mean across publication TEST seeds `{11…101}` × sizes | HIGH |
| Pre-gate vs post-gate | Column sits beside `dbscan_noise_percentage`, `n_predicted_campaign_clusters`, `predicted_campaign_size` — naming indicates **DBSCAN largest cluster size** (raw clustering geometry), **not** gate-accepted campaign count | MEDIUM (emitter `extract_run_metrics` **UNRECOVERABLE**; inference from column family) |
| Equals predicted campaign size? | **No** — strong mean `predicted_campaign_size` ≈ **5.23** vs largest cluster ≈ **33.6** | HIGH |
| S3-specific? | Yes for the ≈34 figure; weak has separate ≈11.1 | HIGH |

Descriptive contrast only (no tuning): reconstructed validation diagnostic often showed `|C_k| > 100` with `r_k ≈ 20`, larger and more fleet-swallowing than publication strong means above.
