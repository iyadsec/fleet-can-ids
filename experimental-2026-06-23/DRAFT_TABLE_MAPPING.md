# Draft table → canonical source mapping

| Draft table | Metric (example) | Draft value | Canonical source | Canonical value | Action |
|-------------|------------------|-------------|------------------|-----------------|--------|
| Table I | Pooled local F1 | 0.868 | `01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv` | ~0.886 pooled | Align wording to balanced P4 |
| Table II | Bandwidth reduction | 53.69% | `table_P5_descriptor_compactness_and_privacy.csv` | verify from P5 | Use P5 |
| Table III | Unrelated incorrect merge | **0.400** | `table_P6_benign_isolated_unrelated_results.csv` | **0.400** | **Keep** (matches balanced) |
| Table IV | Strong F1, cs=5 | **0.867** | `table_P7_strong_campaign_results.csv` | **0.733** | **Update draft** (was original split) |
| Table IV | Strong F1, cs=10 | **0.933** | `table_P7` | **1.000** | **Update draft** |
| Table VI | Weak F1, cs=5 | 0.533 | `table_P8_weak_campaign_results.csv` | 0.500 | Minor update |
| Baseline ablation | M3 strong F1 | 0.406 (old) | `02_baseline_ablation/` (balanced M3) | 0.733 | Use new bundle |
| Baseline ablation | M3 unrelated merge | 0.000 (old) | `02_baseline_ablation/` | 0.400 | Use new bundle |

See `01_primary_ocslab_balanced/audit/original_vs_balanced_split.md` for original→balanced diffs.
