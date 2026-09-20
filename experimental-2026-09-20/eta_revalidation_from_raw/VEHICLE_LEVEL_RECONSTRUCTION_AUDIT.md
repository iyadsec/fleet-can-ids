# VEHICLE_LEVEL_RECONSTRUCTION_AUDIT.md

## Status: **EXECUTED** — scoring MATCH; operating-point F1 differs from table_P4 with explanation

Reconstruction used:

| Step | Implementation | Setting |
|------|----------------|---------|
| Features | `src.features.feature_extractor.BEHAVIOURAL_FEATURE_COLUMNS` (24-D) | Authoritative |
| Split | Exact `balanced_split_manifest.csv` segments | Not random |
| IF fit | `fit_self_supervised_isolation_forest` | Benign **train** only; `n_estimators=200`, `contamination=auto`, `seed=42` |
| Scores | Benign-reference percentile normalisation | Authoritative |
| Threshold | `FPR<=5%` then max recall on **validation** (`SELECTED_METHOD_LABEL`) | Paper / `vehicle_level_evaluation.py` |
| Test | Evaluation only | No fit / no threshold selection |

Artifacts: `artifacts/vehicle_level_test_metrics.csv`, `artifacts/vehicle_level_audit_detail.csv`, `artifacts/threshold_policy_sensitivity.csv`, `models/if_*_seed42.joblib` (local, gitignored).

---

## Reconstructed held-out test metrics (FPR≤5% per vehicle)

| Scope | PR-AUC | Precision | Recall | F1 | Test FPR |
|-------|--------|-----------|--------|----|----------|
| Chevrolet | 0.9936 | 0.9606 | 0.9808 | 0.9706 | 0.0515 |
| Hyundai | 0.8298 | 0.8594 | 0.4554 | 0.5953 | 0.1016 |
| Kia | 0.9850 | 0.9838 | 0.7388 | 0.8439 | 0.0791 |
| pooled | 0.9688 | 0.9685 | 0.7064 | 0.8169 | 0.0848 |

---

## Historical reference — table_P4 (immutable)

| Scope | PR-AUC | Precision | Recall | F1 | Test FPR |
|-------|--------|-----------|--------|----|----------|
| Chevrolet | 0.9936 | 0.8937 | 0.9981 | 0.9430 | 0.1520 |
| Hyundai | 0.8256 | 0.7149 | 0.6686 | 0.6910 | 0.3636 |
| Kia | 0.9854 | 0.9498 | 0.8911 | 0.9195 | 0.3058 |
| pooled | 0.9690 | 0.9112 | 0.8618 | 0.8858 | 0.3096 |

User-cited approximate headlines (F1≈0.868 / per-vehicle ≈0.959/0.798/0.883) are **not** table_P4; numeric audit uses **table_P4**.

---

## Classification vs table_P4

| Metric | Chevrolet | Hyundai | Kia | pooled |
|--------|-----------|---------|-----|--------|
| PR-AUC | **MATCH** (exact to 6+ dp) | **CLOSE** (Δ≈0.004) | **MATCH** | **MATCH** |
| F1 @ FPR≤5% | MATERIAL vs P4 | MATERIAL vs P4 | MATERIAL vs P4 | MATERIAL vs P4 |
| F1 @ hist test-FPR matched | **MATCH** (exact 0.942986) | **CLOSE** (0.695 vs 0.691) | **CLOSE** (0.918 vs 0.920) | — |

### Explanation (evidence-based — not tuning)

1. **IF ranking / scores are reconstructed.** Chevrolet PR-AUC matches table_P4 to machine precision (`0.993597`). Pooled/Kia/Hyundai PR-AUC are MATCH/CLOSE.
2. **table_P4 was not evaluated at the FPR≤5% operating point.** Its test FPRs are 15–36%, incompatible with a validation FPR≤5% gate that yields reconstructed test FPRs ≈5–10%.
3. **Matching each vehicle’s historical test FPR on the reconstructed scores recovers P4 F1** (Chevrolet exact; Hyundai/Kia CLOSE). See `artifacts/threshold_policy_sensitivity.csv`.
4. Master YAML lists `threshold_selection: validation_f1_grid` for the balanced run, while `vehicle_level_evaluation.py` / paper methodology use **FPR≤5% + max recall**. This reconstruction follows the latter (authoritative code path for vehicle-level IDS tables). No thresholds were tuned to chase P4.

### Verdict on proceed / stop

| Question | Answer |
|----------|--------|
| Unexplained MATERIAL_DIFFERENCE in the vehicle pipeline? | **No** — score MATCH; F1 gap explained by operating-point policy |
| `STOP_VEHICLE_PIPELINE_MISMATCH`? | **No** |
| η selection performed? | **No** |

Pipeline is **defensible** for continuing to validation descriptors / scenarios / metric-protocol review.
