# Vehicle-Level Isolation Forest Validation Report

## Intended protocol (FPR-controlled)

1. Train Isolation Forest on **benign-only** training windows (70/15/15 stratified split per vehicle).
2. Score validation and test windows with normalised anomaly scores.
3. On **validation only**, select threshold with FPR ≤ 5%; among feasible thresholds, pick **highest recall**.
4. Apply per-vehicle thresholds to the **held-out test** partition.
5. Report PR-AUC, precision, recall, F1, and FPR per vehicle and pooled.

## Superseded high-FPR Table I sources

| Candidate source | Threshold policy | Typical pooled FPR | Notes |
|------------------|------------------|--------------------|-------|
| `results/vehicle_level_threshold_comparison.csv` (F1-optimal) | F1-optimal on validation | **94.2%** | High F1, impractical FPR |
| `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv` | Fixed strong thresholds (Chevrolet 0.85, Hyundai/Kia 0.7) | **~31%** | Fleet pipeline descriptor gate, not FPR-controlled |
| Draft Table I (manuscript: Chevrolet 0.211, Hyundai 0.436, Kia 0.353, pooled 0.373) | Fixed / F1-oriented threshold (not FPR≤5%) | **~37%** | Inconsistent with deployment policy and Table X |

### Threshold policy diagnosis

- **F1-optimal (validation):** test F1=0.860, test FPR=0.942
- **FPR≤5% (corrected):** test F1=0.624, test FPR=0.040

The draft Table I high-FPR values align with **fixed strong-threshold fleet evaluation** (`table_P4_vehicle_level_results.csv` and `experiments/04_train_vehicle_ids.py` defaults), not the FPR-controlled protocol used for Table X.

### Archived per-vehicle high-FPR table (`/workspace/experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv`)

```
vehicle_model  roc_auc   pr_auc  precision   recall       f1  false_positive_rate  false_negative_rate  latency_sec  test_windows
    Chevrolet 0.993762 0.993597   0.893654 0.998084 0.942986             0.151961             0.001916          0.1           930
      Hyundai 0.738374 0.825614   0.714894 0.668562 0.690952             0.363566             0.331438          0.1          3049
          Kia 0.907120 0.985398   0.949826 0.891075 0.919513             0.305791             0.108925          0.1         10615
       pooled 0.883706 0.968967   0.911218 0.861847 0.885845             0.309570             0.138153          0.1         14594
```

## Corrected FPR-controlled metrics (test)

### Pooled (verified)

- PR-AUC: **0.9273**
- Precision: **0.9724**
- Recall: **0.4597**
- F1: **0.6243**
- FPR: **0.0402**
- Detection latency (ms): **74.82**

### Per-vehicle

Per-vehicle FPR≤5% test metrics were **not previously exported** in curated paper artifacts. They are computed by `run_vehicle_level_evaluation()` when `data/processed/window_features.csv` is available. Re-run:

```bash
export OCSLAB_DATASET_DIR=/path/to/In-Vehicle\ Network\ Intrusion\ Detection\ Challenge
python experiments/01_load_dataset.py --config configs/default.yaml
python experiments/02_generate_windows.py --config configs/default.yaml
python experiments/03_extract_features.py
python scripts/run_vehicle_level_fpr_controlled.py --config configs/default.yaml
```

## Selected thresholds (validation FPR≤5%)

- **Chevrolet:** 0.957090
- **Hyundai:** 0.960466
- **Kia:** 0.950419
- **Pooled reference:** 0.955825 (FPR<=5%)

## Recommendation

**Replace Table I** with `tables/table_vehicle_level_ids_fpr_controlled.tex` and `results/vehicle_level_metrics_fpr_controlled.csv`.
Use the pooled FPR≤5% row for the deployment-oriented summary (matches Table X).
Do not report F1-optimal or fixed strong-threshold metrics as primary vehicle-level IDS results.
