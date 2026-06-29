# Per-Vehicle Isolation Forest Validation Report

## 1. Which old script generated the incorrect high-FPR Table I?

The superseded Table I values were produced by the **fleet end-to-end pipeline**:

- `experiments/04_train_vehicle_ids.py`
- `src/models/vehicle_ids.py::generate_vehicle_anomaly_predictions()`

That path trains one Isolation Forest per vehicle on benign windows, but classifies test windows using **fixed strong/weak score thresholds** (defaults: strong=0.80; balanced Chevrolet strong=0.85, Hyundai/Kia=0.70). It does **not** perform validation FPR≤5% threshold selection.

Archived per-vehicle high-FPR rows matching this policy appear in `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv` (fleet balanced publication run).

The **correct** publication evaluator is:

- `run_vehicle_level_evaluation.py` / `scripts/generate_per_vehicle_fpr_table.py`
- `src/evaluation/vehicle_level_evaluation.py::run_vehicle_level_evaluation()`

## 2. What threshold policy did old Table I use?

| Policy | Mechanism | Pooled test FPR |
|--------|-----------|-----------------|
| **Old manuscript Table I** | Fixed / F1-oriented strong threshold | **~37%** |
| F1-optimal on validation | Max validation F1 | **94.2%** |
| Fixed strong threshold (P4) | Chevrolet 0.85, Hyundai/Kia 0.70 | **~31%** |
| **Final protocol** | Validation FPR≤5%, then max recall | **4.0%** |

Old manuscript Table I (superseded):

| Vehicle | PR-AUC | Precision | Recall | F1 | FPR |
|---------|--------|-----------|--------|-----|-----|
| Chevrolet | 0.994 | 0.921 | 1.000 | 0.959 | 0.211 |
| Hyundai | 0.906 | 0.802 | 0.794 | 0.798 | 0.436 |
| Kia | 0.975 | 0.932 | 0.838 | 0.883 | 0.353 |
| Pooled | 0.963 | 0.900 | 0.838 | 0.868 | 0.373 |

Verified **pooled** FPR-controlled metrics (`results/vehicle_level_threshold_comparison.csv`, FPR≤5% row):

| PR-AUC | Precision | Recall | F1 | FPR |
|--------|-----------|--------|-----|-----|
| 0.9273 | 0.9724 | 0.4597 | 0.6243 | 0.0402 |

Per-vehicle validation thresholds (from prior full OCSLab run, `results/vehicle_level_thresholds.json`):

- Chevrolet: 0.957090
- Hyundai: 0.960466
- Kia: 0.950419

## 3. Corrected per-vehicle FPR-controlled metrics (held-out test)

Per-vehicle test metrics were **not exported** in prior paper bundles. Regenerate with the OCSLab dataset:

```bash
export OCSLAB_DATASET_DIR="/path/to/In-Vehicle Network Intrusion Detection Challenge"
python experiments/01_load_dataset.py --config configs/default.yaml
python experiments/02_generate_windows.py --config configs/default.yaml
python experiments/03_extract_features.py
python scripts/generate_per_vehicle_fpr_table.py --config configs/default.yaml
```

Outputs:

- `results/vehicle_level_metrics_per_vehicle_fpr_controlled.csv`
- `tables/table_vehicle_level_per_vehicle_fpr_controlled.tex`
- `results/vehicle_level_scored_splits.csv` (for audit/recompute)

## 4. FPR ≤ 5% confirmation

Under the final protocol, each vehicle's threshold is chosen on **validation** with FPR≤5% (max recall). Test FPR per vehicle is expected to be ≤5% when evaluated on the held-out test partition with that threshold.

**Verified pooled test FPR = 4.02%** (FPR≤5% protocol). Per-vehicle test FPR values are written when `generate_per_vehicle_fpr_table.py` is run with `data/processed/window_features.csv`.

## Recommendation

Replace Table I with `tables/table_vehicle_level_per_vehicle_fpr_controlled.tex` (Chevrolet, Hyundai, Kia, Pooled rows).
