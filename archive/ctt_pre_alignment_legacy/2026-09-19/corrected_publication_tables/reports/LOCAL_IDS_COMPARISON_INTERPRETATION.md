# Local IDS Comparison Interpretation

Generated: 2026-06-21 20:15 UTC

## 1. Pooled OCSLab result (FPR<=5%, table_01)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9273 |
| Precision | 0.9728 |
| Recall | 0.4596 |
| F1 | 0.6243 |
| FPR | 0.0395 |

## 2. Pooled corrected CTT result (FPR<=5%)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9386 |
| Precision | 0.9465 |
| Recall | 0.2272 |
| F1 | 0.3543 |
| FPR | 0.0497 |

## 3. Why is CTT F1 lower?

CTT F1 (35.4%) is lower than OCSLab (62.4%) despite comparable precision (~94.7% vs ~97.3%). The gap is driven primarily by **recall** (22.7% vs 46.0%): the external validation sets include broader attack diversity, cross-vehicle distribution shift, and windows where `attack_type != benign` but `label = 0`. At the publication operating point (FPR<=5%), the detector ranks attacks well but misses more positives than on the in-domain OCSLab evaluation.

## 4. Is CTT PR-AUC higher or lower than F1 suggests?

CTT pooled PR-AUC (0.9386) is **higher** than OCSLab (0.9273). Threshold-independent ranking remains strong; the F1 gap reflects threshold-dependent recall, not catastrophic score separation failure.

## 5–6. Best / worst CTT vehicle by F1

- **Best:** Subaru Forester (F1=0.5310, PR-AUC=0.9645)
- **Worst:** Chevrolet Impala (F1=0.1950, PR-AUC=0.9100)

## 7–8. Best / worst CTT subset by F1 (available subsets)

- **Best:** test_01 known vehicle / known attack (F1=0.3750)
- **Worst:** test_03 known vehicle / unknown attack (F1=0.3336)

Note: test_02 and test_04 were not exported in the corrected publication local tables (known-vehicle evaluation policy).

## 9–10. Best / worst CTT attack type by F1

- **Best:** combined_spoofing (F1=0.5187)
- **Worst:** gear_spoofing (F1=0.2572)

## 11. Pooled vs per-vehicle for the paper?

Use **pooled metrics for compact cross-dataset comparison** (Table LOCAL_COMP1). Use **per-vehicle and per-subset tables in supplementary material** to show where generalisation degrades. OCSLab per-vehicle FPR<=5% rows are unavailable from curated exports — do not imply parity.

## 12. Recommended main-paper table

**LOCAL_COMP1** (pooled OCSLab vs CTT) plus **LOCAL_COMP4** (CTT attack breakdown) in supplement.

## 13. Recommended main-paper figure

**figure_LOCAL_COMP1_pooled_comparison** — shows PR-AUC remains competitive while F1/recall differ.

## 14. Limitation wording

The lower CTT F1 does not necessarily indicate failure of the framework. It reflects a harder external validation setting with broader attack diversity, cross-vehicle distribution shift, and label inconsistencies. Therefore, local IDS performance is reported using both threshold-independent PR-AUC and threshold-dependent precision, recall, F1, and FPR.

Pooled metrics are reported for compact comparison, while per-vehicle and per-subset results are used to identify where cross-dataset generalisation degrades.

Per-vehicle OCSLab metrics (Hyundai Sonata, Kia Soul, Chevrolet Spark) at the same FPR<=5% protocol were **not available** in `paper/results/table_01_vehicle_level_ids.csv`; only the pooled headline is used for OCSLab.
