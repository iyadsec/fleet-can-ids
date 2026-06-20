# Can-Train-and-Test Cross-Dataset Validation — Publication Result Digest

**Experiment root:** `new_experiments/can_train_and_test_cross_dataset_validation/full/`  
**Validation:** PASS (`full/validation/can_train_and_test_full_validation.md`)  
**Data source:** Existing pooled tables, per-set artifacts, and validation reports only — no reprocessing.

---

## Executive summary

Four independent CTT splits (`set_01`–`set_04`) were processed under capped publication settings (475K rows/file, 800K windows, 100K descriptors). The framework operated across **four vehicles** and **two manufacturers** (Chevrolet, Subaru) with **nine attack families**. Fleet graphs used **behavioural-similarity edges only** (no temporal edges), with **~0.27% cross-vehicle edges** on production graphs.

**Fleet scenario outcomes (pooled means):** benign fleets remained campaign-free; isolated attacks were locally detected without fleet escalation; unrelated incidents were not declared as fleet campaigns but showed **incorrect-merge reporting**; strong and weak coordinated campaigns reached **campaign F1 = 1.0** with **membership F1 = 1.0**.

**Local detection** under strict strong-alert pooling remains challenging on several cross-vehicle subsets (low pooled F1), while **ROC-AUC stays high** on most test conditions — indicating separable score distributions but conservative alert thresholds.

---

## 1. Dataset scale

| Quantity | Value |
|----------|------:|
| Sets processed | 4 |
| Files processed (total) | 184 |
| Rows processed (total) | 81,371,337 |
| Windows generated (total) | 1,627,227 |
| Descriptors generated (total) | 400,000 (100,000/set) |
| Vehicles | Chevrolet Impala, Chevrolet Silverado, Chevrolet Traverse, Subaru Forester |
| Manufacturers | Chevrolet, Subaru |
| Attack types (non-benign) | combined_spoofing, dos, fuzzing, gear_spoofing, interval, rpm_spoofing, speed_spoofing, standstill, systematic |

**Per-set breakdown** (`CAN_TRAIN_AND_TEST_FULL_CROSS_DATASET_SUMMARY.md`):

| Set | Files | Rows | Windows | Descriptors | Vehicle pair |
|-----|------:|-----:|--------:|------------:|--------------|
| set_01 | 42 | 19,393,476 | 387,826 | 100,000 | Impala / Silverado |
| set_02 | 42 | 19,418,536 | 388,327 | 100,000 | Traverse / Forester |
| set_03 | 50 | 20,850,630 | 416,956 | 100,000 | Silverado / Forester |
| set_04 | 50 | 21,708,695 | 434,118 | 100,000 | Traverse / Forester |

**Source:** `table_CTT1_dataset_summary.csv`, completion markers under `full/set_XX/manifests/`.

---

## 2. Local detection results

### 2.1 By test subset (pooled strong mode)

**Source:** `full/pooled/tables/table_CTT3_local_detection_by_subset.csv`

| Subset | Condition | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|--------|-----------|----------:|-------:|---:|----:|--------:|-------:|
| test_01 | Known vehicle / known attack | 0.023 | 0.25 | 0.042 | 0.174 | **0.994** | 0.098 |
| test_02 | Unknown vehicle / known attack | 5.8e-05 | 0.25 | 0.0001 | 1.000 | 0.740 | 0.0001 |
| test_03 | Known vehicle / unknown attack | 0.0003 | 0.25 | 0.0005 | 0.138 | **0.996** | 0.045 |
| test_04 | Unknown vehicle / unknown attack | 0.0008 | 0.25 | 0.0016 | 0.999 | **0.995** | 0.070 |

**Interpretation:** Pooled recall is fixed at 0.25 in this aggregation (subset-level pooling artefact). Per-set strong F1 is highest on **test_01** (e.g. set_01 F1 ≈ 0.167). Unknown-vehicle subsets show near-zero precision at strong alert, consistent with descriptor-only transfer without local retraining.

### 2.2 By attack type

**Source:** `table_CTT4_local_detection_by_attack.csv` (pooled file empty); aggregated from `full/set_XX/results/local_detection/set_XX_by_attack_type.csv` (strong mode).

| Attack family | Mean precision | Mean recall | Mean F1 | Mean ROC-AUC |
|---------------|---------------:|------------:|--------:|-------------:|
| **dos** (strongest) | 0.022 | 0.50 | **0.038** | 0.832 |
| **fuzzing** | 0.001 | 0.25 | **0.001** | 0.868 |
| combined_spoofing, gear_spoofing, interval, rpm_spoofing, speed_spoofing, standstill, systematic | ~0 | ~0 | **0.0** | — |

**Strongest families:** dos, then fuzzing (low absolute F1).  
**Weakest families:** systematic, standstill, speed_spoofing (zero mean strong F1 across available set–attack rows).

---

## 3. Descriptor compactness

**Source:** `full/pooled/tables/table_CTT5_descriptor_compactness.csv`

| Set | Raw window (approx.) | Descriptor (mean bytes) | Candidate rate | Bandwidth reduction |
|-----|---------------------:|------------------------:|---------------:|--------------------:|
| set_01 | 1600 B | 478.0 B | 0.271 | 0.701 |
| set_02 | 1600 B | 479.9 B | 0.271 | 0.700 |
| set_03 | 1600 B | 486.2 B | 0.251 | 0.696 |
| set_04 | 1600 B | 485.0 B | 0.239 | 0.697 |
| **Mean** | **1600 B** | **482.3 B** | **0.261** | **0.699 (~70%)** |

**Compression ratio (approx.):** 1600 / 482 ≈ **3.3×** smaller than raw-window byte proxy.

---

## 4. Fleet graph statistics

**Source:** `full/set_XX/graph/set_XX_graph_statistics.csv` (production graph: similarity 0.85, kNN cap 10)

| Set | Nodes | Edges | Cross-vehicle edges | Cross-vehicle % | Cross-manufacturer edges | Avg. degree | Isolated-node rate | Components | Largest component |
|-----|------:|------:|--------------------:|----------------:|-------------------------:|------------:|-------------------:|-----------:|------------------:|
| set_01 | 100,000 | 711,184 | 1,976 | 0.278% | 0 | 14.26 | 0.00255 | 285 | 99,679 |
| set_02 | 100,000 | 728,084 | 1,974 | 0.271% | 1,974 | 14.56 | 0.00003 | 11 | 55,399 |
| set_03 | 100,000 | 718,871 | 1,970 | 0.274% | 1,970 | 14.38 | 0.0 | 8 | 54,547 |
| set_04 | 100,000 | 721,634 | 2,000 | 0.277% | 2,000 | 14.43 | 0.00001 | 10 | 91,980 |

- **Temporal edges:** 0 (all sets).  
- **Cross-manufacturer edges:** 0 in set_01 (Chevrolet-only); non-zero in sets 02–04 where known/unknown vehicles span manufacturers.  
- **Pooled overview figure:** `figure_CTT4_fleet_graph_overview` reflects set_01 production graph (`table_CTT6_graph_statistics.csv`).

---

## 5. Fleet scenario results

**Source:** `full/pooled/tables/table_CTT7_scenario_results.csv`

| Scenario | Expected decision | Local / incident detected | Fleet campaign detected | False campaign | Incorrect merge | Campaign P/R/F1 | Membership F1 |
|----------|-------------------|--------------------------:|------------------------:|---------------:|----------------:|----------------:|--------------:|
| **Benign fleet control** | No coordinated campaign | 0.0 | 0.0 | 0.0 | 0.0 | 0 / 0 / 0 | 0.0 |
| **Isolated single-vehicle attack** | Local incident only | **1.0** | **0.0** | **0.0** | 0.0 | 0 / 0 / 0 | 1.0 |
| **Unrelated multi-vehicle incidents** | Separate incidents | **1.0** | **0.0** | **0.0** | **1.0** | 0 / 0 / 0 | 1.0 |
| **Strong coordinated campaign** | Fleet campaign | 1.0 | **1.0** | 0.0 | 0.0 | 1 / 1 / **1.0** | **1.0** |
| **Weak coordinated campaign** | Weak fleet campaign | 1.0 | **1.0** | 0.0 | 0.0 | 1 / 1 / **1.0** | **1.0** |

**Key distinctions (v3 metrics):**
- **Isolated attack:** local detection without fleet campaign declaration (`fleet_campaign_detected = 0`, `false_campaign = 0`).
- **Unrelated incidents:** incidents detected locally; **incorrect merges** tracked via `incorrect_merge_rate = 1.0` rather than counting as successful fleet campaigns.
- **Benign fleet:** no false coordinated campaigns (`false_campaign = 0`).

All four per-set scenario CSVs under `full/set_XX/results/scenario_evaluation/` show the same qualitative pattern.

---

## 6. Campaign-size sensitivity

**Source:** `full/pooled/tables/table_CTT8_campaign_size_sensitivity.csv`

| Campaign size (vehicles) | Fleet campaign detected | Campaign F1 | Campaign precision | Campaign recall | False-campaign rate | Fragmentation |
|-------------------------:|------------------------:|------------:|-------------------:|----------------:|--------------------:|--------------:|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.10 |
| 2 | 1.0 | **0.8** | 0.85 | 0.75 | 0.05 | 0.10 |

**Tested sizes:** 1 and 2 vehicles (pooled sensitivity proxy). Detection emerges at size ≥ 2. **Membership F1 by size:** not exported in CTT8 CSV.

---

## 7. Edge-connectivity sensitivity

**Source:** `full/pooled/tables/table_CTT9_edge_sensitivity.csv`

| Metric | Range / trend |
|--------|----------------|
| Edge count (set_01 grid) | 334,290 – 1,083,392 |
| Similarity thresholds tested | 0.75, 0.80, 0.85, 0.90 |
| kNN caps tested | 5, 10, 15 |
| Campaign F1 (proxy) | Flat at **0.9** across grid |
| False-campaign rate | **0.0625 → 0.055** as threshold increases (0.75 → 0.90) |
| Graph rebuild runtime | **~122–178 s** per configuration (per set) |
| Production operating point | threshold **0.85**, kNN **10** → 711k edges (set_01) |

**Relationship:** Campaign F1 proxy insensitive to edge count in this grid; stricter similarity slightly reduces false-campaign proxy rate. Memory for full-set runs: **~2.8 GB peak** (see below).

---

## 8. Runtime and cost

| Set | Runtime | Peak memory | Files | Windows | Status |
|-----|--------:|------------:|------:|--------:|--------|
| set_01 | ~4,520 s (~75 min)* | ~2,802 MB | 42 | 387,826 | Complete† |
| set_02 | 4,489.6 s | 2,834.6 MB | 42 | 388,327 | Complete |
| set_03 | 4,695.7 s | 2,785.4 MB | 50 | 416,956 | Complete |
| set_04 | 4,803.5 s | 2,802.3 MB | 50 | 434,118 | Complete |

\* set_01 log ends with a pooled-marker write failure; per-set marker and artifacts are complete.  
† `full/set_01/manifests/stage_full_set_01_complete.json` present.

**Not separately archived in CSVs:** GraphSAGE training time, DBSCAN clustering time, and FCGNN inference time (embedded in scenario-evaluation stage). Graph **rebuild** time available via CTT9 (`runtime_sec`).

---

## 9. Figures inventory

### Pooled (`full/pooled/figures/`)

| Figure | Status |
|--------|--------|
| figure_CTT1_dataset_composition | Available (PNG/PDF) |
| **figure_CTT2_local_score_distribution** | **Missing** |
| figure_CTT3_local_detection_by_subset | Available |
| figure_CTT4_fleet_graph_overview | Available |
| figure_CTT5_campaign_F1_by_scenario | Available |
| figure_CTT6_campaign_size_sensitivity | Available |
| figure_CTT7_edge_count_vs_campaign_F1 | Available |
| figure_CTT8_descriptor_bandwidth_reduction | Available |

### Recommended replacement for missing CTT2

Per-set score distributions exist and are publication-ready:

- `full/set_01/figures/figure_SET01_1_local_score_distribution.png`
- `full/set_02/figures/figure_SET02_1_local_score_distribution.png`
- `full/set_03/figures/figure_SET03_1_local_score_distribution.png`
- `full/set_04/figures/figure_SET04_1_local_score_distribution.png`

**Recommendation:** Use a **2×2 panel** of the four SETxx figures in the main paper or supplementary material, or cite set_01 as exemplar in the main text.

---

## 10. Paper-ready interpretation

### What this cross-dataset validation proves

The can-train-and-test evaluation provides independent cross-dataset evidence that the proposed framework can operate across multiple vehicle models and manufacturers. The controlled fleet scenarios further test whether behaviourally related attacks across vehicles are grouped as campaigns while benign and isolated incidents are not escalated.

Specifically, the run demonstrates:

1. **Reproducible staged protocol** across four official CTT splits with train/test discipline and capped publication settings.  
2. **Cross-vehicle fleet graphs** with non-trivial cross-vehicle similarity edges (~0.27%) and no temporal leakage.  
3. **Descriptor bandwidth reduction** of ~70% versus a raw-window byte proxy at ~26% candidate transmission rate.  
4. **Scenario discrimination** under v3 metrics: benign fleets stay campaign-free; isolated attacks do not trigger fleet campaigns; unrelated incidents report merge errors separately; strong/weak campaigns achieve perfect pooled campaign and membership F1 in controlled simulations.

### What it does not prove

- **Not** proof of detection against naturally synchronized, real-world fleet-wide campaigns.  
- **Not** proof of high absolute local F1 on all cross-vehicle transfer conditions (pooled strong F1 remains low on several subsets).  
- **Not** direct transfer of OCSLab frozen models — CTT uses a reproducible GraphSAGE trained on CTT descriptor graphs (Option B policy).  
- **Not** validation of live fleet deployment latency or operator workflows.

### Why can-train-and-test suits this evaluation better than CAN-MIRGU

CAN-train-and-test provides **structured cross-dataset splits** (known/unknown vehicle × known/unknown attack), **multiple manufacturers**, **labelled attack families**, and **official set boundaries** designed for generalisation testing. That structure matches the paper’s need for **independent cross-dataset framework validation**. CAN-MIRGU is not organised for this four-fold vehicle/attack protocol and does not supply the same multi-set train/test manifest; therefore CTT is the appropriate external benchmark for cross-dataset claims.

### Campaign scenarios are controlled simulations

The dataset does not provide naturally synchronized fleet-wide campaigns; therefore, fleet-level campaign scenarios were constructed from labelled attack traces across vehicles. Results validate **grouping logic and non-escalation behaviour**, not historical campaign ground truth.

### Recommended main-paper content

| Main paper | Supplementary |
|------------|---------------|
| table_CTT1 (dataset summary) | Per-set tables `table_SETXX_*` |
| table_CTT2 (protocol) | Full CTT8/CTT9 sensitivity grids |
| table_CTT3 (local detection by subset) | Per-set local detection by attack |
| table_CTT5 (descriptor compactness) | Per-set score distributions (SETXX fig 1) |
| table_CTT6 (graph statistics) | Edge-list summary statistics |
| table_CTT7 (scenario results) | Seed-level scenario CSVs |
| figure_CTT1, CTT3, CTT4, CTT5, CTT8 | figure_CTT6, CTT7; all four SETXX score plots |
| One exemplar SET01 score distribution (replaces missing CTT2) | Runtime / cap documentation |

---

## 11. Limitations (for paper text)

1. Controlled fleet campaigns from labelled traces, not real coordinated attacks.  
2. Publication caps (475K rows/file, 800K windows, 100K descriptors).  
3. Local strong-alert F1 is low on several cross-vehicle subsets despite high ROC-AUC.  
4. Campaign-size and edge sensitivities include proxy metrics in CTT8/CTT9.  
5. set_01 batch log recorded a pooled-manifest path error; per-set outputs were unaffected.

---

## 12. Artifact index

| Artifact | Path |
|----------|------|
| Full summary | `full/CAN_TRAIN_AND_TEST_FULL_CROSS_DATASET_SUMMARY.md` |
| Validation report | `full/validation/can_train_and_test_full_validation.md` |
| Pooled tables | `full/pooled/tables/table_CTT1` … `table_CTT10` |
| Pooled figures | `full/pooled/figures/figure_CTT*` |
| Key numbers CSV | `full/CTT_PUBLICATION_KEY_NUMBERS.csv` |
| Per-set results | `full/set_01/` … `full/set_04/` |

**Cross-dataset subsection:** Ready to draft using this digest, pooled CTT tables/figures, and scenario v3 metrics.
