# IEEE Experimental Evaluation — Interpretations

This document supports the four validated contributions in the Experimental Evaluation section.
Claims are limited to what the exported evidence supports.

## Contribution 1 — Vehicle-Level IDS Effectiveness

The self-supervised Isolation Forest achieves **ROC-AUC 0.786** and **PR-AUC 0.927** at an operating point selected for **FPR ≤ 5%**.
Precision is high (**97.3%**) but recall is moderate (**46.0%**, F1 **62.4%**), indicating conservative strong-alert generation.
Per-attack F1 ranges from **39.8% (replay)** to **81.3% (fuzzy)**; replay remains the hardest class at the chosen threshold.

**Interpretation:** The vehicle-level IDS provides a usable local baseline with low false alarms, but does not fully recover weak or replay-dominated attacks without fleet-level correlation.
**Limitation:** Threshold selection trades recall for FPR; weak anomalies are largely deferred to the fleet layer.

## Contribution 2 — Descriptor Compactness and Security

Descriptors compress raw CAN windows by **12.6×** (**92%** bandwidth reduction), with **94%** fleet bandwidth reduction at 100 vehicles.
Frame-level CAN IDs, payloads, and exact message order are not transmitted; only aggregated behavioural statistics and anomaly evidence are uplinked.

Payload-statistic reconstruction from descriptors yields **R² ≈ 0.44**, well below the raw-CAN baseline (**R² = 1.0**), indicating limited inference of payload-derived statistics from the uplink alone.
Vehicle fingerprinting remains high (**≈99.97%**) on behavioural descriptors — **anonymisation is not fully achieved**.

**Interpretation:** The descriptor layer substantially reduces data exposure and communication cost while preserving anomaly evidence.
**Limitation:** Residual vehicle-specific patterns remain; privacy-hardening is future work.

## Contribution 3 — Vehicle-Agnostic Descriptor Generalisation

Leave-one-vehicle-out transfer (Random Forest on behavioural descriptor features) yields mean **ROC-AUC 0.745**, mean **F1 0.896**, and vehicle-agnostic score **0.821**.
Transfer is strongest for Kia→Chevrolet (**ROC-AUC 0.856**) and weakest when Chevrolet is the training platform (small sample size).

Cross-vehicle attack similarity gaps are smallest for **replay** and **fuzzy** (≈0.03 cosine gap), supporting behavioural alignment across platforms.
Embeddings (Figure 5) show attack-type structure spanning multiple vehicle markers.

**Interpretation:** Descriptor features encode attack behaviour that transfers across heterogeneous vehicles, supporting fleet deployment.
**Limitation:** ROC-AUC is moderate for linear models (0.60); Chevrolet's smaller corpus limits some pairs; descriptors still permit vehicle classification.

## Contribution 4 — Fleet-Level Correlation Analysis

On the **full labelled dataset**, fleet graph correlation with **≥3 vehicle cluster gates** does **not** improve strong-alert F1 over local IDS (**0.846 vs 0.846**) under the original top-k similarity graph.
**Behaviour-normalized** graph construction increases cross-vehicle edges from **≈0.02%** to **≈1.08%**, enabling cross-platform correlation that identity-dominated similarity suppresses.

For **weak anomalies**, ungated connected-component promotion inflates FPR; **selective DBSCAN promotion** achieves modest recovery (**≈1.3%**, conservative grid-search point) at **FPR ≈ 1.6%** (operating point) to **2.3%** (optimized conservative).
The full-dataset strong-alert evaluation reports identical local and fleet F1 (**0.846**) at **FPR ≈ 44%** — a different operating context from the vehicle-level threshold in Table 1 (FPR ≤ 5%).
The IEEE recovery target (≥10% at FPR ≤ 10%) was **not achieved** in systematic optimization (max recovery **1.41%**).

**Interpretation:** Fleet correlation adds value primarily through (i) cross-vehicle graph connectivity and (ii) gated weak-anomaly promotion, not through blanket cluster escalation.
**Limitation:** Strong-anomaly fleet gains are null under current gates; weak recovery remains low despite cross-vehicle connectivity improvements.
