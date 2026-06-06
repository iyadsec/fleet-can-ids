# IEEE Experimental Evaluation — Interpretations

This document supports the four validated contributions in the Experimental Evaluation section.
Claims are limited to what the exported evidence supports.

## Contribution 1 — Vehicle-Level IDS Effectiveness

The self-supervised Isolation Forest achieves **ROC-AUC 0.786** and **PR-AUC 0.927** at an operating point selected for **FPR ≤ 5%**.
Precision is high (**97.3%**) but recall is moderate (**46.0%**, F1 **62.4%**), indicating conservative strong-alert generation.
Per-attack F1 ranges from **39.8% (replay)** to **81.3% (fuzzy)**; replay remains the hardest class at the chosen threshold.

**Interpretation:** The vehicle-level IDS provides a usable local baseline with low false alarms, but cannot group cross-vehicle attack behaviour into coordinated campaigns.
**Limitation:** Threshold selection trades recall for FPR; campaign-level reasoning requires the fleet correlation layer.

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

## Contribution 4 — Fleet Campaign Detection

Controlled campaign scenarios were constructed from labelled attack windows across **four attack types** (flooding, fuzzy, replay, malfunction) spanning **2–3 vehicles** each.
These scenarios evaluate fleet-level campaign reasoning; they **do not** represent externally synchronized real-world campaigns.

The behaviour-normalized fleet graph achieves **≈42%** cross-vehicle edges.
DBSCAN clustering on behavioural descriptors yields **one valid cross-vehicle campaign cluster** (flooding, Hyundai+Kia, purity **100%**, mean similarity **0.99**).
Overall **campaign detection rate is 25%** (1/4 scenarios), with **campaign precision 100%** and **false campaign rate 0%** under current gates.
Fuzzy, replay, and malfunction campaigns were **not** recovered as distinct multi-vehicle clusters at the chosen similarity/cohesion thresholds.

Local IDS retains the same per-window attack recall (**≈79%** on campaign windows) but **cannot** perform campaign-level detection (**0%** vs **25%** fleet scenario detection rate).

**Interpretation:** The fleet-aware correlation layer enables campaign-level detection by grouping behaviourally similar anomaly descriptors across multiple vehicles — a capability unavailable to isolated vehicle-level IDS models.
**Limitation:** Detection is strongest for flooding; other attack types overlap behaviourally or form larger mixed clusters; campaign scenarios are synthetically defined from the public dataset labels.
