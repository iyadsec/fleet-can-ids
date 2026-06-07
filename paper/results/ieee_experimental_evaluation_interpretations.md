# IEEE Experimental Evaluation — Interpretations

Evidence supports hypotheses **H1–H4** under the deployment-realistic fleet architecture:
Vehicle IDS → Anomaly Descriptors → Behaviour Graph → GraphSAGE (structure-only) → DBSCAN →
`isolated_attack` / `coordinated_attack`. Attack-type labels are used **only** for evaluation plots and tables.

## H1 — Vehicle-Level IDS Effectiveness (Table 1; Figures 2–3)

The self-supervised Isolation Forest achieves **ROC-AUC 0.786** and **PR-AUC 0.927** at **FPR ≤ 5%**.
Precision is high (**97.3%**) with moderate recall (**46.0%**, F1 **62.4%**), yielding a conservative local alert stream suitable for uplink to the fleet layer.

**Interpretation:** H1 is supported — the vehicle IDS detects suspicious CAN windows locally but cannot classify coordinated multi-vehicle campaigns.

## H2 — Descriptor Compactness and Security (Table 2; Figures 4–5)

Descriptors compress raw CAN windows by **12.6×** (**92%** bandwidth reduction) with **94%** fleet bandwidth reduction at 100 vehicles.
Raw payloads and exact frame order are not transmitted; only aggregated behavioural statistics and anomaly evidence are uplinked.
Payload-statistic reconstruction from descriptors yields **R² ≈ 0.44** vs **R² = 1.0** for raw CAN exposure.

**Interpretation:** H2 is supported — descriptors reduce communication cost and payload disclosure while preserving anomaly evidence.
**Limitation:** Residual vehicle fingerprinting remains high (~99.97%).

## H3 — Cross-Vehicle Descriptor Generalisation (Table 3; Figure 6)

Leave-one-vehicle-out transfer yields mean **ROC-AUC 0.745**, mean **F1 0.896**, and vehicle-agnostic score **0.821**.
Cross-vehicle descriptor embeddings (Figure 6) show attack-behaviour structure spanning vehicle platforms without transmitting raw CAN payloads.

**Interpretation:** H3 is supported — behavioural descriptors generalise across heterogeneous vehicles, enabling fleet-scale correlation without vehicle-identity features in the GNN input.

## H4 — Fleet-Aware GNN Correlation (Tables 4–5; Figures 7–8)

The fleet layer follows: **Vehicle IDS → anomaly descriptors → behaviour-normalized graph → GraphSAGE (structure-only) → DBSCAN → final decision** (`isolated_attack` | `coordinated_attack`).
Runtime decisions use **behavioural cluster cohesion** and multi-vehicle structure — **not** attack-type labels.

On evaluation scenarios (four multi-vehicle attack families), the GNN fleet IDS achieves **100% campaign recall (4/4)** vs **0%** for local IDS alone (Table 5).
**7,267** locally suspicious events are classified as `coordinated_attack`; **49,760** as `isolated_attack` (Figure 8).
Campaign precision is **80%** with behavioural cohesion **0.984** (Table 4); one qualifying cluster is unmatched under evaluation mapping (false campaign rate **20%**).

Figure 7 colours nodes by final fleet decision. Per-attack-type evaluation metrics are reported in Table 5 only.

**Interpretation:** H4 is supported — GraphSAGE fleet correlation adds coordinated-campaign classification beyond isolated local detection.
**Limitation:** Evaluation scenarios are synthetically defined from labelled windows; attack-type names appear only in evaluation tables.
