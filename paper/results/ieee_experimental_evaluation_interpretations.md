# IEEE Experimental Evaluation — Interpretations

Evidence supports hypotheses **H1–H4** under the deployment-realistic fleet architecture:
Vehicle IDS → Anomaly Descriptors → Behaviour Graph → GraphSAGE (structure-only) → DBSCAN →
`isolated_attack` / `coordinated_attack`. Attack-type labels are used **only** for evaluation plots and tables.

## H1 — Vehicle-Level IDS Effectiveness (Table 1; Figure 2)

We introduce a **lightweight, self-supervised** Isolation Forest vehicle IDS that flags suspicious CAN windows without attack labels at training time.
On the held-out test split it achieves **PR-AUC 0.927** (Figure 2) and **ROC-AUC 0.786** (Table 1) with **FPR ≤ 5%**, **97.3% precision**, and **46.0% recall** (F1 **62.4%**).
This produces a compact local alert stream uplinked as anomaly descriptors to the fleet correlation layer.

**Interpretation:** H1 is supported — self-supervised local detection is effective enough to feed the fleet pipeline but does not classify coordinated campaigns.

## H2 — Descriptor Compactness and Security (Table 2; Figures 4–5)

H2 is evaluated on **two complementary axes** — less data sent, and safer content in what is sent:

1. **Compactness (Figure 4; Table 2)** — *How much* leaves the vehicle?
   Per-window uplink drops from **~2,076 bytes** (raw CAN window) to **~165 bytes** (descriptor), i.e. **12.6×** compression and **92%** bandwidth reduction; at 100 vehicles, fleet uplink falls from **~7.6 GB** to **~425 MB** (**94%** reduction, Figure 4).

2. **Security / privacy (Figure 5; Table 2)** — *What sensitive content* is in that uplink?
   Figure 5 compares raw CAN vs descriptor uplink element-by-element: payload bytes and per-frame CAN IDs are **not transmitted**; message order is summarised; anomaly evidence is **preserved** for fleet IDS.

Together, Figure 4 shows the **volume** reduction; Figure 5 shows the **content** reduction. Table 2 reports the numeric compactness metrics plus disclosure rows (e.g. raw payload bytes: Exposed → descriptor: Not transmitted).

**Interpretation:** H2 is supported — descriptors are a smaller *and* safer fleet uplink than raw CAN, while keeping anomaly evidence for correlation.
**Limitation:** Residual vehicle fingerprinting from behavioural patterns remains high (~99.97%); this is a linkability limit, not a claim of full anonymisation.

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
