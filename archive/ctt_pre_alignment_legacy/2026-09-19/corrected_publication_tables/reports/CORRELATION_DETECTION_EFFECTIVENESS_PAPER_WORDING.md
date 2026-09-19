# Correlation Detection Effectiveness — Paper Wording

## Scope

These figures evaluate the **fleet-correlation layer**, not only the local IDS.

1. **Local IDS** identifies individual anomalous windows.
2. **Fleet correlation** determines whether anomalies across multiple vehicles form a coordinated campaign.
3. Corrected CTT uses **OCSLab-aligned 200-node scenario graphs** with **no temporal edges**.
4. Corrected CTT detects **strong and weak campaigns with F1 = 1.0**.
5. Corrected CTT avoids **benign and isolated false campaigns** (`false_campaign = 0`).
6. The **campaign consistency rule** reduces unrelated incident merging from **1.0 to 0.0**.
7. This demonstrates the effectiveness of correlation-based fleet reasoning on the external can-train-and-test dataset.

## Required paragraph

While local IDS performance reflects individual anomaly detection, campaign identification is evaluated at the fleet-correlation layer. The corrected can-train-and-test results show that the proposed correlation layer detects strong and weak behaviourally related campaigns across previously unseen vehicle models while avoiding false escalation of benign and isolated incidents.

## Comparison caveat

The comparison is descriptive rather than a strict benchmark because OCSLab and can-train-and-test differ in vehicle population, attack taxonomy, and labelling structure.

## Graph protocol

Both datasets use 200-node scenario graphs. Temporal edges are not used. Attack labels and attack types are used only for evaluation and diagnostics, not as model inputs.
