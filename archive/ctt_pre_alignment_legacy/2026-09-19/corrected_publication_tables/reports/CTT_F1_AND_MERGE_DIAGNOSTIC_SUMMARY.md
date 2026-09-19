# CTT F1 and Merge Diagnostic Summary

**Output root:** `new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge`  
**Generated:** 2026-06-21 16:47 UTC

**DIAGNOSTIC ONLY — official publication tables not modified.**

## 1. Why was CTT local F1 low?

Pooled CTT3 F1 (~4.2%) reflects **strong-mode, label-only** metrics averaged across four sets. Three sets have **zero label=1 attack windows** on the known vehicle (Silverado/Forester label artefact). set_01 Impala achieves strong F1≈0.17 with recall=1.0 on label=1 but low precision due to high FPR. Many attack windows have **label=0** despite attack_type≠benign, so they never count as positives under official local metrics.

## 2. Was ranking actually good?

**Yes.** ROC-AUC ~0.994 on test_01; PR-AUC lower due to imbalance. High ROC-AUC with low F1 confirms **threshold/ground-truth mismatch**, not ranking failure.

## 3. Which threshold policy improves F1?

Best diagnostic policy on test_01 with eval ground truth: **D_fpr_le_5pct**. FPR≤5% (policy D) aligns with OCSLab calibration. F1-optimal (B) raises F1 substantially on eval labels but requires confirmed ground-truth rule.

## 4. Are CTT labels inconsistent?

**Yes.** 1,553,365 windows have attack_type≠benign but label=0. Silverado attack files commonly have no label=1 windows.

## 5. Is the OCSLab graph comparison unfair?

**Yes.** OCSLab scenario package uses **200-node** graphs with **370–1311 edges**. CTT official validation uses **~100k-node** production graphs — not comparable for scenario merge behaviour.

## 6. Fair 200-node CTT scenario graph results

See `ctt_200_node_scenario_graph_results.csv`. Edge counts at τ=0.85 are orders of magnitude below production graphs and closer to OCSLab scale when capped at 200 nodes.

## 7. Why did unrelated incidents merge?

DBSCAN forms **one multi-vehicle cluster** on dense behaviour-only graphs; `incorrect_merge_rate=1.0` by design when ≥2 vehicles share an attack cluster. Cross-vehicle cosine edges at 0.85 with cap=20 over-connect unrelated families.

## 8. Which graph calibration reduces incorrect_merge_rate?

Graph-only parameter sweeps (1,068 configs) did **not** reduce merge below 1.0. The **campaign consistency rule** drops unrelated incorrect_merge_rate from 1.0→0.0 on all sets (τ=0.88, cap=3, mutual kNN). See `graph_calibration_recommendation.md`.

## 9. Does mutual kNN help?

**Yes** — reduces spurious cross-vehicle edges and unrelated merge in calibration sweep.

## 10. Does limiting cross-vehicle edges per node help?

**Yes** — cap=1–3 materially lowers incorrect_merge_rate vs cap=20.

## 11. Does campaign consistency filtering help?

**Yes** — unrelated incorrect_merge_rate drops 1.0→0.0 on all four sets when enabled; strong/weak campaign F1=1.0 preserved when graph connects campaign nodes.

## 12. Recommended configuration for final CTT validation (diagnostic proposal)

- **Local metrics:** eval ground truth rule + FPR≤5% threshold policy (or per-vehicle F1-optimal under eval labels)
- **Scenario graphs:** 200-node OCSLab-aligned graphs, τ=0.88, k=10, cross_vehicle_cap=3, mutual kNN, campaign consistency gate
- **Do not apply until reviewed and confirmed with a dedicated validation rerun policy**

## 13. What should replace/supplement current comparison?

- Supplement CUR_COMP2/CUR_COMP3 with **diagnostic tables** using eval ground truth and 200-node scenario results
- Do **not** overwrite official CTT3/CTT7 until ground-truth and graph protocol changes are approved

## 14. Remaining limitations

- Label provenance in source CAN-train-and-test files
- 200-node padding uses benign pool — not identical to OCSLab scenario injection
- Graph calibration sweep on 2 sets × 1 seed — confirm on all seeds before publication
- Campaign consistency rule thresholds are heuristic
