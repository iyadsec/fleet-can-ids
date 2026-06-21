# CTT Corrected Publication Summary

**Output:** `new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt`  
**Generated:** 2026-06-21 19:37 UTC

## 1. Old low local F1 cause
Strong-only alerts + label-only GT + pooled across sets with label=0 attack files.

## 2. Corrected ground truth
eval_attack = (label==1) OR (attack_type!='benign') — evaluation only.

## 3. Recommended local policy
**FPR <= 5%** (Policy B).

## 4–5. Corrected pooled F1
- FPR<=5%: 0.3543
- F1-optimal: 0.9333

## 6–7. By subset / attack type
See CTT_CORR2, CTT_CORR3.

## 8. Corrected 200-node scenarios
See CTT_CORR6 — τ=0.88, cap=3, mutual kNN, consistency rule (fallback 0.85/cap5 when needed).

## 9. Unrelated merge
Before rule: 1.000 → After: 0.000

## 10–11. Campaign F1
Strong: 1.000; Weak: 1.000

## 12. Fair OCSLab comparison
200-node graphs (~1k edges) vs OCSLab 370–1311; corrected eval labels for local metrics.

## 13. Limitations
Benign padding to 200 nodes; GNN re-inference on cached descriptors only; confirm all seeds in supplement.

## 14. Replace old tables
CTT_CORR1–CORR7 supplement official CTT3/CTT7; do not overwrite full/.

## 15. Paper figures
CORR4 scenario outcomes; CORR5 unrelated merge ablation; CORR3 local by subset.
