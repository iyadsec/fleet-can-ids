# Local Threshold Policy Recommendation

## Why old pooled F1 was low (~4.2%)

CTT3 used **strong alerts only** with **label-only** ground truth averaged across four sets. Many attack windows have `attack_type != benign` but `label = 0`. Three sets contribute zero label=1 positives on the known vehicle, collapsing pooled recall.

## Why ROC-AUC was high (~99.4%)

Isolation Forest scores rank attack windows well on test_01; the failure was **threshold + ground truth**, not ranking.

## Corrected evaluation ground truth

`eval_attack = (label == 1) OR (attack_type != 'benign')` — **evaluation only**, never model input.

## Recommended publication operating point

**Policy B: FPR <= 5%** — aligns with OCSLab threshold calibration (FPR<=5%).

**Policy D: F1-optimal** — diagnostic only; do not use as primary headline.

## Corrected pooled F1 (eval ground truth)

| Policy | Mean F1 |
|--------|---------|
| FPR <= 5% (official) | 0.3543 |
| F1-optimal (diagnostic) | 0.9333 |
| Existing strong (old) | ~0.1667 on label-only subset |
