# CTT Local F1 Diagnostic Report

Generated: 2026-06-21T16:10:16.609897+00:00

**DIAGNOSTIC ONLY — does not replace official publication tables.**

## Executive summary

The pooled CTT3 F1 (~4.2%) is **not** representative of ranking quality. ROC-AUC ~99.4% on test_01 shows the model ranks attack windows well.

### Root causes of low pooled F1

1. **F1 uses strong alerts only** (97.5th percentile benign validation threshold). Weak candidates are excluded from CTT3.
2. **Ground truth uses `label` column only**, while many CTT attack files carry `label=0` despite non-benign `attack_type` (e.g. DoS, Silverado). Fleet/scenario paths use `_is_attack_window()` but local metrics do not.
3. **Label–attack_type mismatches:** 1,553,365 windows have attack_type≠benign but label=0.
4. **Sets with zero label=1 attack windows on known vehicle:** set_02, set_03. Pooled recall=0.25 is a mean artefact (only set_01 contributes tp>0).
5. **Threshold miscalibration:** set_01 Impala strong F1=0.167 locally but precision is low due to 17%... high FPR from benign windows scoring above strong threshold.
6. **test_02–test_04** include unknown vehicles scored with transferred models; many have no label=1 positives → zero contribution to recall numerators.

### Answers to diagnostic checklist

| # | Question | Finding |
|---|----------|---------|
| 1 | Strong alerts only? | **Yes** — CTT3 aggregates `mode=strong`. |
| 2 | Threshold too high? | **Yes** — 97.5th pct benign; many attack scores (incl. label=0 attacks) fall below strong threshold. |
| 3 | Labels normalized? | Labels copied from source `attack` column; not re-derived from filename. |
| 4 | label=0 attack files? | **Yes** — widespread; Silverado has 0 label=1 attack windows. |
| 5 | Positive class consistent? | **No** — local uses label; fleet uses label OR attack_type. |
| 6 | Extreme imbalance? | **Yes** — millions of benign vs hundreds of label=1 windows. |
| 7 | Dominated by one subset? | set_01/test_01 only set with meaningful label=1 strong detection. |
| 8 | Subsets reported separately? | **Yes** in per-set tables; pooled mean obscures set_01 F1≈0.17. |
| 9 | High ROC-AUC, bad threshold? | **Yes** — ranking good; operating point poor for F1 under label-only GT. |
| 10 | attack_type vs label inconsistent? | **Yes** — see label audit CSV. |
