# Recommended Comparison Paper Wording

Use the following phrasing in the manuscript. All comparisons are **descriptive cross-dataset comparisons** unless metric definitions are confirmed identical (see `metric_compatibility_audit.md`).

## Opening paragraph

> Table COMP1 compares the primary OCSLab / DataChallenge 2019 evaluation with the independent can-train-and-test validation. The comparison is **descriptive** because the datasets differ in vehicle population, attack design, file structure, and scenario construction. The OCSLab experiments provide the primary controlled fleet-campaign evaluation, while can-train-and-test provides external validation across additional vehicles, manufacturers, and attack types.

## Positive cross-dataset evidence

> The can-train-and-test results confirm that the framework can be applied to a larger and more diverse vehicle population, including four vehicles from two manufacturers and nine attack types. Descriptor compactness (~70% bandwidth reduction) and behavioural fleet-graph construction (cross-vehicle edge fraction ~0.27%, zero temporal edges) transfer without structural modification.

## Fleet scenario paragraph

> Both datasets support the controlled evaluation of behaviourally related multi-vehicle campaign scenarios. The proposed fleet layer avoided benign and isolated false campaign escalation in can-train-and-test, while strong and weak campaign scenarios were detected with campaign F1 = 1.0. However, unrelated multi-vehicle incidents showed a high incorrect-merge rate (incorrect_merge_rate = 1.0), indicating that behaviour-only graph construction can over-associate semantically different attacks when temporal constraints are intentionally excluded.

## Local detection paragraph

> Local detection metrics are reported descriptively (Table COMP2). can-train-and-test cross-vehicle test subsets exhibit low pooled F1 at strong alert thresholds despite high ROC-AUC, reflecting conservative transfer rather than failure of the end-to-end pipeline.

## Do not claim

- can-train-and-test contains real synchronized fleet campaigns.
- The two datasets produce directly equivalent benchmark scores.
- Unrelated incident separation is solved (CTT incorrect_merge_rate = 1.0).

## Suggested table/figure references

- **Main paper:** Table COMP1 (dataset roles), Table COMP5 (scenario outcomes), Figure COMP5 (scenario bar chart).
- **Supplementary:** Tables COMP2–COMP4, COMP6–COMP8; Figures COMP1–COMP4, COMP6–COMP7.
