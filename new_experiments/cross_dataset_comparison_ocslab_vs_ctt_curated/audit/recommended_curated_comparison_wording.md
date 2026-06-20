# Recommended Curated Comparison Paper Wording

## Opening

Table CUR_COMP1 compares the primary OCSLab evaluation with the independent can-train-and-test validation. **The comparison is descriptive rather than a strict benchmark** because the two datasets differ in vehicle population, attack design, and scenario construction.

**OCSLab serves as the primary evaluation dataset**, while **can-train-and-test provides independent external validation** across additional vehicle models, manufacturers, and attack families.

## Headline metrics (Table CUR_COMP2)

Report OCSLab vehicle-level and descriptor numbers from the curated paper export alongside CTT pooled validation metrics. Emphasize that local detection columns are *Comparable with caveat*.

## Fleet scenarios (Table CUR_COMP3)

Both datasets support controlled fleet-scenario evaluation. Benign and isolated scenarios remained safe on CTT. Strong and weak coordinated campaigns were detected. **The CTT unrelated-incident scenario produced a high incorrect-merge rate**, indicating that **behaviour-only graph construction may over-associate unrelated attack traces when temporal constraints are not used**.

## Limitations paragraph

Do not claim benchmark equivalence. CTT cross-vehicle local alert rates are conservative at strong thresholds despite high ROC-AUC. Unrelated-incident separation remains an open limitation on CTT.

## Suggested main-paper artifacts

- **Table CUR_COMP1** — dataset roles
- **Table CUR_COMP3** — scenario outcomes (include unrelated limitation)
- **Figure CUR_COMP1** — coverage
- **Figure CUR_COMP3** — scenario outcomes

Supplementary: CUR_COMP2, CUR_COMP4, CUR_COMP5, Figure CUR_COMP2.
