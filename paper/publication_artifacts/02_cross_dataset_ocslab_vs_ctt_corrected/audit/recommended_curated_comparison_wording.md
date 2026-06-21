# Recommended Corrected Comparison Paper Wording

**CTT results use corrected evaluation labels and OCSLab-aligned 200-node scenario graphs.**

The comparison is descriptive rather than a strict benchmark because the two datasets differ in vehicle population, attack design, and scenario construction.

OCSLab serves as the primary evaluation dataset, while can-train-and-test provides independent external validation across additional vehicle models, manufacturers, and attack families.

CTT local metrics use eval_attack = (label==1) OR (attack_type!='benign') with FPR<=5% threshold. CTT fleet scenarios use 200-node graphs (τ=0.88, mutual kNN, cross-vehicle cap=3) and a post-clustering consistency rule that rejects multi-family unrelated merges.

The CTT unrelated-incident scenario no longer reports incorrect_merge_rate=1.0 under the corrected protocol.
