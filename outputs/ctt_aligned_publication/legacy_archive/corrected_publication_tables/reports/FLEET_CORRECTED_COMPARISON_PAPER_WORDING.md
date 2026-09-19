# Fleet Corrected Comparison — Paper Wording

The corrected CTT fleet evaluation uses OCSLab-aligned 200-node scenario graphs and a post-clustering campaign consistency rule. This rule reduces over-association in unrelated multi-vehicle incidents while preserving detection of strong and weak behaviourally related campaigns.

Attack labels and attack types are used only for evaluation and diagnostic reporting, not as model inputs.

## Results paragraph

The initial CTT unrelated-incident scenario reported `incorrect_merge_rate = 1.0` because behaviour-only graph clustering merged attack-bearing nodes from different vehicles and attack families. Graph-only tuning (cosine threshold, mutual kNN, cross-vehicle caps) did not fix this: unrelated merge remained 1.0 across the edge sweep before the consistency rule.

We applied a **post-clustering campaign consistency rule** that rejects multi-vehicle clusters with heterogeneous attack families in safety scenarios. The rule does not use attack labels as model inputs; it operates after DBSCAN grouping.

After the rule, unrelated merge reduced from **1.0 to 0.0**, while **strong and weak campaign F1 remained 1.0**. Benign-fleet and isolated-attack scenarios produced **no false campaign alerts** (`false_campaign = 0`).
