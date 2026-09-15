# Draft response to Reviewer A (Sec 4.5.2)

> “Sec 4.5.2: Would be interesting to add an ablation study to understand how each
> component in the pipeline contributes to the reported metrics”

We thank the reviewer for this suggestion. We have added a **new controlled
component ablation** (not a re-use of mixed historical CSVs) in which all variants
share the same scenarios, seeds, descriptor node sets, clustering/campaign-gate
settings, and evaluation metrics; only the ablated component changes.

The study compares:
1. **local detection** (Isolation Forest alerts only),
2. **descriptor clustering** (no graph / GNN),
3. a **standard GCN** on the same fleet similarity graph, and
4. **GraphSAGE** (full FLEET-GUARD fleet correlation).

Under this shared protocol, direct descriptor clustering already recovers strong
coordinated campaigns well, while GraphSAGE provides additional benefit primarily
on weak campaigns and does not increase incorrect merging of unrelated incidents.
A comparable GCN baseline does not yield the same campaign recovery and increases
incorrect merges. We emphasize that this ablation isolates *relative* component
contributions under a common configuration and does not claim that absolute metric
values match the frozen historical publication tables.
