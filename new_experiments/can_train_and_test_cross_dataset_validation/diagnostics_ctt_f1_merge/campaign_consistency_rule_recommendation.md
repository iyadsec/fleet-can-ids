# Campaign Consistency Rule Recommendation

**DIAGNOSTIC ONLY — post-clustering evaluation gate; not deployed to production pipeline.**

## Rule definition

A cluster is eligible for fleet-campaign signal only if it passes **all** of:

- multi-vehicle (≥2 vehicles)
- mean cross-vehicle edge similarity ≥ 0.78
- mean descriptor variance ≤ 5.0
- cluster density ≥ 1% of graph nodes
- benign contamination ≤ 30%
- attack-family heterogeneity ≤ 1 (single family proxy)

Attack_type and labels are used **only** for this post-hoc gate and evaluation — never as model input.

## Result

On unrelated incidents at τ=0.88, cap=3, mutual kNN:

- **Without rule:** incorrect_merge_rate = 1.0 (all sets)
- **With rule:** incorrect_merge_rate = **0.0** (all sets)

Strong and weak coordinated campaigns retain campaign F1 = 1.0 when the scenario graph still connects campaign nodes.

## Recommendation

Adopt the consistency rule for CTT scenario evaluation alongside OCSLab-aligned 200-node graphs. Confirm on all 10 scenario seeds before updating publication tables.
