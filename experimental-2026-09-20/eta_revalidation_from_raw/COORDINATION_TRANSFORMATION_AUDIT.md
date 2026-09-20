# COORDINATION_TRANSFORMATION_AUDIT.md

Derived **only** from `coordination_strength.py` (`61a8203…`, RECOVERED EXACTLY).

## Exact transformation

Let \(s \in [0,1]\) be `strength` after `float(np.clip(strength, 0.0, 1.0))`.

Let \(C\) be the list of feature columns actually present from
`BEHAVIOURAL_FEATURE_COLUMNS` (24 columns when complete):

`frame_count`, `unique_can_id_count`, `can_id_entropy`, `most_common_can_id_ratio`,
`mean_inter_arrival_time`, `std_inter_arrival_time`, `mean_dlc`, `std_dlc`,
`byte_mean_0…7`, `byte_std_0…7`.

For the target row set \(T\) (boolean `target_mask`):

\[
\begin{aligned}
\mathbf{m} &= \min_{i \in T} \mathbf{d}_i[C] \\
\mathbf{M} &= \max_{i \in T} \mathbf{d}_i[C] \\
\tilde{\mathbf{p}} &= \mathrm{clip}(\mathbf{p},\, \mathbf{m},\, \mathbf{M}) \\
\sigma &= 0.02\,(1-s)
\end{aligned}
\]

where \(\mathbf{p}\) is `campaign_prototype` (mean behavioural vector over the
chosen `attack_type` in the **full descriptor table** passed to
`compute_campaign_prototype`).

For each target index \(i \in T\):

\[
\mathbf{d}'_i[C]
=
\mathrm{clip}\!\Big(
(1-s)\,\mathbf{d}_i[C]
+
s\,\tilde{\mathbf{p}}
+
\boldsymbol{\varepsilon}_i
,\;
\mathbf{m},\;
\mathbf{M}
\Big)
\]

with \(\boldsymbol{\varepsilon}_i \sim \mathcal{N}(\mathbf{0},\, \sigma^2 I)\)
drawn only when \(\sigma > 0\) (i.e. when \(s < 1\)). When \(s=1\), \(\sigma=0\)
and **no noise is added**.

If \(T\) is empty or \(s \le 0\): return descriptors unchanged.

**Equivalent compact form (ignoring clip/noise):**

\[
\mathbf{d}'_i[C] = (1-s)\,\mathbf{d}_i[C] + s\,\mathbf{p}
\]

i.e. convex combination toward the prototype — **not** assumed a priori; this is
exactly lines 67–70 of the recovered function.

---

## What is / is not modified

| Field / aspect | Modified? | Evidence |
|----------------|-----------|----------|
| Behavioural 24-D features in \(C\) | **Yes** | writes `out.loc[idx, cols]` |
| `anomaly_score` | **No** | not in `BEHAVIOURAL_FEATURE_COLUMNS`; only recorded as `original_anomaly_score` in provenance |
| vehicle identity / `vehicle_token` / `scenario_vehicle_id` | **No** | untouched |
| labels / `attack_type` / GT campaign ids | **No** | untouched |
| timing features (`mean/std_inter_arrival_time`, `frame_count`) | **Yes** (they are in \(C\)) | |
| payload-derived byte means/stds | **Yes** (in \(C\)) | |
| all 24-D behavioural features | **Yes**, when present | |
| 9-D GraphSAGE input directly | **No** | blend runs on behavioural table; 9-D is **derived later** |
| Derived later into GNN view | **Indirectly yes** for `message_rate←frame_count`, `burstiness←std/mean IAT`, `payload_entropy←byte_mean_*`; **`anomaly_score` stays original** | `fleet_similarity_features.build_behavior_view_descriptors`; `compute_payload_entropy` |

Normalization: **none** inside `apply_coordination_strength`. Fleet StandardScaler / PCA occur later in graph/clustering.

Clipping: **yes** — prototype and blended vectors clipped to per-feature min/max of the **target subset**.

Randomness: **yes** iff \(s < 1\) (`numpy` Generator with `seed`); **none** at \(s=1\).

Prototype construction: **column-wise arithmetic mean** of behavioural features over all descriptors with `attack_type == …` — not a sampled row, not a median.

---

## Tiny synthetic example (no real data)

2-D vectors, `feat_min/max` from the pair, zero noise for the linear part:

| \(s\) | \(\mathbf{d}_1=(0,10)\) → | \(\mathbf{d}_2=(8,0)\) → | cosine(\(\mathbf{d}'_1,\mathbf{d}'_2\)) | \(\sigma\) |
|------:|---------------------------:|--------------------------:|-----------------------------------------:|-----------:|
| 0.00 | (0, 10) | (8, 0) | 0.000 | 0.020 |
| 0.35 | (1.4, 7.9) | (6.6, 1.4) | 0.375 | 0.013 |
| 1.00 | (4, 4) | (4, 4) | **1.000** | **0** |

At \(s=1\), both targets become the clipped prototype → identical behavioural vectors.

---

## Interpretation of strength values

| strength | Meaning in recovered code |
|----------|---------------------------|
| **1.0** | Full replacement of target behavioural features by clipped prototype; **no noise**; targets become **identical** on \(C\) (subject to clip). Docstring: “moves fully to prototype”. |
| **0.35** | Keep 65% of original behavioural vector + 35% prototype, then add \(\mathcal{N}(0,(0.013)^2)\) noise, then clip. Vectors remain partially distinct → cosine typically ≪ 1. |
| **0.0** | No-op (early return). |
