# Final STOP/GO report — CTT aligned to primary OCSLab FLEET-GUARD

## Verdict: **READY_FOR_ETA_DECISION**

DBSCAN and campaign-gate η are decoupled; CTT will not run until η is supplied explicitly and frozen independently of CTT results. Historical OCSLab / P7 / P8 artifacts are untouched. No CTT experiment was re-run in the η-decoupling correction.

See also: `ETA_DBSCAN_SEPARATION_AUDIT.md`.

## What is aligned (implemented)

- Shared fleet path: `src/evaluation/publication_fleet_core.py`
- Freeze graph: τ=0.95, k_same=2, k_cross=5, cosine, no temporal edges
- 9-D GraphSAGE inputs with publication derivations (`message_rate←frame_count`, burstiness, payload_entropy)
- Structure training: L_link + 0.25·L_score, Adam lr=0.01, wd=5e-4, 30 epochs; no GT labels
- DBSCAN: StandardScaler→PCA(8)→euclidean, eps=0.5, min_samples=2 (**verified**; unchanged)
- Campaign gate: centroid cohesion; freeze `minimum_distinct_vehicles=2`, `minimum_campaign_cohesion=0.5`; **η (`min_campaign_cluster_size`) required explicitly — no default, no DBSCAN inheritance**
- Legacy CTT results archived under `outputs/ctt_aligned_publication/legacy_archive/`
- OCSLab publication artifacts **not** modified

## Prior PR #24 error (corrected)

PR #24 incorrectly set `η := dbscan_min_samples = 2` via `minimum_cluster_size`. PR #25 established that historical η is **UNRECOVERABLE**. That coupling is removed.

## Remaining blockers before a scientific CTT claim

1. **Decide and freeze prospective η** independently of CTT results, then pass `--eta N`.
2. **P7/P8 campaign↔GT matching unrecovered** — ablation greedy Jaccard≥0.5 only.
3. **Real can-train-and-test dataset** required for non-synthetic comparison (DTU DOI 10.11583/DTU.24805533).
4. **`run_refinement_fcgnn` / SharedFleetConfiguration consumer missing** — freeze values applied explicitly in shared config.
5. **λ and weight_decay** only from peer code defaults (not freeze YAML keys).
6. **Fragment consolidation** is an embedding-centroid approximation of the freeze key.

## What would allow a CTT run (still not GO for publication comparison)

1. Freeze η without looking at CTT metrics.
2. Run aligned pipeline with `--eta N` on real CTT data.
3. Separately resolve matcher / dataset / peer-only hypers for comparability claims.

Until η is decided: do **not** run CTT; do **not** invent historical η; do **not** treat prior synthetic tables that used η:=2 as historically justified.
