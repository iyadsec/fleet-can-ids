# Final STOP/GO report — CTT aligned to primary OCSLab FLEET-GUARD

## Verdict: **STOP**

Code alignment to the frozen publication fleet methodology is implemented and unit-tested, but a valid scientific cross-dataset comparison is **not** yet claimable.

## What is aligned (implemented)

- Shared fleet path: `src/evaluation/publication_fleet_core.py`
- Freeze graph: τ=0.95, k_same=2, k_cross=5, cosine, no temporal edges
- 9-D GraphSAGE inputs with publication derivations (`message_rate←frame_count`, burstiness, payload_entropy)
- Structure training: L_link + 0.25·L_score, Adam lr=0.01, wd=5e-4, 30 epochs; no GT labels
- DBSCAN: StandardScaler→PCA(8)→euclidean, eps=0.5, min_samples=2
- Campaign gate: centroid cohesion, min_vehicles=2, cohesion=0.5; no attack_type rule
- Legacy CTT results archived under `outputs/ctt_aligned_publication/legacy_archive/`
- Outputs A–H present under `outputs/ctt_aligned_publication/`
- OCSLab publication artifacts **not** modified

## Remaining blockers (STOP reasons)

1. **P7/P8 campaign↔GT matching unrecovered** — using ablation greedy Jaccard≥0.5 only; not proven identical to balanced publication `extract_run_metrics`.
2. **Real can-train-and-test dataset missing** in this environment — “full” metrics are from a **synthetic fixture**, not DTU DOI 10.11583/DTU.24805533.
3. **η (minimum cluster size) greek mapping unrecovered** — freeze has no `|C_k|` key; this run sets η:=`dbscan_min_samples`=2 (rejects peer default `min_cluster_size=10`).
4. **`run_refinement_fcgnn` / SharedFleetConfiguration consumer missing** — freeze values applied explicitly in shared config rather than via the original missing wiring module.
5. **λ and weight_decay** only from peer code defaults (not freeze YAML keys).
6. **Fragment consolidation** is an embedding-centroid approximation of the freeze key (exact missing-module implementation unrecovered).

## What would flip this to GO

1. Recover or explicitly ratify the P7/P8 matcher as Jaccard≥0.5 (or restore missing metrics module).
2. Run the aligned pipeline end-to-end on the real CTT dataset with frozen hypers (no CTT tuning).
3. Document acceptance of η:=2 and peer λ/wd as the publication stand-in configuration.

Until then, do **not** treat synthetic or old F1=1.0 corrected tables as comparable to OCSLab Section VII.
