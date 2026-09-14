# Real OCSLab Reviewer-A Ablation (M1–M4)

Controlled component ablation answering Reviewer A (Sec 4.5.2) using **real**
OCSLab-derived held-out TEST windows. This result set is separate from the
synthetic-descriptor experiment under `experimental-reviewer-ablation/` and
**must not** overwrite those CSVs.

## Artifact / pipeline used

1. Classic OCSLab Car-Hacking traces under `Dataset/ocslab_classic_only/`
   (`normal_run_data`, `DoS`, `Fuzzy`, `gear`, `RPM`).
2. Car Hacking Challenge Preliminary training CSVs for vehicles D and S under
   `Dataset/ocslab/0_Preliminary/0_Training/`.
3. Current runnable FLEET-GUARD code:
   - 24-D behavioural features (`BEHAVIOURAL_FEATURE_COLUMNS`)
   - benign-only Isolation Forest (`fit_self_supervised_isolation_forest`)
   - 9-D graph descriptors matching the parent ablation `FEATURE_NAMES`
4. Pool builder: `build_scored_pool.py` → `artifacts/test_window_pool.csv`

## Split / provenance

- Per-source **interleaved contiguous blocks** assigned 70/15/15 to
  train / validation / test (preserves inter-arrival stats inside each block).
- IF trained on **train benign only** per physical vehicle.
- Ablation scenarios sample **TEST split only**.
- Physical platforms: `OCSLab_Car`, `Challenge_D`, `Challenge_S`.
- Fleet size 20 is a **virtual fleet**: nodes are sampled TEST windows, not 20
  independently recorded vehicles.

## Strong / weak rule

Publication thresholds (unchanged after seeing results):

- strong: `s ≥ 0.80`
- weak: `0.55 ≤ s < 0.80`

No synthetic descriptors; if weak TEST windows are insufficient, sampling uses
replacement of real weak windows only (documented in scenario meta).

## Common configuration

| Component | Setting |
|-----------|---------|
| Graph | cosine τ=0.95, same-vehicle cap=2, cross-vehicle cap=5 |
| GraphSAGE / GCN | 9→64→32, mean agg, ReLU, 30 epochs, Adam lr=0.01, wd=5e-4, λ=0.25 |
| Clustering | StandardScaler → PCA(8) → DBSCAN eps=0.5, min_samples=2 |
| Campaign gate | **β (min cohesion)=0.5**, fragment centroid merge=0.85 |
| Seeds | 11,23,37,41,53,67,71,83,97,101 |

## Methods

- **M1** Local IF only (no campaign reconstruction; campaign metrics N/A)
- **M2** Real descriptors → Scaler → PCA(8) → DBSCAN → common gate
- **M3** Same real graph → 2-layer GCN → Scaler → PCA(8) → DBSCAN → gate
- **M4** Same real graph → GraphSAGE → Scaler → PCA(8) → DBSCAN → gate

## Reproduce

```bash
python3 experimental-reviewer-ablation/real_ocslab/build_scored_pool.py
python3 experimental-reviewer-ablation/real_ocslab/run_real_ablation.py --smoke   # seed 11
python3 experimental-reviewer-ablation/real_ocslab/run_real_ablation.py           # all 10 seeds
```

## Outputs

- `raw_results.csv`, `summary_results.csv`, `ablation_table.csv`
- `scenario_manifest.csv`, `verification_report.txt`
- `config.yaml`, `provenance.md`, `pool_inventory.json`
- `artifacts/` scored pool + `scenario_cache/` frozen scenarios / predictions
