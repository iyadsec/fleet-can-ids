# Final Report — Real OCSLab Controlled Reviewer Ablation

## Validation checklist (before paper text)

| # | Question | Answer |
|---|----------|--------|
| 1 | All M2/M3/M4 inputs from actual OCSLab held-out TEST observations? | **YES** |
| 2 | Zero synthetic descriptor values? | **YES** |
| 3 | Scenarios identical across methods per seed? | **YES** |
| 4 | GCN and GraphSAGE graphs identical? | **YES** |
| 5 | β distinguished from fragment-merge threshold? | **YES** (0.5 vs 0.85) |
| 6 | Hyperparameters changed after seeing results? | **NO** |
| 7 | Results based on exactly 10 seeds? | **YES** |

Paper subsection drafting is allowed under these checks (all YES). Absolute metrics are **not** claimed to match frozen historical publication tables.

---

## A. Real OCSLab artifact / pipeline

Classic Car-Hacking CSVs/TXT under `Dataset/ocslab_classic_only/` plus Challenge Preliminary D/S CSVs under `Dataset/ocslab/`. Features and scores from current runnable code: 24-D behavioural extractor, benign-only Isolation Forest, 9-D graph view. Pool: `artifacts/test_window_pool.csv` built by `build_scored_pool.py`.

## B. Exact test-data provenance

Interleaved contiguous temporal blocks (70/15/15). IF trained on train-benign only per physical vehicle (`OCSLab_Car`, `Challenge_D`, `Challenge_S`). Ablation scenarios sample **TEST only**. Virtual 20-node fleet = sampled TEST windows (not 20 physical cars).

TEST pool inventory: **7647** windows; strong malicious **2787**; weak malicious **342**; suspicious benign **1997**.

## C. Strong / weak sampling rule

- Strong: label=1 and `anomaly_score ≥ 0.80`
- Weak: label=1 and `0.55 ≤ anomaly_score < 0.80`
- Background: suspicious benign with `anomaly_score ≥ 0.55`
- No score fabrication; weak pool was non-empty after interleaved-block rebuild (342 windows).

## D. Final common configuration

Graph τ=0.95, same-k=2, cross-k=5; GNN 9→64→32, 30 epochs, Adam 0.01, wd 5e-4, λ=0.25; Scaler→PCA(8)→DBSCAN eps=0.5, min_samples=2. See `config.yaml`.

## E. β and fragment-merge thresholds

- **β (minimum campaign cohesion) = 0.5**
- **Fragment centroid merge = 0.85**

Distinct; not both 0.85.

## F. M1/M2/M3/M4 definitions

- **M1** Local IF only — campaign metrics N/A
- **M2** Real descriptors → Scaler → PCA(8) → DBSCAN → common gate
- **M3** Shared real graph → GCN → Scaler → PCA(8) → DBSCAN → gate
- **M4** Same graph → GraphSAGE → Scaler → PCA(8) → DBSCAN → gate

## G. Ten-seed table (mean ± std)

| Method | Strong Campaign F1 | Weak Campaign F1 | Strong Membership F1 | Weak Membership F1 | Independent Merge Rate |
|--------|-------------------:|-----------------:|---------------------:|-------------------:|-----------------------:|
| M1 Local IF | N/A | N/A | N/A | N/A | N/A |
| M2 Descriptor clustering | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| M3 GCN | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| M4 GraphSAGE | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.700 ± 0.483 |

Local F1 (strong scenario): 0.559 ± 0.021 for all methods (shared IF alerts). Campaign precision/recall are in `raw_results.csv` / `summary_results.csv`.

## H. Scenario-sharing verification

All 30 scenario×seed lines in `verification_report.txt` report `shared_ids=OK graph_m3==graph_m4 synthetic=0`. Manifest records source window UIDs, physical vehicle, fleet vehicle id, labels, scores.

## I. Metric definitions

Unchanged from parent harness: greedy Jaccard≥0.5 campaign matching; campaign P/R/F1; membership F1; fragmentation; incorrect merge rate on unrelated incidents; local detection P/R/F1 from IF alerts.

## J. Interpretation

On **real** held-out OCSLab windows with the frozen common gate (β=0.5, fragment=0.85), descriptor clustering and both GNNs fail to recover ground-truth campaigns under this controlled membership protocol: predicted campaigns over-merge (independent merge rate ≈1.0 for M2/M3; 0.70±0.48 for M4), so campaign/membership F1 stay at 0. Local IF still provides moderate window-level detection (F1≈0.56). This does **not** imply the publication’s historical absolute numbers; it answers the reviewer request for a **controlled component ablation on real observations** with shared scenarios/seeds/metrics. GraphSAGE slightly reduces unrelated merges versus GCN/M2 on average, but does not restore campaign F1 under these real-data conditions.
