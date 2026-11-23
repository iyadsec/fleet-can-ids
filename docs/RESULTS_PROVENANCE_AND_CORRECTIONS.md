# Results provenance and documentation corrections (VehicleSec 2027)

This document is the authoritative description of how the archived FLEET-GUARD results were produced and
which earlier documentation claims are corrected. It changes no result: every CSV, configuration and
report under `experimental-2026-06-23/` and `OVERLEAF_CROSS_DATASET_ARTIFACTS/` is left byte-for-byte as
archived. Where an archived report disagrees with this document, this document takes precedence.

## 1. Primary baseline: June 2026 balanced OCSLab run

| Item | Value |
|------|-------|
| Generating commit | `61a8203` (2026-06-15), branch [`cursor/campaign-clustering`](https://github.com/iyadsec/fleet-can-ids/tree/cursor/campaign-clustering) |
| Entry point | `scripts/run_balanced_publication_run.py` → `src/experiments/final_end_to_end_publication_run_balanced/runner.py` |
| Run directory | `new_experiments/final_end_to_end_publication_run_balanced/` (full batch `20260614T201936`) |
| Archived tables | `experimental-2026-06-23/01_primary_ocslab_balanced/tables/` (P4–P12) |
| Runs | 90 = 9 conditions × 10 seeds (11, 23, 37, 41, 53, 67, 71, 83, 97, 101) |

The archived P5, P7, P8, P9 and P10 tables are byte-identical to the tables written by the June run, and
the P7/P8 means recompute exactly from the June per-run `run_level_metrics.csv`. P6 in the bundle reports
S2 incorrect merging 0.400 (4 of 10 seeds), consistent with the per-run metrics; the `table_P6` file in the
local run directory reports 0.0 for the same column and is not used.

### Code that is not in any public commit

`runner.py` at `61a8203` imports three packages that were never committed:
`src/experiments/final_end_to_end_publication_run/` (15 modules, including `source_windows.py`, which
builds the S0–S4 test scenarios), `src/experiments/coordinated_campaign_refinement/` (11 modules, including
`refinement_pipeline.py`: candidate graph, GraphSAGE, DBSCAN, fragment consolidation) and
`src/experiments/final_shared_configuration/` (12 modules, including `metrics.py`). The runner's import
closure needs 25 of these 38 modules plus 4 others that are also uncommitted (`shared_statistics.py`,
`strong_campaign_extended/{__init__,guard,metrics}.py`). All 42 are published in
[`reproducibility/june_2026/`](../reproducibility/june_2026/), with a manifest of SHA-256 hashes.
The local copies' 2026-06-19 modification time is a later touch. The June CPython bytecode records earlier
source dates and matching sizes. 38 modules compile to code identical to bytecode written before the batch.
`final_shared_configuration/shared_config.py` and its empty `__init__.py` match only bytecode recompiled on
16 June, and two modules the runner does not import differ from the June bytecode. These four are marked
unverified. The original `.pyc` files embed local paths and are kept privately; the snapshot publishes a
path-free record of them (`bytecode_record.csv`). A full rerun from raw data with the snapshot reproduced every June detection result exactly,
including the selected shared configuration and all 90 runs. The remaining differences are wall-clock time,
a random scaler label, and the path-dependent P5 descriptor size (see the snapshot README). SHA-256 of the key
local files:

| File | SHA-256 (first 16) | In public commit |
|------|--------------------|------------------|
| `src/experiments/campaign_analysis_corrected.py` | `32f8e52f3a9a3122` | yes (`61a8203`) |
| `src/experiments/coordination_strength.py` | `fd9334697e6dc760` | yes |
| `src/experiments/final_end_to_end_publication_run_balanced/runner.py` | `ad850e38d0d3eea4` | yes |
| `src/evaluation/final_gnn_fleet_decision_experiment.py` | `6628aa2d67ca648c` | yes |
| `src/graph/fleet_similarity_features.py` | `18ce8a68a937557f` | yes (also on `main`) |
| `src/experiments/final_end_to_end_publication_run/source_windows.py` | `78c55075e5702e7e` | no |
| `src/experiments/final_end_to_end_publication_run/runner.py` | `756a4d393fabdfe8` | no |
| `src/experiments/coordinated_campaign_refinement/refinement_pipeline.py` | `bd20e7e9b92aea05` | no |
| `src/experiments/coordinated_campaign_refinement/candidates.py` | `8a664b50a265d6e5` | no |
| `src/experiments/final_shared_configuration/metrics.py` | `7f82c16d2485d5f3` | no |
| run: `results/scenario_evaluation/run_level_metrics.csv` | `94b9507c1d2cda73` | yes |
| run: `configs/final_shared_fleet_configuration.yaml` | `84f501c9ef0d4fef` | yes |
| run: `scalers/fleet_benign_scaler.json` | `c43ee2bf0f3b02b4` | local |
| run: `descriptors/all_descriptors.csv` | `c60abfca405fbb8d` | local |

The numbered pipeline in `experiments/01`–`09` with `configs/default.yaml` is a reference implementation of
the same stages. It is **not** the code that produced Tables P4–P12.

## 2. Implementation as executed in the June run

| Stage | As executed |
|-------|-------------|
| Windows | 100 frames, stride 50; segment-level temporal split with 100-frame guard bands where a recording spans partitions |
| Local IDS | One Isolation Forest per platform, 200 trees, seed 42, fitted on benign training windows |
| Local thresholds | θ_weak = 0.45; θ_strong = 0.70 (Hyundai/Kia), 0.85 (Chevrolet). The 0.55/0.80 values in the master YAML were not used |
| Candidate forwarding | 100% of descriptors in every fleet scenario were forwarded (`fleet_candidate_rate` = 1.0); no local filtering occurred |
| Graph features | 9-D `GNN_FEATURE_COLUMNS`, z-scored with the global benign-train fleet scaler (11,423 rows); `frame_count` and `message_rate` are constant (0) after scaling |
| Cohesion view | `behavior_only_vehicle_normalized` (despite the name, scaled with the same global fleet scaler): up to 8 behaviour features (anomaly score, frame count, message rate, burstiness, mean/std inter-arrival time, CAN-ID entropy, most-common-ID ratio); any feature whose between-/within-vehicle variance ratio exceeds 5 in the scenario is dropped. The June log records 8 features in 425 calls, 7 in 1,090 and 6 in 5 |
| Graph | Cosine constrained kNN: τ = 0.95, k_same = 2, k_cross = 5 |
| GraphSAGE | 9→64→32, 30 epochs, Adam lr 0.01, weight decay 5e-4; self-supervised link loss + 0.25 × MSE to the min–max-normalised anomaly score; no attack labels |
| Clustering | DBSCAN on standardised PCA-8 embeddings, eps 0.5, min_samples 2; fragment consolidation at centroid similarity 0.85 |
| Campaign decision | ≥ 2 distinct vehicles, cross-vehicle support ≥ 1, cohesion ≥ 0.5 (behaviour view), cluster size ≥ 10 (`min_cluster_size` default) |
| Parameter selection | Shared configuration chosen on a separate validation suite (14 candidates), not on the test scenarios |

## 3. Scenario construction (controlled)

* **S0 benign fleet** – one participant with 200 pooled benign windows. 54.3% of these benign windows
  exceed the weak local threshold, so S0 tests whether the fleet stage suppresses many benign local alerts;
  the ≥ 2-vehicle rule makes a campaign structurally impossible here.
* **S1 isolated attack** – three participants, 20 nodes, Chevrolet flooding on one vehicle.
* **S2 unrelated incidents** – five single-vehicle incidents plus one benign participant, downsampled to
  200 nodes. All five incidents are fuzzy attacks; all four incorrect merges (seeds 23, 41, 71, 97) came
  from `Fuzzy_dataset_KIA_Soul_train.csv`. Under a consistent definition both the incorrect-merging rate
  and the false-campaign rate equal **0.400**; the archived scorer's false-campaign value of 0.000 results
  from an inconsistent `n_gt`.
* **S3/S4 strong/weak campaigns** – real labelled OCSLab windows. Before graph construction, a shared
  prototype (mean of all malfunction descriptors across **all** splits, clipped) is blended into the
  descriptors at coordination strength 1.0 for all 10 windows of each attacked participant, including its
  5 benign windows. Anomaly scores are not recomputed. Cosine similarity, GraphSAGE, DBSCAN and the
  campaign decision then run unchanged and without labels. S3/S4 therefore measure detection of
  descriptor-coordinated campaigns, not naturally recorded ones.
* **Replay** has no windows in the test split and was **not** evaluated in any fleet scenario.

## 4. Separate sensitivity analysis (2026-11-21)

`validation_2026-11-21_natural_campaigns/` re-runs S3/S4 with the June artefacts, thresholds and shared
configuration, changing only campaign construction. It does not replace any June number. Condition A
reproduces June exactly. Condition B uses unmodified OCSLab observations (no prototype, benign windows
untouched); these are real attack windows assembled into a scenario, **not** naturally recorded coordinated
campaigns. Removing the controlled descriptor similarity significantly reduced detection (Wilcoxon
p ≤ 0.002 for all comparisons). Details: [`validation_2026-11-21_natural_campaigns/README.md`](../validation_2026-11-21_natural_campaigns/README.md).

## 5. Corrected claims

| Earlier claim (where) | Correction |
|-----------------------|------------|
| 24-D descriptors feed the graph (README, reports) | Graph and GraphSAGE use the 9-D scaled view; cohesion uses a 6–8-feature behaviour view (§2); 24-D is the local descriptor table |
| Descriptor bandwidth reduction 53.69% (`table_P5`) / 92.05% (`table_CUR_COMP2`) / "425 MB vs 7.6 GB" (`CUR_COMP5`) | Superseded. P5 compares a nominal 1,600 B window with the 741 B in-memory size of a descriptor row. Measured per window on 322 test windows from 8 recordings: packed binary 44 B vs 2,035 B raw (97.84% smaller); CSV text 79 B vs 4,679 B (98.32%). This is per-window payload size; because 100% of windows were forwarded and windows overlap (stride 50), it is not a fleet bandwidth figure |
| Local filtering reduces uplink | 100% of windows were forwarded in all fleet scenarios |
| Replay evaluated in fleet scenarios | Not evaluated (no replay test windows) |
| S2 false-campaign rate 0.000 | 0.400 under a consistent definition (see §3) |
| CTT "consistency rule" fixes false campaigns (`03_cross_dataset_ctt`, `OVERLEAF_CROSS_DATASET_ARTIFACTS`) | `apply_campaign_consistency_filter` uses the scenario name and labels and is an oracle. The core CTT `evaluate_campaign` selects clusters using labels and attack types and does not apply the cohesion/size gate. CTT fleet results are ground-truth-assisted scoring, not deployable detection. CTT also used τ 0.88, mutual kNN, cap 3 and 8 GNN epochs |
| Vehicle-level pooled table (PR-AUC 0.9273) | Produced with a random stratified 70/15/15 split of stride-50 overlapping windows (adjacent train/test windows share frames) and per-vehicle thresholds; not the balanced split. Treat as optimistic |
| Edge-connectivity table in the manuscript (380/969/1,810 edges) | From the earlier pre-balanced run (`final_end_to_end_publication_run/tables/table_P10`); its "Best" row mixes τ 0.95/k 20 (strong) with τ 0.85/k 20 (weak). The balanced run's P10 (341–1,886 edges) is in this bundle |
| Method ablation table (M1/M2/M3) | From an earlier configuration (`62bf982`); see `experimental-2026-06-23/02_baseline_ablation/RESULT_PROVENANCE.md` |

## 6. Verification performed for this release

* `git hash-object` of every tracked file under `experimental-2026-06-23/` equals its committed blob.
* Bundle P5/P7/P8/P9/P10 tables are byte-identical to the June run directory tables.
* P7/P8 campaign and membership F1 recompute from the June `run_level_metrics.csv`:
  strong 0.533 / 0.733 / 1.000 (membership 0.490 / 0.725 / 0.928);
  weak 0.067 / 0.500 / 0.717 (membership 0.057 / 0.564 / 0.825).
* Reproduction from raw data with `reproducibility/june_2026/` (2026-11-23): detection results equal June
  exactly. Two reproducibility defects were found and are documented there. The balanced split is seeded with
  Python's per-process `hash()`, so it must be taken from the committed manifests; the committed code can
  regenerate the archived split. Two raw Kia files in the local dataset folder are now 0 bytes, and intact
  copies reproduce the June features exactly.
* `scripts/verify_balanced_campaign_tables.py`: all per-seed checks pass. Its `PR_check` column reports
  FAIL where the mean of per-seed F1 differs from the F1 of mean precision and recall; this is expected.
