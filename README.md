# FLEET-GUARD

**Fleet-Level Graph-Based Anomaly Reasoning for Coordinated CAN Attack Detection**

Research code for the FLEET-GUARD framework: vehicle-level Isolation Forest on benign CAN windows, privacy-aware behavioural descriptors, cosine-similarity fleet graphs, GraphSAGE fleet correlation, DBSCAN clustering, and coordinated campaign decision logic.

**Repository:** [github.com/iyadsec/fleet-can-ids](https://github.com/iyadsec/fleet-can-ids)

## Main contribution

FLEET-GUARD correlates **weak, distributed CAN anomaly evidence across vehicles** at fleet level. Individual vehicles may produce only weak or ambiguous local alerts; the framework abstracts compact behavioural descriptors, builds a fleet similarity graph, learns GraphSAGE embeddings, clusters related descriptors with DBSCAN, and applies campaign decision logic to distinguish isolated anomalies from coordinated multi-vehicle attacks.

## Framework overview

| Layer | Method | Paper role |
|-------|--------|------------|
| Windowing | 100-frame CAN windows (50-frame stride default) | Local temporal context |
| Vehicle IDS | Isolation Forest (benign-trained unsupervised) | Anomaly scores, strong alerts, weak descriptors |
| Descriptors | Compact behaviour-based vectors | Cross-vehicle comparison, reduced raw CAN exposure |
| Fleet graph | Cosine similarity + neighbour caps | Behavioural relationships between descriptors |
| Fleet model | GraphSAGE | Fleet correlation embeddings |
| Clustering | DBSCAN | Group related anomaly descriptors |
| Decision | Campaign logic | Distinguish isolated vs coordinated attacks |

**Primary dataset:** OCSLab Car-Hacking / DataChallenge 2019  
**Cross-dataset validation:** [can-train-and-test](https://doi.org/10.11583/DTU.24805533) (DTU) — not CAN-MIRGU

**Privacy note:** Descriptor abstraction is **privacy-aware** (reduces raw CAN exposure for fleet uplink) but does **not** provide cryptographic privacy, differential privacy, secure aggregation, or formal leakage guarantees.

**Campaign evaluation:** Coordinated fleet campaigns are **experimentally constructed from real labelled CAN windows** because public CAN datasets do not contain naturally recorded synchronized fleet-wide coordinated attacks. Attack labels are used for evaluation and scenario assignment only — not as model inputs.

## Validated headline results

Pooled vehicle-level Isolation Forest metrics use validation **FPR ≤ 5%, max recall** threshold selection (Table I protocol). Fleet campaign metrics are archived in the canonical publication bundle.

| Evaluation | PR-AUC | Precision | Recall | F1 | FPR |
|------------|--------|-----------|--------|-----|-----|
| OCSLab primary (pooled) | 0.9273 | 0.9724 | 0.4597 | 0.6243 | 0.0402 |
| OCSLab cross-dataset export | 0.9273 | 0.9728 | 0.4596 | 0.6243 | 0.0395 |
| can-train-and-test (CTT) | 0.9386 | 0.9465 | 0.2272 | 0.3543 | 0.0497 |

| Campaign type | 2 vehicles | 5 vehicles | 10 vehicles |
|---------------|------------|------------|-------------|
| Strong coordinated (F1) | 0.533 | 0.733 | 1.000 |
| Weak coordinated (F1) | 0.067 | 0.500 | 0.717 |

Sources: `results/vehicle_level_threshold_comparison.csv` (OCSLab FPR≤5% row), `experimental-2026-06-23/03_cross_dataset_ctt/tables_csv/LOCAL_COMP1_pooled_ocslab_vs_ctt.csv`, `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P7_strong_campaign_results.csv`, `table_P8_weak_campaign_results.csv`.

## Repository structure

```
.
├── configs/                    # YAML configuration (default.yaml, fleet_ids.yaml)
├── data/raw/                   # Optional local raw traces (gitignored)
├── data/processed/             # Pipeline outputs (gitignored)
├── docs/                       # datasets.md, paper_pipeline.md
├── experiments/                # Numbered OCSLab pipeline scripts (01–09)
├── experimental-2026-06-23/    # Canonical publication result bundle
├── figures/                    # Regenerated publication figures
├── new_experiments/            # CTT cross-dataset run outputs
├── OVERLEAF_CROSS_DATASET_ARTIFACTS/  # Paper tables/figures for Overleaf
├── outputs/                    # Metrics and figures (gitignored)
├── results/                    # FPR-controlled vehicle-level evaluation outputs
├── scripts/                    # Runners, validation, artifact builders
├── src/                        # Core library (data, features, models, graph, ctt)
├── tables/                     # LaTeX/Markdown vehicle-level tables
└── tests/                      # Smoke tests
```

See [docs/paper_pipeline.md](docs/paper_pipeline.md) for paper-to-code mapping.

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
```

## Dataset requirements

Raw datasets are **not** included in this repository.

1. **OCSLab Car-Hacking / DataChallenge 2019** — primary evaluation dataset.
2. **[can-train-and-test](https://doi.org/10.11583/DTU.24805533)** (DTU) — external cross-dataset validation.

Place datasets as described in [docs/datasets.md](docs/datasets.md), then verify layout:

```bash
python scripts/prepare_datasets.py --check
```

Environment overrides (no hardcoded local paths):

```bash
export OCSLAB_DATASET_DIR=/path/to/ocslab
export CTT_DATASET_ROOT=/path/to/can-train-and-test
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Full OCSLab pipeline: windowing, Isolation Forest, graph, GraphSAGE, DBSCAN, campaign logic |
| `configs/fleet_ids.yaml` | Fleet IDS and scenario evaluation settings |

Key settings in `configs/default.yaml`: random seed (`42`), 100-frame windows, benign-only Isolation Forest, cosine graph similarity, GraphSAGE embedding dimension, DBSCAN parameters. All paths are relative to the repository root or overridable via environment variables.

## Running experiments

### OCSLab primary pipeline (step-by-step)

```bash
python experiments/01_load_dataset.py --config configs/default.yaml
python experiments/01b_validate_clean_dataset.py
python experiments/02_generate_windows.py --config configs/default.yaml
python experiments/03_extract_features.py
python experiments/04_train_vehicle_ids.py --config configs/default.yaml
python experiments/05_generate_descriptors.py --config configs/default.yaml
python experiments/06_build_graph.py --config configs/default.yaml
python experiments/07_train_gnn.py --config configs/default.yaml
python experiments/08_cluster_campaigns.py --config configs/default.yaml
python experiments/09_final_decision.py --config configs/default.yaml
```

### OCSLab end-to-end

```bash
python experiments/run_full_pipeline.py --config configs/default.yaml
```

### FPR-controlled vehicle-level evaluation (Table I)

After generating `data/processed/window_features.csv`:

```bash
python scripts/run_vehicle_level_fpr_controlled.py --config configs/default.yaml
python scripts/generate_per_vehicle_fpr_table.py --config configs/default.yaml
```

Outputs: `results/vehicle_level_threshold_comparison.csv`, `tables/table_vehicle_level_ids.tex`, `results/per_vehicle_validation_report.md`.

### Cross-dataset validation (can-train-and-test)

```bash
python scripts/run_can_train_and_test_cross_dataset.py --stage audit
python scripts/run_can_train_and_test_cross_dataset.py --stage pilot
python scripts/validate_can_train_and_test_cross_dataset.py --stage pilot
python scripts/run_can_train_and_test_cross_dataset.py --stage full
python scripts/build_overleaf_cross_dataset_artifacts.py
```

### Campaign-size experiments (strong and weak coordinated campaigns)

Campaign scenarios are generated from labelled OCSLab windows and evaluated through the full fleet pipeline. Canonical archived results:

```bash
python scripts/verify_balanced_campaign_tables.py
python scripts/build_figure_P4_strong_vs_weak_campaign_f1.py
```

Key outputs: `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P7_strong_campaign_results.csv`, `table_P8_weak_campaign_results.csv`, `table_P9_campaign_size_graph_and_cost.csv`.

### Edge-connectivity sensitivity

OCSLab primary sweep: `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P10_edge_connectivity_performance.csv`  
CTT cross-dataset sweep: `OVERLEAF_CROSS_DATASET_ARTIFACTS/tables_md/CTT_CORR7_corrected_edge_sensitivity.md`

Regenerate supporting artifacts:

```bash
python scripts/consolidate_experimental_results.py
python scripts/build_overleaf_cross_dataset_artifacts.py
```

### Regenerate paper tables and figures

```bash
python scripts/regenerate_paper_artifacts.py
python scripts/validate_repository.py
```

Canonical archived results: `experimental-2026-06-23/`

## Expected outputs

| Location | Contents |
|----------|----------|
| `results/` | FPR-controlled vehicle-level CSVs, validation reports |
| `tables/` | LaTeX/Markdown Table I exports |
| `figures/` | Regenerated publication figures |
| `outputs/metrics/` | Pipeline metrics (vehicle IDS, graph, GNN, clustering, decisions) |
| `outputs/figures/` | Training and diagnostic plots |
| `data/processed/` | Window features, descriptors, fleet graph, embeddings |
| `experimental-2026-06-23/` | Canonical paper tables (P4–P12), figures, summaries |
| `OVERLEAF_CROSS_DATASET_ARTIFACTS/` | Cross-dataset comparison tables and figures |
| `new_experiments/` | CTT run outputs and baseline ablation comparisons |

| Stage | Key outputs |
|-------|-------------|
| Vehicle IDS | `outputs/metrics/vehicle_level_self_supervised_results.csv` (legacy filename; unsupervised IF), `results/vehicle_level_threshold_comparison.csv` |
| Descriptors | `data/processed/anomaly_descriptors.csv` |
| Fleet graph | `data/processed/fleet_graph.pt`, `outputs/metrics/graph_statistics.csv` |
| GraphSAGE | `data/processed/node_embeddings.csv` |
| Clustering | `data/processed/fleet_cluster_results.csv` |
| Final decisions | `outputs/metrics/final_detection_outcomes.csv` |
| Publication bundle | `experimental-2026-06-23/`, `OVERLEAF_CROSS_DATASET_ARTIFACTS/` |

## Reproducibility

- Default random seed: `42` (`configs/default.yaml`)
- Vehicle IDS: benign-trained **unsupervised** Isolation Forest (one-class, no attack labels during fitting); Table I uses validation FPR≤5% threshold selection
- Fleet pipeline (P4–P10): fixed strong/weak alert thresholds from balanced publication run (see bundle README)
- Graph: cosine similarity; fleet model: GraphSAGE; clustering: DBSCAN
- Attack labels are used for **evaluation and scenario assignment only**, not as model inputs
- Datasets, virtual environments, caches, and large processed files are gitignored — download data locally and rerun preprocessing
- No repository paths are hardcoded to a specific machine; use `OCSLAB_DATASET_DIR` and `CTT_DATASET_ROOT`

```bash
python -m pytest tests/ -q
python scripts/validate_repository.py
python experiments/check_pipeline_steps.py --list
```

## Validation scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_environment.py` | Python version and dependencies |
| `scripts/prepare_datasets.py --check` | Dataset folder layout |
| `scripts/run_vehicle_level_fpr_controlled.py` | FPR-controlled OCSLab vehicle-level Table I |
| `scripts/generate_per_vehicle_fpr_table.py` | Per-vehicle Chevrolet/Hyundai/Kia + pooled table |
| `scripts/validate_can_train_and_test_cross_dataset.py` | CTT stage outputs |
| `scripts/validate_repository.py` | Publication bundle and config audit |
| `scripts/verify_balanced_campaign_tables.py` | Strong/weak campaign F1 consistency check |
| `experiments/check_pipeline_steps.py` | OCSLab pipeline step inventory |

## Limitations

- No claim of production deployment readiness or real-time fleet operation.
- Cross-dataset transfer uses dataset-specific graph calibration; OCSLab GraphSAGE weights are not blindly transferred to CTT.
- Descriptor abstraction reduces exposure but is not formal privacy preservation.
- Full reproduction requires downloading licensed/raw datasets locally.
- Per-vehicle Table I rows require local OCSLab preprocessing (`window_features.csv`).

## Citation

See [CITATION.cff](CITATION.cff). If you use this code, please cite the FLEET-GUARD paper when published:

```bibtex
@article{fleetguard2026,
  title   = {FLEET-GUARD: Fleet-Level Graph-Based Anomaly Reasoning for Coordinated CAN Attack Detection},
  author  = {TBD},
  journal = {TBD},
  year    = {2026},
  note    = {Placeholder — update when the paper is published}
}
```

## License and status

- **License:** MIT — see [LICENSE](LICENSE). Confirm with authors before public release if a different license is required.
- **Status:** Research reference implementation aligned with the FLEET-GUARD paper. Canonical results are archived in `experimental-2026-06-23/`. Vehicle-level Table I metrics use the FPR-controlled evaluation protocol; fleet campaign tables use the balanced OCSLab publication run.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
