# FLEET-GUARD

**Fleet-Level Graph-Based Anomaly Reasoning for Coordinated CAN Attack Detection**

Research code for the FLEET-GUARD framework: vehicle-level Isolation Forest on benign CAN windows, privacy-aware behavioural descriptors, cosine-similarity fleet graphs, GraphSAGE fleet correlation, DBSCAN clustering, and coordinated campaign decision logic.

**Repository:** [github.com/iyadsec/fleet-can-ids](https://github.com/iyadsec/fleet-can-ids)

## Framework overview

| Layer | Method | Paper role |
|-------|--------|------------|
| Windowing | 100-frame CAN windows (50-frame stride default) | Local temporal context |
| Vehicle IDS | Isolation Forest (benign-only training) | Anomaly scores, strong alerts, weak descriptors |
| Descriptors | Compact behaviour-based vectors | Cross-vehicle comparison, reduced raw CAN exposure |
| Fleet graph | Cosine similarity + neighbour caps | Behavioural relationships between descriptors |
| Fleet model | GraphSAGE | Fleet correlation embeddings |
| Clustering | DBSCAN | Group related anomaly descriptors |
| Decision | Campaign logic | Distinguish isolated vs coordinated attacks |

**Primary dataset:** OCSLab Car-Hacking / DataChallenge 2019  
**Cross-dataset validation:** [can-train-and-test](https://doi.org/10.11583/DTU.24805533) (DTU) — not CAN-MIRGU

**Privacy note:** Descriptor abstraction is **privacy-aware** (reduces raw CAN exposure for fleet uplink) but does **not** provide cryptographic privacy, differential privacy, secure aggregation, or formal leakage guarantees.

**Campaign evaluation:** Coordinated campaigns are assessed using controlled fleet scenarios because public CAN datasets do not provide naturally recorded synchronized fleet-wide attacks.

## Repository structure

```
.
├── configs/                    # YAML configuration (default.yaml, fleet_ids.yaml)
├── data/raw/                   # Optional local raw traces (gitignored)
├── data/processed/             # Pipeline outputs (gitignored)
├── docs/                       # datasets.md, paper_pipeline.md
├── experiments/                # Numbered OCSLab pipeline scripts (01–09)
├── experimental-2026-06-23/  # Canonical publication result bundle
├── new_experiments/            # CTT cross-dataset run outputs
├── OVERLEAF_CROSS_DATASET_ARTIFACTS/  # Paper tables/figures for Overleaf
├── outputs/                    # Metrics and figures (gitignored)
├── scripts/                    # Runners, validation, artifact builders
├── src/                        # Core library (data, features, models, graph, ctt)
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

## Dataset preparation

Raw datasets are **not** included in this repository.

1. Download OCSLab Car-Hacking / DataChallenge 2019 traces.
2. Download can-train-and-test from DOI [10.11583/DTU.24805533](https://doi.org/10.11583/DTU.24805533).
3. Place them as described in [docs/datasets.md](docs/datasets.md).

```bash
python scripts/prepare_datasets.py --check
```

Environment overrides:

```bash
export OCSLAB_DATASET_DIR=/path/to/ocslab
export CTT_DATASET_ROOT=/path/to/can-train-and-test
```

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

### Cross-dataset validation (can-train-and-test)

```bash
python scripts/run_can_train_and_test_cross_dataset.py --stage audit
python scripts/run_can_train_and_test_cross_dataset.py --stage pilot
python scripts/validate_can_train_and_test_cross_dataset.py --stage pilot
python scripts/run_can_train_and_test_cross_dataset.py --stage full
```

### Regenerate paper tables and figures

```bash
python scripts/regenerate_paper_artifacts.py
python scripts/validate_repository.py
```

Canonical archived results: `experimental-2026-06-23/`

## Expected outputs

| Stage | Key outputs |
|-------|-------------|
| Vehicle IDS | `outputs/metrics/vehicle_level_self_supervised_results.csv` |
| Descriptors | `data/processed/anomaly_descriptors.csv` |
| Fleet graph | `data/processed/fleet_graph.pt`, `outputs/metrics/graph_statistics.csv` |
| GraphSAGE | `data/processed/node_embeddings.csv` |
| Clustering | `data/processed/fleet_cluster_results.csv` |
| Final decisions | `outputs/metrics/final_detection_outcomes.csv` |
| Publication bundle | `experimental-2026-06-23/`, `OVERLEAF_CROSS_DATASET_ARTIFACTS/` |

## Reproducibility

- Default random seed: `42` (`configs/default.yaml`)
- Vehicle IDS: benign-only Isolation Forest; strong threshold 0.80, weak threshold 0.55
- Graph: cosine similarity, threshold 0.85
- Fleet model: GraphSAGE; clustering: DBSCAN
- Attack labels are used for **evaluation and scenario assignment only**, not as model inputs

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
| `scripts/validate_can_train_and_test_cross_dataset.py` | CTT stage outputs |
| `scripts/validate_repository.py` | Publication bundle and config audit |
| `experiments/check_pipeline_steps.py` | OCSLab pipeline step inventory |

## Limitations

- No claim of production deployment readiness or real-time fleet operation.
- Cross-dataset transfer uses dataset-specific graph calibration; OCSLab GraphSAGE weights are not blindly transferred to CTT.
- Descriptor abstraction reduces exposure but is not formal privacy preservation.
- Full reproduction requires downloading licensed/raw datasets locally.

## Citation

See [CITATION.cff](CITATION.cff). If you use this code, please cite the FLEET-GUARD paper when published.

## License

See [LICENSE](LICENSE). SPDX: MIT — confirm with authors before public release if a different license is required.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
