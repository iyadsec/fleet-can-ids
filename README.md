# Fleet-Aware CAN-Bus Intrusion Detection

Research codebase for detecting intrusions on Controller Area Network (CAN) bus traffic with **fleet-level** context. The revised workflow demonstrates how vehicle-level IDS findings can be lifted into a behavioural graph to identify coordinated CAN attack patterns across multiple vehicles, beyond isolated per-vehicle detections.

## Project layout

```
.
├── configs/              # Experiment and pipeline configuration (YAML)
├── data/
│   ├── raw/              # Original datasets (not versioned)
│   └── processed/        # Cleaned, aligned, feature-ready tables
├── experiments/          # Runnable experiment scripts
├── notebooks/            # Exploratory analysis
├── outputs/
│   ├── figures/          # Plots and visualizations
│   └── metrics/          # JSON/CSV evaluation results
└── src/
    ├── data/             # Loading and preprocessing
    ├── features/         # Feature extraction
    ├── graph/            # Fleet / ECU graph construction
    ├── models/           # Vehicle IDS and GNN models
    ├── evaluation/       # Metrics and reporting
    └── utils/            # Config, paths, logging
```

## GitHub

**Repository:** [github.com/iyadsec/fleet-can-ids](https://github.com/iyadsec/fleet-can-ids)

### Sync to GitHub

```bash
./scripts/sync_github.sh "describe your changes"
```

### Auto-sync after each commit

```bash
./scripts/install_git_hooks.sh   # run once per clone
```

After that, every `git commit` on `main` automatically runs `git push origin main`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # optional, after pyproject.toml is added
```

## Configuration

Edit or copy `configs/default.yaml` for dataset paths, splits, and experiment metadata. Override paths with environment variables if needed (see `src/utils/config.py`).

## Running experiments

Experiment entry points live under `experiments/`. The end-to-end research workflow is:

`Raw CAN data → Vehicle anomaly detection → Descriptor generation → Behavioural graph construction → GNN / clustering → Final output classification`

**Load and merge CAN datasets** (external Car Track dataset + `data/raw/`):

```bash
python experiments/01_load_dataset.py --config configs/default.yaml
```

Writes `data/processed/clean_can_data.csv` with standardized columns.

**Generate sliding-window metadata** (default: 100 frames, 50-frame overlap):

```bash
python experiments/02_generate_windows.py --config configs/default.yaml
```

Writes `data/processed/window_metadata.csv` (`window_id`, `vehicle_model`, `attack_type`, `label`, …).

**Extract behavioural window features** (entropy, timing, payload stats):

```bash
python experiments/03_extract_features.py
```

Writes `data/processed/window_features.csv` and figures under `outputs/figures/`.

**Train vehicle-level IDS models** (RF, Isolation Forest, Logistic Regression, Autoencoder):

```bash
python experiments/04_train_vehicle_ids.py
```

Writes `outputs/metrics/vehicle_level_results.csv`, `data/processed/vehicle_anomaly_predictions.csv`, and `outputs/figures/confusion_matrix_vehicle.png`.

`data/processed/vehicle_anomaly_predictions.csv` is the stage hand-off into fleet analysis and contains:

`window_id, vehicle_model, source_file, attack_type, true_label, predicted_label, anomaly_score, is_anomaly`

**Generate anomaly descriptors** (suspicious windows only):

```bash
python experiments/05_generate_descriptors.py
```

Writes `data/processed/anomaly_descriptors.csv` using only rows where `is_anomaly = 1`.

**Build fleet anomaly graph** (behavioural similarity edges, no temporal links):

```bash
python experiments/06_build_graph.py
```

Writes `data/processed/fleet_nodes.csv`, `data/processed/fleet_edges.csv`, `data/processed/fleet_graph.pt`, and `outputs/fleet_graph.graphml`.

**Train GNN** on the fleet graph:

```bash
python experiments/07_train_gnn.py
```

Writes `outputs/embeddings/gcn_node_embeddings.pt`.

**Cluster GNN embeddings** (KMeans + DBSCAN, multi-vehicle suspicious campaigns):

```bash
python experiments/08_cluster_campaigns.py
```

Writes `data/processed/fleet_cluster_results.csv` and campaign cluster figures.

**Final decision stage** classifies each anomaly event:

```bash
python experiments/09_final_decision.py
# or
python experiments/run_full_pipeline.py --only final_decision
```

Writes:

- `outputs/metrics/final_detection_outcomes.csv`
- `outputs/metrics/final_outcome_summary.csv`

Decision rule: if an anomaly belongs to a high-similarity cluster containing anomalies from more than one vehicle, it is classified as `Fleet-level coordinated behavioural pattern`; otherwise it is an `Isolated anomaly`.

**Full pipeline** (all steps end-to-end):

```bash
python experiments/run_full_pipeline.py
```

Optional: `--skip-existing`, `--from-step train_gnn`, `--only cluster_campaigns final_decision generate_report`.

Writes `outputs/metrics/pipeline_report.md` and all intermediate artefacts.

**Research report** (figures + `outputs/experiment_report.md`):

```bash
python experiments/09_generate_research_outputs.py
```

```bash
python experiments/run_baseline.py --config configs/default.yaml
```

## Data

Place raw CAN dumps or public benchmark exports in `data/raw/`. Preprocessing scripts write artifacts to `data/processed/`. Large files are ignored by git (see `.gitignore`).

## Status

| Component        | Status              |
|------------------|---------------------|
| Data pipeline    | Implemented |
| Vehicle IDS      | Implemented |
| Descriptors      | Implemented |
| Behavioural graph | Implemented |
| GNN / clustering | Implemented |
| Final outcomes   | Implemented |

## License

Add your license here.

## Citation

Add citation details when publishing results.
