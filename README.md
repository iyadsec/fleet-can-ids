# Fleet-Aware CAN-Bus Intrusion Detection

Research codebase for detecting intrusions on Controller Area Network (CAN) bus traffic with **fleet-level** context. The workflow demonstrates how vehicle-level IDS findings can be lifted into a behavioural graph to identify coordinated CAN attack patterns across multiple vehicles, beyond isolated per-vehicle detections.

## Reproduce the paper results (quick start)

The project expects **raw datasets to stay outside the GitHub repo**. You provide a local dataset folder path in the config.

### Requirements

- **Python**: 3.11 or 3.12 recommended (3.13 may fail for `torch-geometric` on some platforms)
- OS: macOS / Linux / Windows (commands below show macOS/Linux)

### 1) Clone and set up a virtual environment

```bash
git clone https://github.com/iyadsec/fleet-can-ids.git
cd fleet-can-ids

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sanity check:

```bash
python3 -c "import numpy,pandas,sklearn,matplotlib; print('deps ok')"
```

### 2) Point the config to your local dataset folder

Edit `configs/fleet_ids.yaml`:

```yaml
data:
  external_dataset_dir: "/absolute/path/to/your/Dataset"
```

Notes:
- Use an **absolute path**.
- If your path contains spaces, keep it **quoted** as shown.
- Do not commit datasets to GitHub.

### 3) Run the full end-to-end pipeline + export publication package

```bash
python3 run_all_experiments.py --config configs/fleet_ids.yaml
```

### 4) Outputs you should expect

This command runs the complete pipeline and then exports an IEEE-friendly package:

- **Publication package (paper-ready)**
  - `results/` (CSV + JSON index)
  - `figures/` (PNG)
  - `tables/` (LaTeX `.tex`)
  - `logs/` (logs)
  - `final_experiment_summary.md`

- **Pipeline artifacts (intermediate + research evidence)**
  - `data/processed/`
  - `outputs/metrics/`
  - `outputs/figures/`

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

## Configuration

Two main configs are provided:

- `configs/fleet_ids.yaml`: paper deliverables run (recommended for external users)
- `configs/default.yaml`: default pipeline configuration

To run faster on repeat executions:

```bash
python3 run_all_experiments.py --config configs/fleet_ids.yaml --skip-existing
```

## Running experiments

Experiment entry points live under `experiments/`. The end-to-end research workflow is:

`Raw CAN data → Vehicle anomaly detection → Strong/weak evidence classification → Descriptor generation → Raw-vs-descriptor size comparison → Behavioural graph construction → GNN embedding learning → Clustering → Final outcome classification → Research evidence summary`

**Load and merge CAN datasets** (external Car Track dataset + `data/raw/`):

```bash
python3 experiments/01_load_dataset.py --config configs/default.yaml
```

Writes `data/processed/clean_can_data.csv` with standardized columns.

**Generate sliding-window metadata** (default: 100 frames, 50-frame overlap):

```bash
python3 experiments/02_generate_windows.py --config configs/default.yaml
```

Writes `data/processed/window_metadata.csv` (`window_id`, `vehicle_model`, `attack_type`, `label`, …).

**Extract behavioural window features** (entropy, timing, payload stats):

```bash
python3 experiments/03_extract_features.py
```

Writes `data/processed/window_features.csv` and figures under `outputs/figures/`.

**Train vehicle-level IDS** (proposed self-supervised Isolation Forest):

```bash
python3 experiments/04_train_vehicle_ids.py
```

The proposed vehicle-level IDS uses Isolation Forest trained only on benign CAN windows. Attack labels are used only after inference for evaluation.

Writes `outputs/metrics/vehicle_level_self_supervised_results.csv`, `outputs/models/vehicle_isolation_forest.joblib`, `data/processed/vehicle_anomaly_predictions.csv`, and `outputs/figures/confusion_matrix_vehicle.png`.

`data/processed/vehicle_anomaly_predictions.csv` is the stage hand-off into fleet analysis and contains:

`window_id, vehicle_model, source_file, attack_type, true_label, anomaly_score, local_alert, weak_signal, evidence_level`

**Generate anomaly descriptors** (suspicious windows only):

```bash
python3 experiments/05_generate_descriptors.py
```

Writes `data/processed/anomaly_descriptors.csv` using only `strong_local_anomaly` and `weak_suspicious_signal` rows.

**Build fleet anomaly graph** (behavioural similarity edges, no temporal links):

```bash
python3 experiments/06_build_graph.py
```

Writes `data/processed/fleet_nodes.csv`, `data/processed/fleet_edges.csv`, `data/processed/fleet_graph.pt`, `outputs/fleet_graph.graphml`, and `outputs/metrics/graph_statistics.csv`.

**Train GNN** on the fleet graph:

```bash
python3 experiments/07_train_gnn.py
```

Writes `data/processed/node_embeddings.csv`, `outputs/metrics/gnn_training_metrics.csv`, `outputs/figures/gnn_training_loss.png`, and `outputs/figures/gnn_embeddings_tsne.png`.

**Cluster GNN embeddings** (KMeans + DBSCAN, multi-vehicle suspicious campaigns):

```bash
python3 experiments/08_cluster_campaigns.py
```

Writes `data/processed/fleet_cluster_results.csv` and campaign cluster figures.

**Final decision stage** classifies each anomaly event:

```bash
python3 experiments/09_final_decision.py
# or
python3 experiments/run_full_pipeline.py --only final_decision
```

Writes:

- `outputs/metrics/final_detection_outcomes.csv`
- `outputs/metrics/final_outcome_summary.csv`

Decision rule: strong local alerts that remain single-vehicle become `Isolated anomaly`; weak signals that remain single-vehicle become `Weak isolated signal`; any event in a high-similarity multi-vehicle cluster becomes `Fleet-level coordinated behavioural pattern`.

## Research Evidence Generated by the Pipeline

The key proof generated by the pipeline is that weak suspicious signals that do not trigger local vehicle IDS alerts can be upgraded to fleet-level coordinated behavioural patterns when similar behaviours appear across multiple vehicles.

The main evidence files are:

- `outputs/metrics/fleet_value_summary.csv`
- `outputs/metrics/weak_signal_upgrade_summary.csv`
- `outputs/metrics/cross_vehicle_cluster_summary.csv`
- `outputs/metrics/raw_vs_descriptor_size.csv`
- `outputs/figures/local_vs_fleet_outcomes.png`
- `outputs/figures/weak_signal_upgrade_chart.png`
- `outputs/figures/cross_vehicle_clusters_by_attack_type.png`
- `outputs/figures/fleet_cluster_vehicle_distribution.png`

Validate the evidence package with:

```bash
python3 experiments/validate_research_outputs.py
```

**Full pipeline** (all steps end-to-end):

```bash
python3 experiments/run_full_pipeline.py
```

Optional: `--skip-existing`, `--from-step train_gnn`, `--only cluster_campaigns final_decision generate_report`.

Writes `outputs/metrics/pipeline_report.md` and all intermediate artefacts.

## Paper deliverables (publication package)

To generate the final research deliverables in the required folder structure:

```bash
python3 run_all_experiments.py --config configs/fleet_ids.yaml
```

This runs the end-to-end pipeline and exports:

- `results/*.csv` and `results/*.json`
- `figures/*.png`
- `tables/*.tex`
- `logs/*.txt`
- `final_experiment_summary.md`

**Research report** (figures + `outputs/experiment_report.md`):

```bash
python3 experiments/09_generate_research_outputs.py
```

## Data

Raw datasets should not be committed to GitHub.

- Put small sample files or manual tests under `data/raw/` if needed.
- For the main dataset, set `data.external_dataset_dir` in `configs/fleet_ids.yaml` to a local folder path.
- Preprocessing scripts write derived artifacts to `data/processed/` and `outputs/`.

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

## Troubleshooting

- **`python: command not found`**: use `python3` everywhere.
- **Missing packages (e.g., `matplotlib`)**: ensure the venv is activated: `source .venv/bin/activate`, then `pip install -r requirements.txt`.
- **Dataset not found**: verify `external_dataset_dir` is correct and absolute.
