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

`Raw CAN data → Vehicle anomaly detection → Strong/weak evidence classification → Descriptor generation → Raw-vs-descriptor size comparison → Behavioural graph construction → GNN embedding learning → Clustering → Final outcome classification → Research evidence summary`

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

**Train vehicle-level IDS** (proposed self-supervised Isolation Forest):

```bash
python experiments/04_train_vehicle_ids.py
```

The proposed vehicle-level IDS uses Isolation Forest trained only on benign CAN windows. Attack labels are used only after inference for evaluation.

Writes `outputs/metrics/vehicle_level_self_supervised_results.csv`, `outputs/models/vehicle_isolation_forest.joblib`, `data/processed/vehicle_anomaly_predictions.csv`, and `outputs/figures/confusion_matrix_vehicle.png`.

`data/processed/vehicle_anomaly_predictions.csv` is the stage hand-off into fleet analysis and contains:

`window_id, vehicle_model, source_file, attack_type, true_label, anomaly_score, local_alert, weak_signal, evidence_level`

**Generate anomaly descriptors** (suspicious windows only):

```bash
python experiments/05_generate_descriptors.py
```

Writes `data/processed/anomaly_descriptors.csv` using only `strong_local_anomaly` and `weak_suspicious_signal` rows.

**Build fleet anomaly graph** (behavioural similarity edges, no temporal links):

```bash
python experiments/06_build_graph.py
```

Writes `data/processed/fleet_nodes.csv`, `data/processed/fleet_edges.csv`, `data/processed/fleet_graph.pt`, `outputs/fleet_graph.graphml`, and `outputs/metrics/graph_statistics.csv`.

**Train GNN** on the fleet graph:

```bash
python experiments/07_train_gnn.py
```

Writes `data/processed/node_embeddings.csv`, `outputs/metrics/gnn_training_metrics.csv`, `outputs/figures/gnn_training_loss.png`, and `outputs/figures/gnn_embeddings_tsne.png`.

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
python experiments/validate_research_outputs.py
```

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
