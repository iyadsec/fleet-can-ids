# Fleet-Aware CAN-Bus Intrusion Detection

Research codebase for detecting intrusions on Controller Area Network (CAN) bus traffic with **fleet-level** context—modeling relationships across vehicles, ECUs, and message patterns rather than treating each trace in isolation.

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
    ├── models/           # Detectors (not implemented yet)
    ├── evaluation/       # Metrics and reporting
    └── utils/            # Config, paths, logging
```

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

Experiment entry points live under `experiments/`. Model training and inference are **not implemented yet**; scripts may load data and write placeholders until detectors are added under `src/models/`.

**Load and merge CAN datasets** (external Car Track dataset + `data/raw/`):

```bash
python experiments/01_load_dataset.py --config configs/default.yaml
```

Writes `data/processed/clean_can_data.csv` with standardized columns.

**Validate clean dataset** (row counts, schema, ranges, duplicates):

```bash
python experiments/01b_validate_clean_dataset.py
```

Writes `outputs/metrics/data_validation_report.csv` and `data_validation_summary.txt`.

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

Writes `outputs/metrics/vehicle_level_results.csv` and `outputs/figures/confusion_matrix_vehicle.png`.

**Generate anomaly descriptors** (suspicious windows only):

```bash
python experiments/05_generate_descriptors.py
```

Writes `data/processed/anomaly_descriptors.csv` and `outputs/metrics/window_predictions.csv`.

**Build fleet anomaly graph** (behavioural similarity edges, no temporal links):

```bash
python experiments/06_build_graph.py
```

Writes `data/processed/fleet_graph.pt` and `outputs/fleet_graph.graphml`.

```bash
python experiments/run_baseline.py --config configs/default.yaml
```

## Data

Place raw CAN dumps or public benchmark exports in `data/raw/`. Preprocessing scripts write artifacts to `data/processed/`. Large files are ignored by git (see `.gitignore`).

## Status

| Component        | Status              |
|------------------|---------------------|
| Data pipeline    | Scaffold            |
| Features         | Scaffold            |
| Fleet graph      | Scaffold            |
| Models           | Not implemented     |
| Evaluation       | Scaffold            |

## License

Add your license here.

## Citation

Add citation details when publishing results.
