# Fleet-Aware CAN-Bus Intrusion Detection

Research codebase for **Fleet-Aware Intrusion Detection for Coordinated CAN Attack Campaigns in Connected Vehicles**.

## Architecture (IEEE evaluation)

```
Vehicle CAN traffic
  → Vehicle-Level IDS (Isolation Forest)
  → Anomaly Descriptors (behaviour-only uplink)
  → Behaviour-Normalized Fleet Graph (cosine kNN, cross-vehicle edges)
  → GraphSAGE Fleet Correlation (structure-only training)
  → DBSCAN on GNN embeddings
  → Final decision: isolated_attack | coordinated_attack
```

Runtime fleet decisions use **behavioural cluster cohesion** and **multi-vehicle structure**. Attack-type labels are used **only** for evaluation tables and visualisation (e.g. Figure 6), not for GNN inputs or coordinated-attack gating.

---

## What is implemented

| Layer | Module | Method | IEEE paper |
|-------|--------|--------|------------|
| Data & windows | `src/data/`, `src/features/` | Car-Hacking dataset, sliding windows | H1–H4 |
| Vehicle IDS | `src/models/vehicle_ids.py` | Isolation Forest | **Table 1**, Figs 2–3 |
| Descriptors | `src/features/descriptor_generator.py` | Compact behavioural descriptors | **Table 2**, Figs 4–5 |
| Cross-vehicle transfer | `src/evaluation/cross_vehicle_generalisation_experiment.py` | Leave-one-vehicle-out RF | **Table 3**, Fig 6 |
| Fleet graph | `src/graph/fleet_graph_builder.py` | Cosine similarity, cross-vehicle kNN | H4 |
| GNN fleet correlation | `src/models/gnn_models.py`, `final_gnn_fleet_decision_experiment.py` | GraphSAGE (structure-only) + DBSCAN | **Tables 4–5**, Figs 7–8 |
| Descriptor security | `src/evaluation/descriptor_security_experiment.py` | Compression + privacy metrics | **Table 2**, Figs 4–5 |
| IEEE export | `src/evaluation/ieee_paper_exports.py` | Bundles `paper/` | All |

**Legacy (not in main IEEE bundle):** `run_campaign_detection_evaluation.py` (DBSCAN on descriptors without GNN), `experiments/07_train_gnn.py` (optional PyG step for old pipeline), weak-anomaly recovery scripts.

---

## Reproduce the IEEE paper bundle

### Requirements

- **Python**: 3.11 or 3.12 recommended (3.13+ may need a recent PyTorch / PyG build for GNN training)
- macOS / Linux / Windows

### 1) Clone and install

```bash
git clone https://github.com/iyadsec/fleet-can-ids.git
cd fleet-can-ids

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sanity check:

```bash
python3 -c "import numpy,pandas,sklearn,matplotlib,networkx,torch; print('deps ok')"
```

PyTorch Geometric is required for **`run_final_gnn_fleet_decision_evaluation.py`** (H4). Other H1–H3 scripts do not need PyG.

### 2) Point the config to your local dataset

Edit `configs/fleet_ids.yaml`:

```yaml
data:
  external_dataset_dir: "/absolute/path/to/your/Dataset"
```

Use an **absolute path**. Do not commit datasets to GitHub.

### 3) Ensure processed inputs exist

Minimum artefacts under `data/processed/`:

- `window_features.csv`
- `anomaly_descriptors.csv`
- `vehicle_anomaly_predictions.csv`

If missing, run the data pipeline:

```bash
python3 run_all_experiments.py --config configs/fleet_ids.yaml --skip-existing
```

Or stages `experiments/01_load_dataset.py` … `experiments/05_generate_descriptors.py`.

### 4) Run evaluations (H1–H4) and export

```bash
python3 run_vehicle_level_evaluation.py --config configs/fleet_ids.yaml
python3 run_descriptor_security_experiment.py --config configs/fleet_ids.yaml
python3 run_cross_vehicle_generalisation.py --config configs/fleet_ids.yaml
python3 run_final_gnn_fleet_decision_evaluation.py --config configs/fleet_ids.yaml
python3 run_ieee_paper_exports.py
```

`run_ieee_paper_exports.py` re-runs missing prerequisite experiments automatically.

### 5) Paper-ready outputs (`paper/`)

| Hypothesis | Table | Figures |
|------------|-------|---------|
| **H1** — Vehicle-Level IDS | Table 1 | Figure 2 (ROC), Figure 3 (score distribution) |
| **H2** — Descriptor security | Table 2 | Figure 4 (bandwidth scaling), Figure 5 (payload reconstruction) |
| **H3** — Cross-vehicle generalisation | Table 3 | Figure 6 (descriptor embedding) |
| **H4** — GNN fleet IDS | Tables 4–5 | Figure 7 (GNN campaign graph), Figure 8 (decision distribution) |

Per-attack-type **evaluation** metrics (campaign recall, behavioural cohesion) are in **Table 5 only** — no Figure 9.

See `paper/IEEE_EXPERIMENTAL_EVALUATION_INDEX.md` and `paper/results/ieee_experimental_evaluation_interpretations.md`.

---

## Evaluation scripts (`run_*.py`)

| Script | Purpose | IEEE paper |
|--------|---------|------------|
| `run_vehicle_level_evaluation.py` | Vehicle IDS metrics, ROC, Table 1 | H1 |
| `run_descriptor_security_experiment.py` | Compression, privacy, reconstruction | H2 |
| `run_cross_vehicle_generalisation.py` | Leave-one-vehicle-out transfer | H3 |
| `run_final_gnn_fleet_decision_evaluation.py` | GraphSAGE + DBSCAN → isolated/coordinated | **H4 (main fleet experiment)** |
| `run_ieee_paper_exports.py` | Bundle `paper/` tables, figures, interpretations | All |
| `run_all_experiments.py` | Full data pipeline | Supporting |
| `run_campaign_detection_evaluation.py` | Legacy DBSCAN-on-descriptors path | No (superseded by GNN final decision) |
| `run_fleet_correlation_evaluation.py` | Legacy local vs fleet correlation | No |
| `run_weak_anomaly_recovery_evaluation.py` | Weak-signal recovery ablation | No |

---

## Source modules (`src/`)

```
src/
├── data/                  # Dataset loading, validation
├── features/              # Windows, features, anomaly descriptors
├── graph/                 # Behaviour-normalized fleet similarity graph
├── models/
│   ├── vehicle_ids.py     # Isolation Forest (vehicle-level IDS)
│   └── gnn_models.py      # GraphSAGE fleet correlator (H4)
├── evaluation/
│   ├── vehicle_level_evaluation.py
│   ├── descriptor_security_experiment.py
│   ├── cross_vehicle_generalisation_experiment.py
│   ├── final_gnn_fleet_decision_experiment.py   # H4: final fleet decisions
│   ├── ieee_paper_exports.py
│   ├── campaign_detection_experiment.py         # Legacy fleet experiment
│   └── …
├── pipeline/
└── utils/
```

---

## Configuration (`configs/fleet_ids.yaml`)

Key blocks:

- `final_gnn_fleet_decision:` — GraphSAGE, DBSCAN, behavioural cohesion thresholds, `gnn_supervision: structure`
- `fleet_graph:` — top-k neighbours, similarity threshold
- `campaign_detection:` — legacy experiment (optional)
- `publication:` — `paper/`, `results/`, `figures/`, `tables/` paths

After the first GNN train, set `retrain_gnn: false` to reuse `outputs/models/final_graphsage_fleet.pt`.

---

## Project layout

```
.
├── configs/
├── data/processed/         # Pipeline artefacts
├── paper/                  # IEEE bundle (generated)
├── results/                # Evaluation CSVs
├── figures/                # Source figures copied into paper/
├── tables/                 # LaTeX / markdown tables
├── run_*.py                # Evaluation entry points
└── src/
```

---

## Component status

| Component | Status |
|-----------|--------|
| Vehicle IDS | Implemented |
| Anomaly descriptors | Implemented |
| Fleet behavioural graph | Implemented |
| GNN fleet final decision (H4) | Implemented — `run_final_gnn_fleet_decision_evaluation.py` |
| IEEE paper export | Implemented — `run_ieee_paper_exports.py` |
| Legacy campaign detection (no GNN) | Implemented, not in main paper bundle |

---

## Troubleshooting

- **`python: command not found`**: use `python3`.
- **Missing packages**: `pip install -r requirements.txt` inside `.venv`.
- **Dataset not found**: check `external_dataset_dir` is absolute and correct.
- **PyG / torch errors on H4**: install matching `torch` and `torch-geometric` wheels for your Python/CUDA version.
- **Slow GNN run**: set `retrain_gnn: false` after the first successful train.

---

## License

Add your license here.

## Citation

Add citation details when publishing results.
