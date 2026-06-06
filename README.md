# Fleet-Aware CAN-Bus Intrusion Detection

Research codebase for **Fleet-Aware Intrusion Detection for Coordinated CAN Attack Campaigns in Connected Vehicles**.

The system combines:

1. **Vehicle-level IDS** — self-supervised Isolation Forest on CAN window features  
2. **Compact anomaly descriptors** — uplink-friendly behavioural summaries of suspicious windows  
3. **Fleet behavioural similarity graph** — cosine-similarity edges between descriptor nodes (no raw CAN payloads)  
4. **Fleet campaign detection** — cross-vehicle clustering on behaviour-normalized descriptors (DBSCAN / connected components)

This is **not** an end-to-end learned Fleet CAN GNN (FCGNN). The fleet layer builds an explicit **behavioural similarity graph** and applies **graph-aware clustering**; it does not train a dedicated fleet-correlation GNN as the primary detection method in the IEEE evaluation.

## What is implemented

| Layer | Module | Method | Used in IEEE paper? |
|-------|--------|--------|---------------------|
| Data & windows | `src/data/`, `src/features/` | Car-Hacking dataset loading, sliding windows, behavioural features | Yes |
| Vehicle IDS | `src/models/vehicle_ids.py` | Isolation Forest (+ optional autoencoder) | Yes (Table 1) |
| Descriptors | `src/features/descriptor_generator.py` | Suspicious-window behavioural descriptors | Yes (Table 2) |
| Fleet graph | `src/graph/fleet_graph_builder.py`, `fleet_similarity_features.py` | Cosine similarity, top-k / cross-vehicle kNN | Yes (campaign graph) |
| Campaign detection | `src/evaluation/campaign_detection_experiment.py` | DBSCAN on behaviour-normalized features | Yes (Table 4, Figs 6–7) |
| Cross-vehicle transfer | `src/evaluation/cross_vehicle_generalisation_experiment.py` | Leave-one-vehicle-out RF/LR | Yes (Table 3) |
| Descriptor security | `src/evaluation/descriptor_security_experiment.py` | Compression + privacy metrics | Yes (Table 2) |
| Optional GNN step | `src/models/gnn_models.py` | PyTorch Geometric GraphSAGE / GCN / GAT node embeddings | **No** (legacy pipeline only) |

The optional GNN step (`experiments/07_train_gnn.py`) trains a standard 2-layer PyG encoder on the **pre-built similarity graph**. It exports node embeddings for the legacy `cluster_campaigns` pipeline step. The **IEEE Experimental Evaluation** uses `run_campaign_detection_evaluation.py`, which clusters **behaviour-normalized descriptor features** directly and does **not** depend on GNN training.

---

## Reproduce the IEEE paper bundle (recommended)

### Requirements

- **Python**: 3.11 or 3.12 recommended (3.13 may fail for `torch-geometric` if you run the optional GNN step)
- macOS / Linux / Windows

### 1) Clone and install

```bash
git clone https://github.com/iyadsec/fleet-can-ids.git
cd fleet-can-ids

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sanity check (paper experiments; no PyTorch required):

```bash
python3 -c "import numpy,pandas,sklearn,matplotlib,networkx; print('deps ok')"
```

### 2) Point the config to your local dataset

Edit `configs/fleet_ids.yaml`:

```yaml
data:
  external_dataset_dir: "/absolute/path/to/your/Dataset"
```

Use an **absolute path**. Do not commit datasets to GitHub.

### 3) Ensure processed inputs exist

The evaluation scripts expect pipeline artefacts under `data/processed/` (at minimum `window_features.csv`, `anomaly_descriptors.csv`, and vehicle IDS predictions). If you do not have them yet, run the data pipeline once:

```bash
python3 run_all_experiments.py --config configs/fleet_ids.yaml --skip-existing
```

Or run individual stages under `experiments/01_load_dataset.py` … `experiments/05_generate_descriptors.py`.

### 4) Run the four IEEE evaluation contributions

```bash
python3 run_vehicle_level_evaluation.py --config configs/fleet_ids.yaml
python3 run_descriptor_security_experiment.py --config configs/fleet_ids.yaml
python3 run_cross_vehicle_generalisation.py --config configs/fleet_ids.yaml
python3 run_campaign_detection_evaluation.py --config configs/fleet_ids.yaml
python3 run_ieee_paper_exports.py
```

### 5) Paper-ready outputs

All IEEE assets are collected under `paper/`:

| Contribution | Table | Figures |
|--------------|-------|---------|
| Vehicle-Level IDS | `paper/tables/table_01_*.tex` | `figure_01`, `figure_02` |
| Descriptor Compactness & Security | `paper/tables/table_02_*.tex` | `figure_03`, `figure_04` |
| Cross-Vehicle Generalisation | `paper/tables/table_03_*.tex` | `figure_05` |
| Fleet Campaign Detection | `paper/tables/table_04_*.tex` | `figure_06`, `figure_07` |

See `paper/IEEE_EXPERIMENTAL_EVALUATION_INDEX.md` for the full file list and `paper/results/ieee_experimental_evaluation_interpretations.md` for narrative text.

Supporting CSVs and mirrored LaTeX also appear under `results/`, `figures/`, and `tables/`.

---

## Evaluation scripts (`run_*.py`)

| Script | Purpose | IEEE paper? |
|--------|---------|-------------|
| `run_vehicle_level_evaluation.py` | Vehicle IDS ROC/PR, thresholds, Table 1 | Yes |
| `run_descriptor_security_experiment.py` | Compression, privacy, reconstruction risk | Yes |
| `run_cross_vehicle_generalisation.py` | Leave-one-vehicle-out transfer | Yes |
| `run_campaign_detection_evaluation.py` | Coordinated campaign detection (main fleet experiment) | Yes |
| `run_ieee_paper_exports.py` | Bundle `paper/` tables, figures, interpretations | Yes |
| `run_all_experiments.py` | Full data pipeline + publication CSV export | Supporting |
| `run_fleet_correlation_evaluation.py` | Local vs fleet strong-alert correlation (legacy) | No |
| `run_weak_anomaly_recovery_evaluation.py` | Weak-signal recovery experiment (legacy) | No |
| `run_weak_recovery_optimization.py` | Parameter sweep for weak recovery | No |
| `run_selective_weak_promotion.py` | Gated weak-signal promotion | No |
| `run_behavior_view_similarity.py` | Similarity-view ablation | No |
| `run_graph_construction_comparison.py` | Graph construction comparison | No |
| `run_fleet_similarity_diagnosis.py` | Feature-dominance / bias diagnosis | No |

---

## Source modules (`src/`)

```
src/
├── data/                  # Dataset loading, validation, release loader
├── features/              # Windows, behavioural features, anomaly descriptors
├── graph/                 # Fleet behavioural similarity graph construction
│   ├── fleet_graph_builder.py      # Nodes/edges, PyG export, top-k graphs
│   └── fleet_similarity_features.py  # Behaviour-only / vehicle-normalized views
├── models/
│   ├── vehicle_ids.py     # Isolation Forest vehicle-level IDS (primary)
│   └── gnn_models.py      # Optional PyG GraphSAGE/GCN/GAT (legacy pipeline)
├── evaluation/            # Metrics, experiments, IEEE export
│   ├── vehicle_level_evaluation.py
│   ├── descriptor_security_experiment.py
│   ├── cross_vehicle_generalisation_experiment.py
│   ├── campaign_detection_experiment.py   # IEEE Contribution 4
│   ├── ieee_paper_exports.py
│   ├── campaign_clustering.py             # Legacy clustering (GNN or feature fallback)
│   ├── final_decision.py
│   └── …                  # Additional fleet-correlation / ablation experiments
├── pipeline/              # YAML-driven full pipeline runner
└── utils/                 # Config, paths, logging
```

Legacy scaffold (do not use for paper reproduction): `experiments/run_baseline.py` predates the current implementation.

---

## Full data pipeline (optional)

End-to-end processing from raw CAN logs to research artefacts:

`Raw CAN → windows → features → vehicle IDS → descriptors → fleet similarity graph → [optional GNN] → clustering → final outcomes`

```bash
python3 experiments/run_full_pipeline.py --config configs/fleet_ids.yaml
# or
python3 run_all_experiments.py --config configs/fleet_ids.yaml
```

### Stage-by-stage (`experiments/`)

| Step | Script | Output |
|------|--------|--------|
| Load data | `01_load_dataset.py` | `data/processed/clean_can_data.csv` |
| Windows | `02_generate_windows.py` | `data/processed/window_metadata.csv` |
| Features | `03_extract_features.py` | `data/processed/window_features.csv` |
| Vehicle IDS | `04_train_vehicle_ids.py` | `data/processed/vehicle_anomaly_predictions.csv` |
| Descriptors | `05_generate_descriptors.py` | `data/processed/anomaly_descriptors.csv` |
| **Fleet graph** | `06_build_graph.py` | `data/processed/fleet_graph.pt`, `fleet_edges.csv` |
| **Optional GNN** | `07_train_gnn.py` | `data/processed/node_embeddings.csv` |
| Legacy clustering | `08_cluster_campaigns.py` | `data/processed/fleet_cluster_results.csv` |
| Final outcomes | `09_final_decision.py` | `outputs/metrics/final_detection_outcomes.csv` |

**Build fleet anomaly graph** (`06_build_graph.py`): constructs an undirected graph whose nodes are anomaly descriptors and whose edges connect behaviourally similar pairs (cosine similarity above a threshold, with bounded neighbours). This is a **similarity graph**, not a learned FCGNN.

**Optional GNN** (`07_train_gnn.py`): if PyTorch Geometric is installed, trains a shallow GraphSAGE/GCN/GAT encoder on that graph for node embeddings used by the legacy clustering step. Skip this entirely for IEEE paper reproduction.

**Campaign detection (paper)**: uses `run_campaign_detection_evaluation.py` with a **vehicle-normalized behavioural similarity view** and DBSCAN — independent of the GNN step.

---

## Configuration

- `configs/fleet_ids.yaml` — paper deliverables and evaluation defaults (recommended)
- `configs/default.yaml` — default pipeline configuration

Campaign detection settings: `campaign_detection:` block in `fleet_ids.yaml`.  
Fleet graph similarity views: `fleet_graph:` block (`behavior_only_vehicle_normalized` for paper experiments).

---

## Project layout

```
.
├── configs/              # YAML experiment configuration
├── data/
│   ├── raw/              # Local samples (not versioned)
│   └── processed/        # Derived tables and graph artefacts
├── experiments/          # Pipeline stage scripts (01–09)
├── paper/                # IEEE Experimental Evaluation bundle (generated)
├── results/              # Evaluation CSVs and summaries
├── figures/              # Evaluation figures
├── tables/               # LaTeX / markdown tables
├── run_*.py              # Evaluation entry points
└── src/                  # Library code (see above)
```

---

## Component status

| Component | Status | Notes |
|-----------|--------|-------|
| Data pipeline | Implemented | `src/data/`, `experiments/01–03` |
| Vehicle IDS | Implemented | Isolation Forest in `src/models/vehicle_ids.py` |
| Anomaly descriptors | Implemented | `src/features/descriptor_generator.py` |
| Fleet behavioural graph | Implemented | Similarity graph in `src/graph/`; not a trained FCGNN |
| Fleet campaign detection (paper) | Implemented | `run_campaign_detection_evaluation.py` |
| IEEE paper export | Implemented | `run_ieee_paper_exports.py` → `paper/` |
| Optional GNN embeddings | Implemented (optional) | PyG GraphSAGE/GCN/GAT; legacy pipeline only |
| Legacy fleet correlation / weak recovery | Implemented (supporting) | Not in IEEE main tables |

---

## Data

Raw datasets should not be committed to GitHub.

- Set `data.external_dataset_dir` in `configs/fleet_ids.yaml` to your local Car-Hacking dataset folder.
- Preprocessing writes derived artefacts to `data/processed/` and `outputs/`.

---

## Troubleshooting

- **`python: command not found`**: use `python3`.
- **Missing packages**: activate the venv and run `pip install -r requirements.txt`.
- **Dataset not found**: verify `external_dataset_dir` is correct and absolute.
- **`torch-geometric` install fails**: only required for `experiments/07_train_gnn.py`; IEEE paper scripts do not need it.
- **`experiments/run_baseline.py`**: deprecated scaffold; use `run_all_experiments.py` or the `run_*.py` scripts listed in this README.

---

## License

Add your license here.

## Citation

Add citation details when publishing results.
