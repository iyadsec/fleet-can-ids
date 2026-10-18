# Fleetcan-sim → FLEET-GUARD pipeline (step-by-step)

**Dataset:** [iyadsec/fleet-can-dataset](https://github.com/iyadsec/fleet-can-dataset) release `fleetcan-sim-scale-1.0`  
**Pipeline:** same stages as the original OCSLab / Car Track flow in this repo (`01` → `06`)  
**Constraint:** methodology unchanged; adapter only remaps layout to `STANDARD_COLUMNS`.

## Validation verdict (schema)

| Requirement | fleetcan-sim-scale-1.0 | Status |
|---|---|---|
| `timestamp`, `can_id`, `dlc`, payload bytes | raw `can_frames` | PRESENT (payload = continuous hex → `byte0..7`) |
| `label`, `attack_type` | `ground_truth/frame_labels` (join `frame_uid`) | PRESENT (eval-only) |
| Vehicle identity | `vehicle_id` → `vehicle_model` | PRESENT (`vehicle_01`…) |
| Attack families | flooding, fuzzy, malfunction, replay | PRESENT |
| Scenarios S0–S3 | matrix + `scenario_id` | PRESENT (S4 is detector property, not generated) |
| Coordinated campaigns | `campaigns.csv` + membership | PRESENT (S2/S3) |
| CSV under OEM folders | OCSLab layout | NOT USED — adapter required |

**Gaps closed by adapter (`src/data/fleetcan_adapter.py`):** parquet (or csv) tables; continuous hex payload; separate GT join; `vehicle_id` naming.

**Detector-safe rule:** do not use `scenario_id`, campaign IDs, or labels as features.

## Steps

### Step 0 — Restore dataset

```bash
# from a clone of iyadsec/fleet-can-dataset (needs ~35 GB free, gh auth)
scripts/download_fleetcan_sim_scale_1_0.sh /path/to/fleet-can-ids/data/external
```

Result: `data/external/fleetcan-sim-scale-1.0/data/{raw,ground_truth,metadata,validation}/`.

### Step 1 — Schema validation (pilot sessions)

```bash
python experiments/01c_validate_fleetcan_dataset.py --pilot \
  --release-root data/external/fleetcan-sim-scale-1.0
```

Pilot sessions (N=20, R1):

- `FCS10_N020_S0_none_K000_none_R1`
- `FCS10_N020_S1_flooding_K000_isolated_R1`
- `FCS10_N020_S2_flooding_K000_independent_I05near_R1`
- `FCS10_N020_S3_flooding_K002_simultaneous_R1`

### Step 2 — Load / adapt → `clean_can_data.csv`

```bash
python experiments/01_load_fleetcan_dataset.py --pilot \
  --release-root data/external/fleetcan-sim-scale-1.0 \
  --output data/processed/clean_can_data.csv
```

### Step 3 — Same pipeline as original

```bash
python experiments/01b_validate_clean_dataset.py
python experiments/02_generate_windows.py --config configs/fleetcan_sim.yaml
python experiments/03_extract_features.py --config configs/fleetcan_sim.yaml
python experiments/04_train_vehicle_ids.py --config configs/fleetcan_sim.yaml
python experiments/05_generate_descriptors.py --config configs/fleetcan_sim.yaml
python experiments/06_build_graph.py --config configs/fleetcan_sim.yaml
```

### Step 4 — Scale out

Re-run Steps 1–3 with more sessions (e.g. all N=20 R1, then N=50/100) via `--sessions` / `--all-sessions`. Full matrix is 835 sessions / 444M frames — keep pilot first.

## Evidence labels

| Claim | Grade |
|---|---|
| Schema supports STANDARD_COLUMNS after join | **VERIFIED** (pilot `01c` PASS + synthetic fixture) |
| S0–S3 + four attack families present in freeze | **VERIFIED** (manifest / FREEZE / validation_summary) |
| Full 835-session integrity | **VERIFIED** (`download_fleetcan_sim_scale_1_0.sh` SUCCESS; SHA256 `5d5418b1…de70`; 835 sessions) |
| Pilot pipeline 01→06 on N=20 S0–S3 flooding | **VERIFIED** (see results below) |

## Pilot results (this run)

| Stage | Result |
|---|---|
| Restore | 26,740 files / 835 sessions verified |
| `01c` schema | PASS (4/4 pilot sessions) |
| `01_load_fleetcan` | 699,498 frames; 20 vehicles; attack_free + flooding |
| `01b --schema-only` | PASS (all schema checks) |
| `02` windows | 13,876 windows across 20 vehicles |
| `03` features | 13,876 × 24 behavioural features |
| `04` vehicle IDS | Supervised models on 8 attacked vehicles; IF on all 20 |
| `05` descriptors | 1,945 anomaly descriptors |
| `06` fleet graph | 1,945 nodes, 1,244,454 edges (cosine ≥ 0.85) |

**Note:** In coordinated/isolated fleetcan sessions only a subset of vehicles are attack targets. Benign-only vehicles skip RF/LR (need both classes) and still run Isolation Forest / autoencoder.
