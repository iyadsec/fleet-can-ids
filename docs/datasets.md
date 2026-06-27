# Dataset setup

Raw datasets are **not** committed to GitHub. Download them separately and place them under `Dataset/` at the repository root (this folder is gitignored).

## OCSLab Car-Hacking / DataChallenge 2019 (primary)

**Source:** [OCSLab Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) — DataChallenge 2019 traces.

**Local layout:**

```
Dataset/ocslab/
├── normal_run_data/
│   └── *.txt
├── DoS_dataset/
│   └── *.csv
├── gear_dataset/
│   └── *.csv
├── RPM_dataset/
│   └── *.csv
├── fuzzy_dataset/
│   └── *.csv
└── (other attack folders as provided by OCSLab)
```

**Environment override:**

```bash
export OCSLAB_DATASET_DIR=/path/to/ocslab/root
```

**Config key:** `data.external_dataset_dir` in `configs/default.yaml` (default: `Dataset/ocslab`).

## can-train-and-test (cross-dataset validation)

**Source:** DTU dataset — DOI [10.11583/DTU.24805533](https://doi.org/10.11583/DTU.24805533).

**Local layout:**

```
Dataset/can-train-and-test/
├── set_01/
│   ├── train_01/
│   ├── test_01_known_vehicle_known_attack/
│   ├── test_02_unknown_vehicle_known_attack/
│   ├── test_03_known_vehicle_unknown_attack/
│   └── test_04_unknown_vehicle_unknown_attack/
├── set_02/
│   └── ...
├── set_03/
│   └── ...
└── set_04/
    └── ...
```

Each subset folder contains per-vehicle CSV files with CAN message columns.

**Environment override:**

```bash
export CTT_DATASET_ROOT=/path/to/can-train-and-test
```

## Optional local raw folder

You may also place supplementary traces under `data/raw/` (gitignored). The OCSLab loader merges `Dataset/ocslab/` with `data/raw/` when both exist.

## Verify placement

```bash
python scripts/prepare_datasets.py --check
```
