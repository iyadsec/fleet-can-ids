# Dataset Provenance — Hyundai / Kia / Chevrolet pipeline

**Experiment status: STOPPED before pool construction / M1–M4.**

Manuscript platforms claimed: **Hyundai Sonata**, **Kia Soul**, **Chevrolet Spark**.

---

## 1. Local inventory (`Dataset/ocslab_pipeline/`)

| Directory | Files | Usable for manuscript-aligned run? |
|-----------|------:|------------------------------------|
| `Hyundai/` | **0** (empty) | **No** |
| `Kia/` | **0** (empty) | **No** |
| `Chevrolet/` | 5 files | **No** — classic Car-Hacking copies, not Spark |

### Chevrolet contents (not Spark)

| File | Size | Tokens Sonata/Soul/Spark? |
|------|-----:|---------------------------|
| `attack_free_run.txt` | 87,371,776 | No |
| `flooding_DoS_dataset.csv` | 190,058,708 | No |
| `fuzzy_dataset.csv` | 198,520,767 | No |
| `malfunction_gear_dataset.csv` | 230,320,414 | No |
| `malfunction_RPM_dataset.csv` | 239,581,936 | No |

**Classic identity check:** `Chevrolet/attack_free_run.txt` matches `Dataset/ocslab/normal_run_data/normal_run_data.txt` in size and first-1MB MD5. These are HCRL **classic Car-Hacking** traces renamed into an OEM folder — **not** `Attack_free_CHEVROLET_Spark_train.csv`.

Machine-readable inventory: `artifacts/dataset_inventory.json`.

---

## 2. Sonata / Soul / Spark identity from local evidence

| Question | Answer |
|----------|--------|
| Do local `ocslab_pipeline` filenames contain Sonata/Soul/Spark? | **No** |
| Can model names be asserted from directory names alone? | **No** — directories are empty or hold classic files |
| Do recovered publication manifests name Sonata/Soul/Spark? | **Yes** |

Publication split manifest basenames (examples):

- `Attack_free_HY_Sonata_train.csv`, `Fuzzy_dataset_HY_Sonata_train.csv`, …
- `Attack_free_KIA_Soul_train.csv`, …
- `Attack_free_CHEVROLET_Spark_train.csv`, …

Source paths in manifests point to author OneDrive:

`.../In-Vehicle Network Intrusion Detection Challenge/car_track_{final_1st,final_2nd,preliminary}_train/`

**0 / 21** required car_track basenames exist anywhere under `/workspace`.

---

## 3. Explicitly rejected inputs (per corrected-ablation brief)

Do **not** use for this corrected experiment:

- `OCSLab_Car` (classic Car-Hacking)
- `Challenge_D` / `Challenge_S` (Preliminary tracks D/S)
- PR #22 `real_ocslab` scored pool

Those remain diagnostic provenance under `experimental-reviewer-ablation/real_ocslab/` (PR #22).

---

## 4. Decision

| Item | Status |
|------|--------|
| Manuscript-aligned 3-OEM car_track data locally | **MISSING** |
| Proceed with pool / IF / scenarios / M1–M4 | **STOP** |
| Substitute classic or Challenge D/S | **Forbidden** by task brief |
| Campaign semantics recoverable without raw data | Yes (see `CAMPAIGN_SEMANTICS_AUDIT.md`) |
| Fleet scaler feature match for 9-D `g_i` | Yes (exact) — blocked only by missing windows |

**Required user action:** place car_track CSVs under `Dataset/ocslab_pipeline/{Hyundai,Kia,Chevrolet}/` (or provide a readable path), then continue this agent.
