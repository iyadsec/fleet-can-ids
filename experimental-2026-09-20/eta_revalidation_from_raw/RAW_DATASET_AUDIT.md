# RAW_DATASET_AUDIT.md

**Updated:** 2026-09-19 (continue from PR #28)  
**Experiment:** `experimental-2026-09-20/eta_revalidation_from_raw/`

## OCSLAB_DATASET_DIR as requested

```text
OCSLAB_DATASET_DIR="/Users/iyadatieh/Library/CloudStorage/OneDrive-Personal/University of Reading/CodeRepo/Dataset/In-Vehicle Network Intrusion Detection Challenge"
```

| Check | Result |
|-------|--------|
| Path exists on Cloud Agent VM | **NO** (`No such file or directory`) |
| Readable as dataset root | **NO** |
| `Dataset/In-Vehicle Network Intrusion Detection Challenge/` in workspace | **NO** |
| `Dataset/ocslab/` | **NO** |

**Important:** Cloud Agents run on a **remote Linux VM**. Setting `OCSLAB_DATASET_DIR` to a **local macOS OneDrive path** does not make that filesystem visible to the agent. The dataset must be copied/synced/mounted **inside** the VM (or the task must run on a self-hosted worker that already has the path).

---

## Manifest → raw resolution (34 / 34 required)

Authoritative assignments: `recovered_balanced_split_manifest.csv` (unchanged; **EXACT_SPLIT_RECOVERED**).

Resolution table: `RAW_TRACE_RESOLUTION.csv`

| Status | Count |
|--------|-------|
| RESOLVED | **0** |
| MISSING | **34** |

Every required source segment fails to resolve. Examples of required relative paths:

```text
car_track_preliminary_train/Attack_free_CHEVROLET_Spark_train.csv
car_track_final_1st_train/Attack_free_HY_Sonata_train.csv
car_track_final_1st_train/Attack_free_KIA_Soul_train.csv
car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv
… (full list in RAW_TRACE_RESOLUTION.csv)
```

**STOP condition triggered:** required source traces cannot be resolved → do not reconstruct windows / IF / descriptors / scenarios.

---

## Artifact-inferred layout (unchanged from PR #28)

Still the expected on-disk challenge tree once mounted:

```text
In-Vehicle Network Intrusion Detection Challenge/
├── car_track_preliminary_train/
├── car_track_final_1st_train/
└── car_track_final_2nd_train/
```

Vehicles: Hyundai Sonata, Kia Soul, Chevrolet Spark.  
Attacks in split: attack_free, flooding, fuzzy, malfunction, replay.

Live per-file frame counts / column schema: **still unverified** (no readable files).

---

## Downstream stages

| Stage | Status |
|-------|--------|
| Split reuse | OK (manifest authoritative; not regenerated) |
| Window reconstruction | **BLOCKED** |
| Vehicle-level IF | **BLOCKED** |
| Validation descriptors | **BLOCKED** |
| Validation fleet scenarios | **BLOCKED** |
| η evaluation | **NOT STARTED** (forbidden until metric protocol approval) |
