# RAW_DATASET_AUDIT.md

**Checkpoint:** `experimental-2026-09-20/eta_revalidation_from_raw/`  
**Dataset path claimed by user / configs:**  
`/Users/iyadatieh/Library/CloudStorage/OneDrive-Personal/University of Reading/CodeRepo/Dataset/In-Vehicle Network Intrusion Detection Challenge`

## Accessibility on this Cloud Agent

| Check | Result |
|-------|--------|
| User-provided macOS OneDrive path | **NOT FOUND** on the Cloud Agent VM (`No such file or directory`) |
| `Dataset/ocslab/` | **Absent** |
| `OCSLAB_DATASET_DIR` | **Unset** / empty |
| Direct filesystem scan for OCSLab challenge trees | **None** |

**Conclusion:** The raw OCSLab challenge dataset is **not mounted** in this agent environment. A live directory listing, frame-level schema check, and byte-level file counts from the raw root **cannot** be performed here.

An environment setup action was recorded asking for the dataset to be synced/mounted at `Dataset/ocslab/` or via `OCSLAB_DATASET_DIR`.

---

## Structure recovered from publication artifacts (not live FS)

Authoritative balanced publication manifests embed absolute paths under the same OneDrive challenge root. From  
`recovered_publication_pipeline/.../manifests/balanced_split_manifest.csv` the **relative** layout used by the primary experiment is:

```text
In-Vehicle Network Intrusion Detection Challenge/
├── car_track_preliminary_train/
│   ├── Attack_free_{HYUNDAI_Sonata,KIA_Soul,CHEVROLET_Spark}_train.csv
│   ├── Flooding_{HYUNDAI_Sonata,KIA_Soul,CHEVROLET_Spark}_train.csv
│   ├── Fuzzy_{HYUNDAI_Sonata,KIA_Soul,CHEVROLET_Spark}_train.csv
│   └── Malfunction_{HYUNDAI_Sonata,KIA_Soul,CHEVROLET_Spark}_train.csv
├── car_track_final_1st_train/
│   ├── Attack_free_{HY_Sonata,KIA_Soul}_train.csv
│   ├── Fuzzy_dataset_{HY_Sonata,KIA_Soul}_train.csv
│   └── Malfunction_1st_dataset_{HY_Sonata,KIA_Soul}_train.csv
└── car_track_final_2nd_train/
    ├── Malfunction_2nd_{HY_Sonata,KIA_Soul}_train.csv
    └── Replay_dataset_{HY_Sonata,KIA_Soul}_train.csv
```

### Vehicles / attacks present in the **balanced publication split** (22 unique source basenames → 34 segments)

| Vehicle | Platforms in filenames | Attack types in split |
|---------|------------------------|------------------------|
| Hyundai Sonata | `HY_Sonata`, `HYUNDAI_Sonata` | attack_free, flooding, fuzzy, malfunction, replay |
| Kia Soul | `KIA_Soul` | attack_free, flooding, fuzzy, malfunction, replay |
| Chevrolet Spark | `CHEVROLET_Spark` | attack_free, flooding, fuzzy, malfunction (preliminary only) |

### Folder roles (as used by the balanced runner — do not invent other semantics)

| Folder | Role in balanced split |
|--------|------------------------|
| `car_track_preliminary_train` | Preliminary challenge train traces; Chevrolet coverage + some Hyundai/Kia |
| `car_track_final_1st_train` | Final 1st train traces; long attack-free Sonata/Soul traces segmented into train/val/test |
| `car_track_final_2nd_train` | Final 2nd train traces; malfunction/replay |

**Not observed in the balanced split manifests:** separate `release/` or unlabeled competition-only trees. Those may exist on disk but were **not** part of the recovered balanced publication assignment.

### Segment counts by top folder (from split manifest)

| Top folder | Segments in balanced split |
|------------|----------------------------|
| `car_track_preliminary_train` | 20 |
| `car_track_final_1st_train` | 10 |
| `car_track_final_2nd_train` | 4 |
| **Total** | **34** |

---

## Expected raw schema (from repository loaders — unverified on live files)

`docs/datasets.md` + `src/data/dataset_loader.py` expect OCSLab-style CAN rows with timestamp / CAN ID / DLC / payload (and labels where provided). **Live column verification is pending** until the raw root is mounted.

---

## Comparison to alternate documented layout

`docs/datasets.md` also documents a classic OCSLab Car-Hacking layout (`normal_run_data/`, `DoS_dataset/`, …). The **balanced publication** used the **DataChallenge 2019** tree above (`car_track_*`), not that classic folder naming. Do not conflate the two.

---

## Status

| Item | Status |
|------|--------|
| Live raw audit | **BLOCKED** — path not on VM |
| Artifact-inferred challenge layout | **Documented** |
| Exact per-file raw frame counts from disk | **Unavailable** (window/segment counts available from manifests — see `TRACE_SPLIT_RECOVERY.md`) |
