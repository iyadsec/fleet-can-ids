# Dataset Provenance — Hyundai Sonata / Kia Soul / Chevrolet Spark

**Status: RESOLVED — manuscript car_track traces installed and used.**

## Source

Downloaded from HCRL In-Vehicle Network Intrusion Detection Challenge Dropbox mirror linked from  
https://ocslab.hksecurity.net/Datasets/datachallenge2019/car  

Archive: `In-Vehicle Network Intrusion Detection.zip` → local  
`Dataset/ocslab_pipeline/_source_car_track/{car_track_preliminary_train,car_track_final_1st_train,car_track_final_2nd_train}/`

## Identity evidence

| Platform | Filenames contain | Example |
|----------|-------------------|---------|
| Hyundai | `HY_Sonata` / `HYUNDAI_Sonata` | `Attack_free_HY_Sonata_train.csv` |
| Kia | `KIA_Soul` | `Attack_free_KIA_Soul_train.csv` |
| Chevrolet | `CHEVROLET_Spark` | `Attack_free_CHEVROLET_Spark_train.csv` |

**Sonata / Soul / Spark identity: YES** (filename tokens).  
Publication `balanced_split_manifest.csv` basenames: **21/21 present**.

## Split

Uses recovered publication segment assignments (`balanced_split_manifest.csv`).  
See `split_manifest.csv` for local resolved paths + frame counts. Leakage check: pass.

## Rejected

- Classic Car-Hacking copies previously in `Chevrolet/` (removed)
- `OCSLab_Car` / `Challenge_D` / `Challenge_S` (PR #22 only)
