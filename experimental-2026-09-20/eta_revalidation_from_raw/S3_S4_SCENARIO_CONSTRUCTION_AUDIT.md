# S3_S4_SCENARIO_CONSTRUCTION_AUDIT.md

## Recoverability

Historical builder `build_mixed_validation_suite` (`src.experiments.final_shared_configuration.validation_scenarios`) is **UNRECOVERABLE** — listed in `recovered_publication_pipeline/MISSING_MODULES.md`. This reconstruction uses a **stand-in** in `scripts/reconstruct_validation_pipeline.py` that follows recovered V0–V4 semantics + node budget, **not** claimed bit-identical.

## Intended vs reconstructed construction

| Aspect | Recovered publication intent | This reconstruction | Status |
|--------|------------------------------|----------------------|--------|
| Fleet size / desc per vehicle | 20 / 10 | 20 / 10 | MATCH |
| Mal/benign per attacked | 5 / 5 | 5 / 5 | MATCH |
| S3/S4 campaign size strata | 2, 5, 10 | **5 only** | MISMATCH |
| Seeds | `{11,23,37,41,53,67,71,83,97,101}` (master YAML) | `{131,…,197}` | MISMATCH / alternate list |
| Strong/weak thresholds | 0.80 / 0.55 | 0.80 / 0.55 | MATCH |
| Behavioural coordination | `behavioural_coordination_only: true` | Prototype blend of behavioural features (strength 1.0 / 0.35) toward global attack mean | **KNOWN_PROSPECTIVE / UNVERIFIED vs missing builder** |
| Shared campaign id S3/S4 | One campaign | One `CAMP-*` id | MATCH (semantics) |
| Attack OEMs | Unrecovered exact mix | Hyundai×3 + Kia×2 | UNRECOVERABLE exact |

## Per-run construction snapshot

### val_S3_seed131

- scenario=S3 seed=131 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=27 weak=4
  - H001: n_mal=10 score[min/mean/max]=0.109/0.671/0.998 strong=5 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.291/0.727/0.997 strong=5 weak=1 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.007/0.625/0.998 strong=5 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.025/0.692/1.000 strong=6 weak=1 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.319/0.793/1.000 strong=6 weak=2 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed137

- scenario=S3 seed=137 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=26 weak=6
  - H001: n_mal=10 score[min/mean/max]=0.000/0.593/0.999 strong=5 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.141/0.774/0.996 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.260/0.779/0.998 strong=5 weak=4 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.051/0.684/1.000 strong=5 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.155/0.694/1.000 strong=5 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed149

- scenario=S3 seed=149 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=28 weak=4
  - H001: n_mal=10 score[min/mean/max]=0.122/0.760/0.999 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.004/0.664/0.996 strong=6 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.021/0.655/1.000 strong=5 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.145/0.676/1.000 strong=5 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.417/0.808/1.000 strong=6 weak=2 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed157

- scenario=S3 seed=157 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=30 weak=0
  - H001: n_mal=10 score[min/mean/max]=0.011/0.648/1.000 strong=6 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.264/0.776/0.998 strong=7 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.192/0.628/1.000 strong=5 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.065/0.753/1.000 strong=7 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.092/0.607/1.000 strong=5 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed163

- scenario=S3 seed=163 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=30 weak=4
  - H001: n_mal=10 score[min/mean/max]=0.005/0.671/1.000 strong=6 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.466/0.786/0.999 strong=5 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.005/0.797/0.999 strong=7 weak=1 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.138/0.710/1.000 strong=5 weak=1 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.217/0.774/1.000 strong=7 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed179

- scenario=S3 seed=179 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=30 weak=5
  - H001: n_mal=10 score[min/mean/max]=0.079/0.692/0.999 strong=6 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.031/0.739/0.999 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.325/0.725/0.999 strong=5 weak=1 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.061/0.675/1.000 strong=6 weak=1 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.257/0.797/1.000 strong=7 weak=1 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed181

- scenario=S3 seed=181 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=30 weak=6
  - H001: n_mal=10 score[min/mean/max]=0.087/0.703/0.996 strong=6 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.005/0.735/1.000 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.125/0.764/0.999 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.020/0.707/1.000 strong=5 weak=2 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.372/0.818/1.000 strong=7 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed191

- scenario=S3 seed=191 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=30 weak=4
  - H001: n_mal=10 score[min/mean/max]=0.161/0.770/0.999 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.125/0.800/0.997 strong=8 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.178/0.663/1.000 strong=5 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.324/0.746/1.000 strong=5 weak=2 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.136/0.741/1.000 strong=6 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed193

- scenario=S3 seed=193 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=29 weak=5
  - H001: n_mal=10 score[min/mean/max]=0.057/0.715/0.998 strong=6 weak=1 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.264/0.702/0.998 strong=5 weak=1 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.308/0.799/0.999 strong=6 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.148/0.715/1.000 strong=5 weak=1 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.215/0.782/1.000 strong=7 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S3_seed197

- scenario=S3 seed=197 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=29 weak=3
  - H001: n_mal=10 score[min/mean/max]=0.039/0.685/0.995 strong=6 weak=1 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.126/0.767/0.999 strong=7 weak=0 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.135/0.718/0.998 strong=5 weak=2 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.033/0.683/1.000 strong=6 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.058/0.638/1.000 strong=5 weak=0 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed131

- scenario=S4 seed=131 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=2 weak=29
  - H001: n_mal=10 score[min/mean/max]=0.109/0.571/0.799 strong=0 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.291/0.627/0.797 strong=0 weak=6 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.007/0.525/0.799 strong=0 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.025/0.591/0.935 strong=1 weak=6 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.319/0.692/0.913 strong=1 weak=7 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed137

- scenario=S4 seed=137 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=1 weak=31
  - H001: n_mal=10 score[min/mean/max]=0.000/0.493/0.800 strong=0 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.141/0.674/0.899 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.260/0.680/0.800 strong=0 weak=9 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.051/0.584/0.800 strong=0 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.155/0.592/0.800 strong=0 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed149

- scenario=S4 seed=149 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=3 weak=29
  - H001: n_mal=10 score[min/mean/max]=0.122/0.661/0.842 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.004/0.565/0.940 strong=1 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.021/0.555/0.800 strong=0 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.145/0.575/0.800 strong=0 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.417/0.706/0.850 strong=1 weak=7 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed157

- scenario=S4 seed=157 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=5 weak=25
  - H001: n_mal=10 score[min/mean/max]=0.011/0.549/0.830 strong=1 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.264/0.676/0.869 strong=2 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.192/0.528/0.800 strong=0 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.065/0.653/0.875 strong=2 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.092/0.506/0.800 strong=0 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed163

- scenario=S4 seed=163 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=5 weak=29
  - H001: n_mal=10 score[min/mean/max]=0.005/0.571/0.936 strong=1 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.466/0.686/0.800 strong=0 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.005/0.697/0.935 strong=2 weak=6 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.138/0.608/0.799 strong=0 weak=6 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.217/0.673/0.979 strong=2 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed179

- scenario=S4 seed=179 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=5 weak=30
  - H001: n_mal=10 score[min/mean/max]=0.079/0.592/0.968 strong=1 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.031/0.639/0.838 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.325/0.625/0.800 strong=0 weak=6 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.061/0.575/0.866 strong=1 weak=6 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.257/0.695/0.908 strong=2 weak=6 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed181

- scenario=S4 seed=181 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=5 weak=31
  - H001: n_mal=10 score[min/mean/max]=0.087/0.604/0.948 strong=1 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.005/0.635/0.870 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.125/0.664/0.900 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.020/0.607/0.800 strong=0 weak=7 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.372/0.718/0.908 strong=2 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed191

- scenario=S4 seed=191 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=5 weak=29
  - H001: n_mal=10 score[min/mean/max]=0.161/0.670/0.938 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.125/0.700/0.954 strong=3 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.178/0.563/0.800 strong=0 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.324/0.645/0.800 strong=0 weak=7 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.136/0.640/0.943 strong=1 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed193

- scenario=S4 seed=193 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=4 weak=30
  - H001: n_mal=10 score[min/mean/max]=0.057/0.616/0.961 strong=1 weak=6 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.264/0.602/0.799 strong=0 weak=6 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.308/0.699/0.938 strong=1 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.148/0.614/0.800 strong=0 weak=6 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.215/0.682/0.992 strong=2 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

### val_S4_seed197

- scenario=S4 seed=197 campaign_size=5
- GT vehicles V(G)=['H001', 'H002', 'H003', 'K004', 'K005']
- malicious descriptors=50 strong=4 weak=28
  - H001: n_mal=10 score[min/mean/max]=0.039/0.586/0.804 strong=1 weak=6 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H002: n_mal=10 score[min/mean/max]=0.126/0.667/0.973 strong=2 weak=5 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - H003: n_mal=10 score[min/mean/max]=0.135/0.619/0.800 strong=0 weak=7 types=['attack_free', 'malfunction'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_HY_Sonata_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Malfunction_1st_dataset_HY_Sonata_train.csv']
  - K004: n_mal=10 score[min/mean/max]=0.033/0.581/0.875 strong=1 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']
  - K005: n_mal=10 score[min/mean/max]=0.058/0.538/0.800 strong=0 weak=5 types=['attack_free', 'replay'] sources=['/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train/Attack_free_KIA_Soul_train.csv', '/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train/Replay_dataset_KIA_Soul_train.csv']

## Strong vs weak evidence check

- Mean strong-threshold malicious descriptors per S3 run: **28.9**
- Mean strong-threshold malicious descriptors per S4 run: **3.9**
- Mean weak-band malicious descriptors per S4 run: **29.1**

If S3 and S4 pools are not cleanly separated, the stand-in weak-pool fallback may have used the full attack pool (see builder: weak pool used only if large enough).

## Verdict on construction

**SCENARIO_CONSTRUCTION is a reconstructed stand-in.** Exact historical coordination logic is UNRECOVERABLE. Node budget and thresholds match; campaign-size strata and seed list differ; behavioural blending is a reconstructed proxy.

## Measured coordination effectiveness (diagnostic; no changes)

Across S3/S4 validation runs, mean cross-vehicle cosine among same-GT-campaign
descriptors in the 9-D GNN feature space is **≈0.17**, far below publication
τ=0.95. The reconstructed prototype-blend therefore does **not** reproduce
the historical behavioural-coordination effect that the missing
`build_mixed_validation_suite` must have achieved (historical freeze
selection_report mean V3/V4 Campaign F1 ≈0.65/0.42).

Attack descriptors are present (≈25 malicious descriptors / run; strong/weak
pools populated). The failure is not empty scenarios; it is insufficient
cross-vehicle similarity among GT members after stand-in construction.
