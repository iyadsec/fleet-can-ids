# CTT Set Pilot Validation Report

- [PASS] **audit stage completed**
- [PASS] **pilot stage completed**
- [PASS] **audit validation passed**: []
- [PASS] **pilot validation passed**: []
- [PASS] **set_pilot marker present**
- [PASS] **normalization manifest present**
- [PASS] **only target set processed**: sets=['set_01']
- [PASS] **train/test split respected**
- [PASS] **all expected test subsets present**: missing=[] present=['test_01_known_vehicle_known_attack', 'test_02_unknown_vehicle_known_attack', 'test_03_known_vehicle_unknown_attack', 'test_04_unknown_vehicle_unknown_attack', 'train_01']
- [PASS] **test_04 subset present**
- [PASS] **at least two vehicles in descriptors**: vehicles=2
- [PASS] **descriptor vehicle balance**: counts={'chevrolet_impala': 45175, 'chevrolet_silverado': 54825}
- [PASS] **no test data in training**
- [PASS] **no attack data in training**
- [PASS] **no attack data in thresholding**
- [PASS] **feature matrix excludes forbidden columns**: []
- [PASS] **no temporal edges**
- [PASS] **behavioural similarity edges only**
- [PASS] **cross-vehicle edges not near zero**: cross_vehicle_edge_pct=0.2771%
- [PASS] **benign fleet false campaign rate zero**: mean=0.0
- [PASS] **strong campaign detected**: mean=1.0
- [PASS] **scenario output benign_fleet_control**
- [PASS] **scenario output isolated_attack**
- [PASS] **scenario output unrelated_incidents**
- [PASS] **scenario output strong_campaign**
- [PASS] **scenario output weak_campaign**
- [PASS] **predictions recomputable**
- [PASS] **set tables generated**: count=8
- [PASS] **set figures generated**: count=7
- [PASS] **OCSLab publication output preserved (not present in workspace)**
- [PASS] **summary document present**

**Overall:** PASS