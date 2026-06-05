# Table 4: Fleet-Level Correlation Analysis

| Evaluation | Local IDS F1 | Fleet-aware F1 | FPR (%) | Cross-vehicle edges (%) | Note |
| --- | --- | --- | --- | --- | --- |
| Full-dataset detection (strong alerts) | 0.8456 | 0.8456 | 44.45 | nan | No F1 gain under top-k graph + ≥3 vehicle cluster gates |
| Weak anomalies — local baseline | 0.0 | 0.0013 | 1.55 | nan | Selective DBSCAN promotion (eps=1.2, gated) |
| Weak anomalies — optimized conservative | 0.0 | 0.0255 | 2.27 | nan | Weak recovery 1.31% (grid search, FPR≤5%) |
| Graph connectivity — full descriptor | nan | nan | nan | 0.0232 | Top-k similarity on full descriptor features |
| Graph connectivity — behaviour-normalized | nan | nan | nan | 1.0779 | Vehicle-normalized behavioural similarity view |
