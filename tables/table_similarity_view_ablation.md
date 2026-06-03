# Effect of Descriptor Similarity View on Fleet Graph Construction

| Similarity View | Same-Vehicle Edge % | Cross-Vehicle Edge % | Flooding Cross-Vehicle Similarity | Fleet Recall | Fleet F1 | Weak Recovery Rate | Weak FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full Descriptor | 99.9768 | 0.0232 | 0.955 | 0.7933 | 0.8456 | 0.0 | 0.0 |
| Behaviour-Only | 93.5944 | 6.4056 | 1.0 | 0.7933 | 0.8456 | 100.0 | 1.0 |
| Behaviour-Only + Vehicle Normalization | 98.9221 | 1.0779 | 0.0493 | 1.0 | 0.9146 | 90.2401 | 0.9993 |
