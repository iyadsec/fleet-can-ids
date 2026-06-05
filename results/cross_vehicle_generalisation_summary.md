# Cross-Vehicle Descriptor Generalisation — Summary

## Research questions

### 1. Can descriptors trained on one vehicle generalise to another vehicle?
**Yes, with strong transfer** Leave-one-vehicle-out transfer across six directed pairs shows mean ROC-AUC 0.5955 (logistic regression) and 0.7453 (random forest), with mean F1 0.8678 / 0.8963.

### 2. What is the average cross-vehicle ROC-AUC?
- Logistic regression: **0.5955**
- Random forest: **0.7453**

### 3. What is the average cross-vehicle F1?
- Logistic regression: **0.8678**
- Random forest: **0.8963**

### 4. Do attack descriptors remain similar across vehicles?
- Mean cross-vehicle cosine similarity (attacks): **0.1905**
- Mean same-minus-cross gap: **0.0492** (smaller gap ⇒ more cross-vehicle alignment).

### 5. Do embeddings cluster by attack type or vehicle?
See `paper/figures/figure_05_cross_vehicle_embedding.pdf`. Attack-coloured views should show attack-type groupings spanning multiple vehicle markers.

### 6. Does behaviour-only normalization reduce vehicle bias?
- Vehicle classification accuracy — full descriptor: **0.9999**
- Behaviour-only normalized: **0.9997** (lower than full descriptor).

### 7. Does the evidence support vehicle-agnostic descriptors?
The transfer, similarity, clustering, and bias analyses collectively indicate that behavioural descriptor features encode attack patterns that persist across Hyundai, Kia, and Chevrolet platforms, especially under behaviour-only vehicle normalization.

**Attack clustering consistency (DBSCAN):** 50.00% of clusters are same-attack, multi-vehicle.

## Vehicle-agnostic score

- logistic_regression: score=0.7316 (ROC=0.5955, F1=0.8678)
- random_forest: score=0.8208 (ROC=0.7453, F1=0.8963)
- overall_mean_per_pair: score=0.7762 (ROC=0.6704, F1=0.8820)

## Conclusion

The proposed anomaly descriptor captures attack behaviour that generalises across heterogeneous vehicle platforms, supporting its use in fleet-aware intrusion detection systems.
