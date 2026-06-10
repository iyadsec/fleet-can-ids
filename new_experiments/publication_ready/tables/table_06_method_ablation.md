# Method ablation and primary improvement

| Scenario | Method | Event recall | Event F1 | Vehicle recall | Campaign detection rate | Campaign F1 | False campaign rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S3 | M1 Local IDS | 1.000 $\pm$ 0.000 | 0.771 $\pm$ 0.162 | 1.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 |
| S3 | M2 Descriptor clustering | 1.000 $\pm$ 0.000 | 0.797 $\pm$ 0.161 | 1.000 $\pm$ 0.000 | 0.833 $\pm$ 0.408 | 0.500 $\pm$ 0.408 | 0.833 $\pm$ 0.408 |
| S3 | M3 Standard GNN | 1.000 $\pm$ 0.000 | 0.797 $\pm$ 0.161 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.278 $\pm$ 0.443 | 1.000 $\pm$ 0.000 |
| S3 | M4 Proposed FCGNN (GraphSAGEFleetCorrelator) | 1.000 $\pm$ 0.000 | 0.771 $\pm$ 0.162 | 1.000 $\pm$ 0.000 | 0.857 $\pm$ 0.378 | 0.333 $\pm$ 0.430 | 0.857 $\pm$ 0.378 |
| S3 | Δ FCGNN − Local | +0.000 | +0.000 | +0.000 | +0.857 | +0.333 | +0.857 |
| S4 | M1 Local IDS | 1.000 $\pm$ 0.000 | 0.724 $\pm$ 0.107 | 1.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 |
| S4 | M2 Descriptor clustering | 1.000 $\pm$ 0.000 | 0.742 $\pm$ 0.105 | 1.000 $\pm$ 0.000 | 0.667 $\pm$ 0.516 | 0.611 $\pm$ 0.491 | 0.667 $\pm$ 0.516 |
| S4 | M3 Standard GNN | 1.000 $\pm$ 0.000 | 0.742 $\pm$ 0.105 | 1.000 $\pm$ 0.000 | 0.500 $\pm$ 0.548 | 0.167 $\pm$ 0.408 | 0.500 $\pm$ 0.548 |
| S4 | M4 Proposed FCGNN (GraphSAGEFleetCorrelator) | 1.000 $\pm$ 0.000 | 0.724 $\pm$ 0.107 | 1.000 $\pm$ 0.000 | 0.286 $\pm$ 0.488 | 0.143 $\pm$ 0.378 | 0.286 $\pm$ 0.488 |
| S4 | Δ FCGNN − Local | +0.000 | +0.000 | +0.000 | +0.286 | +0.143 | +0.286 |
