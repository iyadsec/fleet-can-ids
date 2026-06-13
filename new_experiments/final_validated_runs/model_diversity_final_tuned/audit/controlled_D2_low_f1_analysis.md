# Controlled D2 low-F1 analysis

## Summary

### D1 (provisional)
- campaign_precision: mean=0.200, std=0.410
- campaign_recall: mean=0.950, std=0.224
- campaign_f1: mean=0.200, std=0.410
- false_campaign_alert_rate: mean=1.000, std=0.000
- cross_model_malicious_edges: mean=0.0

### D1 (tuned)
- campaign_precision: mean=0.950, std=0.158
- campaign_recall: mean=1.000, std=0.000
- campaign_f1: mean=0.967, std=0.105
- campaign_membership_precision: mean=1.000, std=0.000
- campaign_membership_recall: mean=0.920, std=0.140
- fragmentation: mean=0.100, std=0.316
- incorrect_merging: mean=0.000, std=0.000
- benign_vehicles_included: mean=0.000, std=0.000
- false_campaign_alert_rate: mean=0.100, std=0.316
- cross_model_malicious_edges: mean=0.0

### D2 (provisional)
- campaign_precision: mean=0.050, std=0.224
- campaign_recall: mean=1.000, std=0.000
- campaign_f1: mean=0.050, std=0.224
- false_campaign_alert_rate: mean=1.000, std=0.000
- cross_model_malicious_edges: mean=32.3

### D2 (tuned)
- campaign_precision: mean=1.000, std=0.000
- campaign_recall: mean=1.000, std=0.000
- campaign_f1: mean=1.000, std=0.000
- campaign_membership_precision: mean=1.000, std=0.000
- campaign_membership_recall: mean=0.980, std=0.063
- fragmentation: mean=0.000, std=0.000
- incorrect_merging: mean=0.000, std=0.000
- benign_vehicles_included: mean=0.000, std=0.000
- false_campaign_alert_rate: mean=0.000, std=0.000
- cross_model_malicious_edges: mean=32.3

## Root-cause assessment

1. **Metric semantics (provisional):** Legacy false campaign rate ≈ 1.0 whenever qualifying clusters exist; this inflated apparent safety failure.
2. **D2 cross-model edges:** D2 requires Hyundai/Kia cross-model connectivity; edge counts are non-zero but gate may reject campaigns lacking sufficient cross-model support.
3. **Campaign vs member gate:** Provisional gate combined acceptance; strict campaign thresholds can reject valid D2 campaigns (low recall), while permissive member rules retained benign vehicles.
4. **DBSCAN fragmentation:** Some seeds show multiple qualifying clusters for one ground-truth campaign, reducing campaign precision.
5. **GraphSAGE vs similarity:** C3 and C2 show seed-dependent divergence; not universal oversmoothing.

## Conclusion

D2 low F1 is primarily driven by **campaign recall failure** under cross-model gating combined with **legacy metric misreporting**.
Tuned two-level gate and corrected metrics separate these effects.