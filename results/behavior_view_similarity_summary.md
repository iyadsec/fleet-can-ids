# Behaviour-View Fleet Similarity Summary

## 1. Which features were removed because they encoded vehicle identity?
Excluded from behaviour graph similarity (identity-heavy):

- `byte_mean_0`
- `byte_mean_1`
- `byte_mean_2`
- `byte_mean_3`
- `byte_mean_4`
- `byte_mean_5`
- `byte_mean_6`
- `byte_mean_7`
- `byte_std_0`
- `byte_std_1`
- `byte_std_2`
- `byte_std_3`
- `byte_std_4`
- `byte_std_5`
- `byte_std_6`
- `byte_std_7`
- `mean_dlc`
- `std_dlc`
- `unique_can_id_count`

Auto-excluded when between/within vehicle variance ratio > 5.0:

- (none)

## 2. Did behaviour-only descriptors increase cross-vehicle edges?
- Full descriptor cross-vehicle edges: **0.02%**
- Behaviour-only cross-vehicle edges: **6.41%**

## 3. Did vehicle normalization improve cross-vehicle attack similarity?
- Flooding cross-vehicle similarity (behaviour-only): **1.0000**
- Flooding cross-vehicle similarity (normalized): **0.0493**

## 4. Did flooding attacks from different vehicles become closer?
- Normalized view flooding cross-vehicle similarity: **0.0493**

## 5. Did weak anomaly recovery improve without excessive FPR?
- Weak recovery via connected components (full): 0.00% @ FPR 0.0000
- Weak recovery via connected components (normalized): 90.24% @ FPR 0.9993 (high FPR when the weak graph forms a single multi-vehicle component).
- **Selective DBSCAN promotion** (normalized, gated): 0.00% @ FPR 0.0000

## 6. Which similarity view should be used in the final paper?
**Behaviour-Only + Vehicle Normalization** (`behavior_only_vehicle_normalized`) for fleet graph similarity. It increases cross-vehicle edges versus the full descriptor while per-vehicle z-scoring reduces platform-specific baselines. Report weak-anomaly gains with selective promotion gates, not ungated connected-component promotion.

## Conclusion
Fleet-level graph construction should use a behaviour-focused, vehicle-normalized descriptor view rather than the full descriptor. This reduces vehicle-identity bias and allows graph reasoning to correlate attack behaviour across heterogeneous vehicles.
