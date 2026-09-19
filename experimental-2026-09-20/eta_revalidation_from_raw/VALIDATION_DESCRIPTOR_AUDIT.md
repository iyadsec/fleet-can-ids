# VALIDATION_DESCRIPTOR_AUDIT.md

## Status: **NOT GENERATED** — blocked on raw dataset + IF reconstruction

**2026-09-19 continue:** Still blocked — raw traces unresolved (0 / 34). Validation descriptors were **not** regenerated.

---

## What is known without regenerating

From recovered `balanced_window_manifest.csv`:

| Split | Windows |
|-------|---------|
| validation | **20,669** |
| Chevrolet / Hyundai / Kia (val) | 930 / 6,998 / 12,741 |
| Val label mix (0 / 1) | 3,114 / 17,555 |

These are **window provenance counts**, not descriptor feature matrices.

---

## Required descriptor schemas (to verify on regenerate)

### `d_i` — 24-D behavioural (publication)

Exact order from recovered `feature_extractor.BEHAVIOURAL_FEATURE_COLUMNS` / IF training manifest:

```text
frame_count, unique_can_id_count, can_id_entropy, most_common_can_id_ratio,
mean_inter_arrival_time, std_inter_arrival_time, mean_dlc, std_dlc,
byte_mean_0..7, byte_std_0..7
```

### `g_i` — 9-D GraphSAGE input (publication)

```text
1. anomaly_score
2. message_rate          # derived ← frame_count in behavior view
3. frame_count
4. burstiness            # std_IAT / (|mean_IAT|+1e-9)
5. mean_inter_arrival_time
6. std_inter_arrival_time
7. can_id_entropy
8. most_common_can_id_ratio
9. payload_entropy       # entropy of abs(byte_mean_*)
```

Implemented in `src/evaluation/publication_fleet_core.py` / recovered GNN peer. **Identity check against regenerated columns is pending** until descriptors exist.

---

## Separation rules (for the future run)

- Fit IF / scaler on **train** only.
- Score validation windows with frozen train models.
- Keep validation and test descriptor stores in separate directories under this experiment root.
- Do not mix CTT.
