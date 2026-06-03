# Fleet Similarity Diagnosis

## 1. Are flooding attacks behaviourally similar across vehicles?

- **Same-vehicle flooding similarity (full descriptors):** 0.9870
- **Cross-vehicle flooding similarity (full descriptors):** 0.9550
- **Cross-vehicle gap (same − cross):** 0.0319

Cross-vehicle flooding similarity is comparable to same-vehicle similarity.

## 2. Are top-k neighbours mostly same-vehicle?

- **Top-k edges same-vehicle:** 99.98%
- **Top-k edges cross-vehicle:** 0.02%
- **Flooding top-k neighbours same-vehicle:** 99.99%

Yes — the top-k graph is **strongly biased toward within-vehicle neighbours**, so flooding nodes rarely link across vehicle models.

## 3. Which descriptor features dominate similarity?

Highest vehicle-dominance ratio (between-vehicle / within-vehicle variance) for flooding:

- `unique_can_id_count`: ratio 13.37
- `std_dlc`: ratio 10.97
- `mean_dlc`: ratio 10.20
- `byte_std_0`: ratio 6.66
- `byte_std_1`: ratio 6.49
- `byte_std_2`: ratio 2.76
- `byte_std_5`: ratio 2.65
- `byte_mean_2`: ratio 2.01

Byte-level payload features (unique_can_id_count, std_dlc, mean_dlc) and CAN-ID structure features vary strongly by vehicle platform, dominating cosine similarity.

## 4. Does vehicle-normalization improve cross-vehicle flooding similarity?

- **Raw graph flooding cross-vehicle components:** 1
- **Normalized graph flooding cross-vehicle components:** 1
- **Normalized connected components:** 6

Normalization alone **did not materially increase** cross-vehicle flooding connectivity under current thresholds.

## 5. Does behaviour-only similarity improve fleet correlation?

- **Behaviour-only flooding cross-vehicle components:** 1
- **Behaviour-only connected components:** 1

Behaviour-only cross-vehicle flooding similarity (pairwise means):

- Chevrolet ↔ Hyundai: mean=0.9999, max=1.0000
- Chevrolet ↔ Kia: mean=0.9999, max=1.0000
- Hyundai ↔ Kia: mean=0.9999, max=1.0000

Behaviour-only features **do not yet produce cross-vehicle flooding clusters** at top-k=15, τ=0.95 — similarity may still be too strict or timing/entropy differ by platform.

## Root cause (diagnosis)

Fleet graph clustering fails to connect flooding across vehicles because:

1. **Descriptor design:** Full behavioural vectors include byte means/stds and CAN-ID ratios that encode vehicle platform identity.
2. **Top-k construction:** Nearest neighbours are overwhelmingly same-vehicle, producing vehicle-partitioned components.
3. **Graph threshold:** High cosine threshold (0.95) retains only near-duplicate windows, which are typically same-vehicle.

**Recommendation (diagnosis only — IDS unchanged):** Use behaviour-only, vehicle-normalized features for fleet similarity; consider lower τ or explicit cross-vehicle kNN quotas for fleet correlation (not local IDS).

Parameters: top_k=15, similarity_threshold=0.95.
