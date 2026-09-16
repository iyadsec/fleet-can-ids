# Cosine Validation — seed 11 (PASSED for strong)

Representation: raw 9-D `g_i` → publication `fleet_benign_scaler.json` → cosine.

## strong_campaign

| Bucket | n_pairs | mean | median | min | max | % ≥ 0.95 |
|--------|--------:|-----:|-------:|----:|----:|---------:|
| campaign–campaign | 1225 | 0.711 | 0.779 | 0.110 | 1.000 | 28.82 |
| campaign–benign | 7500 | −0.064 | −0.090 | −0.891 | 0.811 | **0.00** |
| campaign–unrelated | 0 | — | — | — | — | — (no non-campaign malicious in this scenario) |

**PASS:** not the PR #22 condition (≈100% of every bucket ≥ 0.95).

Notes: prototype blend strength=1.0; primary_attack fallback → `fuzzy` (no malfunction in strong TEST band).

## weak_campaign

**UNSUPPORTED:** only 2 weak-band malicious TEST windows (`0.55 ≤ s < 0.80`); need ≥25 for campaign_size=5 × 5 malicious.  
No threshold retuning performed.
