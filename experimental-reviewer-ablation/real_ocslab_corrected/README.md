# real_ocslab_corrected — Reviewer ablation (methodology correction)

Answers Reviewer comment on Sec 4.5.2 (component ablation) using recoverable FLEET-GUARD methodology.

**PR #22 (`real_ocslab/`) is preserved as diagnostic provenance** and must not be used in the paper.

## Current status: STOPPED

| Gate | Result |
|------|--------|
| Campaign semantics recoverable? | **YES** — see `CAMPAIGN_SEMANTICS_AUDIT.md` |
| Manuscript Hyundai/Kia/Chevrolet car_track data present? | **NO** — see `DATASET_PROVENANCE.md` |
| Fleet scaler matches 9-D `g_i`? | **YES** (exact feature order) |
| Seed-11 cosine validation | **BLOCKED** (no TEST pool) |
| M1–M4 ten-seed run | **NOT STARTED** |

## What will run once car_track data is provided

1. Source-trace-level 70/15/15 split (no overlapping-window leakage)
2. Window 100 / stride 50 on contiguous blocks only
3. 24-D behavioural features + IF (benign TRAIN only)
4. Strong ≥0.80 / weak [0.55, 0.80)
5. Publication campaign semantics: instance catalog + composition + **prototype blend strength=1.0**
6. Similarity: raw 9-D `g_i` → `fleet_benign_scaler.json` → cosine → τ=0.95 constrained kNN
7. Seed-11 cosine gate, then seeds `[11,23,37,41,53,67,71,83,97,101]`
8. M1–M4 with shared scenarios, β=0.5, fragment merge=0.85, no retuning

## How to continue

```bash
# Place Sonata / Soul / Spark car_track CSVs under:
#   Dataset/ocslab_pipeline/Hyundai/
#   Dataset/ocslab_pipeline/Kia/
#   Dataset/ocslab_pipeline/Chevrolet/

python experimental-reviewer-ablation/real_ocslab_corrected/check_dataset_gate.py
python experimental-reviewer-ablation/real_ocslab_corrected/run_corrected_ablation.py
```

## Explicit non-goals

- No τ / DBSCAN / gate retuning after seeing results
- No use of OCSLab_Car / Challenge_D / Challenge_S
- No optimization for Campaign F1
