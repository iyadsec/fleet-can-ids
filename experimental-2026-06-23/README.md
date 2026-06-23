# Experimental results bundle (2026-06-23)

Canonical artifacts for the FLEET-GUARD paper. **Do not cite superseded folders in `archive/`.**

## Layout

| Folder | Contents | Use in paper |
|--------|----------|--------------|
| `01_primary_ocslab_balanced/` | Balanced end-to-end publication run (Tables P4–P12) | **Section VII primary OCSLab results** |
| `02_baseline_ablation/` | M1/M2/M3 ablation (see `RESULT_PROVENANCE.md`) | Baseline & ablation subsection |
| `03_cross_dataset_ctt/` | CTT cross-dataset Overleaf bundle | Cross-dataset validation section |
| `DRAFT_TABLE_MAPPING.md` | Draft Table I–IX → file mapping | Manuscript revision |

## Rebuild

```bash
python scripts/build_baseline_ablation_comparison.py
python scripts/consolidate_experimental_results.py
```

## Authoritative vs draft

- Draft Table III (unrelated merge **0.400**) matches the balanced publication run.
- Draft Table IV strong cs=5 F1 (**0.867**) is from the **superseded original split**; update to **0.733** from `table_P7`.
