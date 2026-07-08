# Benchmark Presentation Package

**Vehicle-Level IDS (non-collaborative baseline) vs FLEET-GUARD (collaborative framework)**

Generated: 2026-07-08

## Contents

| File | Description |
|------|-------------|
| `non_collaborative_vs_fleet_guard.csv` | Per-seed detailed benchmark results (both methods) |
| `non_collaborative_vs_fleet_guard_summary.csv` | Scenario-level summary means |
| `non_collaborative_vs_fleet_guard.md` | Validation report and provenance checks |
| `non_collaborative_vs_fleet_guard_overleaf.tex` | Legacy Overleaf snippet (repo tables) |
| `table_non_collaborative_vs_fleet_guard.tex` | Original method-comparison table (repo) |
| `table_scenario_level_comparison.tex` | Original scenario-level table with numeric local metrics (repo) |
| `table_1_capability_comparison.tex` | **Paper Table 1** — capability checkmarks |
| `table_2_benchmark_results.tex` | **Paper Table 2** — headline benchmark metrics |
| `table_3_scenario_level_comparison.tex` | **Paper Table 3** — scenario-level qualitative comparison |
| `benchmark_subsection_overleaf.tex` | Ready-to-paste paper subsection |
| `benchmark_workflow_diagram.mmd` | Mermaid workflow diagram (baseline vs FLEET-GUARD) |

## Source files

This package was built from validated outputs produced by:

```bash
python scripts/run_non_collaborative_vs_fleet_guard.py
```

| Role | Source path |
|------|-------------|
| Detailed results | `results/non_collaborative_vs_fleet_guard.csv` |
| Summary | `results/non_collaborative_vs_fleet_guard_summary.csv` |
| Method table | `tables/table_non_collaborative_vs_fleet_guard.tex` |
| Scenario table | `tables/table_scenario_level_comparison.tex` |
| Validation report | `reports/non_collaborative_vs_fleet_guard.md` |
| Overleaf snippet | `reports/non_collaborative_vs_fleet_guard_overleaf.tex` |
| FLEET-GUARD archive | `experimental-2026-06-23/01_primary_ocslab_balanced/` |

## Key benchmark values

### FLEET-GUARD (campaign size 5, validated archive P7/P8)

| Metric | Value |
|--------|-------|
| Strong campaign F1 | **0.733** |
| Weak campaign F1 | **0.500** |
| False campaign rate (S0) | **0.000** |

### Vehicle-Level IDS baseline (local, θ_strong = 0.80)

| Scenario | Local F1 | FPR |
|----------|----------|-----|
| S0 Benign fleet | N/A | 0.000 |
| S1 Isolated attack | 1.000 | 0.000 |
| S2 Independent attacks | 1.000 | 0.000 |
| S3 Strong campaign | 0.711 | 0.117 |
| S4 Weak campaign | 0.907 | 0.000 |

**Campaign metrics for Vehicle-Level IDS: N/A** (campaign reasoning not supported).

## Including in Overleaf

1. Copy this folder (or the `.tex` files) into your Overleaf project.
2. Ensure your preamble includes:

```latex
\usepackage{booktabs}
\usepackage{amssymb}  % for \checkmark
```

3. Paste the subsection:

```latex
\input{paper_outputs/benchmark_non_collaborative_vs_fleet_guard_2026_07_08/benchmark_subsection_overleaf.tex}
```

4. Include the three paper tables (adjust paths as needed):

```latex
\input{paper_outputs/benchmark_non_collaborative_vs_fleet_guard_2026_07_08/table_1_capability_comparison.tex}
\input{paper_outputs/benchmark_non_collaborative_vs_fleet_guard_2026_07_08/table_2_benchmark_results.tex}
\input{paper_outputs/benchmark_non_collaborative_vs_fleet_guard_2026_07_08/table_3_scenario_level_comparison.tex}
```

5. For the workflow diagram, render `benchmark_workflow_diagram.mmd` with a Mermaid-compatible tool and export as PDF/PNG for `\includegraphics`.

## Reproducibility

Regenerate the underlying benchmark data:

```bash
python scripts/run_non_collaborative_vs_fleet_guard.py
```

Validate archive consistency:

```bash
python scripts/verify_balanced_campaign_tables.py
python scripts/validate_repository.py
```
