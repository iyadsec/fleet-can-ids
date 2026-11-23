# Changelog

## 2026-11-23 (June 2026 reproducibility snapshot)

### Added
- `reproducibility/june_2026/`: the 42 June modules missing from `61a8203` (38 verified against June bytecode, 4 marked unverified), manifest with SHA-256, path-free bytecode verification record (the original `.pyc` files embed local paths and are kept privately), June environment, hashes of uncommitted June intermediates, reproduction and comparison scripts, and the reproduction check (detection results equal June)

### Changed
- `validation_2026-11-21_natural_campaigns`: local directory prefix in the 60 failed-run tracebacks replaced by `<CODE_REPO>/` (no other cell changed)
- README and provenance document: cohesion view described as 6–8 behaviour features (8-D was imprecise); reproduction section points to the snapshot; split `hash()` nondeterminism and truncated raw files documented

### Unchanged
- All historical outputs, configurations and result values

## 2026-11-21 (VehicleSec 2027 documentation and provenance)

### Added
- `docs/RESULTS_PROVENANCE_AND_CORRECTIONS.md`: generating commit, unpublished June modules (SHA-256), executed parameters, scenario construction, corrected claims
- `validation_2026-11-21_natural_campaigns/`: separate S3/S4 descriptor-construction sensitivity analysis and per-window descriptor serialization measurement (summary outputs only)

### Changed
- README rewritten: executed parameters (9-D graph view, 6–8-feature cohesion, actual thresholds), S0–S4 interpretation, controlled S3/S4 prototype, serialization figures, separated November analysis, limitations
- `docs/paper_pipeline.md`: provenance note; P5 53.69% marked superseded; CTT fleet results marked ground-truth-assisted

### Unchanged
- All files under `experimental-2026-06-23/` and `OVERLEAF_CROSS_DATASET_ARTIFACTS/` (historical; corrections are documented separately)

## Unreleased (publication readiness audit)

### Added
- FLEET-GUARD branding and publication-ready README
- Full OCSLab pipeline scripts (`experiments/07`–`09`, `run_full_pipeline.py`)
- `docs/datasets.md` and `docs/paper_pipeline.md`
- `scripts/check_environment.py`, `prepare_datasets.py`, `validate_repository.py`, `regenerate_paper_artifacts.py`
- Smoke tests under `tests/`
- `LICENSE` (MIT), `CITATION.cff`

### Changed
- Vehicle-level primary model: **Isolation Forest** (benign-only training)
- Removed hardcoded OneDrive / `/Users/` paths from configs and loaders
- Cross-dataset dataset documented as **can-train-and-test** (DTU DOI 10.11583/DTU.24805533)
- Privacy wording: **privacy-aware descriptor abstraction** (not formal privacy preservation)

### Notes
- Canonical paper numbers: `experimental-2026-06-23/`
- Cross-dataset Overleaf bundle: `OVERLEAF_CROSS_DATASET_ARTIFACTS/`
