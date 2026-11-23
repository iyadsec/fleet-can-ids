# Changelog

## 2026-11-23 (VehicleSec 2027: June 2026 provenance and reproducibility snapshot)

### Added
- `docs/RESULTS_PROVENANCE_AND_CORRECTIONS.md`: generating commit, unpublished June modules (SHA-256), executed parameters, scenario construction, corrected claims
- `reproducibility/june_2026/`: the 42 June modules missing from `61a8203` (38 verified against June bytecode, 4 marked unverified), manifest with SHA-256, path-free bytecode verification record (the original `.pyc` files embed local paths and are kept privately), June environment, hashes of uncommitted June intermediates, reproduction and comparison scripts, and the reproduction check (detection results equal June)

### Changed
- README rewritten: executed parameters (9-D graph view, 6–8-feature cohesion, actual thresholds), S0–S4 interpretation, controlled S3/S4 prototype, P5 descriptor-size caveat, reproduction from the snapshot, limitations
- `docs/paper_pipeline.md`: provenance note; P5 53.69% marked superseded; CTT fleet results marked ground-truth-assisted

### Unchanged
- All historical outputs, configurations and result values, including every file under `experimental-2026-06-23/` and `OVERLEAF_CROSS_DATASET_ARTIFACTS/` (corrections are documented separately)

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
