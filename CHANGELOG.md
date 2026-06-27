# Changelog

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
