# CTT Split Integrity Report

## Integrity checks

- Train/test file overlap: **0** (expected 0)
- Training files: **60**
- Test files: **176**
- Attack files in train_01 (excluded from benign training): **52**
- No source-row overlap across splits: **verified** (disjoint files per subset)

## Policy

Each set uses `train_01` for benign-only local onboarding and threshold calibration. Test subsets evaluate known/unknown vehicle and attack generalisation per dataset design.
