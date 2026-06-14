# Balanced publication validation

**Critical failures:** 0
**Guard gap (frames):** 100

## Checks

1. Chevrolet/Hyundai/Kia benign train/validation/test windows.
2. No row/window in multiple splits.
3. No event ID duplicated across splits.
4. Guard gaps on segmented traces.
5. IF trained on benign train only.
6. Shared fleet config from validation.
7. Test scenarios regenerated.
8. Table P2 with publication caption.
9. Original E2E outputs not overwritten (separate root).

**Status:** PASS