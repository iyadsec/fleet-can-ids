# RAW_DATASET_AUDIT.md

## Status: **ACCESSIBLE** — 34 / 34 traces resolved

| Field | Value |
|-------|-------|
| `OCSLAB_DATASET_DIR` | `/workspace/Dataset/In-Vehicle Network Intrusion Detection Challenge` |
| Accessible on Cloud Agent | **Yes** |
| Source | OneDrive share downloaded + extracted (gitignored under `Dataset/`) |
| Segments required | 34 |
| Segments resolved | **34** |
| Resolution table | `RAW_TRACE_RESOLUTION.csv` |

### Layout verified

```text
$OCSLAB_DATASET_DIR/
  car_track_preliminary_train/
  car_track_final_1st_train/
  car_track_final_2nd_train/
```

All `source_file` paths from `balanced_split_manifest.csv` remap to local relative paths under this root via the suffix after `In-Vehicle Network Intrusion Detection Challenge/`.

### Notes

- Dataset is **not** committed to Git (`.gitignore` covers `Dataset/`).
- Prior `STOP_OTHER` (macOS OneDrive path invisible on VM) is cleared.
- No raw files were renamed; only archive wrapper extraction was performed.
