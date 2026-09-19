# CHECKPOINT_VERDICT

# **STOP_OTHER**

## One-line reason

The raw OCSLab challenge dataset path provided is a **local macOS OneDrive path that is not mounted on this Cloud Agent VM**, so vehicle-level reconstruction, validation descriptors, and validation fleet scenarios cannot be generated here — even though the **balanced source-trace/segment split is exactly recovered** from committed manifests and a **revised metric protocol has been proposed (unused)**.

---

## Checklist vs requested READY criteria

| Requirement | Status |
|-------------|--------|
| Source-trace split recovered sufficiently | **Yes — EXACT_SPLIT_RECOVERED** from `balanced_split_manifest.csv` |
| Validation data legitimate & usable | **No** — raw frames/descriptors not on VM |
| Vehicle-level reconstruction consistent | **Not run** (blocked) |
| Validation descriptors/scenarios available | **Not generated** (blocked) |
| Metric protocol proposed, not used | **Yes** — `PROPOSED_REVISED_METRIC_PROTOCOL.md` |

Because READY_FOR_METRIC_PROTOCOL_REVIEW requires reconstruction through validation scenarios, this checkpoint is **not** READY.

---

## Why not other STOP codes alone

| Code | Why not primary |
|------|-----------------|
| `STOP_SPLIT_NOT_RECOVERABLE` | Split **is** exactly recovered from artifacts |
| `STOP_VEHICLE_PIPELINE_MISMATCH` | No reconstruction yet; FPR≤5% code matches paper statement |
| `STOP_VALIDATION_DATA_INSUFFICIENT` | Would apply if raw data were present but too thin; here data are **absent from the VM** |
| `READY_FOR_METRIC_PROTOCOL_REVIEW` | Descriptors/scenarios/vehicle reconstruction incomplete |

`STOP_OTHER` captures the environment blocker accurately.

---

## What was completed

1. New tree `experimental-2026-09-20/eta_revalidation_from_raw/` (June historical + 2026-09-19 audits preserved).  
2. Raw-dataset accessibility audit.  
3. Exact balanced split recovery + `WINDOW_PROVENANCE.csv`.  
4. Documented IF FPR≤5% selection code consistency (no run).  
5. Proposed revised metric protocol **without** evaluating η.  
6. η **not** selected; CTT **not** run; final OCSLab fleet tests **not** run.

---

## Required next action (human / environment)

Mount or sync the challenge dataset into the Cloud Agent at e.g. `Dataset/ocslab/` or set `OCSLAB_DATASET_DIR` to a VM-visible path pointing at:

`…/In-Vehicle Network Intrusion Detection Challenge`

Then resume:

1. Window + IF reconstruction under recovered split (write new models here only).  
2. Vehicle-level audit vs table_P4.  
3. Validation descriptors + scenarios.  
4. Stop again at **READY_FOR_METRIC_PROTOCOL_REVIEW** (or an amended approved protocol) **before** any η sweep.

---

## Verdict code

```text
STOP_OTHER
```
