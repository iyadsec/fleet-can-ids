# Descriptor Compactness and Security — Summary

## 1. Compactness
- **Average raw CAN window size:** 2075.75 bytes
- **Transmitted descriptor size:** 165.08 bytes (gzip ~48.51 bytes/record)
- **Compression ratio:** 12.57×
- **Bandwidth reduction:** 92.05%

## 2. Fleet scalability
- **Raw bandwidth @ 100 vehicles:** 7621.93 MB
- **Descriptor bandwidth @ 100 vehicles:** 424.98 MB
- **Fleet bandwidth reduction:** 94.42%

## 3. Information disclosure
Raw CAN transmission exposes timestamp sequences, CAN IDs, DLC, payload bytes, and exact message order. Descriptor transmission sends only aggregated behavioural/statistical features (timing summaries, anomaly scores, evidence flags) and omits frame-level identifiers and raw payloads. See `information_disclosure_comparison.csv` and `table_information_disclosure_comparison.tex` — not the deprecated heuristic exposure score (retained only in `heuristic_exposure_appendix.csv` for debugging).

## 4. Payload-statistic reconstruction
Targets: `byte_mean_0..7`, `byte_std_0..7`, `mean_dlc`, `std_dlc` (statistics derived from payloads, not exact bytes). Attackers: Linear Regression, Random Forest Regressor, MLP Regressor on transmitted descriptor fields excluding byte-level aggregates.
- **Raw baseline R²:** 1.0000 (statistics directly available from raw CAN)
- **Descriptor attacker R² (mean):** 0.4439
- **Random/mean baseline R²:** -0.0000
- **Mean MSE (descriptor attackers, all targets):** 123.0419

Lower descriptor R² indicates **limited ability to infer payload-derived statistics** from the uplink payload; it does **not** claim exact payload-byte recovery is impossible in all settings.

## 5. Vehicle fingerprinting
- **Accuracy (raw CAN windows):** 99.97%
- **Accuracy (descriptors):** 99.97%
- **Reduction:** -0.01%

High descriptor fingerprinting indicates that some vehicle-specific behavioural patterns remain in the transmitted statistics. **Limitation:** future work should investigate descriptor anonymisation, feature coarsening, or differential privacy for stronger unlinkability.

## 6. Conclusion

The proposed descriptor **substantially reduces communication overhead** and **raw CAN data disclosure** while preserving behavioural evidence for fleet-level intrusion detection. It **limits payload-statistic reconstruction** relative to raw CAN and supports **privacy-preserving fleet analysis**, but does **not** guarantee full privacy or complete vehicle anonymisation.
