# Coordinated Campaign Detection — Summary

**Note:** Controlled campaign scenarios are constructed from labelled attack windows
in the public Car-Hacking dataset. They evaluate fleet-level campaign reasoning;
they do not represent externally synchronized real-world campaigns.

1. **Campaign scenarios generated:** 4 (65076 descriptor windows).
2. **Campaign types evaluated:** flooding, fuzzy, malfunction, replay.
3. **Fleet correlation detected coordinated campaigns:** Yes — at least one true campaign matched a valid cross-vehicle cluster.
4. **Best cross-vehicle clustering (attack type):** flooding.
5. **Campaign detection rate:** 25.0% (1/4 scenarios).
6. **Campaign purity (mean dominant-attack ratio):** 1.000.
7. **False campaign rate:** 0.0%.
8. **Added capability vs local IDS:** Local IDS flags individual anomalies but cannot group cross-vehicle behaviourally similar events. Fleet graph clustering (dbscan) achieved cross-vehicle edge share 41.69% and campaign detection rate 25.0%.
9. **Limitations:** Campaign scenarios are synthetically defined from per-vehicle labelled attacks; clustering quality varies by attack type; Chevrolet has fewer replay windows; high similarity thresholds and minimum cluster gates limit recall; false campaign clusters remain when behavioural descriptors overlap across attack classes.

## Conclusion

The fleet-aware correlation layer enables campaign-level detection by grouping behaviourally similar anomaly descriptors across multiple vehicles. This provides a capability that is not available to isolated vehicle-level IDS models.
