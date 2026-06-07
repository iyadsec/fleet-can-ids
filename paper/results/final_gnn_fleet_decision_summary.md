# Final GNN Fleet Decision — Summary

1. **isolated_attack events:** 49760
2. **coordinated_attack events:** 7267
3. **Attack types in detected campaigns (evaluation only):** flooding, fuzzy, malfunction, replay
4. **Vehicles in coordinated campaigns:** 3
5. **GNN fleet IDS added capability:** classifies suspicious activity as isolated vs coordinated using GraphSAGE embeddings, behaviour cohesion, and multi-vehicle campaign clusters (no attack-type metadata in the decision path).
6. **Architecture alignment:** Pipeline follows vehicle IDS → descriptors → behaviour-normalized graph → GraphSAGE (trained on IDS evidence) → DBSCAN on embeddings → behaviour-cohesion campaign gate → final decision.
7. **Final output matches isolated vs coordinated:** Yes — every suspicious event assigned `isolated_attack` or `coordinated_attack`.

## Conclusion

The proposed GNN-based fleet correlation layer extends isolated vehicle-level intrusion detection by learning relational representations over behavioural anomaly descriptors and classifying suspicious activity as either isolated attacks or coordinated multi-vehicle attack campaigns.
