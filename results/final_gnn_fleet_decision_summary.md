# Final GNN Fleet Decision — Summary

1. **isolated_attack events:** 56743
2. **coordinated_attack events:** 284
3. **Attack types forming coordinated campaigns:** flooding
4. **Vehicles in coordinated campaigns:** 2
5. **GNN fleet IDS added capability:** classifies suspicious activity as isolated vs coordinated using GraphSAGE embeddings and multi-vehicle campaign clusters (local IDS campaign detection = 0).
6. **Architecture alignment:** Pipeline follows vehicle IDS → descriptors → behaviour-normalized graph → GraphSAGE → DBSCAN on embeddings → final decision.
7. **Final output matches isolated vs coordinated:** Yes — every suspicious event assigned `isolated_attack` or `coordinated_attack`.

## Conclusion

The proposed GNN-based fleet correlation layer extends isolated vehicle-level intrusion detection by learning relational representations over behavioural anomaly descriptors and classifying suspicious activity as either isolated attacks or coordinated multi-vehicle attack campaigns.
