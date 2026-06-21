# table_CUR_COMP4_fleet_graph_edge_comparison

                                     Metric                             OCSLab curated result                        can-train-and-test corrected result                                                   Interpretation
graph nodes (production / evaluation graph)                         77,233 (full fleet graph)                       200 (OCSLab-aligned scenario graphs) Corrected CTT uses 200-node scenario graphs for fair comparison.
                                graph edges                                           819,914                           ~1,071 mean (200-node scenarios)      Corrected CTT scenario edge counts align with OCSLab scale.
              cross-vehicle edge percentage                               39.68% (full graph)                                                     0.530%          Both use behavioural similarity without temporal edges.
                        temporal edges used                        0 (behavioural similarity)                                                          0                  Both exclude temporal edges in reported graphs.
       scenario graph nodes (fixed package)                                               200                   200-node scenario graphs (corrected CTT)                            Matched 200-node evaluation protocol.
            scenario edge sensitivity range                             370–1311 unique edges         1051–1076 (corrected CTT τ/cap sweep on 200 nodes)                      Comparable graph sizes for scenario sweeps.
            best edge / connectivity region                      S3 ~437 edges; S4 ~370 edges τ=0.88, cap=3, mutual kNN (primary); fallback τ=0.85/cap=5            Campaign F1=1.0 on strong/weak with consistency rule.
      fragmentation / over-connection trend S2 partial incorrect merge; edge sweep documented             Unrelated merge 1.0→0.0 after consistency rule     Post-clustering consistency rule fixes unrelated over-merge.
