# P9 Campaign size cost

| test_condition              |   campaign_size |   unique_edges |   graph_build_time |   end_to_end_latency |   peak_memory |   graph_nodes |
|:----------------------------|----------------:|---------------:|-------------------:|---------------------:|--------------:|--------------:|
| Strong Coordinated Campaign |               2 |          296.1 |           0.122685 |             0.264218 |       1.20707 |           200 |
| Strong Coordinated Campaign |               5 |          426.2 |           0.127178 |             0.275357 |       1.24025 |           200 |
| Strong Coordinated Campaign |              10 |          661.2 |           0.123815 |             0.280977 |       1.30149 |           200 |
| Weak Coordinated Campaign   |               2 |          257.8 |           0.1138   |             0.260692 |       1.19657 |           200 |
| Weak Coordinated Campaign   |               5 |          349.8 |           0.146754 |             0.29168  |       1.21827 |           200 |
| Weak Coordinated Campaign   |              10 |          486   |           0.128201 |             0.283884 |       1.25316 |           200 |