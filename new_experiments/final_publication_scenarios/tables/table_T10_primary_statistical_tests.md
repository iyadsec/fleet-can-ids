# Primary statistical tests

| family        | comparison       | metric                    |   p_value_raw |   p_value_holm |   effect_size |   n_seeds |   paired_seeds |   mean_difference |
|:--------------|:-----------------|:--------------------------|--------------:|---------------:|--------------:|----------:|---------------:|------------------:|
| campaign_size | strong_n2_vs_n5  | campaign_f1               |     0.67831   |              1 |      0.135526 |       nan |             10 |               0.1 |
| campaign_size | strong_n5_vs_n10 | campaign_f1               |     0.508646  |              1 |      0.217643 |       nan |             10 |               0.2 |
| campaign_size | strong_n2_vs_n10 | campaign_f1               |     0.278873  |              1 |      0.364399 |       nan |             10 |               0.3 |
| campaign_size | strong_n2_vs_n5  | campaign_detection_rate   |     0.343436  |              1 |     -0.316228 |       nan |             10 |              -0.1 |
| campaign_size | strong_n5_vs_n10 | campaign_detection_rate   |   nan         |              1 |      0        |       nan |             10 |               0   |
| campaign_size | strong_n2_vs_n10 | campaign_detection_rate   |     0.343436  |              1 |     -0.316228 |       nan |             10 |              -0.1 |
| campaign_size | strong_n2_vs_n5  | false_campaign_alert_rate |     0.343436  |              1 |     -0.316228 |       nan |             10 |              -0.1 |
| campaign_size | strong_n5_vs_n10 | false_campaign_alert_rate |   nan         |              1 |      0        |       nan |             10 |               0   |
| campaign_size | strong_n2_vs_n10 | false_campaign_alert_rate |     0.343436  |              1 |     -0.316228 |       nan |             10 |              -0.1 |
| campaign_size | weak_n2_vs_n5    | campaign_f1               |     0.443332  |              1 |     -0.253546 |       nan |             10 |              -0.2 |
| campaign_size | weak_n5_vs_n10   | campaign_f1               |     0.343436  |              1 |     -0.316228 |       nan |             10 |              -0.1 |
| campaign_size | weak_n2_vs_n10   | campaign_f1               |     0.193422  |              1 |     -0.444478 |       nan |             10 |              -0.3 |
| campaign_size | weak_n2_vs_n5    | campaign_detection_rate   |     0.0811262 |              1 |     -0.621059 |       nan |             10 |              -0.3 |
| campaign_size | weak_n5_vs_n10   | campaign_detection_rate   |   nan         |              1 |      0        |       nan |             10 |               0   |
| campaign_size | weak_n2_vs_n10   | campaign_detection_rate   |     0.0811262 |              1 |     -0.621059 |       nan |             10 |              -0.3 |
| campaign_size | weak_n2_vs_n5    | false_campaign_alert_rate |     0.0811262 |              1 |     -0.621059 |       nan |             10 |              -0.3 |
| campaign_size | weak_n5_vs_n10   | false_campaign_alert_rate |   nan         |              1 |      0        |       nan |             10 |               0   |
| campaign_size | weak_n2_vs_n10   | false_campaign_alert_rate |     0.0811262 |              1 |     -0.621059 |       nan |             10 |              -0.3 |