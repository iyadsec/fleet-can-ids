# P12 Statistical tests

| family               | comparison   | metric               |   paired_seeds | test     |   mean_difference |   p_value_raw |   effect_size | significant   |   p_value_holm |
|:---------------------|:-------------|:---------------------|---------------:|:---------|------------------:|--------------:|--------------:|:--------------|---------------:|
| strong_campaign_size | 2_vs_5       | campaign_f1          |             10 | paired_t |       -0.2        |   0.278873    |     -0.364399 | False         |     1          |
| strong_campaign_size | 2_vs_5       | membership_precision |             10 | paired_t |       -0.265      |   0.162619    |     -0.480938 | False         |     1          |
| strong_campaign_size | 2_vs_5       | membership_recall    |             10 | paired_t |       -0.16       |   0.462036    |     -0.242933 | False         |     1          |
| strong_campaign_size | 2_vs_5       | end_to_end_latency   |             10 | paired_t |       -0.0111394  |   0.557811    |     -0.192468 | False         |     1          |
| strong_campaign_size | 5_vs_10      | campaign_f1          |             10 | paired_t |       -0.266667   |   0.0697074   |     -0.650791 | False         |     1          |
| strong_campaign_size | 5_vs_10      | membership_precision |             10 | paired_t |       -0.275      |   0.0700524   |     -0.649827 | False         |     1          |
| strong_campaign_size | 5_vs_10      | membership_recall    |             10 | paired_t |       -0.12       |   0.317269    |     -0.334829 | False         |     1          |
| strong_campaign_size | 5_vs_10      | end_to_end_latency   |             10 | paired_t |       -0.00562005 |   0.744446    |     -0.106307 | False         |     1          |
| strong_campaign_size | 2_vs_10      | campaign_f1          |             10 | paired_t |       -0.466667   |   0.012799    |     -0.97913  | False         |     0.255981   |
| strong_campaign_size | 2_vs_10      | membership_precision |             10 | paired_t |       -0.54       |   0.00622882  |     -1.12219  | False         |     0.137034   |
| strong_campaign_size | 2_vs_10      | membership_recall    |             10 | paired_t |       -0.28       |   0.120283    |     -0.54267  | False         |     1          |
| strong_campaign_size | 2_vs_10      | end_to_end_latency   |             10 | paired_t |       -0.0167594  |   0.268908    |     -0.372598 | False         |     1          |
| weak_campaign_size   | 2_vs_5       | campaign_f1          |             10 | paired_t |       -0.433333   |   0.0389899   |     -0.763386 | False         |     0.701818   |
| weak_campaign_size   | 2_vs_5       | membership_precision |             10 | paired_t |       -0.56       |   0.0136807   |     -0.966092 | False         |     0.259933   |
| weak_campaign_size   | 2_vs_5       | membership_recall    |             10 | paired_t |       -0.44       |   0.0682972   |     -0.654783 | False         |     1          |
| weak_campaign_size   | 2_vs_5       | end_to_end_latency   |             10 | paired_t |       -0.0309881  |   0.166333    |     -0.476236 | False         |     1          |
| weak_campaign_size   | 5_vs_10      | campaign_f1          |             10 | paired_t |       -0.216667   |   0.274033    |     -0.368351 | False         |     1          |
| weak_campaign_size   | 5_vs_10      | membership_precision |             10 | paired_t |       -0.3        |   0.0811262   |     -0.621059 | False         |     1          |
| weak_campaign_size   | 5_vs_10      | membership_recall    |             10 | paired_t |       -0.23       |   0.0880406   |     -0.604937 | False         |     1          |
| weak_campaign_size   | 5_vs_10      | end_to_end_latency   |             10 | paired_t |        0.00779641 |   0.779978    |      0.091027 | False         |     0.779978   |
| weak_campaign_size   | 2_vs_10      | campaign_f1          |             10 | paired_t |       -0.65       |   0.00262326  |     -1.3008   | False         |     0.0603351  |
| weak_campaign_size   | 2_vs_10      | membership_precision |             10 | paired_t |       -0.86       |   0.000170182 |     -1.94254  | True          |     0.00408437 |
| weak_campaign_size   | 2_vs_10      | membership_recall    |             10 | paired_t |       -0.67       |   0.00651727  |     -1.11307  | False         |     0.136863   |
| weak_campaign_size   | 2_vs_10      | end_to_end_latency   |             10 | paired_t |       -0.0231917  |   0.207755    |     -0.429204 | False         |     1          |