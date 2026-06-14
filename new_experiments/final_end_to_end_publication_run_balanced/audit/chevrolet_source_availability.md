# Chevrolet source availability audit

| Source file | Attack type | Benign | Row count | Window count | Original split | Segmentable |
|-------------|-------------|--------|-----------|--------------|----------------|-------------|
| Attack_free_CHEVROLET_Spark_train.csv | attack_free | yes | 136934 | 2737 | train | yes |
| Flooding_CHEVROLET_Spark_train.csv | flooding | no | 85000 | 1699 | train | yes |
| Fuzzy_CHEVROLET_Spark_train.csv | fuzzy | no | 41000 | 819 | train | yes |
| Malfunction_CHEVROLET_Spark_train.csv | malfunction | no | 51000 | 1019 | test | yes |

## Summary

- Benign source files: 1
- Malicious source files: 3

## Recommendation

Chevrolet has a single benign trace (`Attack_free_CHEVROLET_Spark_train.csv`). Publication-safe validation coverage requires **contiguous segment-level splitting** with **100-frame guard gaps** between train, validation, and test partitions.