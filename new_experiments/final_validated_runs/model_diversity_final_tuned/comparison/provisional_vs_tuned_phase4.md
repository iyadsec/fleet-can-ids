# Provisional vs tuned Phase 4

## Gate
- Provisional: default CampaignGateConfig (not validation-tuned)
- Tuned: `88061ce9aa6101a2`

## False campaign semantics
- Provisional legacy rate ≈ 1.0 when qualifying clusters exist (metric bug)
- Tuned uses decomposed A–D metrics

- false_campaign_alert_rate: provisional mean=0.667, tuned mean=0.020
- campaign_f1: provisional mean=0.093, tuned mean=0.933