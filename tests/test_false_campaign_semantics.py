from src.experiments.model_diversity_final_tuned.false_campaign_metrics import legacy_false_campaign_rate_explanation


def test_legacy_false_campaign_rate_is_one_when_detected():
    info = legacy_false_campaign_rate_explanation(2, expect_campaign=True)
    assert info["legacy_false_campaign_alert_rate"] == 1.0
    assert info["legacy_always_one_when_detected"]
