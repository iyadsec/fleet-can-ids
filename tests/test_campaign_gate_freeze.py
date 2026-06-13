from pathlib import Path

import yaml


def test_gate_frozen_before_test():
    gate = yaml.safe_load(
        Path("new_experiments/final_validated_runs/model_diversity_final_tuned/configs/final_selected_campaign_gate.yaml").read_text()
    )
    assert "frozen_at" in gate
    assert "config_hash" in gate
