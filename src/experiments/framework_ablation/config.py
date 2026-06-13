"""Framework ablation configuration mapping."""

from __future__ import annotations

from typing import Any

# C1–C3 main framework configurations
FRAMEWORK_CONFIGS: dict[str, dict[str, Any]] = {
    "C1": {
        "configuration": "C1",
        "label": "Local-only IDS",
        "method": "local_ids",
        "local_detection": True,
        "behavioural_similarity": False,
        "graph_construction": False,
        "message_passing": False,
        "campaign_decision": False,
        "campaign_metrics_na": True,
    },
    "C2": {
        "configuration": "C2",
        "label": "Similarity-only fleet correlation",
        "method": "descriptor_clustering",
        "local_detection": True,
        "behavioural_similarity": True,
        "graph_construction": True,
        "message_passing": False,
        "campaign_decision": True,
        "campaign_metrics_na": False,
    },
    "C3": {
        "configuration": "C3",
        "label": "Full graph-based fleet correlation (GraphSAGE)",
        "method": "fcgnn",
        "local_detection": True,
        "behavioural_similarity": True,
        "graph_construction": True,
        "message_passing": True,
        "campaign_decision": True,
        "campaign_metrics_na": False,
    },
}

CONFIG_LABELS = {k: v["label"] for k, v in FRAMEWORK_CONFIGS.items()}

METHOD_TO_FRAMEWORK: dict[str, str] = {
    v["method"]: k for k, v in FRAMEWORK_CONFIGS.items()
}

# Supplementary only
SUPPLEMENTARY_METHOD = "standard_gnn"

# Map attack_strength / scenario_key to S-ids
SCENARIO_MAP = {
    "strong": "S3",
    "weak": "S4",
    "S0_benign_control": "S0",
    "S1_isolated": "S1",
    "S2_non_coordinated": "S2",
    "S3_strong_campaign": "S3",
    "S4_weak_campaign": "S4",
}

REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
MAIN_CONFIGS = ["C1", "C2", "C3"]
COORDINATION_STRENGTHS = [0.50, 0.75, 1.00]
