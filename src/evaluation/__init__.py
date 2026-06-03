"""Evaluation metrics and result persistence."""

from src.evaluation.campaign_clustering import (
    load_embedding_table,
    run_campaign_clustering,
    save_campaign_clusters,
)
from src.evaluation.final_decision import (
    classify_final_outcomes,
    save_final_outcomes,
    summarize_final_outcomes,
)
from src.evaluation.metrics import compute_metrics, save_metrics

__all__ = [
    "compute_metrics",
    "save_metrics",
    "load_embedding_table",
    "run_campaign_clustering",
    "save_campaign_clusters",
    "classify_final_outcomes",
    "save_final_outcomes",
    "summarize_final_outcomes",
]
