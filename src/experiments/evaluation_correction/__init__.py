"""Corrected event/campaign evaluation logic for Phase 3 publication."""

from src.experiments.evaluation_correction.promotion import (
    PromotionConfig,
    apply_corrected_event_decisions,
    classify_local_evidence,
)

__all__ = [
    "PromotionConfig",
    "apply_corrected_event_decisions",
    "classify_local_evidence",
]
