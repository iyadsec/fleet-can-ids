"""Benign-attack helpers for fleet scaler fitting (shared publication path)."""

from __future__ import annotations

NORMAL_ATTACK_TYPES = {"normal", "attack_free", "benign", "none", "no_attack"}


def is_benign_attack_type(attack_type: str) -> bool:
    return str(attack_type).strip().lower() in NORMAL_ATTACK_TYPES
