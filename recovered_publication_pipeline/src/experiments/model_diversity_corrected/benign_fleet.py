"""Fixed heterogeneous benign fleet composition for corrected Phase 4."""

from __future__ import annotations

BENIGN_FLEET_COMPOSITION: dict[str, int] = {
    "Hyundai": 5,
    "Kia": 5,
    "Chevrolet": 5,
}

CONTROLLED_ATTACK_MODELS = ("Hyundai", "Kia")
CONTROLLED_ATTACK_TYPE = "malfunction"
