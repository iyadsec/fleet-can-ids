"""Diversity compositions and analysis tiers for corrected Phase 4."""

from __future__ import annotations

from typing import Any

from src.experiments.model_diversity.compositions import (
    D1_STRONG_ROTATIONS,
    D1_WEAK_ROTATIONS,
    D2_STRONG_ROTATIONS,
    D2_WEAK_COMPOSITION,
    D3_STRONG_ROTATIONS,
    REQUIRED_SEEDS,
    composition_label,
)


def analysis_tier(model_composition: dict[str, int], diversity_level: int) -> str:
    if diversity_level == 3 or model_composition.get("Chevrolet", 0) > 0 and diversity_level > 1:
        return "exploratory_mixed_attack"
    return "controlled_same_attack"


def resolve_corrected_composition(
    attack_strength: str,
    diversity_level: int,
    seed: int,
) -> tuple[dict[str, int] | None, str, str, str | None]:
    tier = ""
    if attack_strength == "weak" and diversity_level == 3:
        return None, "", "", "unsupported_by_dataset: no weak Chevrolet malicious descriptors"

    idx = REQUIRED_SEEDS.index(seed) if seed in REQUIRED_SEEDS else seed % 10

    if attack_strength == "strong":
        rotations = {1: D1_STRONG_ROTATIONS, 2: D2_STRONG_ROTATIONS, 3: D3_STRONG_ROTATIONS}
        pool = rotations.get(diversity_level)
        if not pool:
            return None, "", "", f"unsupported diversity {diversity_level}"
        comp = dict(pool[idx % len(pool)])
    elif diversity_level == 1:
        comp = dict(D1_WEAK_ROTATIONS[idx % len(D1_WEAK_ROTATIONS)])
    elif diversity_level == 2:
        comp = dict(D2_WEAK_COMPOSITION)
    else:
        return None, "", "", "unsupported_by_dataset"

    tier = analysis_tier(comp, diversity_level)
    return comp, composition_label(comp), tier, None


def supported_conditions() -> list[dict[str, Any]]:
    rows = []
    for strength in ("strong", "weak"):
        for level in (1, 2, 3):
            for seed in REQUIRED_SEEDS:
                comp, label, tier, reason = resolve_corrected_composition(strength, level, seed)
                rows.append(
                    {
                        "attack_strength": strength,
                        "diversity_level": level,
                        "seed": seed,
                        "supported": comp is not None,
                        "composition": label,
                        "analysis_tier": tier,
                        "unsupported_reason": reason or "",
                    }
                )
    return rows
