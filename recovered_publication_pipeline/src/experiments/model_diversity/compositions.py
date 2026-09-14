"""Vehicle-model diversity compositions for Phase 4 (campaign size 5)."""

from __future__ import annotations

from typing import Any

REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

D1_STRONG_ROTATIONS: list[dict[str, int]] = [
    {"Hyundai": 5, "Kia": 0, "Chevrolet": 0},
    {"Hyundai": 0, "Kia": 5, "Chevrolet": 0},
    {"Hyundai": 0, "Kia": 0, "Chevrolet": 5},
]

D2_STRONG_ROTATIONS: list[dict[str, int]] = [
    {"Hyundai": 3, "Kia": 2, "Chevrolet": 0},
    {"Hyundai": 3, "Kia": 0, "Chevrolet": 2},
    {"Hyundai": 0, "Kia": 3, "Chevrolet": 2},
]

D3_STRONG_ROTATIONS: list[dict[str, int]] = [
    {"Hyundai": 2, "Kia": 2, "Chevrolet": 1},
    {"Hyundai": 2, "Kia": 1, "Chevrolet": 2},
    {"Hyundai": 1, "Kia": 2, "Chevrolet": 2},
]

D1_WEAK_ROTATIONS: list[dict[str, int]] = [
    {"Hyundai": 5, "Kia": 0, "Chevrolet": 0},
    {"Hyundai": 0, "Kia": 5, "Chevrolet": 0},
]

D2_WEAK_COMPOSITION = {"Hyundai": 3, "Kia": 2, "Chevrolet": 0}

CONFIG_LABELS = {
    "C1": "Local-only IDS",
    "C2": "Similarity-only fleet correlation",
    "C3": "GraphSAGE-based fleet correlation",
}

METHOD_TO_CONFIG = {
    "local_ids": "C1",
    "descriptor_clustering": "C2",
    "fcgnn": "C3",
}


def composition_label(comp: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(comp.items()) if v > 0)


def diversity_level_from_composition(comp: dict[str, int]) -> int:
    return sum(1 for v in comp.values() if v > 0)


def resolve_composition(
    attack_strength: str,
    diversity_level: int,
    seed: int,
) -> tuple[dict[str, int] | None, str, str | None]:
    """
    Return (composition, label, unsupported_reason).

    unsupported_reason is set when the condition must not be run.
    """
    if attack_strength == "weak" and diversity_level == 3:
        return None, "", "unsupported_by_dataset: no weak Chevrolet malicious descriptors"

    idx = REQUIRED_SEEDS.index(seed) if seed in REQUIRED_SEEDS else seed % 10

    if attack_strength == "strong":
        rotations = {1: D1_STRONG_ROTATIONS, 2: D2_STRONG_ROTATIONS, 3: D3_STRONG_ROTATIONS}
        pool = rotations.get(diversity_level)
        if not pool:
            return None, "", f"unsupported diversity level {diversity_level}"
        comp = dict(pool[idx % len(pool)])
        return comp, composition_label(comp), None

    if diversity_level == 1:
        comp = dict(D1_WEAK_ROTATIONS[idx % len(D1_WEAK_ROTATIONS)])
        return comp, composition_label(comp), None
    if diversity_level == 2:
        comp = dict(D2_WEAK_COMPOSITION)
        return comp, composition_label(comp), None
    return None, "", "unsupported_by_dataset"


def supported_conditions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strength in ("strong", "weak"):
        for level in (1, 2, 3):
            for seed in REQUIRED_SEEDS:
                comp, label, reason = resolve_composition(strength, level, seed)
                rows.append(
                    {
                        "attack_strength": strength,
                        "diversity_level": level,
                        "seed": seed,
                        "supported": comp is not None,
                        "composition": label,
                        "unsupported_reason": reason or "",
                    }
                )
    return rows
