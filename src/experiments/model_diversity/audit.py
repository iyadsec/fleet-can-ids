"""Phase 4 data-availability audit and source pool manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.campaign_analysis_corrected import (
    DescriptorBudget,
    DEFAULT_BENIGN_PER_ATTACKED,
    DEFAULT_BENIGN_PER_BENIGN,
    DEFAULT_DESCRIPTORS_PER_VEHICLE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_MALICIOUS_PER_ATTACKED,
    _assert_catalog_supports,
)
from src.experiments.data_splits import is_benign_attack_type
from src.experiments.model_diversity.compositions import (
    REQUIRED_SEEDS,
    composition_label,
    resolve_composition,
    supported_conditions,
)
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.vehicle_instance_builder import VEHICLE_MODELS, build_instance_catalog, source_trace_name


def build_source_pool(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    weak_th: float,
    strong_th: float,
) -> pd.DataFrame:
    join_cols = ["window_id", "vehicle_model", "source_file"]
    test = manifest[manifest["split"] == "test"]
    desc = descriptors.merge(test[join_cols + ["split"]].drop_duplicates(), on=join_cols, how="inner")

    rows: list[dict[str, Any]] = []
    for vm in VEHICLE_MODELS:
        for atk in sorted(desc["attack_type"].dropna().unique()):
            for strength in ("strong", "weak"):
                sub = desc[(desc["vehicle_model"] == vm) & (desc["attack_type"] == atk)]
                if sub.empty:
                    continue
                if is_benign_attack_type(atk):
                    if strength != "strong":
                        continue
                    pool = sub
                elif strength == "strong":
                    pool = sub[sub["anomaly_score"] >= strong_th]
                else:
                    pool = sub[(sub["anomaly_score"] >= weak_th) & (sub["anomaly_score"] < strong_th)]
                if pool.empty:
                    continue
                for sf, g in pool.groupby("source_file"):
                    seg = f"{int(g['window_id'].min())}-{int(g['window_id'].max())}"
                    inst_n = len(
                        catalog[
                            (catalog["vehicle_model"] == vm)
                            & (catalog["source_file"].astype(str) == str(sf))
                        ]
                    )
                    rows.append(
                        {
                            "vehicle_model": vm,
                            "attack_type": atk,
                            "attack_strength": strength,
                            "source_trace": source_trace_name(str(sf)),
                            "source_segment": seg,
                            "available_malicious_descriptors": int(len(pool)) if not is_benign_attack_type(atk) else 0,
                            "available_benign_descriptors": int(len(pool)) if is_benign_attack_type(atk) else 0,
                            "independent_instance_count": inst_n,
                            "eligible_diversity_levels": "",
                            "notes": "",
                        }
                    )
    return pd.DataFrame(rows)


def run_model_diversity_audit(
    config: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = config.get("paths", {})
    splits = config.get("splits", {})
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    camp_cfg = config.get("campaign", {})

    descriptors, features = load_descriptor_tables(
        Path(paths["anomaly_descriptors"]),
        Path(paths["window_features"]),
    )
    manifest_path = output_root / "manifests" / "split_manifest.csv"
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=manifest_path,
        seed=int(splits.get("seed", 42)),
        train_ratio=float(splits.get("train", 0.70)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )

    catalog = build_instance_catalog(
        descriptors,
        manifest,
        weak_threshold=weak_th,
        strong_threshold=strong_th,
        min_windows_per_segment=int(camp_cfg.get("min_windows_per_segment", 10)),
    )

    budget = DescriptorBudget(
        DEFAULT_DESCRIPTORS_PER_VEHICLE,
        DEFAULT_MALICIOUS_PER_ATTACKED,
        DEFAULT_BENIGN_PER_ATTACKED,
        DEFAULT_BENIGN_PER_BENIGN,
        DEFAULT_FLEET_SIZE,
    )

    conditions = supported_conditions()
    failures: list[str] = []
    unsupported: list[dict[str, Any]] = []

    for row in conditions:
        row["catalog_ok"] = False
        if not row["supported"]:
            unsupported.append(
                {
                    "attack_strength": row["attack_strength"],
                    "diversity_level": row["diversity_level"],
                    "seed": row["seed"],
                    "reason": row["unsupported_reason"] or "unsupported_by_dataset",
                }
            )
            continue
        comp, _, _ = resolve_composition(row["attack_strength"], row["diversity_level"], row["seed"])
        assert comp is not None
        try:
            _assert_catalog_supports(
                catalog,
                comp,
                n_benign=DEFAULT_FLEET_SIZE - 5,
                attack_strength=row["attack_strength"],  # type: ignore[arg-type]
                budget=budget,
            )
            row["catalog_ok"] = True
        except ValueError as exc:
            row["catalog_ok"] = False
            failures.append(
                f"{row['attack_strength']} D{row['diversity_level']} seed={row['seed']}: {exc}"
            )
            unsupported.append(
                {
                    "attack_strength": row["attack_strength"],
                    "diversity_level": row["diversity_level"],
                    "seed": row["seed"],
                    "reason": str(exc),
                }
            )

    pool = build_source_pool(descriptors, manifest, catalog, weak_th=weak_th, strong_th=strong_th)
    pool_path = output_root / "manifests" / "model_diversity_source_pool.csv"
    pool.to_csv(pool_path, index=False)

    join_cols = ["window_id", "vehicle_model", "source_file"]
    desc_test = descriptors.merge(
        manifest[manifest["split"] == "test"][join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="inner",
    )
    mal = desc_test[~desc_test["attack_type"].map(is_benign_attack_type)]

    per_model: dict[str, Any] = {}
    for vm in VEHICLE_MODELS:
        vm_cat = catalog[catalog["vehicle_model"] == vm]
        strong_mal = mal[(mal["vehicle_model"] == vm) & (mal["anomaly_score"] >= strong_th)]
        weak_mal = mal[
            (mal["vehicle_model"] == vm)
            & (mal["anomaly_score"] >= weak_th)
            & (mal["anomaly_score"] < strong_th)
        ]
        benign = desc_test[
            (desc_test["vehicle_model"] == vm) & desc_test["attack_type"].map(is_benign_attack_type)
        ]
        per_model[vm] = {
            "segments": len(vm_cat),
            "strong_malicious": len(strong_mal),
            "weak_malicious": len(weak_mal),
            "benign": len(benign),
            "strong_by_attack": strong_mal.groupby("attack_type").size().to_dict(),
            "weak_by_attack": weak_mal.groupby("attack_type").size().to_dict(),
        }

    supported_strong = sorted(
        {r["diversity_level"] for r in conditions if r["attack_strength"] == "strong" and r.get("catalog_ok")}
    )
    supported_weak = sorted(
        {r["diversity_level"] for r in conditions if r["attack_strength"] == "weak" and r.get("catalog_ok")}
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_passed": len(failures) == 0,
        "failures": failures,
        "budget": budget.__dict__,
        "supported_strong_diversity_levels": supported_strong,
        "supported_weak_diversity_levels": supported_weak,
        "weak_d3_supported": False,
        "per_model": per_model,
        "expected_strong_runs": 3 * 3 * len(REQUIRED_SEEDS),
        "expected_weak_runs": len(supported_weak) * 3 * len(REQUIRED_SEEDS),
        "catalog_instances": len(catalog),
    }

    md = _render_audit_md(summary, pool, desc_test, unsupported)
    audit_path = output_root / "audit" / "model_diversity_data_availability.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(md, encoding="utf-8")

    if failures:
        raise ValueError(f"Model diversity audit failed ({len(failures)} conditions). First: {failures[0]}")

    unsupported_df = pd.DataFrame(unsupported).drop_duplicates(
        subset=["attack_strength", "diversity_level"], keep="first"
    )
    return catalog, pool, summary


def _render_audit_md(
    summary: dict[str, Any],
    pool: pd.DataFrame,
    desc_test: pd.DataFrame,
    unsupported: list[dict[str, Any]],
) -> str:
    lines = [
        "# Model diversity data availability audit",
        "",
        f"**Generated:** {summary['generated_at']}",
        f"**Audit passed:** {summary['audit_passed']}",
        "",
        "## Fixed experimental conditions",
        "",
        "- Fleet size: 20",
        "- Campaign size: 5 attacked instances",
        "- Descriptor budget: 5 malicious + 5 benign-on-attacked per attacked vehicle; 10 benign per benign vehicle",
        "- Expected nodes: 200",
        "- Coordination strength: 1.0",
        "",
        "## Supported diversity levels",
        "",
        f"- **Strong:** D{summary['supported_strong_diversity_levels']}",
        f"- **Weak:** D{summary['supported_weak_diversity_levels']}",
        "",
        "## Per vehicle model (held-out test)",
        "",
    ]
    for vm, info in summary["per_model"].items():
        lines.extend(
            [
                f"### {vm}",
                "",
                f"- Disjoint segments: {info['segments']}",
                f"- Strong malicious descriptors: {info['strong_malicious']}",
                f"- Weak malicious descriptors: {info['weak_malicious']}",
                f"- Benign descriptors: {info['benign']}",
                f"- Strong by attack: `{info['strong_by_attack']}`",
                f"- Weak by attack: `{info['weak_by_attack']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Attack family across platforms",
            "",
            "- Hyundai/Kia strong and weak: `malfunction`",
            "- Chevrolet strong: `fuzzy` (no weak malicious events in test split)",
            "",
            "## Unsupported configurations",
            "",
        ]
    )
    seen = set()
    for u in unsupported:
        key = (u["attack_strength"], u["diversity_level"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {u['attack_strength']} D{u['diversity_level']}: {u['reason']}")

    lines.extend(
        [
            "",
            "## Source pool",
            "",
            f"Rows in `manifests/model_diversity_source_pool.csv`: **{len(pool)}**",
            "",
            "## Trace inventory",
            "",
        ]
    )
    for sf, g in desc_test.groupby("source_file"):
        lines.append(
            f"- `{source_trace_name(str(sf))}` ({g['vehicle_model'].iloc[0]}): "
            f"{len(g)} windows"
        )
    return "\n".join(lines) + "\n"
