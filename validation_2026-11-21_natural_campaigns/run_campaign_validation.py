"""Separate validation experiment (21 Nov 2026) for S3/S4 coordinated campaigns.

Reuses the June 2026 balanced-run artefacts read-only (descriptors, IF scores,
fleet scaler, shared fleet configuration, instance catalog logic) and changes
only how campaign descriptors are constructed:

  A_reproduction   coordination_strength=1.0 (June code path, reproduction check)
  B_natural        coordination_strength=0.0 (real descriptors, benign untouched,
                   no prototype -> no cross-split information)
  C_partial_train  strength=0.5 applied to malicious windows only; prototype =
                   mean of the June attack family computed on the TRAIN split only

Pre-declared before any results were inspected; no parameter is tuned.
Run with cwd = the Code project root (resolve_project_root uses cwd).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# FLEET_GUARD_CODE_ROOT: checkout of origin/cursor/campaign-clustering (61a8203) plus the
# June run directory and unpublished packages listed in docs/RESULTS_PROVENANCE_AND_CORRECTIONS.md.
CODE = Path(os.environ["FLEET_GUARD_CODE_ROOT"]).resolve()
RUN = CODE / "new_experiments/final_end_to_end_publication_run_balanced"
OUT = Path(__file__).resolve().parent / "outputs"
sys.path.insert(0, str(CODE))


class _GitHeadSourceFinder:
    """Serve committed (HEAD) source for project files truncated to 0 bytes by OneDrive sync."""

    def find_spec(self, fullname, path=None, target=None):
        import importlib.machinery
        import importlib.util
        import subprocess

        if not fullname.startswith("src"):
            return None
        rel = Path(*fullname.split("."))
        for cand, is_pkg in ((rel / "__init__.py", True), (rel.with_suffix(".py"), False)):
            f = CODE / cand
            if f.exists() and f.stat().st_size == 0:
                src = subprocess.run(["git", "-C", str(CODE), "show", f"HEAD:{cand.as_posix()}"],
                                     capture_output=True, text=True).stdout
                if not src:
                    return None
                RESTORED_FROM_GIT.append(cand.as_posix())
                loader = _StringLoader(src, str(f))
                return importlib.util.spec_from_loader(
                    fullname, loader, origin=str(f), is_package=is_pkg)
        return None


class _StringLoader:
    def __init__(self, source, origin):
        self.source, self.origin = source, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = self.origin
        if module.__spec__.submodule_search_locations is not None:
            module.__path__ = [str(Path(self.origin).parent)]
        exec(compile(self.source, self.origin, "exec"), module.__dict__)


RESTORED_FROM_GIT: list[str] = []
sys.meta_path.insert(0, _GitHeadSourceFinder())

import yaml  # noqa: E402

import src.experiments.campaign_analysis_corrected as cac  # noqa: E402
import src.experiments.coordination_strength as cstr  # noqa: E402
from src.experiments.campaign_analysis_corrected import DescriptorBudget, generate_corrected_campaign_scenario  # noqa: E402
from src.experiments.campaign_analysis_generator import STRONG_ATTACK_DEFAULT, WEAK_ATTACK_DEFAULT  # noqa: E402
from src.experiments.coordinated_campaign_refinement.refinement_pipeline import run_refinement_fcgnn  # noqa: E402
from src.experiments.final_end_to_end_publication_run.runner import _experiment_config  # noqa: E402
from src.experiments.final_end_to_end_publication_run.scenario_registry import (  # noqa: E402
    PUBLICATION_SCENARIOS,
    REQUIRED_SEEDS,
)
from src.experiments.final_end_to_end_publication_run.source_windows import build_e2e_test_scenario  # noqa: E402
from src.experiments.final_end_to_end_publication_run_balanced.runner import _load_master_config  # noqa: E402
from src.experiments.final_shared_configuration.metrics import extract_run_metrics  # noqa: E402
from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration  # noqa: E402
from src.experiments.final_shared_configuration.source_windows import _annotate_stats, _attach_split  # noqa: E402
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config  # noqa: E402
from src.experiments.vehicle_instance_builder import build_instance_catalog  # noqa: E402
from src.evaluation.final_gnn_fleet_decision_experiment import GNN_FEATURE_COLUMNS, prepare_gnn_fleet_node_matrix  # noqa: E402
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS  # noqa: E402

CONDITIONS = ("A_reproduction", "B_natural", "C_partial_train")


def load_context():
    master = _load_master_config(RUN, CODE)
    config = _experiment_config(master, RUN)
    models = pd.read_csv(RUN / "manifests/local_model_training_manifest.csv")
    th = json.loads(models.iloc[0]["validation_threshold"])
    config["local_ids"].update({"weak_threshold": th["weak"], "strong_threshold": th["strong"]})

    descriptors = pd.read_csv(RUN / "descriptors/all_descriptors.csv")
    manifest = pd.read_csv(RUN / "manifests/balanced_window_manifest.csv")
    ensure_fleet_scaler_in_config(config, descriptors, manifest)

    orig_sim = cstr.measure_mean_pairwise_similarity

    def _sim_with_scaler(*args, **kwargs):
        kwargs.setdefault("fleet_scaler_provenance", config.get("_fleet_scaler_provenance"))
        return orig_sim(*args, **kwargs)

    cstr.measure_mean_pairwise_similarity = _sim_with_scaler
    cac.measure_mean_pairwise_similarity = _sim_with_scaler

    budget = DescriptorBudget(
        descriptors_per_vehicle=int(master["scenario"]["source_windows_per_vehicle"]),
        malicious_per_attacked=5,
        benign_per_attacked=5,
        benign_per_benign=10,
        total_fleet_size=int(master["scenario"]["fleet_size"]),
    )
    catalog = build_instance_catalog(
        descriptors.drop(columns=["split"], errors="ignore"),
        manifest,
        weak_threshold=float(config["local_ids"]["weak_threshold"]),
        strong_threshold=float(config["local_ids"]["strong_threshold"]),
        min_windows_per_segment=10,
        target_split="test",
    )
    shared_payload = yaml.safe_load((RUN / "configs/final_shared_fleet_configuration.yaml").read_text())
    shared_payload.pop("selection", None)
    shared_payload.pop("master_config_hash", None)
    shared = SharedFleetConfiguration(**shared_payload)
    return config, descriptors, manifest, budget, catalog, shared


def build_scenario(condition, *, strength_label, seed, cs, descriptors, manifest, catalog, config, budget, pub):
    if condition == "A_reproduction":
        return build_e2e_test_scenario(
            pub.internal_key, seed=seed, campaign_size=cs, descriptors=descriptors,
            manifest=manifest, catalog=catalog, config=config, budget=budget,
        )
    desc = _attach_split(descriptors, manifest)
    scenario_df, _, membership, _, _ = generate_corrected_campaign_scenario(
        attack_strength=strength_label, seed=seed, descriptors=desc, manifest=manifest,
        catalog=catalog, config=config, campaign_size=cs, coordination_strength=0.0, budget=budget,
    )
    if condition == "C_partial_train":
        attack = STRONG_ATTACK_DEFAULT if strength_label == "strong" else WEAK_ATTACK_DEFAULT
        train = descriptors[descriptors["split"] == "train"]
        proto = cstr.compute_campaign_prototype(train, attack_type=attack)
        mask = (scenario_df["scenario_role"] == "coordinated") & (scenario_df["ground_truth_malicious"] == 1)
        scenario_df, _ = cstr.apply_coordination_strength(
            scenario_df, strength=0.5, campaign_prototype=proto, target_mask=mask, seed=seed,
        )
    annotated, stats = _annotate_stats(scenario_df, config)
    return annotated, membership, stats


def campaign_diagnostics(source_df, config):
    """Natural-variation diagnostics of malicious campaign windows in the 9-D scaled GNN space."""
    prov = config["_fleet_scaler_provenance"]
    mal = source_df[(source_df["scenario_role"] == "coordinated") & (source_df["ground_truth_malicious"] == 1)]
    ben_att = source_df[(source_df["scenario_role"] == "coordinated") & (source_df["ground_truth_malicious"] == 0)]
    out = {
        "n_campaign_malicious_windows": len(mal),
        "n_unique_malicious_24d_vectors": int(mal[BEHAVIOURAL_FEATURE_COLUMNS].round(9).drop_duplicates().shape[0]),
        "campaign_attack_types": "|".join(sorted(mal["attack_type"].astype(str).unique())),
        "campaign_source_traces": int(mal["source_trace"].nunique()) if "source_trace" in mal else np.nan,
        "campaign_vehicle_models": "|".join(sorted(mal["vehicle_model"].astype(str).unique())),
        "n_benign_windows_on_attacked": len(ben_att),
    }
    if len(mal) >= 2:
        X, _, _ = prepare_gnn_fleet_node_matrix(mal.reset_index(drop=True), fleet_scaler_provenance=prov)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        S = Xn @ Xn.T
        iu = np.triu_indices(len(S), 1)
        out["mal_mean_pairwise_cosine_9d"] = float(S[iu].mean())
        tok = mal["vehicle_token"].to_numpy()
        cross = tok[iu[0]] != tok[iu[1]]
        out["mal_cross_vehicle_pairs_cos_ge_0.95"] = float((S[iu][cross] >= 0.95).mean()) if cross.any() else np.nan
    return out


def run(conditions, seeds):
    config, descriptors, manifest, budget, catalog, shared = load_context()
    rows = []
    for condition in conditions:
        for key in ("strong_campaign", "weak_campaign"):
            pub = PUBLICATION_SCENARIOS[key]
            for seed in seeds:
                for cs in (2, 5, 10):
                    rid = f"{condition}__{key}_cs{cs}_seed{seed}"
                    rd = OUT / "runs" / rid
                    try:
                        source_df, membership, cand_stats = build_scenario(
                            condition, strength_label=pub.attack_strength, seed=seed, cs=cs,
                            descriptors=descriptors, manifest=manifest, catalog=catalog,
                            config=config, budget=budget, pub=pub,
                        )
                        outputs, _, annotated, _ = run_refinement_fcgnn(
                            source_df, membership, config, seed, shared.to_refinement_design()
                        )
                        rd.mkdir(parents=True, exist_ok=True)
                        source_df.to_csv(rd / "source_windows.csv", index=False)
                        outputs.event_predictions.to_csv(rd / "event_predictions.csv", index=False)
                        row = extract_run_metrics(
                            run_id=rid, scenario_id=pub.test_condition, attack_strength=pub.attack_strength,
                            seed=seed, campaign_size=cs, source_df=source_df, membership=membership,
                            outputs=outputs, cand_stats=cand_stats, expect_campaign=True,
                        )
                        row.update(campaign_diagnostics(source_df, config))
                        row.update({"condition": condition, "scenario_key": key, "status": "ok"})
                    except Exception as exc:  # report failures, never drop them
                        row = {"run_id": rid, "condition": condition, "scenario_key": key, "seed": seed,
                               "campaign_size": cs, "status": f"FAIL: {exc}", "trace": traceback.format_exc()}
                    rows.append(row)
                    print(rid, row.get("status"), row.get("campaign_f1"), row.get("membership_f1"), flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"run_level_metrics_{'_'.join(conditions)}.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(REQUIRED_SEEDS))
    a = ap.parse_args()
    run(a.conditions, a.seeds)
