"""Export publication-ready evidence package (IEEE-friendly outputs).

This module does NOT replace existing pipeline outputs under `outputs/` and
`data/processed/`. Instead, it collects key artifacts and exports them into:

  - results/*.csv
  - results/*.json
  - figures/*.png, figures/*.pdf
  - tables/*.tex
  - logs/*.txt
  - final_experiment_summary.md
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PublicationPaths:
    root: Path
    results_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path

    @classmethod
    def from_config(cls, project_root: Path, cfg: dict[str, Any]) -> "PublicationPaths":
        pub = cfg.get("publication") or {}
        root = project_root
        return cls(
            root=root,
            results_dir=root / str(pub.get("results_dir", "results")),
            figures_dir=root / str(pub.get("figures_dir", "figures")),
            tables_dir=root / str(pub.get("tables_dir", "tables")),
            logs_dir=root / str(pub.get("logs_dir", "logs")),
        )

    def ensure(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        logger.warning("Missing artifact: %s", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _df_to_ieee_tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True)
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            body.strip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )


def export_publication_package(
    *,
    project_root: Path,
    config: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Path]:
    """Collect pipeline artifacts and export them to paper-ready folders."""
    pub_cfg = config.get("publication") or {}
    if not bool(pub_cfg.get("enabled", True)):
        logger.info("Publication export disabled in config.")
        return {}

    overwrite = bool(pub_cfg.get("overwrite", True))
    pub_paths = PublicationPaths.from_config(project_root, config)
    pub_paths.ensure()

    # Core result tables (CSV)
    csv_keys = [
        "vehicle_results",
        "graph_statistics",
        "gnn_metrics",
        "final_detection_outcomes",
        "final_outcome_summary",
        "fleet_value_summary",
        "weak_signal_upgrade_summary",
        "cross_vehicle_cluster_summary",
        "raw_vs_descriptor_size",
        # Privacy evidence (attacker-style evaluation)
        "privacy_vehicle_reidentification",
        "privacy_reconstruction_risk",
    ]
    for key in csv_keys:
        if key in artifacts:
            _copy_file(
                project_root / artifacts[key],
                pub_paths.results_dir / f"{key}.csv",
                overwrite=overwrite,
            )

    # Figures (PNG) – copy if present.
    figure_keys = [
        "raw_vs_descriptor_size_figure",
        "gnn_loss_figure",
        "gnn_tsne_figure",
    ]
    for key in figure_keys:
        if key in artifacts:
            src = project_root / artifacts[key]
            _copy_file(src, pub_paths.figures_dir / src.name, overwrite=overwrite)

    # Privacy figures are generated into `outputs/figures/` by the pipeline step.
    # They are not currently tracked in config artifacts (paths are stable), so we
    # copy them by name if present.
    privacy_figure_names = [
        "privacy_reid_confusion_full_descriptor.png",
        "privacy_reid_confusion_transmitted_descriptor.png",
        "privacy_reid_leakage_comparison.png",
        "privacy_reconstruction_risk.png",
    ]
    for name in privacy_figure_names:
        src = project_root / "outputs" / "figures" / name
        _copy_file(src, pub_paths.figures_dir / name, overwrite=overwrite)

    # Lightweight JSON index for reproducibility.
    index = {
        "config": config.get("project", {}),
        "exported_csv": [f"{k}.csv" for k in csv_keys if k in artifacts],
        "exported_figures": [
            (project_root / artifacts[k]).name for k in figure_keys if k in artifacts
        ]
        + [n for n in privacy_figure_names if (project_root / "outputs" / "figures" / n).exists()],
    }
    _write_json(pub_paths.results_dir / "export_index.json", index)

    # IEEE-style TeX tables for a subset of headline evidence.
    written: dict[str, Path] = {}
    tex_tables: list[tuple[str, str, str]] = [
        ("raw_vs_descriptor_size", "Descriptor compression summary.", "tab:compression"),
        ("weak_signal_upgrade_summary", "Weak signal upgrade summary.", "tab:weak-upgrade"),
        ("cross_vehicle_cluster_summary", "Cross-vehicle cluster summary.", "tab:cross-vehicle"),
        ("final_outcome_summary", "Final outcome summary.", "tab:final-outcomes"),
        ("privacy_vehicle_reidentification", "Privacy leakage: vehicle re-identification attack.", "tab:privacy-reid"),
        ("privacy_reconstruction_risk", "Privacy leakage: reconstruction risk from transmitted descriptors.", "tab:privacy-recon"),
    ]
    for name, caption, label in tex_tables:
        path = project_root / artifacts.get(name, "")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        tex = _df_to_ieee_tex_table(df, caption=caption, label=label)
        out = pub_paths.tables_dir / f"{name}.tex"
        out.write_text(tex, encoding="utf-8")
        written[name] = out

    summary_path = pub_paths.root / "final_experiment_summary.md"
    summary_lines = [
        "## Final experiment summary",
        "",
        "This file is generated by `run_all_experiments.py`.",
        "",
        "### Exported artifacts",
        "",
        f"- **results**: `{pub_paths.results_dir.relative_to(pub_paths.root)}/`",
        f"- **figures**: `{pub_paths.figures_dir.relative_to(pub_paths.root)}/`",
        f"- **tables**: `{pub_paths.tables_dir.relative_to(pub_paths.root)}/`",
        f"- **logs**: `{pub_paths.logs_dir.relative_to(pub_paths.root)}/`",
        "",
        "### Notes",
        "",
        "- Vehicle IDS uses **self-supervised Isolation Forest** trained only on benign windows.",
        "- Fleet campaign detection (IEEE) uses a **behaviour-normalized similarity graph** and "
        "**DBSCAN clustering** on descriptor features — not a trained FCGNN.",
        "- An optional PyTorch Geometric GNN step (`train_gnn`) exists for legacy pipeline clustering only.",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "results_dir": pub_paths.results_dir,
        "figures_dir": pub_paths.figures_dir,
        "tables_dir": pub_paths.tables_dir,
        "logs_dir": pub_paths.logs_dir,
        "final_summary": summary_path,
        **{f"table_{k}": v for k, v in written.items()},
    }

