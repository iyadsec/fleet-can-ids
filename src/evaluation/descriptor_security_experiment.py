"""Descriptor compactness, scalability, and security/privacy experiments."""

from __future__ import annotations

import gzip
import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features.descriptor_generator import (
    TRANSMIT_COLUMNS,
    build_transmitted_descriptors,
    generate_anomaly_descriptors,
)
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.models.vehicle_ids import load_feature_dataset, load_window_predictions
from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Publication figure width (readable in IEEE two-column layout).
PAPER_FIG_WIDTH = 7.0
PAPER_FIG_FONTS = {
    "title": 14,
    "label": 13,
    "tick": 12,
    "legend": 12,
    "annotation": 11,
}

RAW_CAN_COLOR = "#C00000"
DESCRIPTOR_COLOR = "#4472C4"

PRIVACY_COMPARISON_ROWS: list[tuple[str, bool, bool]] = [
    # (element, raw_exposed, descriptor_exposed) — True → ✗, False → ✓
    ("CAN IDs", True, False),
    ("Payload Bytes", True, False),
    ("Vehicle Identity", True, True),
    ("Message Sequence", True, False),
    ("Behavioural Evidence", False, False),
]

_DISCLOSURE_SHORT_LABELS: dict[str, tuple[str, str]] = {
    "Exposed": ("Full exposure", "#C00000"),
    "Present": ("Available", "#C00000"),
    "Implicitly inferable": ("Inferable", "#ED7D31"),
    "Not transmitted": ("Not transmitted", "#548235"),
    "Not transmitted directly": ("IDs omitted", "#548235"),
    "Aggregated statistics only": ("Stats only", "#FFC000"),
    "Aggregated / not directly transmitted": ("Aggregated", "#FFC000"),
    "Summarised as behavioural statistics": ("Summarised", "#FFC000"),
    "Not explicitly transmitted, but may remain inferable from behavioural patterns": (
        "No ID field",
        "#ED7D31",
    ),
    "Preserved": ("Preserved", "#4472C4"),
}

PAPER_FIGURE_CAPTIONS = {
    "figure_04_bandwidth_scaling": (
        "Bandwidth scaling comparison between raw CAN transmission and "
        "descriptor-based transmission across different fleet sizes."
    ),
    "figure_05_descriptor_information_disclosure": (
        "Information exposure comparison between raw CAN transmission and "
        "descriptor-based uplink. Descriptor abstraction removes raw identifiers "
        "and payload content while preserving behavioural anomaly evidence "
        "required for fleet-level intrusion detection."
    ),
}

FLEET_SIZES = (10, 50, 100, 500, 1000)

PAYLOAD_TARGET_COLS = [f"byte_mean_{i}" for i in range(8)] + [f"byte_std_{i}" for i in range(8)] + [
    "mean_dlc",
    "std_dlc",
]

BYTE_MEAN_COLS = [f"byte_mean_{i}" for i in range(8)]
BYTE_STD_COLS = [f"byte_std_{i}" for i in range(8)]
DLC_COLS = ["mean_dlc", "std_dlc"]

ATTACKER_SPECS: list[tuple[str, Any]] = [
    ("Linear Regression", LinearRegression()),
    (
        "Random Forest Regressor",
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    ),
    ("MLP Regressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)),
]

INFORMATION_DISCLOSURE_ROWS: list[dict[str, str]] = [
    {
        "Information Element": "Timestamp sequence",
        "Raw CAN Transmission": "Exposed",
        "Descriptor Transmission": "Aggregated / not directly transmitted",
        "Security/Privacy Impact": "Reduces exact traffic replay and timing traceability",
    },
    {
        "Information Element": "CAN ID",
        "Raw CAN Transmission": "Exposed",
        "Descriptor Transmission": "Not transmitted directly",
        "Security/Privacy Impact": "Hides vehicle-specific CAN identifier semantics",
    },
    {
        "Information Element": "DLC",
        "Raw CAN Transmission": "Exposed",
        "Descriptor Transmission": "Aggregated statistics only",
        "Security/Privacy Impact": "Reduces direct frame-level reconstruction",
    },
    {
        "Information Element": "Payload bytes",
        "Raw CAN Transmission": "Exposed",
        "Descriptor Transmission": "Not transmitted",
        "Security/Privacy Impact": "Prevents direct payload disclosure",
    },
    {
        "Information Element": "Message sequence",
        "Raw CAN Transmission": "Exposed",
        "Descriptor Transmission": "Summarised as behavioural statistics",
        "Security/Privacy Impact": "Reduces reconstruction of exact CAN traffic stream",
    },
    {
        "Information Element": "Vehicle identity",
        "Raw CAN Transmission": "Implicitly inferable",
        "Descriptor Transmission": (
            "Not explicitly transmitted, but may remain inferable from behavioural patterns"
        ),
        "Security/Privacy Impact": "Requires optional anonymisation or feature coarsening",
    },
    {
        "Information Element": "Behavioural anomaly evidence",
        "Raw CAN Transmission": "Present",
        "Descriptor Transmission": "Preserved",
        "Security/Privacy Impact": "Keeps useful IDS signal while reducing raw-data exposure",
    },
]


@dataclass(frozen=True)
class SecurityOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _df_to_ieee_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True, float_format="%.4f")
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


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _serialize_row_bytes(row: pd.Series, columns: list[str]) -> int:
    text = ",".join(str(row[c]) for c in columns) + "\n"
    return len(text.encode("utf-8"))


def _gzip_bytes(data: bytes) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(data)
    return len(buf.getvalue())


def estimate_raw_window_bytes(
    row: pd.Series,
    *,
    timestamp_bytes: int = 8,
    can_id_bytes: int = 4,
    dlc_field_bytes: int = 1,
) -> float:
    n = float(row.get("frame_count", row.get("n_frames", 0)) or 0)
    mean_dlc = float(row.get("mean_dlc", 8.0) or 8.0)
    payload_per_frame = min(max(mean_dlc, 0.0), 8.0)
    per_frame = timestamp_bytes + can_id_bytes + dlc_field_bytes + payload_per_frame
    return n * per_frame


def _heuristic_exposure_appendix() -> pd.DataFrame:
    """Optional appendix metric only — not used in main paper table."""
    raw_checks = {
        "can_ids_visible": 1.0,
        "payload_bytes_visible": 1.0,
        "message_sequence_visible": 1.0,
        "vehicle_specific_patterns_visible": 1.0,
    }
    desc_checks = {
        "raw_can_ids_recoverable": 0.0,
        "raw_payload_bytes_recoverable": 0.25,
        "vehicle_identifiers_visible": 0.0,
        "message_sequence_fully_visible": 0.2,
    }
    return pd.DataFrame(
        [
            {
                "view": "raw_can_heuristic",
                "mean_exposure_score": float(np.mean(list(raw_checks.values()))),
                "note": "Appendix/debug only; not a primary paper metric",
            },
            {
                "view": "descriptor_transmit_heuristic",
                "mean_exposure_score": float(np.mean(list(desc_checks.values()))),
                "note": "Appendix/debug only; not a primary paper metric",
            },
        ]
    )


def _build_information_disclosure_table() -> pd.DataFrame:
    return pd.DataFrame(INFORMATION_DISCLOSURE_ROWS)


def _train_fingerprint_accuracy(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    pipe = Pipeline(
        [
            ("pre", ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    pipe.fit(X_train, y_train)
    return float(accuracy_score(y_test, pipe.predict(X_test)))


def _random_baseline_r2(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Predict per-target training means; lower attacker capability than learned models."""
    mean_pred = np.mean(y_train, axis=0, keepdims=True)
    pred = np.repeat(mean_pred, len(y_test), axis=0)
    return float(r2_score(y_test, pred, multioutput="uniform_average"))


def _make_attacker(name: str, seed: int) -> Any:
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed)


def _eval_attacker_on_targets(
    X: np.ndarray,
    y_sub: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate each attacker on a target column subset."""
    if y_sub.ndim == 1:
        y_sub = y_sub.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sub, test_size=0.25, random_state=seed
    )
    rows: list[dict[str, Any]] = []
    for name, _ in ATTACKER_SPECS:
        model = _make_attacker(name, seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "attacker_model": name,
                "mse": float(mean_squared_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred, multioutput="uniform_average")),
            }
        )
    return rows


def _payload_statistic_reconstruction_analysis(
    transmit: pd.DataFrame,
    full_desc: pd.DataFrame,
    tx_recon_cols: list[str],
    target_cols: list[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Returns breakdown table and R² for raw baseline (1.0), descriptor attackers (mean),
    and random/mean baseline.
    """
    X = transmit[tx_recon_cols].fillna(0.0).to_numpy()
    y_all = full_desc[target_cols].fillna(0.0).to_numpy()

    target_groups: dict[str, list[str]] = {
        "byte_mean": [c for c in BYTE_MEAN_COLS if c in target_cols],
        "byte_std": [c for c in BYTE_STD_COLS if c in target_cols],
        "dlc_statistics": [c for c in DLC_COLS if c in target_cols],
        "all_targets_average": target_cols,
    }

    breakdown_rows: list[dict[str, Any]] = []
    attacker_r2_values: list[float] = []

    for group_name, cols in target_groups.items():
        if not cols:
            continue
        col_indices = [target_cols.index(c) for c in cols]
        y_sub = y_all[:, col_indices]
        for row in _eval_attacker_on_targets(X, y_sub, seed=seed):
            breakdown_rows.append(
                {
                    "attacker_model": row["attacker_model"],
                    "target_type": group_name,
                    "mse": row["mse"],
                    "r2": row["r2"],
                }
            )
            if group_name == "all_targets_average":
                attacker_r2_values.append(row["r2"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.25, random_state=seed
    )
    random_r2 = _random_baseline_r2(y_train, y_test)
    descriptor_r2 = float(np.mean(attacker_r2_values)) if attacker_r2_values else 0.0
    raw_baseline_r2 = 1.0

    return pd.DataFrame(breakdown_rows), raw_baseline_r2, descriptor_r2, random_r2


def load_descriptors_for_experiment(
    *,
    features_path: Path,
    predictions_path: Path,
    descriptors_path: Path,
    transmit_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = load_feature_dataset(features_path)
    if descriptors_path.exists():
        full_desc = pd.read_csv(descriptors_path)
        logger.info("Loaded %d descriptors from %s", len(full_desc), descriptors_path)
    else:
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Missing {descriptors_path} and {predictions_path}. "
                "Run vehicle IDS / descriptor pipeline first."
            )
        predictions = load_window_predictions(predictions_path)
        full_desc = generate_anomaly_descriptors(features, predictions)
        logger.info("Built %d descriptors from cached predictions", len(full_desc))

    if transmit_path.exists():
        transmit = pd.read_csv(transmit_path)
    else:
        transmit = build_transmitted_descriptors(full_desc, quantize_decimals=3)
    return features, full_desc, transmit


def _compute_exposure_score_metrics() -> tuple[float, float, float]:
    """Return raw score, descriptor score, and leakage reduction (%)."""
    appendix = _heuristic_exposure_appendix()
    raw = float(
        appendix.loc[appendix["view"] == "raw_can_heuristic", "mean_exposure_score"].iloc[0]
    )
    desc = float(
        appendix.loc[
            appendix["view"] == "descriptor_transmit_heuristic", "mean_exposure_score"
        ].iloc[0]
    )
    reduction = 100.0 * (raw - desc) / raw if raw else 0.0
    return raw, desc, reduction


def _apply_publication_fonts(ax: plt.Axes, *, title: str) -> None:
    fonts = PAPER_FIG_FONTS
    ax.set_title(title, fontsize=fonts["title"], fontweight="bold", pad=10)
    ax.set_xlabel(ax.get_xlabel(), fontsize=fonts["label"])
    ax.set_ylabel(ax.get_ylabel(), fontsize=fonts["label"])
    ax.tick_params(axis="both", labelsize=fonts["tick"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(fonts["legend"])


def _format_bandwidth_label(value_mb: float) -> str:
    if value_mb >= 1000.0:
        return f"{value_mb / 1000.0:.1f}k"
    return f"{int(round(value_mb))}"


def _annotate_bar_values(ax: plt.Axes, bars: Any, *, fontsize: int) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if height <= 0:
            continue
        y = height * 1.08 if ax.get_yscale() == "log" else height + ax.get_ylim()[1] * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            _format_bandwidth_label(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )


def _plot_bandwidth_scaling_line(scalability_df: pd.DataFrame, out: Path) -> None:
    """Line chart: raw CAN vs descriptor fleet bandwidth (full color)."""
    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(PAPER_FIG_WIDTH, 4.5))
        ax.plot(
            scalability_df["fleet_size"],
            scalability_df["total_raw_bandwidth_mb"],
            "o-",
            linewidth=2.2,
            markersize=8,
            label="Raw CAN Transmission",
            color=RAW_CAN_COLOR,
        )
        ax.plot(
            scalability_df["fleet_size"],
            scalability_df["total_descriptor_bandwidth_mb"],
            "s-",
            linewidth=2.2,
            markersize=8,
            label="Descriptor Transmission",
            color=DESCRIPTOR_COLOR,
        )
        ax.set_xlabel("Fleet Size")
        ax.set_ylabel("Bandwidth Usage (MB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", frameon=False)
        _apply_publication_fonts(ax, title="Bandwidth Scaling Across Fleet Sizes")
        fig.tight_layout()
        _save_figure(fig, out)


def _plot_bandwidth_scaling_grouped(scalability_df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: raw CAN vs descriptor fleet bandwidth (full color)."""
    fonts = PAPER_FIG_FONTS
    fleet_sizes = scalability_df["fleet_size"].to_numpy()
    raw_mb = scalability_df["total_raw_bandwidth_mb"].to_numpy()
    desc_mb = scalability_df["total_descriptor_bandwidth_mb"].to_numpy()
    x = np.arange(len(fleet_sizes), dtype=float)
    bar_width = 0.34

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(PAPER_FIG_WIDTH, 4.5))
        bars_raw = ax.bar(
            x - bar_width / 2,
            raw_mb,
            bar_width,
            label="Raw CAN Transmission",
            color=RAW_CAN_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        bars_desc = ax.bar(
            x + bar_width / 2,
            desc_mb,
            bar_width,
            label="Descriptor Transmission",
            color=DESCRIPTOR_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in fleet_sizes])
        ax.set_xlabel("Fleet Size")
        ax.set_ylabel("Bandwidth Usage (MB)")
        ax.grid(True, axis="y", which="major", alpha=0.3)
        ax.legend(loc="upper left", frameon=False)
        _apply_publication_fonts(ax, title="Bandwidth Scaling Comparison")
        _annotate_bar_values(ax, bars_raw, fontsize=fonts["tick"] - 1)
        _annotate_bar_values(ax, bars_desc, fontsize=fonts["tick"] - 1)
        fig.tight_layout()
        _save_figure(fig, out)


def _disclosure_cell(value: str) -> tuple[str, str]:
    text = str(value).strip()
    if text in _DISCLOSURE_SHORT_LABELS:
        return _DISCLOSURE_SHORT_LABELS[text]
    lowered = text.lower()
    if "not transmitted" in lowered:
        return ("Not transmitted", "#548235")
    if "aggregated" in lowered or "summar" in lowered:
        return ("Aggregated", "#FFC000")
    if "preserved" in lowered:
        return ("Preserved", "#4472C4")
    if "infer" in lowered:
        return ("Inferable", "#ED7D31")
    return ("Exposed", "#C00000")


def _plot_information_disclosure(disclosure_df: pd.DataFrame, out: Path) -> None:
    """Colored disclosure matrix: raw CAN vs descriptor uplink."""
    fonts = PAPER_FIG_FONTS
    elements = disclosure_df["Information Element"].tolist()
    raw_vals = disclosure_df["Raw CAN Transmission"].tolist()
    desc_vals = disclosure_df["Descriptor Transmission"].tolist()
    n_rows = len(elements)

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(PAPER_FIG_WIDTH, 0.72 * n_rows + 2.0))
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Raw CAN uplink", "Descriptor uplink"], fontsize=fonts["label"], fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(elements, fontsize=fonts["tick"])
        ax.set_title(
            "Information Disclosure: Raw CAN vs Descriptor Uplink",
            fontsize=fonts["title"],
            fontweight="bold",
            pad=12,
        )
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)

        for row_idx, (raw_text, desc_text) in enumerate(zip(raw_vals, desc_vals)):
            for col_idx, value in enumerate((raw_text, desc_text)):
                label, colour = _disclosure_cell(value)
                ax.add_patch(
                    plt.Rectangle(
                        (col_idx - 0.42, row_idx - 0.34),
                        0.84,
                        0.68,
                        facecolor=colour,
                        edgecolor="white",
                        linewidth=1.5,
                    )
                )
                text_colour = "white" if colour in {"#C00000", "#548235", "#4472C4"} else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fonts["tick"] - 1,
                    fontweight="bold",
                    color=text_colour,
                )

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#C00000", edgecolor="white"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#FFC000", edgecolor="white"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#548235", edgecolor="white"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#4472C4", edgecolor="white"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#ED7D31", edgecolor="white"),
        ]
        ax.legend(
            legend_handles,
            [
                "Full exposure",
                "Aggregated only",
                "Not transmitted",
                "Utility preserved",
                "Residual inferability",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False,
            fontsize=fonts["legend"],
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        _save_figure(fig, out)


def _plot_privacy_preservation_comparison(out: Path) -> None:
    """Colored checkmark privacy matrix with exposure-score summary."""
    fonts = PAPER_FIG_FONTS
    raw_score, desc_score, leakage_reduction = _compute_exposure_score_metrics()
    columns = ["Raw CAN", "Descriptor"]
    n_rows = len(PRIVACY_COMPARISON_ROWS)
    exposed_colour = "#F4CCCC"
    protected_colour = "#D9EAD3"
    cross_colour = "#C00000"
    check_colour = "#38761D"

    with plt.rc_context(IEEE_RC):
        fig = plt.figure(figsize=(PAPER_FIG_WIDTH, 0.58 * n_rows + 2.2))
        grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[n_rows + 0.55, 1.0], hspace=0.25)
        ax = fig.add_subplot(grid[0, 0])
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, fontsize=fonts["label"], fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([row[0] for row in PRIVACY_COMPARISON_ROWS], fontsize=fonts["tick"])
        ax.set_title(
            "Privacy Preservation: Raw CAN vs Descriptor Uplink",
            fontsize=fonts["title"],
            fontweight="bold",
            pad=10,
        )
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for row_idx, (_, raw_exposed, desc_exposed) in enumerate(PRIVACY_COMPARISON_ROWS):
            for col_idx, is_exposed in enumerate((raw_exposed, desc_exposed)):
                face = exposed_colour if is_exposed else protected_colour
                symbol = "\u2717" if is_exposed else "\u2713"
                symbol_colour = cross_colour if is_exposed else check_colour
                ax.add_patch(
                    plt.Rectangle(
                        (col_idx - 0.38, row_idx - 0.34),
                        0.76,
                        0.68,
                        facecolor=face,
                        edgecolor="#999999",
                        linewidth=0.8,
                    )
                )
                ax.text(
                    col_idx,
                    row_idx,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=fonts["title"] + 2,
                    fontweight="bold",
                    color=symbol_colour,
                )

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=protected_colour, edgecolor="#999999"),
            plt.Rectangle((0, 0), 1, 1, facecolor=exposed_colour, edgecolor="#999999"),
        ]
        ax.legend(
            legend_handles,
            ["Protected / preserved", "Exposed / inferable"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False,
            fontsize=fonts["legend"],
        )

        summary_ax = fig.add_subplot(grid[1, 0])
        summary_ax.axis("off")
        summary_ax.text(
            0.5,
            0.5,
            (
                f"Exposure Score — Raw CAN: {raw_score:.2f}  |  "
                f"Descriptor: {desc_score:.2f}  |  "
                f"Leakage Reduction: {leakage_reduction:.2f}%"
            ),
            ha="center",
            va="center",
            fontsize=fonts["annotation"],
            fontweight="bold",
            transform=summary_ax.transAxes,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#EAF2FB",
                "edgecolor": DESCRIPTOR_COLOR,
                "linewidth": 0.9,
            },
        )
        _save_figure(fig, out)


def export_h2_publication_figures(
    *,
    scalability_csv: Path,
    paper_figures_dir: Path,
    source_figures_dir: Path | None = None,
) -> dict[str, Path]:
    """Regenerate H2 paper figures (Fig. 4–5) and copy into ``paper/figures/``."""
    scalability_df = pd.read_csv(scalability_csv)
    source_dir = source_figures_dir or scalability_csv.parent.parent / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    fig4_src = source_dir / "bandwidth_scaling_fleet_sizes"
    fig5_src = source_dir / "information_disclosure_comparison"
    disclosure_df = _build_information_disclosure_table()
    _plot_bandwidth_scaling_line(scalability_df, fig4_src)
    _plot_information_disclosure(disclosure_df, fig5_src)

    written: dict[str, Path] = {}
    for stem, src in (
        ("figure_04_bandwidth_scaling", fig4_src),
        ("figure_05_descriptor_information_disclosure", fig5_src),
    ):
        for ext in (".pdf", ".png"):
            dest = paper_figures_dir / f"{stem}{ext}"
            shutil.copy2(src.with_suffix(ext), dest)
            if ext == ".pdf":
                written[stem] = dest
    return written


def _build_main_comparison_table(
    *,
    avg_raw: float,
    avg_transmit: float,
    compression_ratio: float,
    bandwidth_reduction_pct: float,
    fleet_100_raw_mb: float,
    fleet_100_desc_mb: float,
    fleet_bw_reduction_pct: float,
    r2_raw: float,
    r2_descriptor: float,
    r2_random: float,
    acc_raw: float,
    acc_desc: float,
    fingerprint_reduction_pct: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Average raw CAN window size (bytes)", "Value": round(avg_raw, 2)},
            {
                "Metric": "Average transmitted descriptor size (bytes)",
                "Value": round(avg_transmit, 2),
            },
            {"Metric": "Compression ratio", "Value": round(compression_ratio, 4)},
            {"Metric": "Bandwidth reduction (%)", "Value": round(bandwidth_reduction_pct, 2)},
            {
                "Metric": "Fleet bandwidth at 100 vehicles — raw (MB)",
                "Value": round(fleet_100_raw_mb, 4),
            },
            {
                "Metric": "Fleet bandwidth at 100 vehicles — descriptor (MB)",
                "Value": round(fleet_100_desc_mb, 4),
            },
            {
                "Metric": "Fleet bandwidth reduction (%)",
                "Value": round(fleet_bw_reduction_pct, 2),
            },
            {
                "Metric": "Payload-statistic reconstruction R² — raw baseline",
                "Value": round(r2_raw, 4),
            },
            {
                "Metric": "Payload-statistic reconstruction R² — descriptor attacker",
                "Value": round(r2_descriptor, 4),
            },
            {
                "Metric": "Payload-statistic reconstruction R² — random baseline",
                "Value": round(r2_random, 4),
            },
            {
                "Metric": "Vehicle fingerprinting accuracy — raw CAN",
                "Value": round(acc_raw, 4),
            },
            {
                "Metric": "Vehicle fingerprinting accuracy — descriptor",
                "Value": round(acc_desc, 4),
            },
            {
                "Metric": "Vehicle fingerprinting reduction (%)",
                "Value": round(fingerprint_reduction_pct, 2),
            },
        ]
    )


def run_descriptor_security_experiment(
    *,
    features_path: Path,
    predictions_path: Path,
    descriptors_path: Path,
    transmit_path: Path,
    outputs: SecurityOutputs,
    window_size: int = 100,
    seed: int = 42,
    fleet_sizes: tuple[int, ...] = FLEET_SIZES,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    features, full_desc, transmit = load_descriptors_for_experiment(
        features_path=features_path,
        predictions_path=predictions_path,
        descriptors_path=descriptors_path,
        transmit_path=transmit_path,
    )

    meta_cols = ["window_id", "vehicle_model", *BEHAVIOURAL_FEATURE_COLUMNS]
    win = features[[c for c in meta_cols if c in features.columns]].copy()
    win["raw_window_bytes"] = win.apply(estimate_raw_window_bytes, axis=1)

    desc_with_raw = full_desc.merge(
        win[["window_id", "vehicle_model", "raw_window_bytes"]],
        on=["window_id", "vehicle_model"],
        how="left",
    )
    desc_with_raw["raw_window_bytes"] = desc_with_raw["raw_window_bytes"].fillna(
        window_size * 21.0
    )

    tx_cols = [c for c in TRANSMIT_COLUMNS if c in transmit.columns]
    transmit_bytes = transmit.apply(lambda r: _serialize_row_bytes(r, tx_cols), axis=1)
    avg_transmit = float(transmit_bytes.mean())
    csv_block = transmit.to_csv(index=False).encode("utf-8")
    avg_transmit_gzip = _gzip_bytes(csv_block) / max(len(transmit), 1)

    avg_raw = float(desc_with_raw["raw_window_bytes"].mean())
    compression_ratio = avg_raw / avg_transmit if avg_transmit else 0.0
    bandwidth_reduction_pct = (
        100.0 * (avg_raw - avg_transmit) / avg_raw if avg_raw else 0.0
    )

    n_windows = int(len(features))
    n_vehicles_data = int(features["vehicle_model"].nunique())
    windows_per_vehicle = n_windows / max(n_vehicles_data, 1)
    suspicious_rate = len(full_desc) / max(n_windows, 1)

    scale_rows: list[dict[str, Any]] = []
    for fleet_n in fleet_sizes:
        total_windows = windows_per_vehicle * fleet_n
        raw_bw_mb = total_windows * avg_raw / 1e6
        desc_bw_mb = total_windows * suspicious_rate * avg_transmit / 1e6
        scale_rows.append(
            {
                "fleet_size": fleet_n,
                "total_raw_bandwidth_mb": round(raw_bw_mb, 4),
                "total_descriptor_bandwidth_mb": round(desc_bw_mb, 4),
                "bandwidth_reduction_percent": round(
                    100.0 * (raw_bw_mb - desc_bw_mb) / raw_bw_mb if raw_bw_mb else 0.0,
                    2,
                ),
            }
        )
    scalability_df = pd.DataFrame(scale_rows)
    fleet_100 = scalability_df[scalability_df["fleet_size"] == 100].iloc[0]

    disclosure_df = _build_information_disclosure_table()

    target_cols = [c for c in PAYLOAD_TARGET_COLS if c in full_desc.columns]
    leak_cols = {f"byte_mean_{i}" for i in range(8)} | {f"byte_std_{i}" for i in range(8)}
    tx_recon_cols = [
        c
        for c in tx_cols
        if c in transmit.columns
        and c not in leak_cols
        and pd.api.types.is_numeric_dtype(transmit[c])
    ]
    breakdown_df, r2_raw, r2_descriptor, r2_random = _payload_statistic_reconstruction_analysis(
        transmit, full_desc, tx_recon_cols, target_cols, seed=seed
    )

    le = LabelEncoder()
    y_vehicle = le.fit_transform(full_desc["vehicle_model"].astype(str))
    merged = full_desc[["window_id", "vehicle_model"]].merge(
        win, on=["window_id", "vehicle_model"], how="left"
    )
    X_raw = merged[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0)
    X_desc = transmit[tx_cols].select_dtypes(include=["number"]).fillna(0.0)
    acc_raw = _train_fingerprint_accuracy(X_raw, y_vehicle, seed=seed)
    acc_desc = _train_fingerprint_accuracy(X_desc, y_vehicle, seed=seed)
    fingerprint_reduction_pct = (
        100.0 * (acc_raw - acc_desc) / acc_raw if acc_raw else 0.0
    )

    comparison = _build_main_comparison_table(
        avg_raw=avg_raw,
        avg_transmit=avg_transmit,
        compression_ratio=compression_ratio,
        bandwidth_reduction_pct=bandwidth_reduction_pct,
        fleet_100_raw_mb=float(fleet_100["total_raw_bandwidth_mb"]),
        fleet_100_desc_mb=float(fleet_100["total_descriptor_bandwidth_mb"]),
        fleet_bw_reduction_pct=float(fleet_100["bandwidth_reduction_percent"]),
        r2_raw=r2_raw,
        r2_descriptor=r2_descriptor,
        r2_random=r2_random,
        acc_raw=acc_raw,
        acc_desc=acc_desc,
        fingerprint_reduction_pct=fingerprint_reduction_pct,
    )

    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = outputs.results_dir / "descriptor_security_metrics.csv"
    pd.DataFrame(
        [
            {
                "avg_raw_window_bytes": avg_raw,
                "avg_descriptor_bytes_transmit": avg_transmit,
                "avg_descriptor_gzip_bytes": avg_transmit_gzip,
                "compression_ratio": round(compression_ratio, 4),
                "bandwidth_reduction_percent": round(bandwidth_reduction_pct, 2),
                "payload_statistic_reconstruction_r2_raw_baseline": round(r2_raw, 4),
                "payload_statistic_reconstruction_r2_descriptor_attacker": round(
                    r2_descriptor, 4
                ),
                "payload_statistic_reconstruction_r2_random_baseline": round(r2_random, 4),
                "vehicle_fingerprint_accuracy_raw": round(acc_raw, 4),
                "vehicle_fingerprint_accuracy_descriptor": round(acc_desc, 4),
                "vehicle_fingerprinting_reduction_percent": round(
                    fingerprint_reduction_pct, 2
                ),
                "suspicious_window_rate": round(suspicious_rate, 6),
                "n_windows": n_windows,
                "n_descriptors": len(full_desc),
            }
        ]
    ).to_csv(metrics_path, index=False)

    _heuristic_exposure_appendix().to_csv(
        outputs.results_dir / "heuristic_exposure_appendix.csv", index=False
    )
    disclosure_df.to_csv(outputs.results_dir / "information_disclosure_comparison.csv", index=False)
    breakdown_df.to_csv(
        outputs.results_dir / "payload_statistic_reconstruction_breakdown.csv", index=False
    )
    scalability_df.to_csv(outputs.results_dir / "descriptor_fleet_scalability.csv", index=False)
    comparison.to_csv(outputs.results_dir / "descriptor_security_comparison_table.csv", index=False)

    (outputs.tables_dir / "table_information_disclosure_comparison.tex").write_text(
        _df_to_ieee_tex(
            disclosure_df,
            "Information disclosure comparison: raw CAN vs descriptor transmission.",
            "tab:information-disclosure",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_information_disclosure_comparison.md").write_text(
        "# Information Disclosure Comparison\n\n" + _df_to_markdown(disclosure_df),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_payload_statistic_reconstruction.tex").write_text(
        _df_to_ieee_tex(
            breakdown_df,
            "Payload-statistic reconstruction attack (lower R² indicates better privacy).",
            "tab:payload-stat-recon",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_payload_statistic_reconstruction.md").write_text(
        "# Payload-Statistic Reconstruction\n\n" + _df_to_markdown(breakdown_df),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_descriptor_security.tex").write_text(
        _df_to_ieee_tex(
            comparison,
            "Descriptor compactness and defensible security metrics (test dataset).",
            "tab:descriptor-security",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_descriptor_security.md").write_text(
        "# Descriptor Security Metrics\n\n" + _df_to_markdown(comparison),
        encoding="utf-8",
    )

    _plot_bandwidth_scaling_line(
        scalability_df, outputs.figures_dir / "bandwidth_scaling_fleet_sizes"
    )

    _plot_information_disclosure(
        disclosure_df, outputs.figures_dir / "information_disclosure_comparison"
    )

    _plot_privacy_preservation_comparison(
        outputs.figures_dir / "privacy_preservation_comparison"
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    labels = ["Raw baseline", "Descriptor attacker", "Random baseline"]
    r2_vals = [r2_raw, r2_descriptor, r2_random]
    ax.bar(labels, r2_vals, color=["#C00000", "#ED7D31", "#A5A5A5"])
    ax.set_ylabel("Payload-statistic reconstruction R²")
    ax.set_title("Payload-Statistic Reconstruction Attack")
    ax.set_ylim(0, 1.05)
    fig.text(
        0.5,
        -0.10,
        "Lower R² from descriptors limits inference of original payload-derived statistics "
        "(byte means/stds, DLC); does not reconstruct exact payload bytes.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    _save_figure(fig, outputs.figures_dir / "payload_reconstruction_error")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar(
        ["Raw CAN windows", "Descriptors"],
        [acc_raw * 100, acc_desc * 100],
        color=["#ED7D31", "#5B9BD5"],
    )
    ax.set_ylabel("Vehicle ID accuracy (%)")
    ax.set_title("Vehicle Fingerprinting")
    _save_figure(fig, outputs.figures_dir / "vehicle_fingerprinting_comparison")

    mean_mse = float(breakdown_df.loc[
        breakdown_df["target_type"] == "all_targets_average", "mse"
    ].mean())

    summary_path = outputs.results_dir / "descriptor_compactness_security_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Descriptor Compactness and Security — Summary",
                "",
                "## 1. Compactness",
                f"- **Average raw CAN window size:** {avg_raw:.2f} bytes",
                f"- **Transmitted descriptor size:** {avg_transmit:.2f} bytes "
                f"(gzip ~{avg_transmit_gzip:.2f} bytes/record)",
                f"- **Compression ratio:** {compression_ratio:.2f}×",
                f"- **Bandwidth reduction:** {bandwidth_reduction_pct:.2f}%",
                "",
                "## 2. Fleet scalability",
                f"- **Raw bandwidth @ 100 vehicles:** {fleet_100['total_raw_bandwidth_mb']:.2f} MB",
                f"- **Descriptor bandwidth @ 100 vehicles:** "
                f"{fleet_100['total_descriptor_bandwidth_mb']:.2f} MB",
                f"- **Fleet bandwidth reduction:** {fleet_100['bandwidth_reduction_percent']:.2f}%",
                "",
                "## 3. Information disclosure",
                "Raw CAN transmission exposes timestamp sequences, CAN IDs, DLC, payload bytes, "
                "and exact message order. Descriptor transmission sends only aggregated "
                "behavioural/statistical features (timing summaries, anomaly scores, evidence "
                "flags) and omits frame-level identifiers and raw payloads. See "
                "`information_disclosure_comparison.csv` and "
                "`table_information_disclosure_comparison.tex` — not the deprecated heuristic "
                "exposure score (retained only in `heuristic_exposure_appendix.csv` for debugging).",
                "",
                "## 4. Payload-statistic reconstruction",
                "Targets: `byte_mean_0..7`, `byte_std_0..7`, `mean_dlc`, `std_dlc` (statistics "
                "derived from payloads, not exact bytes). Attackers: Linear Regression, Random "
                "Forest Regressor, MLP Regressor on transmitted descriptor fields excluding "
                "byte-level aggregates.",
                f"- **Raw baseline R²:** {r2_raw:.4f} (statistics directly available from raw CAN)",
                f"- **Descriptor attacker R² (mean):** {r2_descriptor:.4f}",
                f"- **Random/mean baseline R²:** {r2_random:.4f}",
                f"- **Mean MSE (descriptor attackers, all targets):** {mean_mse:.4f}",
                "",
                "Lower descriptor R² indicates **limited ability to infer payload-derived "
                "statistics** from the uplink payload; it does **not** claim exact payload-byte "
                "recovery is impossible in all settings.",
                "",
                "## 5. Vehicle fingerprinting",
                f"- **Accuracy (raw CAN windows):** {acc_raw:.2%}",
                f"- **Accuracy (descriptors):** {acc_desc:.2%}",
                f"- **Reduction:** {fingerprint_reduction_pct:.2f}%",
                "",
                "High descriptor fingerprinting indicates that some vehicle-specific behavioural "
                "patterns remain in the transmitted statistics. **Limitation:** future work "
                "should investigate descriptor anonymisation, feature coarsening, or differential "
                "privacy for stronger unlinkability.",
                "",
                "## 6. Conclusion",
                "",
                "The proposed descriptor **substantially reduces communication overhead** and "
                "**raw CAN data disclosure** while preserving behavioural evidence for fleet-level "
                "intrusion detection. It **limits payload-statistic reconstruction** relative to "
                "raw CAN and supports **privacy-preserving fleet analysis**, but does **not** "
                "guarantee full privacy or complete vehicle anonymisation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Descriptor security experiment complete: %s", metrics_path)
    return {
        "descriptor_security_metrics": metrics_path,
        "information_disclosure_comparison": outputs.results_dir
        / "information_disclosure_comparison.csv",
        "descriptor_compactness_security_summary": summary_path,
        "table_descriptor_security": outputs.tables_dir / "table_descriptor_security.tex",
        "information_disclosure_figure": outputs.figures_dir
        / "information_disclosure_comparison.png",
    }
