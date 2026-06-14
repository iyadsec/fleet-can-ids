"""Publication tables for balanced E2E run (custom P2 split summary)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.final_end_to_end_publication_run.tables import generate_tables as _generate_base_tables
from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration

P2_CAPTION = (
    "Distribution of source traces or non-overlapping trace segments and "
    "observation windows across the final dataset partitions."
)
P2_NOTE = (
    "Chevrolet coverage was obtained using disjoint contiguous trace segments "
    "with guard gaps to prevent overlapping windows."
)


def build_p2_table(platform_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in platform_summary.sort_values(["vehicle_model", "split"]).iterrows():
        rows.append(
            {
                "vehicle_model": r["vehicle_model"],
                "partition": r["split"],
                "source_traces": int(r["source_trace_count"]),
                "source_segments": int(r["source_segment_count"]),
                "benign_windows": int(r["benign_window_count"]),
                "malicious_windows": int(r["malicious_window_count"]),
                "total_windows": int(r["total_window_count"]),
            }
        )
    return pd.DataFrame(rows)


def write_p2_table(out_dir: Path, platform_summary: pd.DataFrame) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p2 = build_p2_table(platform_summary)
    stem = out_dir / "table_P2_dataset_and_split_summary"
    p2.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(
        f"# P2 Dataset and split\n\n{p2.to_markdown(index=False)}\n",
        encoding="utf-8",
    )
    chev_seg = int(platform_summary.loc[platform_summary.vehicle_model == "Chevrolet", "source_segment_count"].sum())
    note_block = f"\\\\footnote{{{P2_NOTE}}}" if chev_seg > 0 else ""
    tex_body = p2.to_latex(index=False, float_format="%.0f")
    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{P2_CAPTION}{note_block}}}\n"
        f"{tex_body}\n"
        "\\end{table}\n"
    )
    stem.with_suffix(".tex").write_text(tex, encoding="utf-8")
    return stem.with_suffix(".tex")


def generate_tables(
    out_dir: Path,
    *,
    df: pd.DataFrame,
    shared: SharedFleetConfiguration,
    platform_summary: pd.DataFrame,
    vehicle_metrics: pd.DataFrame,
    descriptor_metrics: pd.DataFrame,
    master_hash: str,
) -> list[str]:
    out_dir = Path(out_dir)
    trace_stub = platform_summary.assign(
        source_trace=platform_summary["source_trace_count"],
        window_count=platform_summary["total_window_count"],
    )
    generated = _generate_base_tables(
        out_dir,
        df=df,
        shared=shared,
        trace_df=trace_stub,
        vehicle_metrics=vehicle_metrics,
        descriptor_metrics=descriptor_metrics,
        master_hash=master_hash,
    )
    write_p2_table(out_dir, platform_summary)
    if "P2" not in generated:
        generated.insert(1, "P2")
    return generated
