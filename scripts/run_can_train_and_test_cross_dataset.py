#!/usr/bin/env python3
"""Run can-train-and-test cross-dataset validation (staged: audit | pilot | full)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import DEFAULT_CTT_DATASET_ROOT, OUTPUT_ROOT
from src.ctt.progress_logger import ProgressLogger
from src.ctt.run_config import RunConfig
from src.ctt.stages import (
    _check_prior_stages,
    run_stage_audit,
    run_stage_full,
    run_stage_pilot,
)
from src.ctt.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CTT cross-dataset validation (default: audit stage only)",
    )
    parser.add_argument(
        "--stage",
        choices=["audit", "pilot", "full"],
        default="audit",
        help="Execution stage (default: audit)",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-files", type=int, default=None, help="Cap number of files processed")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Cap rows read per file")
    parser.add_argument("--max-windows", type=int, default=None, help="Cap total windows generated")
    parser.add_argument("--max-graph-nodes", type=int, default=None, help="Cap fleet graph nodes")
    parser.add_argument("--max-descriptors", type=int, default=None, help="Cap fleet candidate descriptors")
    parser.add_argument("--resume", action="store_true", help="Resume from existing artifacts")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing shards")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root or Path(os.environ.get("CTT_DATASET_ROOT", DEFAULT_CTT_DATASET_ROOT))
    output_root = ensure_dir(args.output_root)

    cfg = RunConfig.for_stage(
        args.stage,
        dataset_root=dataset_root,
        output_root=output_root,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
        max_windows=args.max_windows,
        max_graph_nodes=args.max_graph_nodes,
        max_descriptors=args.max_descriptors,
        resume=args.resume,
        skip_existing=args.skip_existing,
    )

    progress = ProgressLogger(cfg.log_path(), cfg.stage)
    progress.stage_start()

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        if cfg.stage == "audit":
            audit_result = run_stage_audit(cfg, progress)
            progress.info(
                f"Audit complete: {audit_result['total_files']} files, "
                f"{audit_result['total_rows']:,} rows, "
                f"vehicles={audit_result['vehicles_found']}"
            )

        elif cfg.stage == "pilot":
            run_stage_pilot(cfg, progress)

        elif cfg.stage == "full":
            ok, msg = _check_prior_stages(cfg)
            if not ok:
                progress.info(f"ABORT: {msg}")
                progress.stage_end("BLOCKED")
                return 1
            from scripts.validate_can_train_and_test_cross_dataset import validate

            for prior_stage in ("audit", "pilot"):
                vr = validate(output_root, prior_stage)
                if vr.critical_failures:
                    progress.info(
                        f"ABORT: Stage 3 blocked — {prior_stage} validation failed: "
                        f"{vr.critical_failures}"
                    )
                    progress.stage_end("BLOCKED")
                    return 1
            run_stage_full(cfg, progress)

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)

        progress.info(f"Runtime: {elapsed:.1f}s | Peak memory: {peak_mb:.1f} MB")
        progress.stage_end("OK")
        print(f"\nStage '{cfg.stage}' complete in {elapsed:.1f}s")
        print(f"Output: {output_root.resolve()}")
        print(f"Log: {cfg.log_path()}")
        return 0

    except Exception as exc:
        progress.info(f"FAILED: {exc}")
        progress.stage_end("FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
