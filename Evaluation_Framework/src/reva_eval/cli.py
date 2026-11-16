"""Command Line Interface for ReVA Universal Evaluation Framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from reva_eval.dataset_loader import normalize_annotation, save_canonical_ground_truth
from reva_eval.evaluator import evaluate
from reva_eval.experiment_tracker import ExperimentTracker
from reva_eval.reporting import generate_model_comparison_report


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="reva-eval",
        description="ReVA Universal Evaluation Framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: prepare-ground-truth
    prep_p = subparsers.add_parser(
        "prepare-ground-truth",
        help="Normalize official ReVA annotation JSON into canonical ground truth format",
    )
    prep_p.add_argument("--annotation", "-a", required=True, type=str, help="Path to raw annotation JSON file")
    prep_p.add_argument("--output", "-o", required=True, type=str, help="Path to write canonical ground truth JSON")
    prep_p.add_argument("--split", "-s", default="validation", type=str, help="Split name (default: validation)")
    prep_p.add_argument("--dataset", "-d", default="ReVA", type=str, help="Dataset identifier (default: ReVA)")

    # Command: evaluate
    eval_p = subparsers.add_parser(
        "evaluate",
        help="Evaluate a model's canonical predictions against canonical ground truth",
    )
    eval_p.add_argument("--ground-truth", "-g", required=True, type=str, help="Path to canonical ground truth JSON")
    eval_p.add_argument("--predictions", "-p", required=True, type=str, help="Path to predictions file (JSON, JSONL, CSV)")
    eval_p.add_argument("--output-dir", "-o", required=True, type=str, help="Directory to save evaluation reports and plots")
    eval_p.add_argument("--format", "-f", default="auto", choices=["auto", "json", "jsonl", "csv"], help="Prediction format (default: auto)")
    eval_p.add_argument("--strict", action="store_true", help="Enable strict validation mode (fails clearly on missing or invalid records)")
    eval_p.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate prediction IDs (diagnostic mode)")
    eval_p.add_argument("--strict-probabilities", action="store_true", help="Enforce probability sum == 1 strictly")
    eval_p.add_argument("--tolerance", type=float, default=1e-4, help="Probability sum error tolerance (default: 1e-4)")
    eval_p.add_argument("--no-plots", action="store_true", help="Disable PNG plot generation")
    eval_p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing output directory")
    eval_p.add_argument("--track", action="store_true", help="Record run in local experiment tracking registry")
    eval_p.add_argument("--experiment-id", type=str, default=None, help="Optional custom identifier for the experiment run")

    # Command: compare
    comp_p = subparsers.add_parser(
        "compare",
        help="Compare multiple model prediction files against the same ground truth",
    )
    comp_p.add_argument("--ground-truth", "-g", required=True, type=str, help="Path to canonical ground truth JSON")
    comp_p.add_argument(
        "--predictions", "-p",
        nargs="+",
        required=True,
        help="List of model predictions in form ModelName=PATH or simply PATH",
    )
    comp_p.add_argument("--output-dir", "-o", required=True, type=str, help="Directory to save comparison reports")
    comp_p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing output directory")
    comp_p.add_argument("--no-plots", action="store_true", help="Disable PNG plot generation")

    # Command: track
    track_p = subparsers.add_parser(
        "track",
        help="Manage and view tracked evaluation experiments in the local registry",
    )
    track_p.add_argument("--registry-dir", default="experiments", help="Directory for experiment registry (default: experiments)")
    track_p.add_argument("--list", action="store_true", help="List all tracked experiment runs")
    track_p.add_argument("--split", type=str, default=None, help="Filter by split")
    track_p.add_argument("--export", type=str, default=None, help="Export comparison table to CSV or Markdown path")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == "prepare-ground-truth":
            print(f"[reva-eval] Normalizing annotation: {parsed_args.annotation}")
            gt = normalize_annotation(
                file_path=parsed_args.annotation,
                split=parsed_args.split,
                dataset=parsed_args.dataset,
            )
            out_path = save_canonical_ground_truth(gt, parsed_args.output)
            print(f"[reva-eval] Successfully saved canonical ground truth with {len(gt.samples)} samples to: {out_path}")
            return 0

        elif parsed_args.command == "evaluate":
            print(f"[reva-eval] Running evaluation...")
            print(f"  Ground truth: {parsed_args.ground_truth}")
            print(f"  Predictions:  {parsed_args.predictions}")
            print(f"  Output dir:   {parsed_args.output_dir}")
            print(f"  Mode:         {'Strict' if parsed_args.strict else 'Non-strict (permissive)'}")

            report = evaluate(
                ground_truth=parsed_args.ground_truth,
                predictions=parsed_args.predictions,
                output_dir=parsed_args.output_dir,
                file_format=parsed_args.format,
                strict_mode=parsed_args.strict,
                allow_duplicates=parsed_args.allow_duplicates,
                strict_probabilities=parsed_args.strict_probabilities,
                probability_sum_tolerance=parsed_args.tolerance,
                save_plots=not parsed_args.no_plots,
                overwrite=not parsed_args.no_overwrite,
                experiment_id=parsed_args.experiment_id,
                track_experiment=parsed_args.track,
            )

            print(f"[reva-eval] Evaluation status: {report.status}")
            if report.ground_truth_labeled and report.metrics.accuracy is not None:
                print(f"[reva-eval] Overall Accuracy: {report.metrics.accuracy:.2%} ({report.metrics.correct}/{report.metrics.total_samples})")
            else:
                print(f"[reva-eval] Split is unlabeled: accuracy omitted.")
            print(f"[reva-eval] Valid Prediction Coverage: {report.metrics.valid_prediction_coverage:.2%}")
            print(f"[reva-eval] Reports successfully generated in: {parsed_args.output_dir}")
            return 0

        elif parsed_args.command == "compare":
            print(f"[reva-eval] Comparing model predictions...")
            model_reports = {}
            for item in parsed_args.predictions:
                if "=" in item:
                    name, path = item.split("=", 1)
                else:
                    path = item
                    name = Path(path).stem

                print(f"  Evaluating candidate '{name}' from: {path}")
                rep = evaluate(
                    ground_truth=parsed_args.ground_truth,
                    predictions=path,
                    output_dir=None,
                    save_plots=False,
                )
                model_reports[name] = rep

            csv_out, md_out = generate_model_comparison_report(
                reports=model_reports,
                output_dir=parsed_args.output_dir,
                save_plots=not parsed_args.no_plots,
            )
            print(f"[reva-eval] Comparison reports written to:")
            print(f"  CSV: {csv_out}")
            print(f"  Markdown: {md_out}")
            return 0

        elif parsed_args.command == "track":
            tracker = ExperimentTracker(registry_dir=parsed_args.registry_dir)
            if parsed_args.export:
                out_path = tracker.export_summary(parsed_args.export)
                print(f"[reva-eval] Exported experiment summary to: {out_path}")
                return 0

            runs = tracker.list_runs(split=parsed_args.split)
            print(f"[reva-eval] Tracked Experiments ({len(runs)} runs in '{parsed_args.registry_dir}'):")
            df = tracker.compare_runs(split=parsed_args.split)
            if df.empty:
                print("  No tracked runs found.")
            else:
                print(df.to_string(index=False))
            return 0

    except Exception as exc:
        print(f"[reva-eval] Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
