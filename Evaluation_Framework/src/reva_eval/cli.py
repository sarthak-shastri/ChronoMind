"""Command Line Interface for ReVA Universal Evaluation Framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from reva_eval.dataset_loader import normalize_annotation, save_canonical_ground_truth
from reva_eval.evaluator import evaluate
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
    eval_p.add_argument("--predictions", "-p", required=True, type=str, help="Path to canonical predictions JSON")
    eval_p.add_argument("--output-dir", "-o", required=True, type=str, help="Directory to save evaluation reports and plots")
    eval_p.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate prediction IDs (diagnostic mode)")
    eval_p.add_argument("--strict-probabilities", action="store_true", help="Enforce probability sum == 1 strictly")
    eval_p.add_argument("--tolerance", type=float, default=1e-4, help="Probability sum error tolerance (default: 1e-4)")
    eval_p.add_argument("--no-plots", action="store_true", help="Disable PNG plot generation")
    eval_p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing output directory")

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

            report = evaluate(
                ground_truth=parsed_args.ground_truth,
                predictions=parsed_args.predictions,
                output_dir=parsed_args.output_dir,
                allow_duplicates=parsed_args.allow_duplicates,
                strict_probabilities=parsed_args.strict_probabilities,
                probability_sum_tolerance=parsed_args.tolerance,
                save_plots=not parsed_args.no_plots,
                overwrite=not parsed_args.no_overwrite,
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
                    output_dir=None,  # Don't export single model artifact directories
                    save_plots=False,
                )
                model_reports[name] = rep

            csv_out, md_out = generate_model_comparison_report(
                reports=model_reports,
                output_dir=parsed_args.output_dir,
            )
            print(f"[reva-eval] Comparison reports written to:")
            print(f"  CSV: {csv_out}")
            print(f"  Markdown: {md_out}")
            return 0

    except Exception as exc:
        print(f"[reva-eval] Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
