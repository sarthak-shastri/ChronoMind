"""Core evaluation pipeline for ReVA VideoQA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from reva_eval.category_metrics import (
    category_metrics_to_dataframe,
    compute_group_metrics,
    task_metrics_to_dataframe,
)
from reva_eval.dataset_loader import (
    load_canonical_ground_truth,
    load_raw_annotation_file,
    normalize_annotation,
)
from reva_eval.error_analysis import (
    analyze_error_patterns,
    export_aligned_predictions_dataframe,
    generate_incorrect_predictions_dataframe,
    save_error_analysis_summary,
)
from reva_eval.experiment_tracker import ExperimentTracker
from reva_eval.metrics import (
    align_predictions_with_ground_truth,
    compute_confusion_matrix,
    compute_primary_metrics,
    compute_runtime_stats,
    compute_score_metadata,
    compute_supplementary_metrics,
)
from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    EvaluationReport,
    ValidationSummary,
)
from reva_eval.prediction_loader import load_predictions
from reva_eval.reporting import (
    create_evaluation_report,
    generate_markdown_report,
    save_evaluation_json,
)
from reva_eval.validation import (
    validate_ground_truth,
    validate_predictions_against_ground_truth,
    ValidationResult,
)
from reva_eval.visualization import (
    plot_category_accuracy,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_task_accuracy,
)


def evaluate(
    ground_truth: Union[str, Path, CanonicalGroundTruth],
    predictions: Union[str, Path, CanonicalPredictions],
    output_dir: Optional[Union[str, Path]] = None,
    allow_duplicates: bool = False,
    strict_probabilities: bool = False,
    probability_sum_tolerance: float = 1e-4,
    strict_mode: bool = False,
    file_format: Optional[str] = None,
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    case_fold: bool = True,
    save_plots: bool = True,
    overwrite: bool = True,
    experiment_id: Optional[str] = None,
    track_experiment: bool = False,
    experiment_tracker_dir: Optional[Union[str, Path]] = None,
) -> EvaluationReport:
    """Run complete evaluation pipeline and export all specified artifacts.

    Args:
        ground_truth: CanonicalGroundTruth instance or path to ground-truth / annotation JSON.
        predictions: CanonicalPredictions instance or path to predictions (JSON, JSONL, CSV).
        output_dir: Directory where reports, CSVs, and plots will be written.
        allow_duplicates: If True, evaluates first duplicate and logs warnings rather than raising error.
        strict_probabilities: If True, enforce probability sum == 1 strictly.
        probability_sum_tolerance: Margin of error for probability sum checks.
        strict_mode: If True, any missing IDs, unexpected IDs, or invalid labels fail clearly.
        file_format: Format of prediction file ('json', 'jsonl', 'csv', 'auto').
        strip_whitespace: Normalize whitespace on prediction answer labels.
        normalize_unicode: Apply Unicode NFKC normalization to prediction labels.
        case_fold: Convert prediction labels to uppercase (e.g. 'b' -> 'B').
        save_plots: Generate headless PNG visualization plots.
        overwrite: Allow overwriting existing output directory.
        experiment_id: Custom unique identifier for the experiment run.
        track_experiment: If True, records the evaluation run in the local experiment tracker registry.
        experiment_tracker_dir: Directory for the experiment tracker registry (defaults to 'experiments').

    Returns:
        EvaluationReport object containing full metrics and validation diagnostics.
    """
    # 1. Resolve Ground Truth
    if isinstance(ground_truth, CanonicalGroundTruth):
        gt = ground_truth
    else:
        gt_path = Path(ground_truth)
        _, _, detected_fmt = load_raw_annotation_file(gt_path)
        if detected_fmt == "canonical":
            gt = load_canonical_ground_truth(gt_path)
        else:
            gt = normalize_annotation(gt_path)

    # 2. Resolve Predictions
    if isinstance(predictions, CanonicalPredictions):
        preds = predictions
    else:
        preds = load_predictions(
            file_path=predictions,
            file_format=file_format,
            strip_whitespace=strip_whitespace,
            normalize_unicode=normalize_unicode,
            case_fold=case_fold,
        )

    # 3. Validate
    gt_val_res = validate_ground_truth(gt)
    if not gt_val_res.is_valid:
        raise ValueError(f"Ground truth integrity validation failed: {gt_val_res.errors}")

    val_res = validate_predictions_against_ground_truth(
        predictions=preds,
        ground_truth=gt,
        allow_duplicates=allow_duplicates,
        strict_probabilities=strict_probabilities,
        probability_sum_tolerance=probability_sum_tolerance,
        strict_mode=strict_mode,
    )

    if not val_res.is_valid:
        raise ValueError(
            f"Prediction validation failed with {len(val_res.errors)} error(s):\n"
            + "\n".join(f"- {e}" for e in val_res.errors[:10])
        )

    # 4. Alignment
    df_aligned, align_meta = align_predictions_with_ground_truth(gt, preds)

    # 5. Metrics calculation
    is_labeled = gt.labeled
    primary_metrics = compute_primary_metrics(df_aligned, val_res, is_labeled=is_labeled)
    category_metrics = compute_group_metrics(df_aligned, group_col="category", is_labeled=is_labeled)
    task_metrics = compute_group_metrics(df_aligned, group_col="task", is_labeled=is_labeled)

    raw_cm, norm_cm, cm_counts = compute_confusion_matrix(df_aligned, is_labeled=is_labeled)
    runtime_stats = compute_runtime_stats(df_aligned)
    score_meta = compute_score_metadata(df_aligned, preds.score_semantics)

    supplementary_metrics = compute_supplementary_metrics(
        df_aligned=df_aligned,
        category_metrics=category_metrics,
        task_metrics=task_metrics,
        is_labeled=is_labeled,
        overall_accuracy=primary_metrics.accuracy,
    )

    error_summary = analyze_error_patterns(df_aligned, is_labeled=is_labeled)

    val_summary = ValidationSummary(
        errors=len(val_res.errors),
        warnings=len(val_res.warnings),
    )

    # 6. Build Evaluation Report
    report = create_evaluation_report(
        dataset=gt.dataset,
        split=gt.split,
        model_metadata=preds.model,
        primary_metrics=primary_metrics,
        category_metrics=category_metrics,
        task_metrics=task_metrics,
        runtime_stats=runtime_stats,
        score_metadata=score_meta,
        validation_summary=val_summary,
        is_labeled=is_labeled,
        experiment_id=experiment_id,
        validation_mode="strict" if strict_mode else "non-strict",
        supplementary_metrics=supplementary_metrics,
        alignment_statistics=align_meta,
        notes=val_res.warnings,
    )

    # 7. Export artifacts if output_dir is given
    if output_dir is not None:
        out_p = Path(output_dir)
        if out_p.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory '{out_p}' already exists and overwrite is False."
            )
        out_p.mkdir(parents=True, exist_ok=True)

        # 7.1 evaluation.json
        save_evaluation_json(report, out_p / "evaluation.json")

        # 7.2 validation_issues.json
        with open(out_p / "validation_issues.json", "w", encoding="utf-8") as f:
            json.dump(val_res.to_dict(), f, indent=2)

        # 7.3 category_metrics.csv
        df_cat = category_metrics_to_dataframe(category_metrics)
        df_cat.to_csv(out_p / "category_metrics.csv", index=False)

        # 7.4 task_metrics.csv
        df_task = task_metrics_to_dataframe(task_metrics)
        df_task.to_csv(out_p / "task_metrics.csv", index=False)

        # 7.5 predictions_aligned.csv
        df_export_aligned = export_aligned_predictions_dataframe(df_aligned, preds.score_semantics)
        df_export_aligned.to_csv(out_p / "predictions_aligned.csv", index=False)

        # 7.6 incorrect_predictions.csv
        df_incorrect = generate_incorrect_predictions_dataframe(df_aligned, preds.score_semantics)
        df_incorrect.to_csv(out_p / "incorrect_predictions.csv", index=False)

        # 7.7 error_patterns.json
        save_error_analysis_summary(error_summary, out_p / "error_patterns.json")

        # 7.8 confusion_matrix.csv
        if raw_cm is not None:
            cm_df = raw_cm.copy()
            cm_df.to_csv(out_p / "confusion_matrix.csv")
        else:
            pd.DataFrame().to_csv(out_p / "confusion_matrix.csv")

        # 7.9 evaluation_report.md
        generate_markdown_report(report, out_p / "evaluation_report.md")

        # 7.10 Visualizations (PNG)
        if save_plots:
            plot_confusion_matrix(
                raw_matrix=raw_cm,
                norm_matrix=norm_cm,
                output_path=out_p / "confusion_matrix.png",
                model_name=preds.model.name,
            )
            plot_category_accuracy(
                category_metrics=category_metrics,
                output_path=out_p / "category_accuracy.png",
                model_name=preds.model.name,
            )
            plot_task_accuracy(
                task_metrics=task_metrics,
                output_path=out_p / "task_accuracy.png",
                model_name=preds.model.name,
            )
            plot_error_distribution(
                error_summary=error_summary,
                output_path=out_p / "error_distribution.png",
                model_name=preds.model.name,
            )

    # 8. Experiment Tracking
    if track_experiment:
        tracker = ExperimentTracker(registry_dir=experiment_tracker_dir or "experiments")
        tracker.log_run(
            report=report,
            output_report_dir=output_dir,
            experiment_id=experiment_id,
        )

    return report
