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
    export_aligned_predictions_dataframe,
    generate_incorrect_predictions_dataframe,
)
from reva_eval.metrics import (
    align_predictions_with_ground_truth,
    compute_confusion_matrix,
    compute_primary_metrics,
    compute_runtime_stats,
    compute_score_metadata,
)
from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    EvaluationReport,
    ValidationSummary,
)
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
    plot_task_accuracy,
)


def evaluate(
    ground_truth: Union[str, Path, CanonicalGroundTruth],
    predictions: Union[str, Path, CanonicalPredictions],
    output_dir: Optional[Union[str, Path]] = None,
    allow_duplicates: bool = False,
    strict_probabilities: bool = False,
    probability_sum_tolerance: float = 1e-4,
    save_plots: bool = True,
    overwrite: bool = True,
) -> EvaluationReport:
    """Run complete evaluation pipeline and export all specified artifacts."""
    # 1. Resolve Ground Truth
    if isinstance(ground_truth, CanonicalGroundTruth):
        gt = ground_truth
    else:
        gt_path = Path(ground_truth)
        # Attempt to load canonical; if format is raw annotation, normalize it
        _, _, detected_fmt = load_raw_annotation_file(gt_path)
        if detected_fmt == "canonical":
            gt = load_canonical_ground_truth(gt_path)
        else:
            gt = normalize_annotation(gt_path)

    # 2. Resolve Predictions
    if isinstance(predictions, CanonicalPredictions):
        preds = predictions
    else:
        p_path = Path(predictions)
        if not p_path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {p_path}")
        with open(p_path, "r", encoding="utf-8") as f:
            raw_p = json.load(f)
        preds = CanonicalPredictions.model_validate(raw_p)

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

        # 7.7 confusion_matrix.csv
        if raw_cm is not None:
            # Combine raw counts and normalized percentages into clean CSV
            cm_df = raw_cm.copy()
            cm_df.to_csv(out_p / "confusion_matrix.csv")
        else:
            pd.DataFrame().to_csv(out_p / "confusion_matrix.csv")

        # 7.8 evaluation_report.md
        generate_markdown_report(report, out_p / "evaluation_report.md")

        # 7.9 Visualizations (PNG)
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

    return report

