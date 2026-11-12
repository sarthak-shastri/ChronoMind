"""Metrics calculation, sample alignment, and confusion matrix generation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    PredictionRecord,
    PrimaryMetrics,
    RuntimeStats,
    ScoreMetadata,
    VALID_LABELS,
)
from reva_eval.validation import ValidationResult


def align_predictions_with_ground_truth(
    ground_truth: CanonicalGroundTruth,
    predictions: CanonicalPredictions,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align predictions with ground truth using sample_id (left join from ground truth).

    Returns:
        (aligned_dataframe, metadata_dict)
    """
    # Extract unique predictions (first occurrence wins if allow_duplicates was used)
    pred_map: Dict[str, PredictionRecord] = {}
    for p in predictions.predictions:
        if p.sample_id not in pred_map:
            pred_map[p.sample_id] = p

    gt_ids = {s.sample_id for s in ground_truth.samples}
    unexpected_pred_ids = [p_id for p_id in pred_map if p_id not in gt_ids]

    rows = []
    for s in ground_truth.samples:
        pred_rec = pred_map.get(s.sample_id)
        pred_label = pred_rec.prediction if pred_rec is not None else None
        scores = pred_rec.scores if pred_rec is not None else None
        latency = pred_rec.inference_time_seconds if pred_rec is not None else None

        is_provided = pred_rec is not None
        is_valid_pred = pred_label in VALID_LABELS if is_provided else False

        # Status definition: correct, wrong, missing, invalid
        if not is_provided:
            status = "missing"
            is_correct = False
        elif not is_valid_pred:
            status = "invalid"
            is_correct = False
        else:
            if ground_truth.labeled and s.ground_truth is not None:
                is_correct = (pred_label == s.ground_truth)
                status = "correct" if is_correct else "wrong"
            else:
                is_correct = None
                status = "unlabeled"

        row = {
            "sample_id": s.sample_id,
            "video_path": s.video_path,
            "dataset_name": s.dataset_name,
            "subdir": s.subdir,
            "category": s.category or "Unknown",
            "task": s.task or "Unknown",
            "question": s.question,
            "options": s.options,
            "ground_truth": s.ground_truth,
            "prediction": pred_label,
            "scores": scores,
            "inference_time_seconds": latency,
            "is_provided": is_provided,
            "is_valid_prediction": is_valid_pred,
            "is_correct": is_correct,
            "status": status,
        }
        rows.append(row)

    if not rows:
        columns = [
            "sample_id", "video_path", "dataset_name", "subdir",
            "category", "task", "question", "options", "ground_truth",
            "prediction", "scores", "inference_time_seconds",
            "is_provided", "is_valid_prediction", "is_correct", "status",
        ]
        df_aligned = pd.DataFrame(columns=columns)
    else:
        df_aligned = pd.DataFrame(rows)

    meta = {
        "unexpected_prediction_ids": unexpected_pred_ids,
        "unexpected_prediction_count": len(unexpected_pred_ids),
    }
    return df_aligned, meta


def compute_primary_metrics(
    df_aligned: pd.DataFrame,
    validation_res: ValidationResult,
    is_labeled: bool,
) -> PrimaryMetrics:
    """Compute overall primary evaluation metrics."""
    total_samples = len(df_aligned)

    if total_samples == 0:
        return PrimaryMetrics(
            total_samples=0,
            provided_predictions=0,
            valid_predictions=0,
            correct=None if is_labeled else None,
            incorrect=None if is_labeled else None,
            accuracy=None,
            prediction_coverage=0.0,
            valid_prediction_coverage=0.0,
            missing_predictions=0,
            invalid_predictions=0,
            unexpected_predictions=len(validation_res.unexpected_sample_ids),
            duplicate_prediction_occurrences=len(validation_res.duplicate_sample_ids),
        )

    provided_count = int(df_aligned["is_provided"].sum())
    valid_count = int(df_aligned["is_valid_prediction"].sum())
    missing_count = int((~df_aligned["is_provided"]).sum())
    invalid_count = int((df_aligned["is_provided"] & ~df_aligned["is_valid_prediction"]).sum())

    prediction_coverage = round(provided_count / total_samples, 6)
    valid_prediction_coverage = round(valid_count / total_samples, 6)

    if is_labeled:
        correct_count = int((df_aligned["is_correct"] == True).sum())
        incorrect_count = total_samples - correct_count
        accuracy = round(correct_count / total_samples, 6)
    else:
        correct_count = None
        incorrect_count = None
        accuracy = None

    return PrimaryMetrics(
        total_samples=total_samples,
        provided_predictions=provided_count,
        valid_predictions=valid_count,
        correct=correct_count,
        incorrect=incorrect_count,
        accuracy=accuracy,
        prediction_coverage=prediction_coverage,
        valid_prediction_coverage=valid_prediction_coverage,
        missing_predictions=missing_count,
        invalid_predictions=invalid_count,
        unexpected_predictions=len(validation_res.unexpected_sample_ids),
        duplicate_prediction_occurrences=len(validation_res.duplicate_sample_ids),
    )


def compute_confusion_matrix(
    df_aligned: pd.DataFrame,
    is_labeled: bool,
    labels: Tuple[str, ...] = VALID_LABELS,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, int]]:
    """Compute 4x4 confusion matrix (raw and normalized) for valid predictions only.

    Returns:
        (raw_matrix_df, normalized_matrix_df, extra_counts)
    """
    extra_counts = {
        "missing_predictions": int((~df_aligned["is_provided"]).sum()),
        "invalid_predictions": int((df_aligned["is_provided"] & ~df_aligned["is_valid_prediction"]).sum()),
    }

    if not is_labeled or len(df_aligned) == 0:
        return None, None, extra_counts

    # Filter to samples with valid prediction and valid ground truth
    valid_df = df_aligned[
        df_aligned["is_valid_prediction"] & df_aligned["ground_truth"].isin(labels)
    ]

    # Initialize 4x4 count matrix
    matrix = pd.DataFrame(0, index=list(labels), columns=list(labels), dtype=int)
    for true_lbl in labels:
        subset = valid_df[valid_df["ground_truth"] == true_lbl]
        pred_counts = subset["prediction"].value_counts()
        for pred_lbl in labels:
            matrix.loc[true_lbl, pred_lbl] = int(pred_counts.get(pred_lbl, 0))

    # Normalized matrix: row sums (per true class)
    row_sums = matrix.sum(axis=1)
    norm_matrix = matrix.astype(float).copy()
    for true_lbl in labels:
        total = row_sums[true_lbl]
        if total > 0:
            norm_matrix.loc[true_lbl] = np.round(norm_matrix.loc[true_lbl] / total, 4)
        else:
            norm_matrix.loc[true_lbl] = 0.0

    return matrix, norm_matrix, extra_counts


def compute_runtime_stats(df_aligned: pd.DataFrame) -> RuntimeStats:
    """Calculate latency and timing statistics."""
    latencies = df_aligned["inference_time_seconds"].dropna().tolist()
    valid_latencies = [float(x) for x in latencies if not math.isnan(x) and not math.isinf(x) and x >= 0]

    if not valid_latencies:
        return RuntimeStats(
            timed_samples=0,
            mean_seconds=None,
            median_seconds=None,
            p95_seconds=None,
        )

    arr = np.array(valid_latencies)
    return RuntimeStats(
        timed_samples=len(valid_latencies),
        mean_seconds=round(float(np.mean(arr)), 4),
        median_seconds=round(float(np.median(arr)), 4),
        p95_seconds=round(float(np.percentile(arr, 95)), 4),
    )


def compute_score_metadata(df_aligned: pd.DataFrame, score_semantics: Optional[str]) -> ScoreMetadata:
    """Compute summary metadata for model confidence / scores."""
    samples_with_scores = int(df_aligned["scores"].apply(lambda s: s is not None and isinstance(s, dict)).sum())
    return ScoreMetadata(
        score_semantics=score_semantics,
        samples_with_scores=samples_with_scores,
    )
