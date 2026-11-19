"""Tests for error analysis and failure export."""

from reva_eval.error_analysis import (
    export_aligned_predictions_dataframe,
    generate_incorrect_predictions_dataframe,
)
from reva_eval.metrics import align_predictions_with_ground_truth
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


def test_error_analysis_dataframe_generation():
    """Verify that error analysis dataframe contains incorrect, missing, and invalid samples with required columns."""
    gt = make_synthetic_ground_truth(num_samples=4)
    # Make a prediction with 1 wrong, 1 missing, 1 invalid, 1 correct
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")
    # Sample 1: wrong
    preds.predictions[0].prediction = "B"  # GT is A
    # Sample 2: invalid
    preds.predictions[1].prediction = "Z"  # GT is B
    # Sample 3: missing (pop)
    preds.predictions.pop(2)               # GT is C
    # Sample 4: correct (GT is D, pred is D)

    df_aligned, _ = align_predictions_with_ground_truth(gt, preds)
    df_err = generate_incorrect_predictions_dataframe(df_aligned, score_semantics="probabilities")

    # There should be 3 error rows
    assert len(df_err) == 3

    # Check required columns
    expected_cols = [
        "sample_id", "video_path", "dataset_name", "subdir",
        "question", "options_A", "options_B", "options_C", "options_D",
        "category", "task", "ground_truth", "prediction",
        "scores", "score_semantics", "inference_time_seconds",
        "error_status", "manual_error_type", "reviewer_notes",
    ]
    for col in expected_cols:
        assert col in df_err.columns

    # Verify status labels
    statuses = set(df_err["error_status"].tolist())
    assert "incorrect" in statuses
    assert "invalid" in statuses
    assert "missing" in statuses


def test_aligned_dataframe_export():
    """Verify full aligned dataframe contains all samples."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="mixed")

    df_aligned, _ = align_predictions_with_ground_truth(gt, preds)
    df_full = export_aligned_predictions_dataframe(df_aligned, score_semantics="probabilities")

    assert len(df_full) == 4
    assert "status" in df_full.columns
    assert "is_correct" in df_full.columns

