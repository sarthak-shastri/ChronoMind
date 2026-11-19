"""Tests for core metrics calculation, alignment, and confusion matrix."""

from reva_eval.evaluator import evaluate
from reva_eval.metrics import (
    align_predictions_with_ground_truth,
    compute_confusion_matrix,
    compute_primary_metrics,
    compute_runtime_stats,
)
from reva_eval.models import CanonicalGroundTruth, VALID_LABELS
from reva_eval.validation import validate_predictions_against_ground_truth
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


def test_all_correct_synthetic():
    """All correct predictions yield accuracy 1.0 and coverage 1.0."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4
    assert report.metrics.correct == 4
    assert report.metrics.incorrect == 0
    assert report.metrics.accuracy == 1.0
    assert report.metrics.prediction_coverage == 1.0
    assert report.metrics.valid_prediction_coverage == 1.0
    assert report.metrics.missing_predictions == 0
    assert report.metrics.invalid_predictions == 0


def test_all_wrong_synthetic():
    """All wrong predictions yield accuracy 0.0."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="all_wrong")

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4
    assert report.metrics.correct == 0
    assert report.metrics.incorrect == 4
    assert report.metrics.accuracy == 0.0
    assert report.metrics.prediction_coverage == 1.0
    assert report.metrics.valid_prediction_coverage == 1.0


def test_mixed_synthetic():
    """Mixed predictions (3 correct, 1 wrong) yield accuracy 0.75."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="mixed")

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4
    assert report.metrics.correct == 3
    assert report.metrics.incorrect == 1
    assert report.metrics.accuracy == 0.75


def test_missing_prediction_counted_as_incorrect_in_denominator():
    """Missing prediction must be included in denominator and count as incorrect."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="missing")  # 3 provided, all 3 correct

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4
    assert report.metrics.provided_predictions == 3
    assert report.metrics.correct == 3
    assert report.metrics.incorrect == 1  # 4 - 3 = 1 (missing counts as incorrect)
    assert report.metrics.accuracy == 0.75
    assert report.metrics.prediction_coverage == 0.75
    assert report.metrics.missing_predictions == 1


def test_invalid_prediction_counted_as_incorrect_in_denominator():
    """Invalid prediction label counts as incorrect in denominator."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="invalid_label")  # 1 invalid label 'Z', 3 correct

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4
    assert report.metrics.provided_predictions == 4
    assert report.metrics.valid_predictions == 3
    assert report.metrics.invalid_predictions == 1
    assert report.metrics.correct == 3
    assert report.metrics.incorrect == 1  # invalid counts as incorrect
    assert report.metrics.accuracy == 0.75
    assert report.metrics.valid_prediction_coverage == 0.75


def test_unexpected_prediction_id_excluded_from_denominator():
    """Unexpected predictions not present in GT are excluded from denominator."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="unexpected_id")  # 4 correct + 1 unexpected

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4  # denominator remains 4!
    assert report.metrics.correct == 4
    assert report.metrics.accuracy == 1.0
    assert report.metrics.unexpected_predictions == 1


def test_empty_predictions_list():
    """Empty predictions list does not crash and gives accuracy 0.0, coverage 0.0."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="empty")

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 4
    assert report.metrics.provided_predictions == 0
    assert report.metrics.correct == 0
    assert report.metrics.incorrect == 4
    assert report.metrics.accuracy == 0.0
    assert report.metrics.prediction_coverage == 0.0
    assert report.metrics.missing_predictions == 4


def test_empty_ground_truth():
    """Empty ground truth returns 0 total samples, null accuracy, and does not crash."""
    gt = CanonicalGroundTruth(
        schema_version="1.0",
        dataset="ReVA",
        split="validation",
        labeled=True,
        samples=[],
    )
    preds = make_synthetic_predictions(gt, prediction_mode="empty")

    report = evaluate(gt, preds)
    assert report.metrics.total_samples == 0
    assert report.metrics.accuracy is None


def test_reordered_predictions_yield_identical_metrics():
    """Reordering predictions produces identical results due to ID-based join."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds_ordered = make_synthetic_predictions(gt, prediction_mode="mixed")
    
    # Create reversed copy of the exact same prediction records
    reversed_records = list(reversed(preds_ordered.predictions))
    preds_reversed = preds_ordered.model_copy(update={"predictions": reversed_records})

    rep1 = evaluate(gt, preds_ordered)
    rep2 = evaluate(gt, preds_reversed)

    assert rep1.metrics.accuracy == rep2.metrics.accuracy
    assert rep1.metrics.correct == rep2.metrics.correct
    assert rep1.metrics.incorrect == rep2.metrics.incorrect
    assert rep1.metrics.prediction_coverage == rep2.metrics.prediction_coverage


def test_absent_timing_handled_gracefully():
    """When timing values are absent, runtime statistics are safely null."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct", include_timing=False)

    report = evaluate(gt, preds)
    assert report.runtime.timed_samples == 0
    assert report.runtime.mean_seconds is None
    assert report.runtime.median_seconds is None
    assert report.runtime.p95_seconds is None


def test_confusion_matrix_structure():
    """Confusion matrix adheres to fixed 4x4 A-D order and separates missing/invalid."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="missing")  # 3 valid, 1 missing

    df_aligned, _ = align_predictions_with_ground_truth(gt, preds)
    raw_cm, norm_cm, extra_counts = compute_confusion_matrix(df_aligned, is_labeled=True)

    assert list(raw_cm.index) == list(VALID_LABELS)
    assert list(raw_cm.columns) == list(VALID_LABELS)
    assert extra_counts["missing_predictions"] == 1
    assert extra_counts["invalid_predictions"] == 0
    # Sum of elements in raw confusion matrix is exactly 3 (valid predictions)
    assert raw_cm.values.sum() == 3


def test_unlabeled_test_split_omits_accuracy():
    """Unlabeled ground truth must omit accuracy and report unlabeled status."""
    gt = make_synthetic_ground_truth(num_samples=4, labeled=False, split="test")
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")

    report = evaluate(gt, preds)
    assert report.ground_truth_labeled is False
    assert report.status == "unlabeled_ground_truth"
    assert report.metrics.accuracy is None
    assert report.metrics.correct is None
    assert report.metrics.incorrect is None
    assert report.metrics.prediction_coverage == 1.0
