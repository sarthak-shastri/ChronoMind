"""Tests for validation rules, ID alignment, and prediction schemas."""

import pytest
from pydantic import ValidationError

from reva_eval.models import CanonicalPredictions, ModelMetadata, PredictionRecord
from reva_eval.validation import (
    validate_ground_truth,
    validate_predictions_against_ground_truth,
)
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


def test_duplicate_prediction_id_fails_by_default():
    """Duplicate prediction IDs must fail validation by default."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="duplicate_id")

    res = validate_predictions_against_ground_truth(preds, gt, allow_duplicates=False)
    assert not res.is_valid
    assert len(res.duplicate_sample_ids) > 0
    assert any("Duplicate IDs fail validation by default" in e for e in res.errors)


def test_duplicate_prediction_id_allowed_in_diagnostic_mode():
    """Duplicate prediction IDs pass validation with warning if allow_duplicates=True."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="duplicate_id")

    res = validate_predictions_against_ground_truth(preds, gt, allow_duplicates=True)
    assert res.is_valid
    assert len(res.duplicate_sample_ids) > 0
    assert any("Allowed by diagnostic mode" in w for w in res.warnings)


def test_unexpected_and_missing_prediction_ids():
    """Verify detection of missing and unexpected sample IDs."""
    gt = make_synthetic_ground_truth(num_samples=4)
    # Missing last sample, plus unexpected sample
    preds = make_synthetic_predictions(gt, prediction_mode="missing")
    # Add an unexpected prediction
    preds.predictions.append(
        PredictionRecord(sample_id="extra_sample_999", prediction="A")
    )

    res = validate_predictions_against_ground_truth(preds, gt)
    assert "synth_000004" in res.missing_sample_ids
    assert "extra_sample_999" in res.unexpected_sample_ids


def test_invalid_prediction_answer_label():
    """Verify detection of invalid answer choice labels (e.g. 'Z')."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="invalid_label")

    res = validate_predictions_against_ground_truth(preds, gt)
    assert "synth_000001" in res.invalid_sample_ids
    assert any("invalid choice labels" in w for w in res.warnings)


def test_score_keys_validation():
    """Scores dictionary must have keys exactly ['A', 'B', 'C', 'D']."""
    gt = make_synthetic_ground_truth(num_samples=2)
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")
    # Malform score keys for first sample
    preds.predictions[0].scores = {"A": 0.5, "B": 0.5, "E": 0.0}

    res = validate_predictions_against_ground_truth(preds, gt)
    assert not res.is_valid
    assert any("scores keys must be exactly" in e for e in res.errors)


def test_probability_sum_tolerance_warning_and_strict_error():
    """Verify probability sum validation under standard and strict modes."""
    gt = make_synthetic_ground_truth(num_samples=2)
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")
    # Violate sum to 1.0 (sums to 0.7)
    preds.predictions[0].scores = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.1}

    # Standard mode: warning
    res_std = validate_predictions_against_ground_truth(
        preds, gt, strict_probabilities=False, probability_sum_tolerance=1e-4
    )
    assert res_std.is_valid
    assert len(res_std.probability_sum_violations) == 1
    assert any("not summing to 1.0" in w for w in res_std.warnings)

    # Strict mode: error
    res_strict = validate_predictions_against_ground_truth(
        preds, gt, strict_probabilities=True, probability_sum_tolerance=1e-4
    )
    assert not res_strict.is_valid
    assert any("not summing to 1.0" in e for e in res_strict.errors)


def test_nan_or_negative_latency_rejected():
    """NaN or negative latency must be rejected by model validation."""
    with pytest.raises(ValidationError):
        PredictionRecord(sample_id="s1", prediction="A", inference_time_seconds=-0.5)

    with pytest.raises(ValidationError):
        PredictionRecord(sample_id="s1", prediction="A", inference_time_seconds=float("nan"))


def test_invalid_score_semantics_rejected():
    """Invalid score_semantics string must be rejected."""
    gt = make_synthetic_ground_truth(num_samples=1)
    with pytest.raises(ValidationError):
        make_synthetic_predictions(gt, score_semantics="arbitrary_scores")

