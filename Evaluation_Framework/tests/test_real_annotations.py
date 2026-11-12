"""Tests verifying dataset loader and evaluator on real workspace annotation files."""

from pathlib import Path
import pytest

from reva_eval.dataset_loader import normalize_annotation
from reva_eval.evaluator import evaluate
from tests.fixtures.synthetic_fixtures import make_synthetic_predictions


VAL_PATH = Path("data/annotations/val.json")
TEST_PATH = Path("data/annotations/test.json")


@pytest.mark.skipif(not VAL_PATH.is_file(), reason="data/annotations/val.json not present")
def test_real_validation_annotation():
    """Verify loading real 2000-sample validation split."""
    gt = normalize_annotation(VAL_PATH, split="validation")
    assert gt.dataset == "ReVA"
    assert gt.split == "validation"
    assert gt.labeled is True
    assert len(gt.samples) == 2000

    categories = {s.category for s in gt.samples}
    assert categories == {
        "Factual Perception",
        "Temporal Understanding",
        "Spatial Reasoning",
        "Causal Reasoning",
    }

    tasks = {s.task for s in gt.samples}
    assert len(tasks) == 11

    # Evaluate against synthetic all_correct predictions
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct", model_name="RealValCheck")
    rep = evaluate(gt, preds)
    assert rep.metrics.total_samples == 2000
    assert rep.metrics.correct == 2000
    assert rep.metrics.accuracy == 1.0
    assert rep.metrics.prediction_coverage == 1.0
    assert len(rep.category_metrics) == 4
    assert len(rep.task_metrics) == 11


@pytest.mark.skipif(not TEST_PATH.is_file(), reason="data/annotations/test.json not present")
def test_real_test_annotation_unlabeled():
    """Verify loading real 4000-sample unlabeled test split."""
    gt = normalize_annotation(TEST_PATH, split="test")
    assert gt.dataset == "ReVA"
    assert gt.split == "test"
    assert gt.labeled is False
    assert len(gt.samples) == 4000

    preds = make_synthetic_predictions(gt, prediction_mode="all_correct", model_name="RealTestCheck")
    rep = evaluate(gt, preds)
    assert rep.metrics.total_samples == 4000
    assert rep.ground_truth_labeled is False
    assert rep.status == "unlabeled_ground_truth"
    assert rep.metrics.accuracy is None
    assert rep.metrics.correct is None
    assert rep.metrics.prediction_coverage == 1.0

