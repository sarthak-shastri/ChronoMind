"""Tests for category and task grouping metrics."""

from reva_eval.category_metrics import compute_group_metrics
from reva_eval.evaluator import evaluate
from reva_eval.metrics import align_predictions_with_ground_truth
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


def test_category_and_task_denominator_with_missing_prediction():
    """Group denominators must include ALL ground-truth samples in the group, including missing predictions."""
    # 4 samples: each in a different category and task
    # sample 1: Factual Perception, Object and Land Cover Recognition, gt=A
    # sample 2: Temporal Understanding, Temporal Grounding, gt=B
    # sample 3: Spatial Reasoning, Geometric Relation, gt=C
    # sample 4: Causal Reasoning, Causation Reasoning, gt=D
    gt = make_synthetic_ground_truth(num_samples=4)
    # Missing sample 4 (Causal Reasoning)
    preds = make_synthetic_predictions(gt, prediction_mode="missing")

    df_aligned, _ = align_predictions_with_ground_truth(gt, preds)
    cat_metrics = compute_group_metrics(df_aligned, group_col="category", is_labeled=True)

    # Find Causal Reasoning group
    causal = next(c for c in cat_metrics if c["category"] == "Causal Reasoning")
    assert causal["total"] == 1  # Denominator is 1, not 0!
    assert causal["correct"] == 0
    assert causal["incorrect"] == 1  # Missing counts as incorrect!
    assert causal["accuracy"] == 0.0
    assert causal["coverage"] == 0.0
    assert causal["missing"] == 1

    # Factual Perception (sample 1) is provided and correct
    factual = next(c for c in cat_metrics if c["category"] == "Factual Perception")
    assert factual["total"] == 1
    assert factual["correct"] == 1
    assert factual["accuracy"] == 1.0
    assert factual["coverage"] == 1.0
    assert factual["missing"] == 0


def test_dynamic_taxonomy_no_hardcoding():
    """Categories and tasks with arbitrary/custom names must be grouped properly."""
    gt = make_synthetic_ground_truth(num_samples=2)
    gt.samples[0].category = "CustomCategory_A"
    gt.samples[0].task = "CustomTask_1"
    gt.samples[1].category = "CustomCategory_B"
    gt.samples[1].task = "CustomTask_2"

    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")
    report = evaluate(gt, preds)

    observed_cats = [c["category"] for c in report.category_metrics]
    assert "CustomCategory_A" in observed_cats
    assert "CustomCategory_B" in observed_cats

    observed_tasks = [t["task"] for t in report.task_metrics]
    assert "CustomTask_1" in observed_tasks
    assert "CustomTask_2" in observed_tasks

