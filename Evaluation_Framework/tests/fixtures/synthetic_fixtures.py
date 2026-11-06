"""Synthetic test data and fixture generators for ReVA Evaluation tests."""

from __future__ import annotations

from typing import Any, Dict, List
from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    GroundTruthSample,
    ModelMetadata,
    PredictionRecord,
)


def make_synthetic_ground_truth(
    num_samples: int = 4,
    labeled: bool = True,
    split: str = "validation",
) -> CanonicalGroundTruth:
    """Build a synthetic CanonicalGroundTruth instance with deterministic values."""
    categories = [
        "Factual Perception",
        "Temporal Understanding",
        "Spatial Reasoning",
        "Causal Reasoning",
    ]
    tasks = [
        "Object and Land Cover Recognition",
        "Temporal Grounding",
        "Geometric Relation",
        "Causation Reasoning",
    ]
    gt_answers = ["A", "B", "C", "D"]

    samples: List[GroundTruthSample] = []
    for i in range(num_samples):
        idx = i + 1
        cat = categories[i % len(categories)]
        task = tasks[i % len(tasks)]
        ans = gt_answers[i % len(gt_answers)] if labeled else None

        s = GroundTruthSample(
            sample_id=f"synth_{idx:06d}",
            video_path=f"DroneDataset/scene_{idx:03d}.mp4",
            dataset_name="SyntheticUAV",
            subdir=f"Scene_{idx}",
            question=f"Synthetic question {idx}?",
            options={
                "A": f"Option A {idx}",
                "B": f"Option B {idx}",
                "C": f"Option C {idx}",
                "D": f"Option D {idx}",
            },
            category=cat,
            task=task,
            ground_truth=ans,
        )
        samples.append(s)

    return CanonicalGroundTruth(
        schema_version="1.0",
        dataset="ReVA",
        split=split,
        source_annotation_file="synthetic_fixture.json",
        labeled=labeled,
        samples=samples,
    )


def make_synthetic_predictions(
    ground_truth: CanonicalGroundTruth,
    prediction_mode: str = "all_correct",
    model_name: str = "SyntheticModel",
    include_scores: bool = True,
    include_timing: bool = True,
    score_semantics: str = "probabilities",
) -> CanonicalPredictions:
    """Build a synthetic CanonicalPredictions instance aligned with ground truth.

    prediction_mode can be:
    - 'all_correct'
    - 'all_wrong'
    - 'mixed': 3 correct, 1 wrong (for 4 samples: 1->A, 2->C(wrong), 3->C, 4->D)
    - 'missing': omits the last prediction
    - 'invalid_label': sets one prediction label to 'Z'
    - 'unexpected_id': includes an extra prediction ID not in GT
    - 'duplicate_id': duplicates the first prediction ID
    - 'empty': no predictions
    - 'reordered': reversed prediction list
    """
    records: List[PredictionRecord] = []
    wrong_mapping = {"A": "B", "B": "C", "C": "D", "D": "A"}

    for idx, s in enumerate(ground_truth.samples):
        gt_ans = s.ground_truth or "A"

        if prediction_mode == "all_correct":
            pred_lbl = gt_ans
        elif prediction_mode == "all_wrong":
            pred_lbl = wrong_mapping.get(gt_ans, "B")
        elif prediction_mode == "mixed":
            # For 4 samples: idx 1 is wrong, others correct
            pred_lbl = wrong_mapping.get(gt_ans, "B") if idx == 1 else gt_ans
        elif prediction_mode == "missing" and idx == len(ground_truth.samples) - 1:
            continue  # Skip last sample
        elif prediction_mode == "invalid_label" and idx == 0:
            pred_lbl = "Z"
        else:
            pred_lbl = gt_ans

        scores = None
        if include_scores:
            if pred_lbl in ("A", "B", "C", "D"):
                # Probabilities summing to 1.0
                scores = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}
                scores[pred_lbl] = 0.7
            else:
                scores = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        timing = (1.0 + 0.1 * idx) if include_timing else None

        records.append(
            PredictionRecord(
                sample_id=s.sample_id,
                prediction=pred_lbl,
                scores=scores,
                inference_time_seconds=timing,
            )
        )

    if prediction_mode == "unexpected_id":
        records.append(
            PredictionRecord(
                sample_id="unexpected_sample_999999",
                prediction="A",
                scores={"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1} if include_scores else None,
                inference_time_seconds=1.2 if include_timing else None,
            )
        )

    if prediction_mode == "duplicate_id" and records:
        # Append a duplicate of the first record
        first = records[0]
        records.append(
            PredictionRecord(
                sample_id=first.sample_id,
                prediction=first.prediction,
                scores=first.scores,
                inference_time_seconds=first.inference_time_seconds,
            )
        )

    if prediction_mode == "reordered":
        records.reverse()

    if prediction_mode == "empty":
        records = []

    return CanonicalPredictions(
        schema_version="1.0",
        dataset="ReVA",
        split=ground_truth.split,
        model=ModelMetadata(name=model_name, version="1.0", run_id="SYNTH_001"),
        score_semantics=score_semantics if include_scores else None,
        option_order=["A", "B", "C", "D"],
        predictions=records,
    )

