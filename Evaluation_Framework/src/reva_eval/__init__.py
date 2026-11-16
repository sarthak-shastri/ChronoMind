"""ReVA Universal Evaluation Framework.

Universal model-output normalization and evaluation framework for the
ReVA drone-video multiple-choice VideoQA challenge.
"""

__version__ = "1.0.0"

from reva_eval.evaluator import evaluate
from reva_eval.experiment_tracker import ExperimentRecord, ExperimentTracker
from reva_eval.interfaces import DummyBaselineModelAdapter, ReVAModelInterface
from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    EvaluationReport,
    GroundTruthSample,
    PredictionRecord,
)
from reva_eval.prediction_loader import load_predictions

__all__ = [
    "evaluate",
    "load_predictions",
    "CanonicalGroundTruth",
    "CanonicalPredictions",
    "EvaluationReport",
    "GroundTruthSample",
    "PredictionRecord",
    "ExperimentTracker",
    "ExperimentRecord",
    "ReVAModelInterface",
    "DummyBaselineModelAdapter",
]
