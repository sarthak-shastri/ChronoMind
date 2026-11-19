"""ReVA Universal Evaluation Framework.

Universal model-output normalization and evaluation framework for the
ReVA drone-video multiple-choice VideoQA challenge.
"""

__version__ = "1.0.0"

from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    EvaluationReport,
    GroundTruthSample,
    PredictionRecord,
)

__all__ = [
    "CanonicalGroundTruth",
    "CanonicalPredictions",
    "EvaluationReport",
    "GroundTruthSample",
    "PredictionRecord",
]

