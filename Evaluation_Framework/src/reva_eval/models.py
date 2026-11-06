"""Data models for ReVA Universal Evaluation Framework.

Provides Pydantic models for ground truth, predictions, metrics, and evaluation reports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VALID_LABELS = ("A", "B", "C", "D")
SCORE_SEMANTICS_VALUES = (
    "probabilities",
    "logits",
    "ranking_scores",
    "unknown",
    None,
)


class ModelMetadata(BaseModel):
    """Metadata describing the evaluated model."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1, description="Model identifier")
    version: Optional[str] = Field(None, description="Optional model/checkpoint version")
    run_id: Optional[str] = Field(None, description="Optional experiment run ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional arbitrary metadata")


class PredictionRecord(BaseModel):
    """A single sample prediction record."""

    model_config = ConfigDict(extra="allow")

    sample_id: str = Field(..., min_length=1, description="Exact sample/qa identifier matching ground truth")
    prediction: Optional[str] = Field(..., description="Answer prediction label (A, B, C, D)")
    scores: Optional[Dict[str, float]] = Field(
        None,
        description="Optional label-keyed scores mapping A, B, C, D to numerical values",
    )
    inference_time_seconds: Optional[float] = Field(
        None,
        description="Optional inference latency in seconds for this sample",
    )

    @field_validator("inference_time_seconds")
    @classmethod
    def validate_latency(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if math.isnan(v) or math.isinf(v) or v < 0:
                raise ValueError(f"inference_time_seconds must be a non-negative finite number, got {v}")
        return v

    def is_valid_label(self) -> bool:
        """Return True if prediction is one of the valid canonical choice labels (A, B, C, D)."""
        return self.prediction in VALID_LABELS


class CanonicalPredictions(BaseModel):
    """Canonical model predictions file structure."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field("1.0", description="Contract schema version")
    dataset: str = Field("ReVA", description="Dataset identifier")
    split: str = Field(..., min_length=1, description="Split name (e.g. validation, test)")
    model: ModelMetadata = Field(..., description="Model metadata")
    score_semantics: Optional[str] = Field(
        None,
        description="Semantics of scores (probabilities, logits, ranking_scores, unknown)",
    )
    option_order: Optional[List[str]] = Field(
        default=["A", "B", "C", "D"],
        description="Explicit ordering of options",
    )
    predictions: List[PredictionRecord] = Field(..., description="List of prediction records")

    @field_validator("score_semantics")
    @classmethod
    def validate_score_semantics(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in SCORE_SEMANTICS_VALUES:
            raise ValueError(
                f"score_semantics must be one of {SCORE_SEMANTICS_VALUES}, got '{v}'"
            )
        return v


class GroundTruthSample(BaseModel):
    """A single canonical ground-truth sample record."""

    model_config = ConfigDict(extra="allow")

    sample_id: str = Field(..., min_length=1, description="Unique question identifier")
    video_path: str = Field(..., min_length=1, description="Relative video path")
    dataset_name: Optional[str] = Field(None, description="Source drone dataset name")
    subdir: Optional[str] = Field(None, description="Subdirectory or scene")
    question: Optional[str] = Field(None, description="Question text")
    options: Optional[Dict[str, str]] = Field(
        None,
        description="Dictionary mapping choice labels A, B, C, D to choice text",
    )
    category: Optional[str] = Field("Unknown", description="High-level category taxonomy")
    task: Optional[str] = Field("Unknown", description="Subtask taxonomy")
    ground_truth: Optional[str] = Field(
        None,
        description="Correct answer label (A, B, C, D) or None if unlabeled",
    )


class CanonicalGroundTruth(BaseModel):
    """Canonical ground-truth dataset structure."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field("1.0", description="Contract schema version")
    dataset: str = Field("ReVA", description="Dataset identifier")
    split: str = Field(..., min_length=1, description="Split name (validation, test, train)")
    source_annotation_file: Optional[str] = Field(None, description="Originating annotation file path/name")
    labeled: bool = Field(True, description="Whether ground-truth labels are present")
    samples: List[GroundTruthSample] = Field(..., description="Ground truth sample records")


class PrimaryMetrics(BaseModel):
    """Primary overall evaluation metrics."""

    total_samples: int = Field(..., ge=0)
    provided_predictions: int = Field(..., ge=0)
    valid_predictions: int = Field(..., ge=0)
    correct: Optional[int] = Field(None, ge=0)
    incorrect: Optional[int] = Field(None, ge=0)
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    prediction_coverage: float = Field(..., ge=0.0, le=1.0)
    valid_prediction_coverage: float = Field(..., ge=0.0, le=1.0)
    missing_predictions: int = Field(..., ge=0)
    invalid_predictions: int = Field(..., ge=0)
    unexpected_predictions: int = Field(..., ge=0)
    duplicate_prediction_occurrences: int = Field(..., ge=0)


class GroupMetric(BaseModel):
    """Category or task breakdown metrics."""

    group_name: str
    total: int = Field(..., ge=0)
    correct: Optional[int] = Field(None, ge=0)
    incorrect: Optional[int] = Field(None, ge=0)
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    coverage: float = Field(..., ge=0.0, le=1.0)


class RuntimeStats(BaseModel):
    """Timing and latency summary statistics."""

    timed_samples: int = Field(..., ge=0)
    mean_seconds: Optional[float] = Field(None, ge=0.0)
    median_seconds: Optional[float] = Field(None, ge=0.0)
    p95_seconds: Optional[float] = Field(None, ge=0.0)


class ScoreMetadata(BaseModel):
    """Metadata regarding prediction scores."""

    score_semantics: Optional[str] = None
    samples_with_scores: int = Field(0, ge=0)


class ValidationSummary(BaseModel):
    """Summary of validation counts."""

    errors: int = Field(0, ge=0)
    warnings: int = Field(0, ge=0)


class EvaluationReport(BaseModel):
    """Canonical evaluation report format."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    dataset: str = "ReVA"
    split: str
    model: ModelMetadata
    status: Literal["completed", "unlabeled_ground_truth", "failed"]
    ground_truth_labeled: bool
    metrics: PrimaryMetrics
    category_metrics: List[Dict[str, Any]]
    task_metrics: List[Dict[str, Any]]
    runtime: RuntimeStats
    score_metadata: ScoreMetadata
    validation: ValidationSummary
    generated_at_utc: str
    notes: List[str] = Field(default_factory=list)

