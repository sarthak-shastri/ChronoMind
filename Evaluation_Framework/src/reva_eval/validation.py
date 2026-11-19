"""Validation engine and ID alignment checks for ReVA Ground Truth and Predictions."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    PredictionRecord,
    SCORE_SEMANTICS_VALUES,
    VALID_LABELS,
)


class ValidationResult:
    """Encapsulates validation diagnostics, errors, warnings, and issue lists."""

    def __init__(
        self,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        missing_sample_ids: Optional[List[str]] = None,
        invalid_sample_ids: Optional[List[str]] = None,
        unexpected_sample_ids: Optional[List[str]] = None,
        duplicate_sample_ids: Optional[List[str]] = None,
        probability_sum_violations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.errors = errors or []
        self.warnings = warnings or []
        self.missing_sample_ids = missing_sample_ids or []
        self.invalid_sample_ids = invalid_sample_ids or []
        self.unexpected_sample_ids = unexpected_sample_ids or []
        self.duplicate_sample_ids = duplicate_sample_ids or []
        self.probability_sum_violations = probability_sum_violations or []

    @property
    def is_valid(self) -> bool:
        """Return True if no validation errors were found."""
        return len(self.errors) == 0

    def to_dict(self, max_ids: int = 100) -> Dict[str, Any]:
        """Convert validation issues to a serializable dictionary for validation_issues.json."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "counts": {
                "missing": len(self.missing_sample_ids),
                "invalid": len(self.invalid_sample_ids),
                "unexpected": len(self.unexpected_sample_ids),
                "duplicates": len(self.duplicate_sample_ids),
                "probability_violations": len(self.probability_sum_violations),
            },
            "sample_ids": {
                "missing": self.missing_sample_ids[:max_ids],
                "invalid": self.invalid_sample_ids[:max_ids],
                "unexpected": self.unexpected_sample_ids[:max_ids],
                "duplicates": self.duplicate_sample_ids[:max_ids],
            },
            "probability_violations_sample": self.probability_sum_violations[:max_ids],
        }


def validate_ground_truth(gt: CanonicalGroundTruth) -> ValidationResult:
    """Validate ground truth integrity."""
    errors: List[str] = []
    warnings: List[str] = []

    if not gt.samples:
        warnings.append("Ground truth dataset contains 0 samples.")

    seen_ids: Set[str] = set()
    dup_ids: List[str] = []

    for s in gt.samples:
        if not s.sample_id:
            errors.append("Encountered ground truth sample with empty sample_id.")
        elif s.sample_id in seen_ids:
            dup_ids.append(s.sample_id)
        seen_ids.add(s.sample_id)

        if not s.video_path:
            errors.append(f"Sample {s.sample_id} is missing video_path.")

        if gt.labeled and s.ground_truth not in VALID_LABELS:
            warnings.append(
                f"Sample {s.sample_id} has unexpected ground_truth value '{s.ground_truth}' in labeled split."
            )

    if dup_ids:
        errors.append(f"Ground truth has {len(dup_ids)} duplicate sample_id(s): {dup_ids[:5]}")

    return ValidationResult(
        errors=errors,
        warnings=warnings,
        duplicate_sample_ids=dup_ids,
    )


def validate_predictions_against_ground_truth(
    predictions: CanonicalPredictions,
    ground_truth: CanonicalGroundTruth,
    allow_duplicates: bool = False,
    strict_probabilities: bool = False,
    probability_sum_tolerance: float = 1e-4,
) -> ValidationResult:
    """Perform rigorous validation and alignment diagnostics between predictions and ground truth."""
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Dataset & Split compatibility
    if predictions.dataset.lower() != ground_truth.dataset.lower():
        warnings.append(
            f"Dataset mismatch: predictions specify '{predictions.dataset}', "
            f"ground truth specifies '{ground_truth.dataset}'"
        )

    # Standardize split names (val vs validation)
    p_split = predictions.split.lower().replace("val", "validation")
    g_split = ground_truth.split.lower().replace("val", "validation")
    if p_split != g_split:
        warnings.append(
            f"Split mismatch: predictions specify '{predictions.split}', "
            f"ground truth specifies '{ground_truth.split}'"
        )

    # 2. Check duplicate predictions
    id_counts = Counter(p.sample_id for p in predictions.predictions)
    duplicate_ids: List[str] = []
    for s_id, count in id_counts.items():
        if count > 1:
            duplicate_ids.extend([s_id] * (count - 1))

    if duplicate_ids:
        msg = f"Found {len(duplicate_ids)} duplicate prediction records across {len(set(duplicate_ids))} unique ID(s)."
        if allow_duplicates:
            warnings.append(f"{msg} (Allowed by diagnostic mode --allow-duplicates; only first occurrence will be evaluated)")
        else:
            errors.append(f"{msg} Duplicate IDs fail validation by default.")

    # 3. Ground truth alignment
    gt_ids = {s.sample_id for s in ground_truth.samples}
    first_pred_by_id: Dict[str, PredictionRecord] = {}

    for p in predictions.predictions:
        if p.sample_id not in first_pred_by_id:
            first_pred_by_id[p.sample_id] = p

    pred_ids = set(first_pred_by_id.keys())

    missing_sample_ids = sorted(list(gt_ids - pred_ids))
    unexpected_sample_ids = sorted(list(pred_ids - gt_ids))

    if missing_sample_ids:
        warnings.append(
            f"Missing predictions for {len(missing_sample_ids)} / {len(gt_ids)} ground truth samples."
        )

    if unexpected_sample_ids:
        warnings.append(
            f"Found {len(unexpected_sample_ids)} unexpected prediction IDs not present in ground truth."
        )

    # 4. Prediction label and score validity
    invalid_sample_ids: List[str] = []
    prob_violations: List[Dict[str, Any]] = []

    is_prob_semantics = predictions.score_semantics == "probabilities"

    for s_id in (pred_ids & gt_ids):
        pred_rec = first_pred_by_id[s_id]
        if not pred_rec.is_valid_label():
            invalid_sample_ids.append(s_id)

        # Validate scores if present
        if pred_rec.scores is not None:
            # Check keys
            if sorted(pred_rec.scores.keys()) != sorted(VALID_LABELS):
                errors.append(
                    f"Sample '{s_id}' scores keys must be exactly {sorted(VALID_LABELS)}, "
                    f"got {sorted(pred_rec.scores.keys())}"
                )
            else:
                for lbl, score_val in pred_rec.scores.items():
                    if math.isnan(score_val) or math.isinf(score_val):
                        errors.append(f"Sample '{s_id}' has non-finite score for label '{lbl}': {score_val}")

                if is_prob_semantics:
                    # Check range [0, 1]
                    out_of_range = {k: v for k, v in pred_rec.scores.items() if v < -1e-6 or v > 1.0 + 1e-6}
                    score_sum = sum(pred_rec.scores.values())
                    diff = abs(score_sum - 1.0)
                    if out_of_range or diff > probability_sum_tolerance:
                        violation = {
                            "sample_id": s_id,
                            "scores": pred_rec.scores,
                            "sum": score_sum,
                            "delta_from_one": diff,
                        }
                        prob_violations.append(violation)

    if invalid_sample_ids:
        warnings.append(
            f"Found {len(invalid_sample_ids)} predictions with invalid choice labels (not in A, B, C, D)."
        )

    if prob_violations:
        prob_msg = (
            f"Found {len(prob_violations)} samples with probability scores not summing to 1.0 "
            f"(tolerance: {probability_sum_tolerance})."
        )
        if strict_probabilities:
            errors.append(prob_msg)
        else:
            warnings.append(prob_msg)

    return ValidationResult(
        errors=errors,
        warnings=warnings,
        missing_sample_ids=missing_sample_ids,
        invalid_sample_ids=invalid_sample_ids,
        unexpected_sample_ids=unexpected_sample_ids,
        duplicate_sample_ids=duplicate_ids,
        probability_sum_violations=prob_violations,
    )

