"""Generic JSON adapter for converting heterogeneous model predictions into canonical format."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from reva_eval.adapters.base import BaseAdapter
from reva_eval.models import (
    CanonicalPredictions,
    ModelMetadata,
    PredictionRecord,
    SCORE_SEMANTICS_VALUES,
    VALID_LABELS,
)


class GenericJSONAdapter(BaseAdapter):
    """Converts standard and semi-structured JSON model outputs to canonical prediction format."""

    def convert(
        self,
        raw_data: Any,
        model_name: str,
        split: str,
        dataset: str = "ReVA",
        model_version: Optional[str] = None,
        run_id: Optional[str] = None,
        score_semantics: Optional[str] = None,
        option_order: Optional[Sequence[str]] = None,
        allow_duplicates: bool = False,
        **kwargs: Any,
    ) -> CanonicalPredictions:
        """Convert raw prediction JSON into CanonicalPredictions.

        Supported input shapes:
        1. Canonical predictions dictionary (pass-through validation).
        2. Dictionary with a 'predictions' or 'results' list.
        3. Raw list of dicts: [{'sample_id'/'qa_id': ..., 'prediction'/'answer': ..., ...}].
        4. Dictionary mapping sample_id -> prediction label or dict.
        """
        # Validate score semantics if provided
        if score_semantics is not None and score_semantics not in SCORE_SEMANTICS_VALUES:
            raise ValueError(
                f"Invalid score_semantics: '{score_semantics}'. Must be one of {SCORE_SEMANTICS_VALUES}"
            )

        # Validate option_order if provided
        validated_option_order = ["A", "B", "C", "D"]
        if option_order is not None:
            opt_list = list(option_order)
            if sorted(opt_list) != sorted(VALID_LABELS):
                raise ValueError(
                    f"Invalid option_order: {opt_list}. Must contain exactly labels ['A', 'B', 'C', 'D']"
                )
            validated_option_order = opt_list

        raw_records: List[Dict[str, Any]] = []

        if isinstance(raw_data, dict):
            # Check if it's already structured canonical or wrapper
            if "predictions" in raw_data and isinstance(raw_data["predictions"], list):
                raw_records = raw_data["predictions"]
                if not model_name and "model" in raw_data:
                    m = raw_data["model"]
                    if isinstance(m, dict):
                        model_name = m.get("name", model_name)
                        model_version = model_version or m.get("version")
                        run_id = run_id or m.get("run_id")
                score_semantics = score_semantics or raw_data.get("score_semantics")
                if "option_order" in raw_data and isinstance(raw_data["option_order"], list):
                    validated_option_order = raw_data["option_order"]
            elif "results" in raw_data and isinstance(raw_data["results"], list):
                raw_records = raw_data["results"]
            else:
                # Check if it's a mapping of sample_id -> prediction (or sample_id -> dict)
                # Sample check: are all keys strings and values strings or dicts?
                for s_id, s_val in raw_data.items():
                    if isinstance(s_val, str):
                        raw_records.append({"sample_id": s_id, "prediction": s_val})
                    elif isinstance(s_val, dict):
                        rec = dict(s_val)
                        rec["sample_id"] = s_id
                        raw_records.append(rec)
                    else:
                        raise ValueError(f"Unsupported record format for key '{s_id}' in prediction dictionary")
        elif isinstance(raw_data, list):
            raw_records = raw_data
        else:
            raise ValueError(f"Unsupported raw prediction data type: {type(raw_data)}")

        if not model_name:
            model_name = "UnknownModel"

        seen_ids = set()
        duplicate_ids = []
        predictions: List[PredictionRecord] = []

        for idx, rec in enumerate(raw_records):
            if not isinstance(rec, dict):
                raise ValueError(f"Prediction record at index {idx} must be a dictionary, got {type(rec)}")

            # Extract sample_id
            sample_id = rec.get("sample_id") or rec.get("qa_id") or rec.get("id")
            if not sample_id:
                raise ValueError(f"Prediction record at index {idx} is missing 'sample_id' or 'qa_id'")
            sample_id = str(sample_id)

            if sample_id in seen_ids:
                duplicate_ids.append(sample_id)
            seen_ids.add(sample_id)

            # Extract prediction
            prediction = rec.get("prediction")
            if prediction is None:
                prediction = rec.get("answer") or rec.get("pred")

            if prediction is not None:
                prediction = str(prediction).strip()

            # Extract scores
            raw_scores = rec.get("scores") or rec.get("probabilities") or rec.get("logits")
            scores_dict: Optional[Dict[str, float]] = None

            if raw_scores is not None:
                if isinstance(raw_scores, dict):
                    # Validate keys are A, B, C, D
                    scores_dict = {}
                    for k, v in raw_scores.items():
                        cleaned_k = str(k).strip()
                        try:
                            scores_dict[cleaned_k] = float(v)
                        except (ValueError, TypeError) as e:
                            raise ValueError(f"Score for label '{k}' in sample '{sample_id}' is not numeric: {v}") from e
                    if sorted(scores_dict.keys()) != sorted(VALID_LABELS):
                        raise ValueError(
                            f"Sample '{sample_id}' scores dict must have keys exactly {sorted(VALID_LABELS)}, "
                            f"got {sorted(scores_dict.keys())}"
                        )
                elif isinstance(raw_scores, (list, tuple)):
                    if len(raw_scores) != 4:
                        raise ValueError(
                            f"Sample '{sample_id}' score list must have length 4, got {len(raw_scores)}"
                        )
                    # When mapping a score list, use explicit option_order without reordering
                    scores_dict = {}
                    for opt_label, score_val in zip(validated_option_order, raw_scores):
                        try:
                            scores_dict[opt_label] = float(score_val)
                        except (ValueError, TypeError) as e:
                            raise ValueError(
                                f"Score value for '{opt_label}' in sample '{sample_id}' is not numeric: {score_val}"
                            ) from e
                else:
                    raise ValueError(f"Scores for sample '{sample_id}' must be a dict or list, got {type(raw_scores)}")

            # Extract timing
            raw_time = rec.get("inference_time_seconds")
            if raw_time is None:
                raw_time = rec.get("inference_time") or rec.get("latency") or rec.get("time")

            timing_val: Optional[float] = None
            if raw_time is not None:
                try:
                    timing_val = float(raw_time)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Timing for sample '{sample_id}' is not numeric: {raw_time}") from e

            # IMPORTANT: Do not include ground truth in prediction records
            # Strip fields like 'ground_truth', 'correct_answer'
            pred_record = PredictionRecord(
                sample_id=sample_id,
                prediction=prediction,
                scores=scores_dict,
                inference_time_seconds=timing_val,
            )
            predictions.append(pred_record)

        if duplicate_ids and not allow_duplicates:
            raise ValueError(
                f"Prediction validation failed: found {len(duplicate_ids)} duplicate sample_id(s) "
                f"(e.g., {duplicate_ids[:5]}). Use allow_duplicates=True if inspecting malformed submissions."
            )

        model_meta = ModelMetadata(
            name=model_name,
            version=model_version,
            run_id=run_id,
        )

        return CanonicalPredictions(
            schema_version="1.0",
            dataset=dataset,
            split=split,
            model=model_meta,
            score_semantics=score_semantics,
            option_order=validated_option_order,
            predictions=predictions,
        )

