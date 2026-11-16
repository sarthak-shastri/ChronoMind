"""Multi-format prediction loader and answer normalizer for ReVA VideoQA.

Supports canonical and non-canonical JSON, JSONL, and CSV prediction formats.
Performs clean, non-destructive answer normalization (Unicode NFKC, whitespace stripping,
optional case-folding) while preserving original raw prediction strings.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import unicodedata

from reva_eval.models import (
    CanonicalPredictions,
    ModelMetadata,
    PredictionRecord,
    SCORE_SEMANTICS_VALUES,
    VALID_LABELS,
)


def normalize_answer_string(
    raw_answer: Optional[str],
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    case_fold: bool = True,
) -> Optional[str]:
    """Normalize a candidate answer prediction string safely.

    Preserves semantic meaning while resolving trivial formatting differences:
    - Unicode NFKC normalization (e.g. full-width characters, ligature normalization).
    - Whitespace trimming (leading/trailing/newlines/tabs).
    - Case conversion (e.g. 'a' -> 'A') if configured.

    Returns None if raw_answer is None or empty after trimming.
    """
    if raw_answer is None:
        return None

    ans = str(raw_answer)

    if normalize_unicode:
        ans = unicodedata.normalize("NFKC", ans)

    if strip_whitespace:
        # Strip ASCII whitespace as well as zero-width and non-breaking spaces
        ans = ans.strip().strip("\u200b\ufeff\u00a0\r\n\t")

    if not ans:
        return None

    if case_fold:
        ans = ans.upper()

    return ans


def load_predictions_from_json(
    file_path: Union[str, Path],
    model_name: Optional[str] = None,
    split: Optional[str] = None,
    dataset: str = "ReVA",
    score_semantics: Optional[str] = None,
    option_order: Optional[Sequence[str]] = None,
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    case_fold: bool = True,
) -> CanonicalPredictions:
    """Load predictions from a JSON file (canonical or generic structure)."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse prediction JSON file {path.name}: {exc}") from exc

    # Fast path: already structured CanonicalPredictions
    if (
        isinstance(data, dict)
        and "schema_version" in data
        and "predictions" in data
        and "model" in data
        and isinstance(data["predictions"], list)
    ):
        # Validate and apply normalization to records
        try:
            canonical = CanonicalPredictions.model_validate(data)
        except Exception:
            # Fall back to adaptive conversion
            canonical = None

        if canonical is not None:
            # Apply normalization to prediction labels while preserving raw string
            for p in canonical.predictions:
                if p.raw_prediction is None:
                    p.raw_prediction = p.prediction
                p.prediction = normalize_answer_string(
                    p.prediction,
                    strip_whitespace=strip_whitespace,
                    normalize_unicode=normalize_unicode,
                    case_fold=case_fold,
                )
            return canonical

    # Adaptive parsing using GenericJSONAdapter logic
    from reva_eval.adapters.generic_json import GenericJSONAdapter
    adapter = GenericJSONAdapter()
    return adapter.convert(
        raw_data=data,
        model_name=model_name or "Model",
        split=split or "validation",
        dataset=dataset,
        score_semantics=score_semantics,
        option_order=option_order,
        strip_whitespace=strip_whitespace,
        normalize_unicode=normalize_unicode,
        case_fold=case_fold,
    )


def load_predictions_from_jsonl(
    file_path: Union[str, Path],
    model_name: Optional[str] = None,
    split: Optional[str] = None,
    dataset: str = "ReVA",
    score_semantics: Optional[str] = None,
    option_order: Optional[Sequence[str]] = None,
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    case_fold: bool = True,
) -> CanonicalPredictions:
    """Load predictions from line-delimited JSON (JSONL / NDJSON).

    Each line should be a JSON object containing at least:
    - 'sample_id' (or 'qa_id' / 'id')
    - 'prediction' (or 'answer' / 'pred')
    - Optional: 'scores' (dict or list), 'inference_time_seconds'
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction JSONL file not found: {path}")

    raw_records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in JSONL file {path.name}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_no} in JSONL file {path.name} must be a JSON object, got {type(record)}"
                )
            raw_records.append(record)

    from reva_eval.adapters.generic_json import GenericJSONAdapter
    adapter = GenericJSONAdapter()
    return adapter.convert(
        raw_data=raw_records,
        model_name=model_name or path.stem,
        split=split or "validation",
        dataset=dataset,
        score_semantics=score_semantics,
        option_order=option_order,
        strip_whitespace=strip_whitespace,
        normalize_unicode=normalize_unicode,
        case_fold=case_fold,
    )


def load_predictions_from_csv(
    file_path: Union[str, Path],
    model_name: Optional[str] = None,
    split: Optional[str] = None,
    dataset: str = "ReVA",
    score_semantics: Optional[str] = None,
    option_order: Optional[Sequence[str]] = None,
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    case_fold: bool = True,
) -> CanonicalPredictions:
    """Load predictions from a CSV file.

    Expected CSV columns:
    - 'sample_id' or 'qa_id' or 'id'
    - 'prediction' or 'answer' or 'pred'
    - Optional scores:
      - Column 'scores' as JSON string '{"A": 0.7, ...}'
      - OR individual columns 'score_A', 'score_B', 'score_C', 'score_D'
    - Optional latency: 'inference_time_seconds', 'latency', 'inference_time'
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV file not found: {path}")

    raw_records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV file {path.name} is empty or missing headers.")

        # Identify required column mappings
        fields = [fn.strip() for fn in reader.fieldnames]
        field_map = {fn.lower().strip(): fn for fn in fields}

        id_col = None
        for cand in ("sample_id", "qa_id", "id"):
            if cand in field_map:
                id_col = field_map[cand]
                break

        pred_col = None
        for cand in ("prediction", "answer", "pred", "predicted_answer"):
            if cand in field_map:
                pred_col = field_map[cand]
                break

        if id_col is None:
            raise ValueError(
                f"CSV {path.name} is missing a sample identifier column (expected 'sample_id', 'qa_id', or 'id'). "
                f"Found columns: {fields}"
            )
        if pred_col is None:
            raise ValueError(
                f"CSV {path.name} is missing a prediction column (expected 'prediction', 'answer', or 'pred'). "
                f"Found columns: {fields}"
            )

        # Check for individual score columns: score_a, score_b, score_c, score_d
        has_score_cols = all(f"score_{l.lower()}" in field_map for l in VALID_LABELS)
        score_json_col = field_map.get("scores") or field_map.get("probabilities") or field_map.get("logits")

        # Check for latency column
        latency_col = None
        for cand in ("inference_time_seconds", "inference_time", "latency", "time"):
            if cand in field_map:
                latency_col = field_map[cand]
                break

        for row_idx, row in enumerate(reader, start=2):
            s_id = row.get(id_col)
            if s_id is not None:
                s_id = s_id.strip()

            p_val = row.get(pred_col)

            # Scores extraction
            scores_obj = None
            if has_score_cols:
                try:
                    scores_obj = {
                        lbl: float(row[field_map[f"score_{lbl.lower()}"]])
                        for lbl in VALID_LABELS
                    }
                except (ValueError, TypeError):
                    scores_obj = None
            elif score_json_col and row.get(score_json_col):
                val_str = row[score_json_col].strip()
                if val_str:
                    try:
                        scores_obj = json.loads(val_str)
                    except json.JSONDecodeError:
                        scores_obj = None

            # Latency extraction
            latency_val = None
            if latency_col and row.get(latency_col):
                try:
                    latency_val = float(row[latency_col])
                except (ValueError, TypeError):
                    latency_val = None

            record: Dict[str, Any] = {
                "sample_id": s_id,
                "prediction": p_val,
            }
            if scores_obj is not None:
                record["scores"] = scores_obj
            if latency_val is not None:
                record["inference_time_seconds"] = latency_val

            raw_records.append(record)

    from reva_eval.adapters.generic_json import GenericJSONAdapter
    adapter = GenericJSONAdapter()
    return adapter.convert(
        raw_data=raw_records,
        model_name=model_name or path.stem,
        split=split or "validation",
        dataset=dataset,
        score_semantics=score_semantics,
        option_order=option_order,
        strip_whitespace=strip_whitespace,
        normalize_unicode=normalize_unicode,
        case_fold=case_fold,
    )


def load_predictions(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    model_name: Optional[str] = None,
    split: Optional[str] = None,
    dataset: str = "ReVA",
    score_semantics: Optional[str] = None,
    option_order: Optional[Sequence[str]] = None,
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    case_fold: bool = True,
) -> CanonicalPredictions:
    """Universal prediction file loader with automatic format detection.

    Supported formats:
    - 'json': Canonical or raw JSON prediction files.
    - 'jsonl' / 'ndjson': Line-delimited JSON prediction files.
    - 'csv': Comma-separated value prediction files.
    - 'auto' / None: Detect format from file extension.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    fmt = (file_format or "auto").lower().strip()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            fmt = "jsonl"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix == ".json":
            fmt = "json"
        else:
            # Inspect first byte / line to detect JSON vs JSONL vs CSV
            try:
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if first_line.startswith("{") and not first_line.endswith("}"):
                    fmt = "json"
                elif first_line.startswith("{") and first_line.endswith("}"):
                    fmt = "jsonl"
                elif "," in first_line:
                    fmt = "csv"
                else:
                    fmt = "json"
            except Exception:
                fmt = "json"

    if fmt == "json":
        return load_predictions_from_json(
            file_path=path,
            model_name=model_name,
            split=split,
            dataset=dataset,
            score_semantics=score_semantics,
            option_order=option_order,
            strip_whitespace=strip_whitespace,
            normalize_unicode=normalize_unicode,
            case_fold=case_fold,
        )
    elif fmt in ("jsonl", "ndjson"):
        return load_predictions_from_jsonl(
            file_path=path,
            model_name=model_name,
            split=split,
            dataset=dataset,
            score_semantics=score_semantics,
            option_order=option_order,
            strip_whitespace=strip_whitespace,
            normalize_unicode=normalize_unicode,
            case_fold=case_fold,
        )
    elif fmt == "csv":
        return load_predictions_from_csv(
            file_path=path,
            model_name=model_name,
            split=split,
            dataset=dataset,
            score_semantics=score_semantics,
            option_order=option_order,
            strip_whitespace=strip_whitespace,
            normalize_unicode=normalize_unicode,
            case_fold=case_fold,
        )
    else:
        raise ValueError(
            f"Unsupported prediction format: '{fmt}'. Supported formats are: 'json', 'jsonl', 'csv', 'auto'."
        )

