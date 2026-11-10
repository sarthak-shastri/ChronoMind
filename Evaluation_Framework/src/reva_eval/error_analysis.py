"""Error analysis and diagnostic failure export."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_incorrect_predictions_dataframe(
    df_aligned: pd.DataFrame,
    score_semantics: Optional[str] = None,
) -> pd.DataFrame:
    """Extract all incorrect, missing, or invalid ground-truth samples for error analysis.

    Columns conform to specification Section 9:
    - sample_id
    - video_path
    - dataset_name
    - subdir
    - question
    - options (or options_A, options_B, options_C, options_D)
    - category
    - task
    - ground_truth
    - prediction (None if missing)
    - scores
    - score_semantics
    - inference_time_seconds
    - error_status ('incorrect', 'missing', 'invalid')
    - manual_error_type (empty/None for manual annotation)
    - reviewer_notes (empty/None for reviewer notes)
    """
    if len(df_aligned) == 0:
        cols = [
            "sample_id", "video_path", "dataset_name", "subdir",
            "question", "options_A", "options_B", "options_C", "options_D",
            "category", "task", "ground_truth", "prediction",
            "scores", "score_semantics", "inference_time_seconds",
            "error_status", "manual_error_type", "reviewer_notes",
        ]
        return pd.DataFrame(columns=cols)

    # Errors include: missing, invalid, or wrong (status == 'wrong' or is_correct == False)
    error_mask = df_aligned["status"].isin(["wrong", "missing", "invalid"])
    sub_df = df_aligned[error_mask].copy()

    records = []
    for _, row in sub_df.iterrows():
        status_val = row["status"]
        error_status = "incorrect" if status_val == "wrong" else status_val

        options = row["options"] if isinstance(row["options"], dict) else {}
        scores_str = json.dumps(row["scores"]) if row["scores"] is not None else None

        records.append({
            "sample_id": row["sample_id"],
            "video_path": row["video_path"],
            "dataset_name": row["dataset_name"],
            "subdir": row["subdir"],
            "question": row["question"],
            "options_A": options.get("A"),
            "options_B": options.get("B"),
            "options_C": options.get("C"),
            "options_D": options.get("D"),
            "category": row["category"],
            "task": row["task"],
            "ground_truth": row["ground_truth"],
            "prediction": row["prediction"] if row["is_provided"] else None,
            "scores": scores_str,
            "score_semantics": score_semantics,
            "inference_time_seconds": row["inference_time_seconds"],
            "error_status": error_status,
            "manual_error_type": "",
            "reviewer_notes": "",
        })

    return pd.DataFrame(records)


def export_aligned_predictions_dataframe(
    df_aligned: pd.DataFrame,
    score_semantics: Optional[str] = None,
) -> pd.DataFrame:
    """Format the complete aligned predictions table for export to predictions_aligned.csv."""
    if len(df_aligned) == 0:
        return pd.DataFrame()

    out_df = df_aligned.copy()
    if "scores" in out_df.columns:
        out_df["scores"] = out_df["scores"].apply(lambda s: json.dumps(s) if s is not None else None)
    out_df["score_semantics"] = score_semantics
    return out_df

