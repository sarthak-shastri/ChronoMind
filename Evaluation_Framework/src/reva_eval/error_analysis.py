"""Error analysis and diagnostic failure export."""

from __future__ import annotations

import json
from pathlib import Path
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


def analyze_error_patterns(
    df_aligned: pd.DataFrame,
    is_labeled: bool = True,
) -> Dict[str, Any]:
    """Perform diagnostic error pattern analysis on aligned evaluation results.

    Identifies:
    - Overall error counts and status breakdown (incorrect, missing, invalid)
    - Frequent confusion pairs (e.g. True 'B' -> Predicted 'C')
    - Error rates across categories and tasks
    - Most frequent incorrect answers
    """
    if df_aligned.empty or not is_labeled:
        return {
            "total_errors": 0,
            "error_breakdown": {"incorrect": 0, "missing": 0, "invalid": 0},
            "category_error_rates": {},
            "task_error_rates": {},
            "frequent_confusion_pairs": [],
            "most_frequent_incorrect_answers": {},
        }

    error_mask = df_aligned["status"].isin(["wrong", "missing", "invalid"])
    err_df = df_aligned[error_mask]
    total_errors = len(err_df)

    status_counts = {
        "incorrect": int((df_aligned["status"] == "wrong").sum()),
        "missing": int((df_aligned["status"] == "missing").sum()),
        "invalid": int((df_aligned["status"] == "invalid").sum()),
    }

    # Category error rates
    cat_errors: Dict[str, Any] = {}
    for cat in df_aligned["category"].dropna().unique():
        sub = df_aligned[df_aligned["category"] == cat]
        total = len(sub)
        errs = int((sub["status"].isin(["wrong", "missing", "invalid"])).sum())
        cat_errors[str(cat)] = {
            "total": total,
            "errors": errs,
            "error_rate": round(errs / total, 4) if total > 0 else 0.0,
        }

    # Task error rates
    task_errors: Dict[str, Any] = {}
    for t in df_aligned["task"].dropna().unique():
        sub = df_aligned[df_aligned["task"] == t]
        total = len(sub)
        errs = int((sub["status"].isin(["wrong", "missing", "invalid"])).sum())
        task_errors[str(t)] = {
            "total": total,
            "errors": errs,
            "error_rate": round(errs / total, 4) if total > 0 else 0.0,
        }

    # Confusion pairs among wrong predictions with valid labels
    wrong_valid = df_aligned[(df_aligned["status"] == "wrong") & df_aligned["is_valid_prediction"]]
    confusion_pairs = []
    if not wrong_valid.empty:
        pair_counts = (
            wrong_valid.groupby(["ground_truth", "prediction"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )
        for _, row in pair_counts.iterrows():
            confusion_pairs.append({
                "true": str(row["ground_truth"]),
                "predicted": str(row["prediction"]),
                "count": int(row["count"]),
            })

    # Most frequent incorrect answers
    incorrect_pred_counts = {}
    if not wrong_valid.empty:
        vc = wrong_valid["prediction"].value_counts()
        for k, v in vc.items():
            incorrect_pred_counts[str(k)] = int(v)

    return {
        "total_errors": total_errors,
        "error_breakdown": status_counts,
        "category_error_rates": cat_errors,
        "task_error_rates": task_errors,
        "frequent_confusion_pairs": confusion_pairs,
        "most_frequent_incorrect_answers": incorrect_pred_counts,
    }


def save_error_analysis_summary(
    summary: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save error patterns summary to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path

