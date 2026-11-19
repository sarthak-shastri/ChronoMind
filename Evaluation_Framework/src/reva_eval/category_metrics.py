"""Grouped evaluation metrics for categories and tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd


def compute_group_metrics(
    df_aligned: pd.DataFrame,
    group_col: str,
    is_labeled: bool,
) -> List[Dict[str, Any]]:
    """Compute breakdown metrics dynamically for any grouping column (e.g. 'category' or 'task').

    Rules:
    - Denominator includes ALL ground-truth samples in each group.
    - Missing and invalid predictions count as incorrect in local accuracy.
    - If is_labeled is False, correct, incorrect, and accuracy are None.
    """
    if len(df_aligned) == 0:
        return []

    results: List[Dict[str, Any]] = []
    # Sort groups deterministically
    unique_groups = sorted(df_aligned[group_col].dropna().unique().tolist())

    for grp in unique_groups:
        sub = df_aligned[df_aligned[group_col] == grp]
        total = len(sub)
        provided = int(sub["is_provided"].sum())
        valid_pred = int(sub["is_valid_prediction"].sum())
        missing = int((~sub["is_provided"]).sum())
        invalid = int((sub["is_provided"] & ~sub["is_valid_prediction"]).sum())
        coverage = round(provided / total, 6) if total > 0 else 0.0

        if is_labeled:
            correct = int((sub["is_correct"] == True).sum())
            incorrect = total - correct
            accuracy = round(correct / total, 6) if total > 0 else None
        else:
            correct = None
            incorrect = None
            accuracy = None

        item: Dict[str, Any] = {
            group_col: str(grp),
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "coverage": coverage,
            "missing": missing,
            "invalid": invalid,
        }

        # If grouping by task, also include associated category if consistent
        if group_col == "task" and "category" in sub.columns:
            cats = sub["category"].unique()
            if len(cats) == 1:
                item["category"] = str(cats[0])

        results.append(item)

    return results


def category_metrics_to_dataframe(cat_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert category metrics list to DataFrame for CSV export."""
    if not cat_metrics:
        return pd.DataFrame(columns=["category", "total", "correct", "incorrect", "accuracy", "coverage", "missing", "invalid"])
    return pd.DataFrame(cat_metrics)


def task_metrics_to_dataframe(task_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert task metrics list to DataFrame for CSV export."""
    if not task_metrics:
        cols = ["task", "category", "total", "correct", "incorrect", "accuracy", "coverage", "missing", "invalid"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(task_metrics)
    # Ensure preferred column order
    preferred = ["task", "category", "total", "correct", "incorrect", "accuracy", "coverage", "missing", "invalid"]
    existing_cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[existing_cols]

