"""Plotting and visualization generators for ReVA Evaluation reports."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union

# Set safe MPLCONFIGDIR and non-interactive backend for headless environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(
    raw_matrix: Optional[pd.DataFrame],
    norm_matrix: Optional[pd.DataFrame],
    output_path: Union[str, Path],
    model_name: str = "Model",
    dpi: int = 300,
) -> Optional[Path]:
    """Generate and save confusion matrix heatmap."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if raw_matrix is None or norm_matrix is None or raw_matrix.empty:
        # Create an informative placeholder image
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5,
            "Confusion Matrix Unavailable\n(Dataset is unlabeled or has 0 valid samples)",
            ha="center", va="center", fontsize=12, color="gray",
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_p, dpi=dpi)
        plt.close(fig)
        return out_p

    labels = list(raw_matrix.columns)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(norm_matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Accuracy", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")

    ax.set_xlabel("Predicted Choice", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Ground Truth Choice", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            raw_val = int(raw_matrix.iloc[i, j])
            norm_val = float(norm_matrix.iloc[i, j])
            text_color = "white" if norm_val > 0.5 else "black"
            cell_text = f"{raw_val}\n({norm_val:.1%})"
            ax.text(j, i, cell_text, ha="center", va="center", color=text_color, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_p, dpi=dpi)
    plt.close(fig)
    return out_p


def plot_category_accuracy(
    category_metrics: List[Dict[str, Any]],
    output_path: Union[str, Path],
    model_name: str = "Model",
    dpi: int = 300,
) -> Optional[Path]:
    """Generate and save bar chart of accuracy across categories."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not category_metrics:
        return None

    cats = [c["category"] for c in category_metrics]
    accs = [c["accuracy"] if c["accuracy"] is not None else 0.0 for c in category_metrics]
    totals = [c["total"] for c in category_metrics]
    has_labels = any(c["accuracy"] is not None for c in category_metrics)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(cats))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))

    bars = ax.barh(y_pos, accs, color=colors, height=0.55, edgecolor="black", linewidth=0.8)

    ax.set_yticks(y_pos)
    labels = [f"{cat} (N={n})" for cat, n in zip(cats, totals)]
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(f"Category Accuracy — {model_name}", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, acc in zip(bars, accs):
        if has_labels:
            ax.text(
                bar.get_width() + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}",
                va="center", fontsize=10, fontweight="bold",
            )
        else:
            ax.text(0.02, bar.get_y() + bar.get_height() / 2, "Unlabeled", va="center", color="gray")

    fig.tight_layout()
    fig.savefig(out_p, dpi=dpi)
    plt.close(fig)
    return out_p


def plot_task_accuracy(
    task_metrics: List[Dict[str, Any]],
    output_path: Union[str, Path],
    model_name: str = "Model",
    dpi: int = 300,
) -> Optional[Path]:
    """Generate and save horizontal bar chart of accuracy across tasks."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not task_metrics:
        return None

    tasks = [t["task"] for t in task_metrics]
    accs = [t["accuracy"] if t["accuracy"] is not None else 0.0 for t in task_metrics]
    totals = [t["total"] for t in task_metrics]
    has_labels = any(t["accuracy"] is not None for t in task_metrics)

    fig, ax = plt.subplots(figsize=(10, max(5, len(tasks) * 0.45)))
    y_pos = np.arange(len(tasks))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(tasks)))

    bars = ax.barh(y_pos, accs, color=colors, height=0.6, edgecolor="black", linewidth=0.7)

    ax.set_yticks(y_pos)
    labels = [f"{t} (N={n})" for t, n in zip(tasks, totals)]
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()  # Top-down order
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(f"Task Breakdown Accuracy — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, acc in zip(bars, accs):
        if has_labels:
            ax.text(
                bar.get_width() + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}",
                va="center", fontsize=9.5, fontweight="bold",
            )
        else:
            ax.text(0.02, bar.get_y() + bar.get_height() / 2, "Unlabeled", va="center", color="gray")

    fig.tight_layout()
    fig.savefig(out_p, dpi=dpi)
    plt.close(fig)
    return out_p


def plot_error_distribution(
    error_summary: Dict[str, Any],
    output_path: Union[str, Path],
    model_name: str = "Model",
    dpi: int = 300,
) -> Optional[Path]:
    """Generate a breakdown visualization of errors by status and category."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    status_counts = error_summary.get("error_breakdown", {})
    cat_rates = error_summary.get("category_error_rates", {})

    if not status_counts and not cat_rates:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Error Status Pie / Donut
    statuses = [k.capitalize() for k, v in status_counts.items() if v > 0]
    counts = [v for v in status_counts.values() if v > 0]
    if counts:
        colors = ["#e74c3c", "#f39c12", "#95a5a6"][:len(statuses)]
        ax1.pie(counts, labels=statuses, autopct="%1.1f%%", colors=colors, startangle=140,
                wedgeprops=dict(width=0.4, edgecolor="white"))
        ax1.set_title("Error Breakdown by Type", fontsize=11, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No Errors Recorded", ha="center", va="center", color="gray")
        ax1.set_axis_off()

    # 2. Category Error Rates
    if cat_rates:
        cats = list(cat_rates.keys())
        rates = [cat_rates[c]["error_rate"] for c in cats]
        y_pos = np.arange(len(cats))
        ax2.barh(y_pos, rates, color="#e67e22", height=0.55, edgecolor="black", linewidth=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(cats, fontsize=9.5)
        ax2.set_xlim(0.0, 1.05)
        ax2.set_xlabel("Error Rate", fontsize=10, fontweight="bold")
        ax2.set_title("Category Error Rate", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.6)
        for idx, r in enumerate(rates):
            ax2.text(r + 0.02, idx, f"{r:.1%}", va="center", fontsize=9, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No Category Data", ha="center", va="center", color="gray")
        ax2.set_axis_off()

    fig.suptitle(f"Error Diagnostic Distribution — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_p, dpi=dpi)
    plt.close(fig)
    return out_p


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Union[str, Path],
    metric_col: str = "accuracy",
    dpi: int = 300,
) -> Optional[Path]:
    """Generate a multi-model comparative bar chart."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if comparison_df.empty or "model_name" not in comparison_df.columns:
        return None

    valid_df = comparison_df.dropna(subset=[metric_col])
    if valid_df.empty:
        return None

    models = valid_df["model_name"].tolist()
    values = valid_df[metric_col].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4.5))
    x_pos = np.arange(len(models))
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(models)))

    bars = ax.bar(x_pos, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold", rotation=15 if len(models) > 3 else 0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_title(f"Model Comparison — {metric_col.replace('_', ' ').title()}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2%}" if metric_col == "accuracy" else f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_p, dpi=dpi)
    plt.close(fig)
    return out_p
