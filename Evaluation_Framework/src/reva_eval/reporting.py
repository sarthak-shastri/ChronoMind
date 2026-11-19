"""Reporting generators for JSON, Markdown, and model comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from reva_eval.models import (
    EvaluationReport,
    ModelMetadata,
    PrimaryMetrics,
    RuntimeStats,
    ScoreMetadata,
    ValidationSummary,
)


def create_evaluation_report(
    dataset: str,
    split: str,
    model_metadata: ModelMetadata,
    primary_metrics: PrimaryMetrics,
    category_metrics: List[Dict[str, Any]],
    task_metrics: List[Dict[str, Any]],
    runtime_stats: RuntimeStats,
    score_metadata: ScoreMetadata,
    validation_summary: ValidationSummary,
    is_labeled: bool,
    notes: Optional[List[str]] = None,
) -> EvaluationReport:
    """Instantiate a validated EvaluationReport structure."""
    status = "completed" if is_labeled else "unlabeled_ground_truth"

    all_notes = list(notes or [])
    if is_labeled:
        all_notes.append("Missing and invalid predictions count as incorrect in overall accuracy.")
    else:
        all_notes.append("Ground truth is unlabeled (e.g. test split). Accuracy metrics are omitted.")

    now_utc = datetime.now(timezone.utc).isoformat()

    return EvaluationReport(
        schema_version="1.0",
        dataset=dataset,
        split=split,
        model=model_metadata,
        status=status,
        ground_truth_labeled=is_labeled,
        metrics=primary_metrics,
        category_metrics=category_metrics,
        task_metrics=task_metrics,
        runtime=runtime_stats,
        score_metadata=score_metadata,
        validation=validation_summary,
        generated_at_utc=now_utc,
        notes=all_notes,
    )


def save_evaluation_json(report: EvaluationReport, output_path: Union[str, Path]) -> Path:
    """Save evaluation report to evaluation.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    return path


def generate_markdown_report(report: EvaluationReport, output_path: Union[str, Path]) -> Path:
    """Render a human-readable markdown evaluation report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = report.metrics
    lines: List[str] = [
        f"# ReVA Evaluation Report — {report.model.name}",
        "",
        f"- **Dataset**: {report.dataset}",
        f"- **Split**: {report.split}",
        f"- **Model**: `{report.model.name}`" + (f" (Version: {report.model.version})" if report.model.version else ""),
        f"- **Run ID**: `{report.model.run_id or 'N/A'}`",
        f"- **Status**: `{report.status}`",
        f"- **Ground Truth Labeled**: `{report.ground_truth_labeled}`",
        f"- **Generated At (UTC)**: `{report.generated_at_utc}`",
        "",
        "## 1. Overall Metrics Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Ground-Truth Samples** | {m.total_samples:,} |",
        f"| **Provided Predictions** | {m.provided_predictions:,} |",
        f"| **Valid Predictions (A-D)** | {m.valid_predictions:,} |",
    ]

    if report.ground_truth_labeled:
        acc_str = f"**{m.accuracy:.2%}** ({m.correct:,} / {m.total_samples:,})" if m.accuracy is not None else "N/A"
        lines.extend([
            f"| **Overall Accuracy** | {acc_str} |",
            f"| **Correct Predictions** | {m.correct:,} |",
            f"| **Incorrect Predictions** | {m.incorrect:,} |",
        ])
    else:
        lines.append("| **Overall Accuracy** | *Unlabeled Ground Truth (Omitted)* |")

    lines.extend([
        f"| **Prediction Coverage** | {m.prediction_coverage:.2%} |",
        f"| **Valid Prediction Coverage** | {m.valid_prediction_coverage:.2%} |",
        f"| **Missing Predictions** | {m.missing_predictions:,} |",
        f"| **Invalid Predictions** | {m.invalid_predictions:,} |",
        f"| **Unexpected Predictions** | {m.unexpected_predictions:,} |",
        f"| **Duplicate Prediction Occurrences** | {m.duplicate_prediction_occurrences:,} |",
        "",
    ])

    # Category Breakdown
    lines.extend([
        "## 2. Category Performance Breakdown",
        "",
        "| Category | Total | Correct | Accuracy | Coverage | Missing | Invalid |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for c in report.category_metrics:
        acc = f"{c['accuracy']:.2%}" if c["accuracy"] is not None else "N/A"
        cor = f"{c['correct']:,}" if c["correct"] is not None else "N/A"
        lines.append(
            f"| **{c['category']}** | {c['total']:,} | {cor} | {acc} | {c['coverage']:.2%} | {c.get('missing', 0):,} | {c.get('invalid', 0):,} |"
        )
    lines.append("")

    # Task Breakdown
    lines.extend([
        "## 3. Task Performance Breakdown",
        "",
        "| Task | Category | Total | Correct | Accuracy | Coverage |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for t in report.task_metrics:
        acc = f"{t['accuracy']:.2%}" if t["accuracy"] is not None else "N/A"
        cor = f"{t['correct']:,}" if t["correct"] is not None else "N/A"
        cat = t.get("category", "-")
        lines.append(
            f"| **{t['task']}** | {cat} | {t['total']:,} | {cor} | {acc} | {t['coverage']:.2%} |"
        )
    lines.append("")

    # Runtime & Confidence
    lines.extend([
        "## 4. Runtime Latency & Confidence Metadata",
        "",
        f"- **Timed Samples**: {report.runtime.timed_samples:,}",
    ])
    if report.runtime.timed_samples > 0:
        mean_s = f"{report.runtime.mean_seconds:.3f} s" if report.runtime.mean_seconds is not None else "N/A"
        med_s = f"{report.runtime.median_seconds:.3f} s" if report.runtime.median_seconds is not None else "N/A"
        p95_s = f"{report.runtime.p95_seconds:.3f} s" if report.runtime.p95_seconds is not None else "N/A"
        lines.extend([
            f"- **Mean Latency**: {mean_s}",
            f"- **Median Latency**: {med_s}",
            f"- **95th Percentile Latency**: {p95_s}",
        ])
    lines.extend([
        f"- **Score Semantics**: `{report.score_metadata.score_semantics or 'None'}`",
        f"- **Samples with Scores**: {report.score_metadata.samples_with_scores:,}",
        "",
        "## 5. Validation & Notes",
        "",
        f"- **Validation Errors**: {report.validation.errors}",
        f"- **Validation Warnings**: {report.validation.warnings}",
    ])
    for note in report.notes:
        lines.append(f"- *Note*: {note}")

    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def generate_model_comparison_report(
    reports: Dict[str, EvaluationReport],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Generate comparative CSV and Markdown reports for multiple evaluated models."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    splits = set()
    total_samples_set = set()

    for name, r in reports.items():
        splits.add(r.split)
        total_samples_set.add(r.metrics.total_samples)
        summary_rows.append({
            "model_name": name,
            "split": r.split,
            "total_samples": r.metrics.total_samples,
            "accuracy": r.metrics.accuracy,
            "correct": r.metrics.correct,
            "incorrect": r.metrics.incorrect,
            "prediction_coverage": r.metrics.prediction_coverage,
            "valid_prediction_coverage": r.metrics.valid_prediction_coverage,
            "missing_predictions": r.metrics.missing_predictions,
            "invalid_predictions": r.metrics.invalid_predictions,
            "mean_latency_seconds": r.runtime.mean_seconds,
            "p95_latency_seconds": r.runtime.p95_seconds,
        })

    df_comp = pd.DataFrame(summary_rows)
    csv_path = out_dir / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False)

    md_path = out_dir / "model_comparison.md"
    md_lines = [
        "# ReVA Model Performance Comparison",
        "",
        f"- **Evaluated Models**: {', '.join(f'`{k}`' for k in reports.keys())}",
        f"- **Splits**: {', '.join(splits)}",
    ]

    if len(total_samples_set) > 1:
        md_lines.append(
            "> [!WARNING]\n"
            "> **Ground Truth Sample Size Discrepancy**: Models were evaluated against differing numbers of ground truth samples! "
            "Ensure comparisons use identical splits and sample IDs for fair benchmarking.\n"
        )

    md_lines.extend([
        "",
        "## Overall Model Comparison",
        "",
        "| Model | Accuracy | Valid Coverage | Total Samples | Correct | Missing | Invalid | Mean Latency (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, row in df_comp.iterrows():
        acc = f"**{row['accuracy']:.2%}**" if pd.notna(row["accuracy"]) else "N/A"
        cov = f"{row['valid_prediction_coverage']:.2%}" if pd.notna(row["valid_prediction_coverage"]) else "N/A"
        lat = f"{row['mean_latency_seconds']:.3f}" if pd.notna(row["mean_latency_seconds"]) else "-"
        cor = f"{int(row['correct']):,}" if pd.notna(row['correct']) else "N/A"
        md_lines.append(
            f"| **{row['model_name']}** | {acc} | {cov} | {row['total_samples']:,} | {cor} | {row['missing_predictions']:,} | {row['invalid_predictions']:,} | {lat} |"
        )

    # Category Comparison Table
    all_categories = sorted(list({
        c["category"]
        for r in reports.values()
        for c in r.category_metrics
    }))

    if all_categories:
        md_lines.extend([
            "",
            "## Category Accuracy Comparison",
            "",
            "| Category | " + " | ".join(f"**{name}**" for name in reports.keys()) + " |",
            "| :--- | " + " | ".join(":---" for _ in reports.keys()) + " |",
        ])
        for cat in all_categories:
            vals = []
            for r in reports.values():
                cat_metric = next((c for c in r.category_metrics if c["category"] == cat), None)
                if cat_metric and cat_metric["accuracy"] is not None:
                    vals.append(f"{cat_metric['accuracy']:.2%}")
                else:
                    vals.append("-")
            md_lines.append(f"| **{cat}** | " + " | ".join(vals) + " |")

    md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return csv_path, md_path

