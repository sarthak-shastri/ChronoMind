"""Lightweight local experiment tracking system for ReVA model evaluations.

Maintains a local JSONL registry of evaluation experiments, tracking model configurations,
reproducible metrics, latencies, and output report paths across runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import pandas as pd
from pydantic import BaseModel, Field

from reva_eval.models import EvaluationReport


class ExperimentRecord(BaseModel):
    """Metadata and metrics recorded for a single evaluation experiment run."""

    experiment_id: str = Field(..., description="Unique experiment run identifier")
    model_name: str = Field(..., description="Name of the evaluated model")
    model_version: Optional[str] = Field(None, description="Model/checkpoint version")
    run_id: Optional[str] = Field(None, description="Optional upstream run ID")
    dataset: str = Field("ReVA", description="Dataset identifier")
    split: str = Field(..., description="Evaluation split")
    timestamp_utc: str = Field(..., description="Evaluation timestamp in ISO 8601 UTC")
    status: str = Field(..., description="Evaluation completion status")
    total_samples: int = Field(..., ge=0)
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    prediction_coverage: float = Field(..., ge=0.0, le=1.0)
    valid_prediction_coverage: float = Field(..., ge=0.0, le=1.0)
    missing_predictions: int = Field(..., ge=0)
    invalid_predictions: int = Field(..., ge=0)
    unexpected_predictions: int = Field(..., ge=0)
    mean_latency_seconds: Optional[float] = None
    p95_latency_seconds: Optional[float] = None
    output_report_dir: Optional[str] = None
    validation_mode: str = "non-strict"
    config: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class ExperimentTracker:
    """Local, reproducible experiment registry for tracking and comparing runs."""

    def __init__(self, registry_dir: Union[str, Path] = "experiments") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.jsonl"

    def log_run(
        self,
        report: EvaluationReport,
        output_report_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> ExperimentRecord:
        """Log an evaluation report as an experiment record in the registry."""
        exp_id = experiment_id or report.experiment_id or f"EXP_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        all_notes = list(report.notes)
        if notes:
            all_notes.extend(notes)

        record = ExperimentRecord(
            experiment_id=exp_id,
            model_name=report.model.name,
            model_version=report.model.version,
            run_id=report.model.run_id,
            dataset=report.dataset,
            split=report.split,
            timestamp_utc=report.generated_at_utc,
            status=report.status,
            total_samples=report.metrics.total_samples,
            accuracy=report.metrics.accuracy,
            prediction_coverage=report.metrics.prediction_coverage,
            valid_prediction_coverage=report.metrics.valid_prediction_coverage,
            missing_predictions=report.metrics.missing_predictions,
            invalid_predictions=report.metrics.invalid_predictions,
            unexpected_predictions=report.metrics.unexpected_predictions,
            mean_latency_seconds=report.runtime.mean_seconds,
            p95_latency_seconds=report.runtime.p95_seconds,
            output_report_dir=str(output_report_dir) if output_report_dir else None,
            validation_mode=report.validation_mode,
            config=config or {},
            notes=all_notes,
        )

        # Append to registry.jsonl
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

        return record

    def list_runs(
        self,
        split: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[ExperimentRecord]:
        """List all tracked runs, optionally filtering by split or model name."""
        if not self.registry_file.is_file():
            return []

        runs: List[ExperimentRecord] = []
        with open(self.registry_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    rec = ExperimentRecord.model_validate(data)
                    if split and rec.split.lower() != split.lower():
                        continue
                    if model_name and rec.model_name.lower() != model_name.lower():
                        continue
                    runs.append(rec)
                except Exception:
                    continue
        return runs

    def get_run(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve a specific experiment record by ID."""
        for run in self.list_runs():
            if run.experiment_id == experiment_id:
                return run
        return None

    def compare_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate a comparison DataFrame across specified experiment IDs or all experiments."""
        runs = self.list_runs(split=split)
        if experiment_ids:
            runs = [r for r in runs if r.experiment_id in set(experiment_ids)]

        if not runs:
            return pd.DataFrame()

        rows = []
        for r in runs:
            rows.append({
                "experiment_id": r.experiment_id,
                "model_name": r.model_name,
                "version": r.model_version or "N/A",
                "split": r.split,
                "accuracy": r.accuracy,
                "valid_coverage": r.valid_prediction_coverage,
                "total_samples": r.total_samples,
                "missing": r.missing_predictions,
                "invalid": r.invalid_predictions,
                "mean_latency_s": r.mean_latency_seconds,
                "timestamp_utc": r.timestamp_utc,
                "report_dir": r.output_report_dir or "N/A",
            })
        return pd.DataFrame(rows)

    def export_summary(self, output_path: Union[str, Path]) -> Path:
        """Export all tracked experiments to a CSV or Markdown summary table."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.compare_runs()

        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# ReVA Experiment Registry Summary\n\n")
                if df.empty:
                    f.write("No experiments recorded.\n")
                else:
                    headers = [str(c) for c in df.columns]
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("| " + " | ".join([":---"] * len(headers)) + " |\n")
                    for _, row in df.iterrows():
                        f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
                    f.write("\n")
        return path
