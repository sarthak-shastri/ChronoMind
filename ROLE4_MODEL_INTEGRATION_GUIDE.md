# ReVA Universal Evaluation Framework — Role 4 Model Integration Guide
**Project**: ChronoMind — MAVU Workshop 2027 ReVA Drone Video Question Answering Challenge  
**Role**: Role 4 — Evaluation, Optimization and Competition Submission  
**Target Audience**: Teammates implementing models (Role 1, Role 2, Role 3)  
**Author**: Shubham (Branch: `shubham`)  
**Version**: 1.0.0  

---

## 1. Executive Summary

This guide defines the authoritative integration contract for teammates developing models for the ChronoMind project (such as BIMBA or advanced multimodal reasoning architectures).

By adhering to this standard interface, your model can be evaluated, benchmarked, and tracked by the evaluation harness with **zero modifications to your model's internal architecture** or the evaluation framework.

---

## 2. Integration Architecture

Models integrate with the evaluation framework via two alternative pathways:

```text
PATH A: Python API Integration (Recommended for full pipeline automation)
┌─────────────────────────────────┐
│ Subclass ReVAModelInterface     │
│ - load_checkpoint()             │
│ - predict_sample()              │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ model.evaluate_model(...)       │ ──► Generates canonical predictions and full reports
└─────────────────────────────────┘

PATH B: File-Based Integration (Recommended for standalone inference scripts)
┌─────────────────────────────────┐
│ Your Custom Inference Script    │
│ (PyTorch, Hugging Face, etc.)   │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Export Predictions File         │
│ (JSON, JSONL, or CSV)           │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ reva-eval evaluate ...          │ ──► Generates full evaluation artifacts
└─────────────────────────────────┘
```

---

## 3. Path A: Python API Integration (`ReVAModelInterface`)

The evaluation package provides an abstract base class `ReVAModelInterface` in `reva_eval.interfaces`.

### 3.1 Interface Contract Definition

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reva_eval.interfaces import ReVAModelInterface
from reva_eval.models import GroundTruthSample, PredictionRecord

class MyChronoMindModel(ReVAModelInterface):
    """Custom ReVA VideoQA model implementation."""

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """Load weights, tokenizers, and video encoders."""
        # 1. Load your model weights
        # self.model = load_weights(checkpoint_path)
        # self.device = device
        self.is_loaded = True

    def predict_sample(
        self,
        sample: GroundTruthSample,
        **kwargs: Any,
    ) -> PredictionRecord:
        """Execute inference on a single VideoQA question."""
        # Input sample fields available:
        # sample.sample_id   -> e.g. "val_000001"
        # sample.video_path  -> e.g. "Hawk_UAV/NJ/DJI_0157_d4_01.mp4"
        # sample.question    -> Question text string
        # sample.options     -> Dict {"A": "...", "B": "...", "C": "...", "D": "..."}
        # sample.category    -> Taxonomy category string
        # sample.task        -> Taxonomy task string

        # Execute inference:
        # predicted_label, probs, latency = self.infer(sample.video_path, sample.question, sample.options)

        return PredictionRecord(
            sample_id=sample.sample_id,
            prediction="B",  # Must be one of "A", "B", "C", "D"
            scores={
                "A": 0.10,
                "B": 0.70,
                "C": 0.15,
                "D": 0.05,
            },  # Optional: label-keyed probability or logit mapping
            inference_time_seconds=1.42,  # Optional latency in seconds
        )
```

### 3.2 Running Batch Evaluation from Python

Once implemented, evaluate your model with a single call:

```python
model = MyChronoMindModel(model_name="BIMBA_Base", model_version="v1.0", run_id="EXP_001")
model.load_checkpoint("checkpoints/bimba_checkpoint.pt", device="cuda")

# Generate predictions and evaluate end-to-end
report = model.evaluate_model(
    ground_truth="Evaluation_Framework/data/annotations/val.json",
    output_dir="reports/BIMBA_Base/validation",
    strict_mode=False,
    track_experiment=True,
)

print(f"Overall Accuracy: {report.metrics.accuracy:.2%}")
```

---

## 4. Path B: File-Based Integration

If you prefer to run inference independently via your own CLI or batch scripts, export your predictions to any of the supported formats.

### 4.1 Required Fields
For each sample, provide:
1. `sample_id` (or `qa_id`): Exact string matching the annotation `qa_id` (e.g. `val_000001`).
2. `prediction` (or `answer`): Predicted choice label. Must resolve to `"A"`, `"B"`, `"C"`, or `"D"`.
3. `scores` (optional): Confidence scores. If provided, must be label-keyed `{"A": ..., "B": ..., "C": ..., "D": ...}`.
4. `inference_time_seconds` (optional): Non-negative float representing per-sample latency.

### 4.2 Format Examples

#### Option 1: Canonical JSON
```json
{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "model": {
    "name": "BIMBA",
    "version": "1.0",
    "run_id": "RUN_001"
  },
  "score_semantics": "probabilities",
  "predictions": [
    {
      "sample_id": "val_000001",
      "prediction": "B",
      "scores": {"A": 0.1, "B": 0.7, "C": 0.15, "D": 0.05},
      "inference_time_seconds": 1.42
    }
  ]
}
```

#### Option 2: JSONL (NDJSON)
```jsonl
{"sample_id": "val_000001", "prediction": "B", "scores": {"A": 0.1, "B": 0.7, "C": 0.15, "D": 0.05}}
{"sample_id": "val_000002", "prediction": "A", "scores": {"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}}
```

#### Option 3: CSV
```csv
sample_id,prediction,inference_time_seconds,scores
val_000001,B,1.42,"{""A"": 0.1, ""B"": 0.7, ""C"": 0.15, ""D"": 0.05}"
val_000002,A,1.15,"{""A"": 0.8, ""B"": 0.1, ""C"": 0.05, ""D"": 0.05}"
```

### 4.3 Running Evaluation via CLI
```bash
reva-eval evaluate \
  --ground-truth Evaluation_Framework/data/annotations/val.json \
  --predictions outputs/bimba_val_predictions.csv \
  --output-dir reports/BIMBA/validation \
  --track \
  --experiment-id EXP_BIMBA_VAL_01
```

---

## 5. Synthetic Dummy Adapter for Pipeline Verification

For verification before model weights are available, the framework provides `DummyBaselineModelAdapter`:

```python
from reva_eval.interfaces import DummyBaselineModelAdapter

# Test your evaluation workflow with a mock adapter
dummy = DummyBaselineModelAdapter(model_name="MockVerificationModel")
dummy.load_checkpoint("dummy_weights.pt")

report = dummy.evaluate_model(
    ground_truth="Evaluation_Framework/data/annotations/val.json",
    output_dir="reports/VerificationRun",
    save_plots=True,
)
```

> [!WARNING]
> `DummyBaselineModelAdapter` is strictly for interface and pipeline testing. It generates deterministic synthetic predictions and must never be cited as an actual competition benchmark.

---

## 6. Model Evaluation Checklist for Teammates

Before submitting your predictions or integrating your checkpoint:
- [ ] Are all 2,000 question IDs from `val.json` present in your predictions?
- [ ] Are all prediction IDs unique (no duplicates)?
- [ ] Are all prediction values strictly `"A"`, `"B"`, `"C"`, or `"D"`?
- [ ] If probability scores are included, do they sum to 1.0 within tolerance?
- [ ] Is ground truth strictly excluded from your prediction file?
- [ ] Did you evaluate locally using `reva-eval evaluate` and inspect `evaluation_report.md`?
- [ ] Did you review `incorrect_predictions.csv` to analyze error cases?

