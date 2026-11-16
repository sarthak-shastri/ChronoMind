# ReVA Universal Evaluation Framework — Role 4 Evaluation Guide
**Project**: ChronoMind — MAVU Workshop 2027 ReVA Drone Video Question Answering Challenge  
**Role**: Role 4 — Evaluation, Optimization and Competition Submission  
**Author**: Shubham (Branch: `shubham`)  
**Version**: 1.1.0  
**Status**: Authoritative Documentation  

---

## 1. Introduction and Architectural Overview

The **ReVA Universal Evaluation Framework** (`reva-eval`) is a model-agnostic normalization, validation, diagnostic, and evaluation system engineered for the **ReVA Drone Video Question Answering challenge** at the MAVU Workshop 2027.

The framework provides a mathematically rigorous, reproducible benchmarking harness that decouples model inference from evaluation. Any model developed by team members—including baseline models (BIMBA), advanced reasoning models, or external baselines—can be evaluated under identical, immutable criteria without requiring GPU hardware, model weights, or video decoding.

### Core Architectural Principles:
1. **Deterministic Alignment by Immutable ID**: Predictions and ground-truth questions are joined strictly on `sample_id` (or `qa_id`). Prediction ordering has zero impact on evaluation metrics.
2. **Strict Ground-Truth Denominator**: Accuracy is strictly computed over the entire ground-truth sample count ($N$). Missing and invalid predictions are treated as incorrect ($0$ score) in overall accuracy.
3. **Unlabeled Test Protection**: For splits where answers are withheld (`correct_answer: ""`, e.g. `test.json`), accuracy is strictly set to `null` and status is set to `unlabeled_ground_truth`. Accuracy is never guessed or fabricated.
4. **Dynamic Taxonomy Discovery**: Category (4) and Task (11) metrics are dynamically discovered from annotation records, avoiding hardcoded assumptions.
5. **Multi-Format Input Support**: Natively loads prediction files in canonical JSON, generic JSON, line-delimited JSON (JSONL), and CSV.
6. **Separation of Official and Supplementary Metrics**: Official multiple-choice accuracy is maintained as the primary metric, while research ablation metrics (macro-averaged accuracy, Cohen's Kappa, choice distributions) are explicitly isolated in supplementary diagnostics.

---

## 2. Framework Pipeline Architecture

```text
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ Raw ReVA Annotations (val.json) │       │ Raw Model Predictions           │
│ (QA List / Legacy Videos Map)   │       │ (JSON, JSONL, or CSV)           │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 ▼ (prepare-ground-truth)                  ▼ (load_predictions)
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ Canonical Ground Truth JSON     │       │ Canonical Predictions JSON      │
│ (schemas/ground_truth.json)     │       │ (schemas/predictions.json)      │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    reva_eval.evaluator      │
                      └──────────────┬──────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │   Validation Engine     │             │      ID Aligner         │
    │  - Strict Mode          │             │  - Left join on GT IDs  │
    │  - Permissive Mode      │             │  - Order invariant      │
    │  - Probability Sums     │             │  - Diagnostic counters  │
    └────────────┬────────────┘             └────────────┬────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Metrics & Error Engine   │
                      │  - Primary Accuracy         │
                      │  - Category / Task Metrics  │
                      │  - 4x4 Confusion Matrix     │
                      │  - Error Pattern Analysis   │
                      │  - Supplementary Metrics    │
                      └──────────────┬──────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ Structured Artifacts    │             │ Experiment Registry     │
    │ - evaluation.json       │             │ - experiments/          │
    │ - evaluation_report.md  │             │   registry.jsonl        │
    │ - category_metrics.csv  │             │ - Run comparisons       │
    │ - task_metrics.csv      │             │ - Summary exports       │
    │ - incorrect_preds.csv   │             └─────────────────────────┘
    │ - error_patterns.json   │
    │ - confusion_matrix.csv  │
    │ - Visualizations (.png) │
    └─────────────────────────┘
```

---

## 3. Data Contracts and Formats

### 3.1 Ground Truth Formats

#### A. Raw Official ReVA Annotation (`val.json`, `train.json`, `test.json`)
The official dataset layout provides a top-level dictionary containing `metadata` and a `QA` list:
```json
{
  "metadata": {
    "total_videos": 379,
    "total_questions": 2000
  },
  "QA": [
    {
      "qa_id": "val_000001",
      "video_path": "Hawk_UAV/NJ/DJI_0157_d4_01.mp4",
      "subdir": "NJ",
      "dataset_name": "Hawk_UAV",
      "category": "Temporal Understanding",
      "task": "Temporal Grounding",
      "question": "At what point in the video does the road with moving vehicles first appear?",
      "options": {
        "A": "At 5.0 seconds",
        "B": "At 4.2 seconds",
        "C": "At 2.5 seconds",
        "D": "At 3.8 seconds"
      },
      "correct_answer": "B"
    }
  ]
}
```

#### B. Canonical Ground Truth Format (`schemas/ground_truth.schema.json`)
Normalized representation generated by `reva-eval prepare-ground-truth`:
```json
{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "source_annotation_file": "val.json",
  "labeled": true,
  "samples": [
    {
      "sample_id": "val_000001",
      "video_path": "Hawk_UAV/NJ/DJI_0157_d4_01.mp4",
      "dataset_name": "Hawk_UAV",
      "subdir": "NJ",
      "question": "At what point in the video does the road with moving vehicles first appear?",
      "options": {
        "A": "At 5.0 seconds",
        "B": "At 4.2 seconds",
        "C": "At 2.5 seconds",
        "D": "At 3.8 seconds"
      },
      "category": "Temporal Understanding",
      "task": "Temporal Grounding",
      "ground_truth": "B"
    }
  ]
}
```
*Note for Test Splits*: When `correct_answer: ""`, `ground_truth` is normalized to `null` and `labeled: false`.

---

### 3.2 Prediction Formats

#### A. Canonical JSON Format (`schemas/predictions.schema.json`)
```json
{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "model": {
    "name": "BIMBA",
    "version": "v1.0",
    "run_id": "EXP_001"
  },
  "score_semantics": "probabilities",
  "option_order": ["A", "B", "C", "D"],
  "predictions": [
    {
      "sample_id": "val_000001",
      "prediction": "B",
      "scores": {
        "A": 0.10,
        "B": 0.70,
        "C": 0.15,
        "D": 0.05
      },
      "inference_time_seconds": 1.42
    }
  ]
}
```

#### B. Line-Delimited JSON (JSONL / NDJSON)
Each line represents an independent JSON object:
```jsonl
{"sample_id": "val_000001", "prediction": "B", "scores": {"A": 0.1, "B": 0.7, "C": 0.15, "D": 0.05}, "inference_time_seconds": 1.42}
{"sample_id": "val_000002", "prediction": "A", "scores": {"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}, "inference_time_seconds": 1.15}
```

#### C. Comma-Separated Values (CSV)
Supported column headers:
- Identifier: `sample_id` or `qa_id`
- Prediction: `prediction` or `answer`
- Optional scores: `scores` (JSON string) or individual columns `score_a`, `score_b`, `score_c`, `score_d`
- Optional latency: `inference_time_seconds` or `latency`

```csv
sample_id,prediction,inference_time_seconds,scores
val_000001,B,1.42,"{""A"": 0.10, ""B"": 0.70, ""C"": 0.15, ""D"": 0.05}"
val_000002,A,1.15,"{""A"": 0.80, ""B"": 0.10, ""C"": 0.05, ""D"": 0.05}"
```

---

## 4. Answer Normalization and Validation

### 4.1 Answer Normalization
Raw model outputs often carry whitespace or case artifacts. The loader performs configurable, non-destructive normalization:
- **Unicode NFKC Normalization**: Resolves full-width characters (e.g. `\uff21` -> `A`) and ligatures.
- **Whitespace Stripping**: Strips ASCII whitespace, zero-width spaces (`\u200b`), and non-breaking spaces.
- **Case Folding**: Converts choice letters to uppercase (e.g. `"b"` -> `"B"`).
- **Audit Preservation**: The original unnormalized prediction is preserved in `raw_prediction` for error analysis.

### 4.2 Validation Modes: Strict vs Non-Strict

| Condition | Non-Strict (Permissive, Default) | Strict Mode (`--strict`) |
| :--- | :--- | :--- |
| **Duplicate Prediction IDs** | Fails by default; allowed only with `--allow-duplicates` | Fails immediately |
| **Missing Prediction IDs** | Recorded in diagnostics; penalized as incorrect in accuracy | Raises `ValueError` and halts |
| **Unexpected Prediction IDs** | Recorded in diagnostics; excluded from denominator | Raises `ValueError` and halts |
| **Invalid Choice Labels** | Recorded in diagnostics; penalized as incorrect in accuracy | Raises `ValueError` and halts |
| **Probability Sum Violations** | Recorded in warnings (or error if `--strict-probabilities`) | Raises `ValueError` and halts |
| **Empty Predictions File** | Generates 0.0% accuracy report with 100% missing count | Raises error or produces empty diagnostic |

---

## 5. Metrics Formulation

### 5.1 Official Competition Metrics
1. **Overall Multiple-Choice Accuracy**:
   $$\text{Accuracy} = \frac{\text{Correct Valid Predictions}}{N_{\text{total\_gt}}}$$
   Missing and invalid predictions are included in the denominator and assigned a score of 0.
2. **Prediction Coverage**:
   $$\text{Coverage}_{\text{pred}} = \frac{|\text{Provided IDs} \cap \text{GT IDs}|}{N_{\text{total\_gt}}}$$
3. **Valid Prediction Coverage**:
   $$\text{Coverage}_{\text{valid}} = \frac{|\text{Valid A-D Predictions} \cap \text{GT IDs}|}{N_{\text{total\_gt}}}$$

### 5.2 Dynamic Taxonomy Metrics
Accuracy is computed for each high-level category (4) and subtask (11):
$$\text{Accuracy}_{\text{group}} = \frac{\text{Correct Valid Predictions in Group}}{N_{\text{group\_gt}}}$$
Denominators strictly include all ground-truth samples within that group.

### 5.3 Supplementary Research Metrics
To assist in model ablation and error diagnosis, the framework computes supplementary metrics:
- **Macro-Averaged Category Accuracy**: Unweighted mean of category accuracies:
  $$\text{MacroAcc}_{\text{cat}} = \frac{1}{|C|} \sum_{c \in C} \text{Accuracy}_c$$
- **Macro-Averaged Task Accuracy**: Unweighted mean of task accuracies:
  $$\text{MacroAcc}_{\text{task}} = \frac{1}{|T|} \sum_{t \in T} \text{Accuracy}_t$$
- **Chance-Adjusted Accuracy (Cohen's Kappa)**: Chance adjustment with uniform prior $p_e = 0.25$:
  $$\kappa = \frac{\text{Accuracy} - 0.25}{1.0 - 0.25}$$
- **Choice Distribution Bias**: Empirical proportions of model predicted choices across $\{A, B, C, D\}$.

---

## 6. Generated Output Artifacts

For each evaluation run, the output directory contains 13 reproducible artifacts:

| Artifact File | Format | Description |
| :--- | :--- | :--- |
| `evaluation.json` | JSON | Authoritative machine-readable evaluation metrics conforming to `schemas/evaluation_report.schema.json`. |
| `evaluation_report.md` | Markdown | Comprehensive human-readable report with summaries, breakdowns, and diagnostic alerts. |
| `validation_issues.json` | JSON | Exhaustive validation diagnostics (missing IDs, unexpected IDs, duplicate counts). |
| `category_metrics.csv` | CSV | Category-by-category breakdown table (total, correct, incorrect, accuracy, coverage). |
| `task_metrics.csv` | CSV | Task-by-task breakdown table (total, correct, incorrect, accuracy, coverage). |
| `predictions_aligned.csv` | CSV | Full sample-level alignment table with ground truth, prediction, status, and latency. |
| `incorrect_predictions.csv` | CSV | Filtered failure cases (`incorrect`, `missing`, `invalid`) with editable review columns. |
| `error_patterns.json` | JSON | Diagnostic error pattern summary (confusion pairs, error rates per category/task). |
| `confusion_matrix.csv` | CSV | 4x4 matrix of ground-truth choice vs predicted choice. |
| `confusion_matrix.png` | PNG | 300 DPI heatmap with raw cell counts and row-normalized percentages. |
| `category_accuracy.png` | PNG | 300 DPI bar chart displaying accuracy across categories. |
| `task_accuracy.png` | PNG | 300 DPI horizontal bar chart displaying accuracy across all 11 subtasks. |
| `error_distribution.png` | PNG | 300 DPI visualization illustrating error status breakdown and category error rates. |

---

## 7. Experiment Tracking System

The framework includes a built-in local experiment tracking registry (`ExperimentTracker`).

### Logging an Experiment:
```bash
reva-eval evaluate \
  --ground-truth data/ground_truth/validation.json \
  --predictions data/predictions/bimba_run1.json \
  --output-dir reports/BIMBA/run1 \
  --track \
  --experiment-id EXP_BIMBA_001
```

### Viewing and Comparing Tracked Experiments:
```bash
# List all tracked experiments
reva-eval track --list

# Filter by split
reva-eval track --list --split validation

# Export experiment registry summary to Markdown or CSV
reva-eval track --export reports/experiment_summary.md
```

All runs are recorded in `experiments/registry.jsonl` with timestamps, model metadata, metrics, and report paths.

---

## 8. CLI Command Reference

### 8.1 `prepare-ground-truth`
Normalizes raw annotations into canonical ground truth format:
```bash
reva-eval prepare-ground-truth \
  --annotation data/annotations/val.json \
  --output data/ground_truth/validation.json \
  --split validation
```

### 8.2 `evaluate`
Evaluates prediction files against ground truth:
```bash
reva-eval evaluate \
  --ground-truth data/ground_truth/validation.json \
  --predictions data/predictions/model_predictions.csv \
  --output-dir reports/Model/validation \
  --format csv \
  --strict \
  --track
```

### 8.3 `compare`
Performs side-by-side comparative benchmarking across multiple models:
```bash
reva-eval compare \
  --ground-truth data/ground_truth/validation.json \
  --predictions BIMBA=reports/BIMBA/preds.json \
                Advanced=reports/Advanced/preds.json \
  --output-dir reports/comparison
```
Generates `model_comparison.csv`, `model_comparison.md`, and `model_comparison.png`.

---

## 9. Troubleshooting and Diagnostic Guide

| Symptom / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `ValueError: Prediction validation failed ... duplicate sample_id(s)` | Prediction file contains multiple predictions for the same question. | Deduplicate predictions in upstream inference, or use `--allow-duplicates` to evaluate the first occurrence. |
| `ValueError: Strict validation failure: Missing predictions ...` | Model omitted predictions for some ground-truth samples under `--strict`. | Ensure model inference runs over all sample IDs in ground truth, or run without `--strict` to penalize missing predictions in accuracy. |
| `ValueError: CSV is missing a sample identifier column` | CSV file headers do not include `sample_id`, `qa_id`, or `id`. | Rename the column to `sample_id` or `qa_id`. |
| `ImportError: No module named 'reva_eval'` | Package not installed in current Python environment. | Run `pip install -e Evaluation_Framework --no-build-isolation --no-deps`. |
| `Accuracy: null, status: unlabeled_ground_truth` | Evaluated against `test.json` where answers are withheld (`""`). | Expected behavior. Accuracy is never calculated on unlabeled splits. |
| `Probability sum violation` | Probabilities do not sum to 1.0 within tolerance. | Check softmax calibration or adjust tolerance using `--tolerance 0.001`. |

