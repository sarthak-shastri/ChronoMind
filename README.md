# ReVA Universal Evaluation Framework

A model-agnostic normalization, validation, and evaluation framework for the **ReVA drone-video multiple-choice VideoQA challenge**.

This package provides an authoritative, reproducible evaluation harness so any teammate or model (BIMBA, advanced reasoning models, or baselines) can export predictions in a single versioned canonical format and be evaluated under identical, mathematically rigorous criteria.

---

## 1. Scope & Boundaries

- **In Scope**:
  - Ground truth loading and normalization from official ReVA annotation JSON formats.
  - Strict validation of canonical prediction contracts, answer labels (`A`, `B`, `C`, `D`), and option order.
  - Deterministic alignment by immutable sample ID (`sample_id`), never row position.
  - Overall accuracy, prediction coverage, valid prediction coverage, missing/invalid prediction accounting.
  - Dynamic category and task metrics derived directly from annotations.
  - Confusion matrix (4x4 fixed `A,B,C,D` order; missing/invalid separated).
  - Error analysis export (`incorrect_predictions.csv` with editable columns for manual review).
  - Headless visualization generation (`confusion_matrix.png`, `category_accuracy.png`, `task_accuracy.png`).
  - Comparative multi-model benchmarking (`reva-eval compare`).
  - Handling of unlabeled test sets (`status: "unlabeled_ground_truth"`).
- **Out of Scope (By Design)**:
  - Video decoding, frame extraction, or downloading large video files.
  - Model weights, training, fine-tuning, or inference execution.
  - Codabench packaging (the exact live competition platform submission contract has not been independently verified; this framework targets local evaluation).

---

## 2. Installation

Install the package in editable mode from the repository root:

```bash
# In an offline or sandboxed environment:
pip install -e . --no-build-isolation --no-deps

# Or in an unrestricted environment:
pip install -e .
```

Verify installation:
```bash
python -m reva_eval.cli --help
# Or directly:
reva-eval --help
```

---

## 3. Canonical Contracts

### 3.1 Canonical Prediction Contract (`schemas/predictions.schema.json`)
Every model adapter or exporter must produce predictions matching this exact format:

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

**Key Prediction Rules**:
- `sample_id` and `prediction` are strictly required for each record.
- `prediction` must be one of `"A"`, `"B"`, `"C"`, `"D"`.
- Scores are optional. When present, they must be label-keyed (`{"A": ..., "B": ..., "C": ..., "D": ...}`), eliminating answer-choice-order ambiguity.
- `score_semantics` must be one of `"probabilities"`, `"logits"`, `"ranking_scores"`, `"unknown"`, or `null`.
- Ground truth must not be included in prediction files.
- Duplicate sample IDs fail validation by default (override with `--allow-duplicates` for diagnostic inspection).

### 3.2 Canonical Ground Truth Contract (`schemas/ground_truth.schema.json`)

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

For test splits where `correct_answer` is empty (`""`), `ground_truth` is set to `null` and `labeled` is `false`.

---

## 4. CLI Usage & Commands

### 4.1 Normalize Ground Truth
Converts official annotations (`QA` list format or legacy `videos` dictionary) into canonical ground truth:

```bash
reva-eval prepare-ground-truth \
  --annotation data/annotations/val.json \
  --output data/ground_truth/validation.json \
  --split validation
```

### 4.2 Evaluate Model Predictions
Evaluates a canonical prediction file against ground truth and generates all reports, CSVs, and plots:

```bash
reva-eval evaluate \
  --ground-truth data/ground_truth/validation.json \
  --predictions data/predictions/bimba_val_predictions.json \
  --output-dir reports/BIMBA/validation
```

*CLI Options*:
- `--allow-duplicates`: Permit duplicate prediction IDs (evaluates first occurrence, records duplicates in diagnostics).
- `--strict-probabilities`: Raise an error if probability scores do not sum to 1.0 within tolerance.
- `--tolerance 0.0001`: Error margin for probability sum checks.
- `--no-plots`: Disable PNG visualization generation for headless speed.

### 4.3 Multi-Model Comparison
Compare multiple candidate models side-by-side:

```bash
reva-eval compare \
  --ground-truth data/ground_truth/validation.json \
  --predictions BIMBA=reports/BIMBA/validation/predictions.json \
                Advanced=reports/Advanced/validation/predictions.json \
  --output-dir reports/comparison
```

Generates:
- `model_comparison.csv`
- `model_comparison.md`

---

## 5. Generated Artifacts

For every evaluated model/split, the output directory contains:

| Artifact | Purpose |
| :--- | :--- |
| `evaluation.json` | Authoritative machine-readable evaluation metrics conforming to `schemas/evaluation_report.schema.json`. |
| `evaluation_report.md` | Human-readable markdown report with summary tables. |
| `category_metrics.csv` | Total, correct, incorrect, accuracy, and coverage per category. |
| `task_metrics.csv` | Total, correct, incorrect, accuracy, and coverage per task. |
| `predictions_aligned.csv` | Sample-by-sample alignment table with prediction, ground truth, status, and scores. |
| `incorrect_predictions.csv` | Filtered failure cases (`wrong`, `missing`, `invalid`) with review columns. |
| `confusion_matrix.csv` | 4x4 matrix of ground-truth choice vs predicted choice. |
| `confusion_matrix.png` | Heatmap visualization with cell counts and row-normalized percentages. |
| `category_accuracy.png` | Bar chart of accuracy across categories. |
| `task_accuracy.png` | Bar chart of accuracy across subtasks. |
| `validation_issues.json` | Detailed validation diagnostics (missing IDs, unexpected IDs, duplicates). |

---

## 6. Synthetic Fixture Walkthrough & Expected Output

The repository includes synthetic verification fixtures:
- `examples/sample_ground_truth.json` (4 samples across 4 categories: `A`, `B`, `D`, `C`)
- `examples/sample_predictions.json` (4 predictions: `A`, `C`, `D`, `C`)

Run evaluation on the synthetic fixture:
```bash
reva-eval evaluate \
  --ground-truth examples/sample_ground_truth.json \
  --predictions examples/sample_predictions.json \
  --output-dir reports/ExampleModel/validation
```

**Expected Results**:
- **Total Samples**: 4
- **Provided Predictions**: 4
- **Valid Predictions**: 4
- **Correct Predictions**: 3 (`valid_000001`: A==A, `valid_000003`: D==D, `valid_000004`: C==C)
- **Incorrect Predictions**: 1 (`valid_000002`: true B, predicted C)
- **Overall Accuracy**: **75.00%** (3 / 4)
- **Prediction Coverage**: 100.00%
- **Category Accuracies**:
  - `Factual Perception`: 100.00% (1/1)
  - `Temporal Understanding`: 0.00% (0/1)
  - `Spatial Reasoning`: 100.00% (1/1)
  - `Causal Reasoning`: 100.00% (1/1)
- **Mean Latency**: 1.200 seconds

---

## 7. Mathematical & Correctness Principles

1. **Alignment by ID**: Predictions and ground truth are linked exclusively via `sample_id`. Shuffling or reordering prediction rows produces identical metrics.
2. **Strict Denominators**: For labeled validation data, the denominator $N$ is the count of **all unique ground-truth samples**. Missing and invalid predictions count as incorrect ($0$ score) in accuracy:
   $$\text{Accuracy} = \frac{\text{Correct Valid Predictions}}{N}$$
3. **Coverage Accounting**:
   $$\text{Prediction Coverage} = \frac{\text{Provided Intersecting Predictions}}{N}$$
   $$\text{Valid Prediction Coverage} = \frac{\text{Valid A-D Predictions}}{N}$$
4. **Group Integrity**: Category and task denominators include all ground-truth samples in that group, including missing and invalid predictions.
5. **Unlabeled Test Protection**: On unlabeled test splits (`correct_answer: ""`), accuracy is never calculated or hallucinated. The evaluator returns status `unlabeled_ground_truth` and reports only coverage and validation diagnostics.

---

## 8. Running Tests

Execute the comprehensive test suite (41 tests):

```bash
python -m pytest -q
```

All tests pass without external network access or GPU requirements.
