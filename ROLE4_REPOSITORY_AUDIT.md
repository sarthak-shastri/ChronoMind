# Role 4 — Repository and Documentation Audit
**Project**: ChronoMind — MAVU Workshop 2027 ReVA Drone Video Question Answering Challenge  
**Role**: Role 4 — Evaluation, Optimization and Competition Submission  
**Author**: Shubham (Branch: `shubham`)  
**Audit Date**: 2026-11-16  
**Status**: Authoritative Baseline Established  

---

## 1. Executive Summary

This audit establishes the baseline of the ChronoMind repository for the **ReVA drone-video multiple-choice Video Question Answering (VideoQA)** challenge at the MAVU Workshop 2027.

The repository currently provides the foundational architecture for the **ReVA Universal Evaluation Framework** (`reva-eval`), consisting of:
1. Pydantic-based schemas and contracts for Ground Truth and Predictions.
2. Annotation loader and normalizer supporting the official ReVA annotation format.
3. ID-based left-join alignment and metrics computation engine.
4. Error analysis export, markdown/JSON reporting, and headless matplotlib visualization.
5. Pytest test suite with 41 existing passing unit and integration tests.

Other teammates' model implementations (such as BIMBA or advanced reasoning models), model weights/checkpoints, and actual prediction outputs are not yet present in this workspace. In accordance with the project directives, all work proceeds independently, without fabricating model outputs, metrics, or competition contracts.

---

## 2. Repository Directory Structure

```text
ChronoMind/
├── .git/                                     # Git tracking repository (Branch: shubham)
├── README.md                                 # Universal Evaluation Framework documentation
├── Evaluation_Framework/
│   ├── .gitignore                            # Ignored paths (venv, data, reports, cache)
│   ├── ANTIGRAVITY_MASTER_PROMPT.md          # Architectural and operational guidance
│   ├── ANTIGRAVITY_MASTER_PROMPT.txt         # Text duplicate of master prompt
│   ├── IMPLEMENTATION_SPEC.md                # 506-line detailed implementation specification
│   ├── pyproject.toml                        # Package definition (reva-eval v1.0.0, Python >=3.10)
│   ├── ReVA_Universal_Evaluation_Framework_Specification.docx # Document specification
│   ├── configs/
│   │   └── default.yaml                      # Configuration defaults (validation, metrics, reporting)
│   ├── data/
│   │   └── annotations/                      # Real challenge annotation splits
│   │       ├── val.json                      # Validation set (2,000 QA pairs, perfectly balanced)
│   │       ├── test.json                     # Test set (4,000 QA pairs, unlabeled correct_answer: "")
│   │       └── train.json                    # Train set (15,773 QA pairs)
│   ├── docs/
│   │   └── DATASET_SCHEMA_AUDIT.md           # Schema audit of val.json, test.json, train.json
│   ├── examples/
│   │   ├── sample_evaluation_report.json     # Sample output schema report
│   │   ├── sample_ground_truth.json          # 4-sample synthetic ground truth
│   │   ├── sample_predictions.json           # 4-sample synthetic predictions
│   │   └── sample_predictions_advanced.json  # 4-sample predictions with logits/scores
│   ├── schemas/
│   │   ├── evaluation_report.schema.json     # JSON Schema for evaluation reports
│   │   ├── ground_truth.schema.json          # JSON Schema for canonical ground truth
│   │   └── predictions.schema.json           # JSON Schema for canonical predictions
│   ├── src/
│   │   └── reva_eval/
│   │       ├── __init__.py                   # Package exports
│   │       ├── models.py                     # Pydantic v2 domain models
│   │       ├── dataset_loader.py             # Annotation parser and normalizer
│   │       ├── validation.py                 # Ground truth & prediction validation engine
│   │       ├── metrics.py                    # Primary metrics, alignment, confusion matrix
│   │       ├── category_metrics.py           # Dynamic category and task metrics
│   │       ├── error_analysis.py             # CSV error exports (incorrect_predictions.csv)
│   │       ├── visualization.py              # Matplotlib headless plot generators
│   │       ├── reporting.py                  # JSON, Markdown, and model comparison reporting
│   │       ├── evaluator.py                  # High-level evaluation orchestration function
│   │       ├── cli.py                        # Command line interface (prepare, evaluate, compare)
│   │       └── adapters/
│   │           ├── __init__.py               # Adapter exports
│   │           ├── base.py                   # Abstract BaseAdapter interface
│   │           └── generic_json.py           # Multi-shape JSON input adapter
│   └── tests/
│       ├── fixtures/
│       │   ├── __init__.py
│       │   └── synthetic_fixtures.py         # Deterministic synthetic generators
│       ├── test_category_metrics.py          # Category and task calculation tests
│       ├── test_dataset_loader.py            # Raw annotation parsing & normalization tests
│       ├── test_end_to_end.py                # CLI commands and 10-artifact generation tests
│       ├── test_error_analysis.py            # Error CSV and aligned table tests
│       ├── test_metrics.py                   # Primary metrics and mathematical accuracy tests
│       ├── test_real_annotations.py          # Validation against actual data/annotations/val.json
│       ├── test_reporting.py                 # Markdown, JSON, and comparison report tests
│       └── test_validation.py                # Prediction schema, ID mismatch, and duplicate tests
```

---

## 3. Module-by-Module Description

| Module Path | Primary Responsibility | Key Classes / Functions |
| :--- | :--- | :--- |
| `src/reva_eval/models.py` | Type definitions and validation models using Pydantic v2. | `ModelMetadata`, `PredictionRecord`, `CanonicalPredictions`, `GroundTruthSample`, `CanonicalGroundTruth`, `PrimaryMetrics`, `GroupMetric`, `RuntimeStats`, `ScoreMetadata`, `ValidationSummary`, `EvaluationReport`. |
| `src/reva_eval/dataset_loader.py` | Parsing raw annotations (`QA` list, legacy `videos` dictionary) into canonical ground truth. | `load_raw_annotation_file`, `normalize_annotation`, `save_canonical_ground_truth`, `load_canonical_ground_truth`. |
| `src/reva_eval/validation.py` | Strict validation of prediction contracts, scores, duplicate IDs, missing IDs, and alignment. | `ValidationResult`, `validate_ground_truth`, `validate_predictions_against_ground_truth`. |
| `src/reva_eval/metrics.py` | Deterministic sample-by-sample alignment by `sample_id` (left join from GT), primary accuracy, confusion matrix. | `align_predictions_with_ground_truth`, `compute_primary_metrics`, `compute_confusion_matrix`, `compute_runtime_stats`, `compute_score_metadata`. |
| `src/reva_eval/category_metrics.py` | Dynamic category and task metric computations; DataFrame exports. | `compute_group_metrics`, `category_metrics_to_dataframe`, `task_metrics_to_dataframe`. |
| `src/reva_eval/error_analysis.py` | Diagnostic exports of failure cases (`wrong`, `missing`, `invalid`) with manual review columns. | `generate_incorrect_predictions_dataframe`, `export_aligned_predictions_dataframe`. |
| `src/reva_eval/visualization.py` | Non-interactive headless matplotlib rendering of confusion matrix, category bar chart, task bar chart. | `plot_confusion_matrix`, `plot_category_accuracy`, `plot_task_accuracy`. |
| `src/reva_eval/reporting.py` | Comprehensive report compilation: `evaluation.json`, `evaluation_report.md`, and multi-model comparisons. | `create_evaluation_report`, `save_evaluation_json`, `generate_markdown_report`, `generate_model_comparison_report`. |
| `src/reva_eval/evaluator.py` | Master evaluation pipeline orchestrator connecting loader, validator, aligner, metrics, and exporters. | `evaluate()`. |
| `src/reva_eval/cli.py` | Command line entrypoint providing subcommands `prepare-ground-truth`, `evaluate`, and `compare`. | `main()`, `build_parser()`. |
| `src/reva_eval/adapters/base.py` | Abstract base class defining conversion contract for raw model outputs. | `BaseAdapter`. |
| `src/reva_eval/adapters/generic_json.py` | Format adapter converting diverse JSON shapes (dicts, lists, wrappers) to canonical predictions. | `GenericJSONAdapter`. |

---

## 4. Current Evaluation Workflow

The existing workflow executes as follows:
```text
[Official ReVA Annotation JSON (val.json)]
                   │
                   ▼  (reva-eval prepare-ground-truth)
       [Canonical Ground Truth JSON]
                   │
                   ├───────────────────────────────────┐
                   ▼                                   ▼
        [Canonical Predictions JSON]         [Raw Predictions File]
                   │                                   │
                   │                        (GenericJSONAdapter)
                   │                                   │
                   ▼ ◄─────────────────────────────────┘
        [Evaluator (reva_eval.evaluator.evaluate)]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   [Validation]       [ID Alignment (sample_id)]
         │                   │
         └─────────┬─────────┘
                   ▼
       [Metrics & Diagnostics Calculation]
                   │
                   ▼
        [Artifact Generation in reports/<model>/<split>/]
         ├── evaluation.json
         ├── evaluation_report.md
         ├── validation_issues.json
         ├── category_metrics.csv
         ├── task_metrics.csv
         ├── predictions_aligned.csv
         ├── incorrect_predictions.csv
         ├── confusion_matrix.csv
         ├── confusion_matrix.png
         ├── category_accuracy.png
         └── task_accuracy.png
```

---

## 5. Supported Metrics and Mathematical Formulations

1. **Overall Multiple-Choice Accuracy**:
   $$\text{Accuracy} = \frac{\text{Correct Valid Predictions}}{N}$$
   where $N$ is the count of all unique ground-truth samples in the split. Missing and invalid predictions count as incorrect ($0$ score) in local accuracy.
2. **Prediction Coverage**:
   $$\text{Coverage}_{\text{pred}} = \frac{\text{Provided Intersecting Predictions}}{N}$$
3. **Valid Prediction Coverage**:
   $$\text{Coverage}_{\text{valid}} = \frac{\text{Valid A-D Predictions}}{N}$$
4. **Group Breakdown Metrics (Categories & Tasks)**:
   $$\text{Accuracy}_{\text{group}} = \frac{\text{Correct Valid Predictions in Group}}{N_{\text{group}}}$$
5. **Confusion Matrix (4x4 fixed A, B, C, D)**:
   - Raw count matrix $C_{i,j}$ where row $i \in \{A, B, C, D\}$ represents the true ground truth label and column $j \in \{A, B, C, D\}$ represents the predicted label.
   - Row-normalized percentage matrix: $P_{i,j} = \frac{C_{i,j}}{\sum_{k} C_{i,k}}$.
   - Missing and invalid predictions are explicitly tracked in separate counters and not injected into arbitrary classes.
6. **Inference Latency Statistics**:
   - Mean, Median, and 95th-percentile (p95) latency across timed samples.
7. **Unlabeled Split Protection**:
   - For splits with unlabeled ground truth (`correct_answer: ""`, e.g. `test.json`), accuracy is strictly set to `null` and status is set to `unlabeled_ground_truth`. Accuracy is never fabricated.

---

## 6. Supported Input and Output Formats

### 6.1 Input Formats
- **Canonical Predictions JSON**: Top-level object with `schema_version`, `dataset`, `split`, `model` metadata, optional `score_semantics`, `option_order`, and array of `PredictionRecord` (`sample_id`, `prediction`, optional `scores`, optional `inference_time_seconds`).
- **Raw JSON via `GenericJSONAdapter`**: Supports dictionaries with `predictions` or `results`, lists of prediction objects, and key-value mappings of `sample_id -> prediction`.
- **Annotation JSON**: Top-level object with `metadata` and `QA` list, or legacy `videos` dictionary.

### 6.2 Output Formats
- **Machine-Readable**: JSON (`evaluation.json`, `validation_issues.json`), CSV (`category_metrics.csv`, `task_metrics.csv`, `predictions_aligned.csv`, `incorrect_predictions.csv`, `confusion_matrix.csv`, `model_comparison.csv`).
- **Human-Readable**: Markdown (`evaluation_report.md`, `model_comparison.md`).
- **Visualizations**: 300 DPI headless PNGs (`confusion_matrix.png`, `category_accuracy.png`, `task_accuracy.png`).

---

## 7. Test Coverage Baseline

The existing test suite comprises **41 unit and integration tests** located in `Evaluation_Framework/tests`:

- `test_category_metrics.py`: 2 tests (dynamic grouping, empty dataset handling).
- `test_dataset_loader.py`: 5 tests (QA format, legacy format, unlabeled test split, duplicate sample IDs in GT, missing ID handling).
- `test_end_to_end.py`: 5 tests (CLI prepare ground truth, full 10-artifact generation, model comparison command, raw adapter integration, invalid CLI arguments).
- `test_error_analysis.py`: 2 tests (error dataframe filtering, aligned predictions dataframe export).
- `test_metrics.py`: 12 tests (all-correct accuracy 1.0, all-wrong accuracy 0.0, mixed predictions, ID ordering invariance, missing predictions denominator penalty, invalid prediction labels, empty predictions, empty GT, confusion matrix math, runtime latency stats, score metadata).
- `test_real_annotations.py`: 2 tests (real `val.json` parsing and taxonomy verification, end-to-end evaluation with synthetic predictions on real `val.json`).
- `test_reporting.py`: 5 tests (report generation, markdown rendering, model comparison CSV/MD, ground truth sample discrepancy warning, unlabeled test markdown).
- `test_validation.py`: 8 tests (canonical predictions schema, duplicate prediction rejection, `--allow-duplicates` bypass, probability sum validation, probability tolerance, unexpected prediction IDs, missing prediction IDs, invalid choice labels).

**Execution result**: 41 passed in 3.26 seconds. Zero failures.

---

## 8. Strengths of Current Implementation

1. **Mathematical Rigor**: Evaluates accuracy strictly with $N = \text{total ground truth}$, properly penalizing missing and invalid predictions.
2. **Deterministic Alignment**: Joined exclusively on immutable `sample_id`, completely order-independent.
3. **Pydantic Validation**: Uses Pydantic v2 schemas for strong data modeling, field checking, and serialization.
4. **Unlabeled Data Safety**: Explicitly prohibits calculating or hallucinating accuracy on test splits where answers are withheld.
5. **Dynamic Taxonomy**: Derives categories (4) and tasks (11) dynamically from annotation records rather than hardcoding static lists.
6. **Headless Safe**: Configures matplotlib to use non-interactive `Agg` backend and custom temporary `MPLCONFIGDIR`.

---

## 9. Identified Weaknesses and Opportunities for Improvement

1. **Prediction File Formats Limited in Evaluator**:
   - `evaluator.py` only accepts canonical JSON files. If a user provides a JSONL (`.jsonl`) or CSV (`.csv`) file, or a raw unadapted JSON, `evaluator.py` fails with a JSON decode error or validation error.
   - Prediction loading should natively support JSON, JSONL, and CSV, with automatic detection or format specification.
2. **Missing Validation Strictness Controls**:
   - The framework has hardcoded behavior for errors vs warnings. It lacks a centralized strict vs non-strict validation mode configuration.
   - In strict mode, any data anomaly (unexpected IDs, invalid labels, missing predictions) should be raised or fail cleanly.
   - In non-strict mode, invalid records should be flagged in warnings and excluded from numerator while maintaining denominator integrity, without silently discarding problematic records.
3. **Answer Normalization**:
   - Model predictions with lowercase labels (`"a"`, `"b"`) or leading/trailing whitespace/unicode whitespace are rejected as invalid labels unless manually cleaned.
   - Configurable normalization (unicode normalization NFKC, whitespace stripping, case conversion) should be available with raw predictions preserved for auditability.
4. **Supplementary Metrics**:
   - Beyond standard multiple-choice accuracy, model comparison benefits from macro-averaged category accuracy, macro-averaged task accuracy, chance-adjusted accuracy (Cohen's Kappa), and predicted class distribution / choice bias metrics.
   - These must be clearly designated as supplementary, not official competition scores.
5. **No Persistent Experiment Tracking**:
   - Currently, there is no persistent local experiment tracking registry. Model comparison requires passing all files simultaneously to `reva-eval compare`.
   - A lightweight experiment tracking registry (`ExperimentTracker`) is needed to record experiment runs, configurations, metrics, and timestamps, allowing incremental logging and comparison over time without heavy external dependencies.
6. **Visualizations**:
   - Lacks an error distribution visualization and a multi-model comparison chart.
7. **Model Integration Interface**:
   - Currently, only `BaseAdapter` exists for file conversion. There is no unified model evaluation interface (`ReVAModelInterface`) for future model pipelines (BIMBA, etc.) that defines how models load weights, run inference over ReVA batches, and output canonical predictions.

---

## 10. Missing Dependencies and Blocked Tasks

The following resources are currently unavailable in the local environment and represent external dependencies:
1. **BIMBA Codebase & Checkpoints**: Role 1 / Role 2 deliverable. Blocked until teammate integrates model weights.
2. **Trained Checkpoints of Advanced Models**: Role 3 deliverable. Blocked until teammate trains and publishes models.
3. **Live Codabench Submission Bundle**: Competition organizers deliverable. The exact Codabench scoring container contract has not been released.
4. **Drone Video Files**: Video files referenced in `video_path` (e.g. `VisDrone/uav...mp4`) are not downloaded locally. The evaluation framework is intentionally designed to evaluate annotations and prediction files without requiring video decoding.

---

## 11. Conclusion and Next Steps

The repository foundation is clean, modular, and mathematically sound. We will proceed to implement the identified enhancements incrementally:
1. Prediction loading: Add JSONL and CSV support, clean normalization.
2. Validation engine: Configurable strict / non-strict validation modes.
3. Metrics: Add clearly marked supplementary metrics (macro-average, Kappa, class bias).
4. Experiment tracking: Implement local `ExperimentTracker` with registry.
5. Visualization: Add error distribution and model comparison charts.
6. Model integration interface: Provide `ReVAModelInterface` and a synthetic `DummyBaselineModelAdapter`.
7. Comprehensive tests and documentation.

