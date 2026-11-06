# ReVA Universal Evaluation Framework — Implementation Specification

Version: 1.0  
Prepared: 6 November 2026  
Scope: Role 4 foundation only — model-output normalization, validation, and evaluation of multiple-choice ReVA VideoQA predictions.

## 1. Purpose and scope

Build a reusable Python package that evaluates prediction files from any ReVA VideoQA model (BIMBA, advanced model, or future models) against a single authoritative ground-truth annotation split.

The framework must be model-agnostic. Model-specific inference code is out of scope. Model-specific output formats are handled only by small adapters that convert raw outputs into the canonical prediction contract.

### In scope
- Inspect and load official ReVA annotation JSON.
- Normalize annotation records into canonical ground truth.
- Define and validate a versioned prediction contract.
- Convert raw model outputs via adapters.
- Align predictions and ground truth by immutable QA/sample ID, never by row order.
- Calculate overall, category, and task/subtask metrics.
- Detect duplicate, missing, extra, invalid, and malformed predictions.
- Export machine-readable and human-readable reports.
- Generate confusion matrix and category/task visualizations.
- Provide tests and a reproducible CLI.

### Explicitly out of scope for this milestone
- Training or fine-tuning BIMBA or any other model.
- Running video inference or decoding videos.
- Changing teammate model internals.
- Ensembling, calibration, routing, and inference optimization.
- Codabench packaging/submission adapter.
- Using hidden test labels or estimating test accuracy when labels are absent.

Do not expand scope without asking the project owner.

## 2. Research findings and authoritative facts

### 2.1 ReVA dataset format

The current Hugging Face dataset card describes a unified layout containing `train.json`, `valid.json`, and `test.json` (the repository listing may show `val.json`; inspect the actual downloaded file names rather than hardcoding one spelling). The annotation JSON is described as a top-level object with `metadata` and a `QA` list.

Common QA fields documented by the dataset card:
- `qa_id`: unique question identifier, e.g. `train_000001`.
- `video_path`: path relative to the dataset root.
- `subdir`, `dataset_name`.
- `category`: upper-level taxonomy.
- `task`: lower-level taxonomy.
- `question`.
- `options`: mapping with labels A, B, C, D.
- `correct_answer`: correct label; the released test annotation may contain an empty string.

Important: do not assume a `video_id` field exists in every current annotation version. Preserve `video_path` and derive a video grouping key only if needed, documenting the derivation. Do not treat the filename stem as an official ID unless the source confirms it.

The dataset card's examples and current repository metadata must be inspected at implementation time. The earlier `videos`-dictionary format and current `QA`-list format may differ across revisions. Implement an explicit schema adapter for the actual downloaded version; do not silently guess.

### 2.2 Dataset size and paper

The ReVA paper abstract reports 2,438 drone videos, 18 cities, approximately 580K frames, and 22K QA pairs over 11 QA tasks. These are full-dataset research-paper figures, not necessarily the exact size of the workshop challenge subset. Do not hardcode these counts as expected split sizes.

The workshop challenge page describes multiple-choice VideoQA on drone videos, identifies BIMBA as a baseline in associated challenge materials, and states overall multiple-choice accuracy as the primary metric with additional reasoning/task analysis. Confirm the exact challenge scoring instructions from the live Codabench phase/starter kit before claiming that a local report exactly reproduces the leaderboard score.

### 2.3 Taxonomy

The role brief specifies four main categories:
- Factual Perception
- Temporal Understanding
- Spatial Reasoning
- Causal Reasoning

It also requires metrics for 11 subtasks using actual annotation labels.

Implementation rule: read the actual `category` and `task` values from the annotation files. Build category/task reports dynamically from observed labels. Do not hardcode invented subtask names, mappings, or expected counts. If the source contains unexpected/blank taxonomy values, report them explicitly and do not silently map them.

### 2.4 BIMBA

The public BIMBA repository describes BIMBA as a long-range VideoQA multimodal model, provides inference/evaluation scripts and a released checkpoint. Its README describes Python 3.10 / Conda environment setup. This is relevant only to a future adapter; this evaluation package must not import BIMBA or require its heavyweight dependencies.

### 2.5 Codabench

General Codabench documentation distinguishes code submissions (ingestion plus scoring) from result/prediction submissions. This project's current scope is a local evaluation framework, not a Codabench submission package. Do not infer the competition's exact submission contract from generic Codabench documentation. That requires the live competition's own rules, starter kit, and phase configuration.

### 2.6 Open-source award requirement

The workshop page states that final award eligibility is limited to open-source-only models and top teams may be asked to submit code and weights for reproduction. This is not part of the evaluation framework implementation, but preserve model/checkpoint/source metadata in experiment/report metadata where available. Do not make legal conclusions from a repository being public.

## 3. Architecture

Pipeline:

1. Official annotation file
2. Ground-truth loader/normalizer
3. Raw model output file
4. Model-specific adapter (only format conversion)
5. Canonical prediction file
6. Validator and ID alignment
7. Universal evaluation engine
8. Reports: JSON, CSV, Markdown, PNG

The evaluator must never import model classes, load checkpoints, decode videos, or depend on BIMBA internals.

## 4. Repository structure

```text
reva-universal-evaluation/
├── README.md
├── pyproject.toml                 # or requirements.txt; choose one primary dependency source
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── ground_truth/              # local data; gitignored
│   └── predictions/               # local prediction files; gitignored
├── schemas/
│   ├── ground_truth.schema.json
│   ├── predictions.schema.json
│   └── evaluation_report.schema.json
├── src/
│   └── reva_eval/
│       ├── __init__.py
│       ├── models.py
│       ├── dataset_loader.py
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── generic_json.py
│       ├── validation.py
│       ├── metrics.py
│       ├── category_metrics.py
│       ├── error_analysis.py
│       ├── visualization.py
│       ├── reporting.py
│       ├── evaluator.py
│       └── cli.py
├── tests/
│   ├── fixtures/
│   ├── test_dataset_loader.py
│   ├── test_validation.py
│   ├── test_metrics.py
│   ├── test_category_metrics.py
│   ├── test_error_analysis.py
│   ├── test_reporting.py
│   └── test_end_to_end.py
├── examples/
│   ├── sample_ground_truth.json
│   └── sample_predictions.json
└── reports/                        # generated; gitignored
```

Use a `src` layout and install in editable mode (`pip install -e .`) so imports work consistently. Use pathlib for paths. Never commit full videos, checkpoints, secrets, or private test labels.

## 5. Canonical ground-truth contract

Use a versioned JSON object, not an unversioned loose array.

```json
{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "source_annotation_file": "valid.json",
  "samples": [
    {
      "sample_id": "valid_000001",
      "video_path": "VisDrone/example.mp4",
      "dataset_name": "VisDrone",
      "subdir": "VisDrone",
      "question": "Example question?",
      "options": {
        "A": "Choice A",
        "B": "Choice B",
        "C": "Choice C",
        "D": "Choice D"
      },
      "category": "Factual Perception",
      "task": "Actual annotation task label",
      "ground_truth": "B"
    }
  ]
}
```

Mapping from current documented dataset fields:
- `qa_id` -> `sample_id`
- `video_path` -> `video_path`
- `dataset_name` -> `dataset_name`
- `subdir` -> `subdir`
- `question` -> `question`
- `options` -> `options`
- `category` -> `category`
- `task` -> `task`
- `correct_answer` -> `ground_truth`

Keep `video_path` as the stable video reference. `video_id` may be an optional derived field only if an upstream pipeline supplies a stable ID; never require it for core evaluation.

For test split records where `correct_answer` is empty, set ground_truth to null and mark the split as unlabeled. The evaluator must refuse to calculate accuracy for unlabeled data and must not infer labels.

## 6. Canonical prediction contract (critical interface)

One prediction file per model and split. Use this exact top-level structure:

```json
{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "model": {
    "name": "BIMBA",
    "version": "optional model/checkpoint identifier",
    "run_id": "optional experiment/run identifier"
  },
  "score_semantics": "probabilities",
  "option_order": ["A", "B", "C", "D"],
  "predictions": [
    {
      "sample_id": "valid_000001",
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

Required:
- `schema_version`, `dataset`, `split`, `model.name`, `predictions`.
- Each prediction has `sample_id` and `prediction`.
- `prediction` must be one of A/B/C/D (case-normalize only if explicitly configured; otherwise reject invalid labels).
- Preserve sample IDs exactly; do not regenerate or reorder IDs.
- A missing score vector is allowed.
- Scores are optional and may be null. If present, require all four labels A/B/C/D exactly once.
- `score_semantics` must be one of `probabilities`, `logits`, `ranking_scores`, `unknown`, or null.
- Never call logits probabilities. Never normalize or softmax scores silently.
- Scores are represented as a label-keyed object, not an unlabelled list, to eliminate answer-choice-order ambiguity.
- If an adapter receives a list of scores, it must also know and validate the exact option order before mapping them to labels.
- `inference_time_seconds` is optional, numeric, finite, and non-negative.
- Extra model metadata may be stored under `model.metadata`, but do not put ground truth into prediction files.

Duplicate prediction IDs are invalid and should fail validation by default. Provide a deliberate `--allow-duplicates` diagnostic mode only if useful; never silently choose one duplicate.

Extra prediction IDs not present in ground truth must be reported as `unexpected_prediction_ids`. Do not count them in the denominator.

## 7. Validation and alignment rules

The evaluator must align by `sample_id` using a left join from ground truth. Never compare rows by position.

For a labeled split:
- `N = number of unique ground-truth samples`.
- `provided_count = number of unique prediction IDs that intersect ground truth` (after duplicate validation).
- `valid_count = number of intersecting predictions with valid A/B/C/D labels`.
- `correct_count = valid predictions matching ground_truth`.
- `incorrect_count = N - correct_count` (includes wrong, missing, and invalid predictions).
- `accuracy = correct_count / N`; if N=0, return null and an explicit empty-dataset status.
- `prediction_coverage = provided_count / N`.
- `valid_prediction_coverage = valid_count / N`.
- `missing_prediction_count = number of GT IDs with no prediction`.
- `invalid_prediction_count = number of GT IDs with a prediction but invalid answer label or malformed record`.
- `unexpected_prediction_count = prediction IDs absent from GT`.
- `duplicate_prediction_count = number of duplicate occurrences beyond the first; validation should fail unless diagnostic mode is enabled.
- Report all counts and ID lists (with configurable cap for report size).

The official overall accuracy is multiple-choice accuracy. For local framework consistency, missing/invalid predictions count as incorrect in the denominator. State this explicitly in the report; confirm exact competition scorer behavior from the live Codabench scoring program before claiming parity.

Ground truth must contain unique sample IDs and valid answer labels for labeled splits. For unlabeled test data, accuracy fields are null and status is `unlabeled_ground_truth`.

Validate:
- required keys and types;
- unique non-empty sample IDs;
- valid answer labels;
- `options` includes A/B/C/D when supplied;
- category/task non-empty for category reports (otherwise count as `Unknown` and report);
- no NaN/Infinity in scores/timing;
- scores have exactly A/B/C/D keys when present;
- probability scores are in [0,1] and sum to 1 within documented tolerance (default 1e-4); if not, validation warning/error according to strictness config;
- timing >= 0;
- split and dataset compatibility;
- model name consistency within a prediction file.

## 8. Metrics

### Primary
- total ground-truth samples
- correct
- incorrect (including missing/invalid)
- accuracy
- prediction coverage
- valid prediction coverage
- missing prediction count
- invalid prediction count
- unexpected prediction count
- duplicate prediction count

### Grouped
- accuracy by `category`
- accuracy by `task` (the actual annotation field; display as "task" or "subtask" in report)
- sample count, correct, incorrect, missing, invalid, coverage for each group
- calculate denominators from all ground-truth samples in each group, not just valid predictions

### Confusion matrix
Use fixed class order A, B, C, D. Include only valid predicted labels; missing/invalid should be separately counted and not silently added to an arbitrary class. Save raw counts and normalized matrix. Label axes clearly (true answer vs predicted answer).

### Optional descriptive metrics
- accuracy by dataset_name/subdir
- accuracy by answer label
- mean/median inference time and p95 where timing values exist
- score availability count
- confidence summaries only if scores are probabilities or a clearly defined confidence measure
Do not add unsupported claims or metrics that require video metadata not in the annotations.

## 9. Error analysis

Export one row per ground-truth sample that is incorrect, missing, or invalid. Include:
- sample_id
- video_path
- dataset_name
- subdir
- question
- options A-D
- category
- task
- ground_truth
- prediction (null if missing)
- scores
- score_semantics
- inference_time_seconds
- error_status: `incorrect`, `missing`, or `invalid`
- manual_error_type: null initially
- reviewer_notes: null initially

Do not automatically assign semantic failure labels such as counting or temporal reasoning based only on the category. Provide editable CSV columns for manual review. Optionally include a sampled video frame inspection link/path only if a separate upstream tool supplies it; video decoding is out of scope.

## 10. Output files and exact report JSON

For each model/split, write to `reports/<model_name>/<split>/`:
- `evaluation.json`
- `category_metrics.csv`
- `task_metrics.csv`
- `predictions_aligned.csv`
- `incorrect_predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `category_accuracy.png`
- `task_accuracy.png`
- `evaluation_report.md`
- `validation_issues.json`

Evaluation JSON example:

```json
{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "model": {
    "name": "BIMBA",
    "version": null,
    "run_id": null
  },
  "status": "completed",
  "ground_truth_labeled": true,
  "metrics": {
    "total_samples": 4,
    "provided_predictions": 4,
    "valid_predictions": 4,
    "correct": 3,
    "incorrect": 1,
    "accuracy": 0.75,
    "prediction_coverage": 1.0,
    "valid_prediction_coverage": 1.0,
    "missing_predictions": 0,
    "invalid_predictions": 0,
    "unexpected_predictions": 0,
    "duplicate_prediction_occurrences": 0
  },
  "category_metrics": [
    {
      "category": "Factual Perception",
      "total": 1,
      "correct": 1,
      "incorrect": 0,
      "accuracy": 1.0,
      "coverage": 1.0
    }
  ],
  "task_metrics": [],
  "runtime": {
    "timed_samples": 4,
    "mean_seconds": 1.2,
    "median_seconds": 1.1,
    "p95_seconds": 1.8
  },
  "score_metadata": {
    "score_semantics": "probabilities",
    "samples_with_scores": 4
  },
  "validation": {
    "errors": 0,
    "warnings": 0
  },
  "generated_at_utc": "ISO-8601 timestamp",
  "notes": [
    "Missing and invalid predictions count as incorrect in overall accuracy."
  ]
}
```

For unlabeled test data:
- `ground_truth_labeled` is false.
- `status` is `unlabeled_ground_truth`.
- accuracy, correct, incorrect, category metrics and task metrics must be null or omitted with an explicit reason.
- prediction format/ID/label validation and coverage diagnostics may still run.
- Never calculate or fabricate accuracy.

For multiple model comparison, write `reports/model_comparison.csv` and `.md`. Compare only the intersection of the same ground-truth sample IDs or, preferably, require identical split sample IDs and report any mismatch. Do not rank models evaluated on different sample sets as if directly comparable.

## 11. CLI requirements

Provide commands similar to:

```bash
# Convert official validation annotations to canonical ground truth
reva-eval prepare-ground-truth \
  --annotation data/raw/valid.json \
  --output data/ground_truth/validation.json \
  --split validation

# Evaluate a canonical prediction file
reva-eval evaluate \
  --ground-truth data/ground_truth/validation.json \
  --predictions data/predictions/bimba_validation.json \
  --output-dir reports/BIMBA/validation

# Compare model prediction files
reva-eval compare \
  --ground-truth data/ground_truth/validation.json \
  --predictions BIMBA=data/predictions/bimba_validation.json \
               Advanced=data/predictions/advanced_validation.json \
  --output-dir reports/comparison
```

If implementing a CLI library is too much, use argparse. CLI must provide helpful errors and non-zero exit codes for invalid input.

## 12. Tests and acceptance criteria

The implementation is complete only when all of these pass:

1. Load actual current ReVA `QA` list format.
2. If encountered, handle the legacy `videos` mapping only via an explicit adapter and tests; do not silently mix formats.
3. Ground-truth normalization preserves `qa_id`, video_path, options, category, task, and correct_answer.
4. Test split empty `correct_answer` is treated as unlabeled.
5. Canonical prediction JSON validates.
6. Score label order is unambiguous and score semantics are preserved.
7. Reordered prediction records produce identical metrics (ID-based join).
8. Duplicate ground-truth IDs fail.
9. Duplicate prediction IDs fail.
10. Missing prediction is counted and treated as incorrect for local accuracy.
11. Invalid prediction is counted and treated as incorrect.
12. Unexpected prediction IDs are reported and excluded from denominator.
13. All-correct synthetic test gives 1.0 accuracy.
14. All-wrong synthetic test gives 0.0 accuracy.
15. Empty prediction list does not crash; coverage 0 and accuracy 0 for labeled nonempty GT.
16. Empty ground truth returns null accuracy and explicit status.
17. Category and task metrics are derived from annotation values, not invented mappings.
18. Group denominators include all GT samples in the group.
19. Confusion matrix uses A/B/C/D fixed order and separates invalid/missing counts.
20. Unlabeled test split does not report accuracy.
21. Error CSV includes missing, invalid, and wrong predictions.
22. Runtime stats handle absent timings.
23. JSON output is valid and schema-conformant.
24. Full test suite passes with `pytest`.
25. README explains setup, schemas, CLI, metrics, limitations, and examples.
26. A single command can evaluate any compliant model prediction file without model-specific evaluator changes.

## 13. Engineering constraints

- Python 3.10+ recommended; avoid dependencies from BIMBA.
- Use dataclasses or Pydantic (choose one; do not create redundant validation layers).
- Prefer pandas/numpy/matplotlib/scikit-learn only where useful.
- Pin or constrain dependencies in pyproject.toml/requirements.
- Add type hints, docstrings, structured logging, and clear exceptions.
- Deterministic outputs except generated timestamps.
- Use UTF-8 JSON and `ensure_ascii=False`.
- Never silently overwrite existing reports unless `--overwrite` is explicit.
- Add `.gitignore` for venv, cache, raw dataset, videos, checkpoints, predictions, and reports as appropriate.
- Do not fabricate any performance result. Synthetic fixtures must be labeled synthetic.
- No network calls at evaluation runtime.
- Never upload private dataset files, videos, checkpoints, or secrets to Git.

## 14. Source references

1. MAVU workshop/challenge page: https://zyaocoder.github.io/MAVU-workshop-2027/#challenge
2. ReVA dataset card: https://huggingface.co/datasets/ReVA-Benchmark/ReVA
3. ReVA current README: https://huggingface.co/datasets/ReVA-Benchmark/ReVA/blob/main/README.md
4. ReVA paper: https://arxiv.org/abs/2609.35507
5. ReVA research repository: https://github.com/zyaocoder/ReVA
6. BIMBA repository: https://github.com/md-mohaiminul/BIMBA
7. Codabench competition: https://www.codabench.org/competitions/18274/#/pages-tab
8. Codabench competition design documentation: https://docs.codabench.org/dev/Organizers/Benchmark_Creation/Competition-Design/

Research caveats:
- The live Codabench competition page was not readable through public web retrieval during preparation. Exact competition submission contract and scorer behavior remain unverified.
- The paper and challenge pages describe different dataset scales because the challenge may use a subset; do not assume the challenge split equals the full research dataset.
- Dataset README revisions differ in file names and annotation layout. Inspect the downloaded data and support explicit versioned adapters.
