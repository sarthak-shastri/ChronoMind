# ANTIGRAVITY MASTER PROMPT — Implement ReVA Universal Evaluation Framework

You are a senior Python ML infrastructure engineer. Implement the project in the currently open workspace using the attached `ReVA_Universal_Evaluation_Framework_Specification.docx` (or `IMPLEMENTATION_SPEC.md`) as the authoritative technical specification.

## Mission — strict scope

Build ONLY the universal model-output normalization and evaluation framework for the ReVA drone-video multiple-choice VideoQA challenge.

Do not implement/train BIMBA, advanced model architectures, video inference, ensembling, calibration, optimization, or Codabench submission packaging. Do not change teammate model internals. The goal is that any teammate can export predictions in the canonical contract and the same evaluator will evaluate them.

## First actions — inspect, don't assume

1. Inspect the current workspace and preserve existing user work. Do not delete/reinitialize repositories.
2. Read the full implementation specification and all supplied examples/schemas.
3. Inspect the actual ReVA annotation JSON if it exists in the workspace. Determine whether it uses the current top-level `QA` list or a legacy `videos` mapping. Record the actual observed field names and sample values in `docs/DATASET_SCHEMA_AUDIT.md`.
4. Do not download the 29.9 GB video dataset just to build the evaluator. Annotation JSON is enough. Do not make network access a runtime dependency.
5. If the actual annotation file is not present, implement against the documented current `QA`-list format, add synthetic fixtures, and clearly record that real-data validation is pending. Do not pretend actual annotations were inspected.
6. Inspect the actual `category` and `task` values if annotations are available. Do not invent task names or mappings. Build group metrics dynamically from observed values.

## Required deliverables

Implement the full repository structure in the specification, including:
- package under `src/reva_eval/`;
- ground-truth loader/normalizer;
- versioned canonical prediction schema;
- generic adapter interface and at least one generic JSON adapter;
- robust validation and ID alignment;
- overall metrics;
- category and task metrics;
- confusion matrix;
- error analysis export;
- report generation;
- CLI commands;
- tests;
- README and developer documentation;
- example ground truth, prediction, and evaluation report JSON.

Use Python 3.10+ compatibility. Choose one validation strategy (dataclasses plus explicit validation OR Pydantic) and implement it consistently. Use type hints and clear exceptions. Keep inference/model dependencies out of this package.

## Canonical prediction format — do not change without explaining

Every model adapter must produce this format:

{
  "schema_version": "1.0",
  "dataset": "ReVA",
  "split": "validation",
  "model": {
    "name": "BIMBA",
    "version": null,
    "run_id": null
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

Rules:
- `sample_id` and `prediction` are required.
- prediction is one of A/B/C/D.
- preserve IDs exactly from source `qa_id`.
- Scores are optional; when present they are label-keyed A/B/C/D, not an ambiguous array.
- score_semantics is one of probabilities/logits/ranking_scores/unknown/null.
- Never silently softmax, normalize, reorder, or reinterpret scores.
- If converting a score list, adapter must receive explicit source option order and validate it.
- Keep ground truth out of prediction files.
- Reject duplicate prediction IDs by default.
- Report unexpected IDs and missing IDs.

Canonical ground truth format must preserve source QA information:
- qa_id -> sample_id
- video_path
- dataset_name
- subdir
- question
- options
- category
- task
- correct_answer -> ground_truth
Do not require a video_id; it is not guaranteed in the current dataset card's QA-list schema. Preserve video_path as the video reference. Test rows with empty correct_answer are unlabeled.

## Core correctness rules

- Align by `sample_id`, never by row order.
- For labeled validation: denominator is ALL unique ground-truth samples.
- Missing and invalid predictions count as incorrect in local accuracy.
- accuracy = correct / total_ground_truth_samples.
- prediction_coverage = predictions present for GT IDs / total GT.
- valid_prediction_coverage = valid A-D predictions for GT IDs / total GT.
- Report missing, invalid, unexpected, duplicate counts separately.
- Unexpected predictions are excluded from denominator.
- Duplicate IDs fail validation; do not silently select first/last.
- Category/task denominators include all ground-truth samples in each group, including missing/invalid predictions.
- For unlabeled test data: do not calculate accuracy, correct/incorrect, or category/task accuracy. Return status `unlabeled_ground_truth`; validation and coverage diagnostics may still run.
- Categories and tasks come from actual annotations. No invented mappings.
- Confusion matrix fixed label order A,B,C,D; missing/invalid are separate counts, not coerced into a label.
- Runtime stats only if valid timing values exist.
- Never fabricate results.

## Output artifacts

For each model/split, generate:
- evaluation.json (must conform to spec)
- category_metrics.csv
- task_metrics.csv
- predictions_aligned.csv
- incorrect_predictions.csv
- confusion_matrix.csv
- confusion_matrix.png
- category_accuracy.png
- task_accuracy.png
- evaluation_report.md
- validation_issues.json

For comparison, generate model_comparison.csv and model_comparison.md. Require same ground-truth split/sample IDs or clearly disclose differences; do not make unfair rankings.

## CLI

Implement usable commands equivalent to:

reva-eval prepare-ground-truth --annotation PATH --output PATH --split validation

reva-eval evaluate --ground-truth PATH --predictions PATH --output-dir PATH

reva-eval compare --ground-truth PATH --predictions BIMBA=PATH Advanced=PATH --output-dir PATH

If argparse is simpler, use it. Include `python -m reva_eval.cli` fallback. Errors must be clear and exit non-zero.

## Testing — mandatory

Create synthetic tests for:
- all correct;
- all wrong;
- mixed;
- missing prediction;
- invalid answer label;
- unexpected prediction ID;
- duplicate GT ID;
- duplicate prediction ID;
- empty prediction file/list;
- empty ground truth;
- reordered predictions;
- category/task denominator with missing prediction;
- unlabeled test split;
- score key validation and score semantics;
- absent timing;
- malformed JSON / missing required fields;
- end-to-end CLI output generation.

Run the full suite. Fix failures. Do not merely write tests; execute them and report actual results.

## Quality gates

1. Install package in editable mode in the current environment if possible.
2. Run formatting/lint only if configured; don't add tool sprawl.
3. Run `pytest -q`.
4. Run CLI on synthetic fixture and inspect every generated artifact.
5. Validate JSON by parsing it back.
6. Check output counts and accuracy by hand for the synthetic fixture.
7. Confirm there are no hardcoded subtask labels.
8. Confirm no BIMBA imports or GPU/model packages are required.
9. Confirm no dataset/videos/checkpoints/secrets are accidentally committed.
10. Update README with exact commands and expected synthetic outputs.
11. Provide a concise completion report listing files created, tests run, pass/fail counts, limitations, and any blockers.

## Safety and execution behavior

- Never overwrite or delete existing files without inspection and explicit need.
- Do not run destructive Git commands.
- Do not create fake competition results.
- Do not claim Codabench compliance; that is outside this milestone and the live competition's exact contract has not been verified.
- If a required input is missing, continue with synthetic fixtures and clearly state the limitation.
- Implement the working system end-to-end, not just a plan or pseudocode.
- Do not stop after scaffolding. Write code, run tests, fix errors, and produce actual reports.

Start by inspecting the workspace and reading the spec. Then implement incrementally, run the tests, and finish with a verified working local evaluation framework.
