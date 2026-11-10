# ReVA Evaluation Framework

Standalone evaluator for ReVA multiple-choice VideoQA predictions.

## Install

From this directory, install the package and optional test dependencies:

```powershell
python -m pip install -e ".[dev]"
```

## Input and evaluation

Use `reva-eval --help` for command options. The package normalizes ReVA `{"metadata": ..., "QA": [...]}` annotations and supported prediction JSON formats, aligns records by QA/sample ID, and emits evaluation metrics/reports. The public ReVA test annotation has blank answer labels; the evaluator marks it as unlabeled and must not report accuracy for it.

Annotation files are not bundled here. Point the evaluator at your local pinned dataset files. Before training, use ChronoMind's grouped training split and split-integrity checker; this evaluation package does not repair the official split overlap.

## Tests

```powershell
python -m pytest tests -q
```
