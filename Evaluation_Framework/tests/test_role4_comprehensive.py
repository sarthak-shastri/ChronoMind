"""Comprehensive automated test suite for Role 4 evaluation framework.

Covers all 20 required test scenarios specified in Role 4 contract:
1. Valid prediction files (JSON, JSONL, CSV)
2. Empty prediction files
3. Missing IDs
4. Duplicate IDs
5. Invalid answer types
6. Extra prediction records (unexpected IDs)
7. Missing prediction records
8. Unordered predictions
9. Ground-truth mismatch (dataset/split mismatch)
10. Empty answers
11. Unicode and whitespace handling
12. Invalid file formats
13. Empty datasets
14. Single-record datasets
15. Per-category evaluation
16. Report generation
17. Experiment tracking
18. Error-analysis exports
19. Strict validation mode
20. Non-strict validation mode
Plus Model Integration Interface tests (DummyBaselineModelAdapter).

All fixtures in this file are strictly synthetic and labeled as such.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from reva_eval.evaluator import evaluate
from reva_eval.experiment_tracker import ExperimentTracker
from reva_eval.interfaces import DummyBaselineModelAdapter, ReVAModelInterface
from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    GroundTruthSample,
    ModelMetadata,
    PredictionRecord,
)
from reva_eval.prediction_loader import (
    load_predictions,
    load_predictions_from_csv,
    load_predictions_from_json,
    load_predictions_from_jsonl,
    normalize_answer_string,
)
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


# ==============================================================================
# Case 1: Valid Prediction Files (JSON, JSONL, CSV)
# ==============================================================================

def test_case_01_valid_prediction_files_json_jsonl_csv(tmp_path: Path):
    """Test loading and evaluating valid prediction files in JSON, JSONL, and CSV formats."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")

    # 1. JSON
    json_path = tmp_path / "preds.json"
    json_path.write_text(json.dumps(preds.model_dump()), encoding="utf-8")
    loaded_json = load_predictions(json_path)
    assert len(loaded_json.predictions) == 4
    rep_json = evaluate(gt, json_path, output_dir=tmp_path / "out_json", save_plots=False)
    assert rep_json.metrics.accuracy == 1.0

    # 2. JSONL
    jsonl_path = tmp_path / "preds.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in preds.predictions:
            f.write(json.dumps(p.model_dump()) + "\n")
    loaded_jsonl = load_predictions(jsonl_path)
    assert len(loaded_jsonl.predictions) == 4
    rep_jsonl = evaluate(gt, jsonl_path, output_dir=tmp_path / "out_jsonl", save_plots=False)
    assert rep_jsonl.metrics.accuracy == 1.0

    # 3. CSV
    csv_path = tmp_path / "preds.csv"
    csv_content = (
        "sample_id,prediction,inference_time_seconds,scores\n"
        'synth_000001,A,1.0,"{""A"": 0.7, ""B"": 0.1, ""C"": 0.1, ""D"": 0.1}"\n'
        'synth_000002,B,1.1,"{""A"": 0.1, ""B"": 0.7, ""C"": 0.1, ""D"": 0.1}"\n'
        'synth_000003,C,1.2,"{""A"": 0.1, ""B"": 0.1, ""C"": 0.7, ""D"": 0.1}"\n'
        'synth_000004,D,1.3,"{""A"": 0.1, ""B"": 0.1, ""C"": 0.1, ""D"": 0.7}"\n'
    )
    csv_path.write_text(csv_content, encoding="utf-8")
    loaded_csv = load_predictions(csv_path)
    assert len(loaded_csv.predictions) == 4
    rep_csv = evaluate(gt, csv_path, output_dir=tmp_path / "out_csv", save_plots=False)
    assert rep_csv.metrics.accuracy == 1.0


# ==============================================================================
# Case 2: Empty Prediction Files
# ==============================================================================

def test_case_02_empty_prediction_files(tmp_path: Path):
    """Test evaluation behavior when prediction file is completely empty or has 0 records."""
    gt = make_synthetic_ground_truth(num_samples=4)
    empty_preds = make_synthetic_predictions(gt, prediction_mode="empty")

    rep = evaluate(gt, empty_preds, output_dir=tmp_path / "out_empty", save_plots=False)
    assert rep.metrics.total_samples == 4
    assert rep.metrics.provided_predictions == 0
    assert rep.metrics.valid_predictions == 0
    assert rep.metrics.correct == 0
    assert rep.metrics.incorrect == 4
    assert rep.metrics.accuracy == 0.0
    assert rep.metrics.prediction_coverage == 0.0
    assert rep.metrics.missing_predictions == 4


# ==============================================================================
# Case 3: Missing IDs in Predictions
# ==============================================================================

def test_case_03_missing_prediction_ids(tmp_path: Path):
    """Test handling of prediction records missing sample_id."""
    raw_json = [
        {"sample_id": "synth_000001", "prediction": "A"},
        {"prediction": "B"},  # Missing sample_id!
    ]
    p_path = tmp_path / "missing_ids.json"
    p_path.write_text(json.dumps(raw_json), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'sample_id'"):
        load_predictions(p_path)


# ==============================================================================
# Case 4: Duplicate IDs
# ==============================================================================

def test_case_04_duplicate_ids(tmp_path: Path):
    """Test that duplicate prediction IDs fail by default and are permitted in diagnostic mode."""
    gt = make_synthetic_ground_truth(num_samples=4)
    dup_preds = make_synthetic_predictions(gt, prediction_mode="duplicate_id")

    # Default fails
    with pytest.raises(ValueError, match="Duplicate IDs fail validation by default"):
        evaluate(gt, dup_preds, allow_duplicates=False)

    # Allowed in diagnostic mode
    rep = evaluate(gt, dup_preds, allow_duplicates=True, save_plots=False)
    assert rep.metrics.duplicate_prediction_occurrences >= 1
    assert any("Allowed by diagnostic mode" in w for w in rep.notes or [])


# ==============================================================================
# Case 5: Invalid Answer Types
# ==============================================================================

def test_case_05_invalid_answer_types(tmp_path: Path):
    """Test that invalid answer choices (non-A-D labels) are handled without crashing."""
    gt = make_synthetic_ground_truth(num_samples=4)
    invalid_preds = make_synthetic_predictions(gt, prediction_mode="invalid_label")  # 1st label is 'Z'

    rep = evaluate(gt, invalid_preds, output_dir=tmp_path / "out_invalid", save_plots=False)
    assert rep.metrics.valid_predictions == 3
    assert rep.metrics.invalid_predictions == 1
    # Invalid counts as incorrect in accuracy
    assert rep.metrics.correct == 3
    assert rep.metrics.incorrect == 1
    assert rep.metrics.accuracy == 0.75


# ==============================================================================
# Case 6: Extra Prediction Records (Unexpected IDs)
# ==============================================================================

def test_case_06_extra_prediction_records(tmp_path: Path):
    """Test that predictions with IDs not in ground truth are reported and excluded from denominator."""
    gt = make_synthetic_ground_truth(num_samples=4)
    extra_preds = make_synthetic_predictions(gt, prediction_mode="unexpected_id")

    rep = evaluate(gt, extra_preds, output_dir=tmp_path / "out_extra", save_plots=False)
    assert rep.metrics.total_samples == 4  # Denominator remains GT size
    assert rep.metrics.unexpected_predictions == 1
    assert rep.metrics.correct == 4
    assert rep.metrics.accuracy == 1.0


# ==============================================================================
# Case 7: Missing Prediction Records
# ==============================================================================

def test_case_07_missing_prediction_records(tmp_path: Path):
    """Test that GT samples with omitted predictions are counted as missing and penalize accuracy."""
    gt = make_synthetic_ground_truth(num_samples=4)
    missing_preds = make_synthetic_predictions(gt, prediction_mode="missing")  # 3 of 4 samples

    rep = evaluate(gt, missing_preds, output_dir=tmp_path / "out_missing", save_plots=False)
    assert rep.metrics.total_samples == 4
    assert rep.metrics.provided_predictions == 3
    assert rep.metrics.missing_predictions == 1
    assert rep.metrics.correct == 3
    assert rep.metrics.incorrect == 1
    assert rep.metrics.accuracy == 0.75
    assert rep.metrics.prediction_coverage == 0.75


# ==============================================================================
# Case 8: Unordered Predictions
# ==============================================================================

def test_case_08_unordered_predictions(tmp_path: Path):
    """Test that reordering or reversing prediction records produces identical metrics (ID join invariance)."""
    gt = make_synthetic_ground_truth(num_samples=4)
    norm_preds = make_synthetic_predictions(gt, prediction_mode="mixed")
    rev_preds = norm_preds.model_copy(deep=True)
    rev_preds.predictions = list(reversed(norm_preds.predictions))

    rep_norm = evaluate(gt, norm_preds, save_plots=False)
    rep_rev = evaluate(gt, rev_preds, save_plots=False)

    assert rep_norm.metrics.accuracy == rep_rev.metrics.accuracy
    assert rep_norm.metrics.correct == rep_rev.metrics.correct
    assert rep_norm.metrics.incorrect == rep_rev.metrics.incorrect
    assert rep_norm.category_metrics == rep_rev.category_metrics


# ==============================================================================
# Case 9: Ground-Truth Mismatch
# ==============================================================================

def test_case_09_ground_truth_mismatch(tmp_path: Path):
    """Test detection and warning on split or dataset name mismatches."""
    gt = make_synthetic_ground_truth(num_samples=4, split="validation")
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")
    preds.split = "test"  # Mismatch!

    rep = evaluate(gt, preds, save_plots=False)
    # Generates a split mismatch warning
    assert any("Split mismatch" in w for w in rep.notes or [])


# ==============================================================================
# Case 10: Empty Answers
# ==============================================================================

def test_case_10_empty_answers(tmp_path: Path):
    """Test that empty string answer '' or None is treated as an invalid prediction rather than crashing."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="all_correct")
    preds.predictions[0].prediction = ""  # Empty answer

    rep = evaluate(gt, preds, save_plots=False)
    assert rep.metrics.invalid_predictions == 1
    assert rep.metrics.valid_predictions == 3


# ==============================================================================
# Case 11: Unicode and Whitespace Handling
# ==============================================================================

def test_case_11_unicode_and_whitespace_handling(tmp_path: Path):
    """Test normalization of whitespace, lowercase, and unicode characters in predictions."""
    assert normalize_answer_string("  b  ", case_fold=True) == "B"
    assert normalize_answer_string("\u3000A\u200b", case_fold=True) == "A"
    assert normalize_answer_string("c", case_fold=True) == "C"
    assert normalize_answer_string(None) is None
    assert normalize_answer_string("   ") is None

    # In CSV with messy whitespace and lowercase
    csv_path = tmp_path / "messy.csv"
    csv_content = (
        "sample_id,prediction\n"
        "synth_000001,  a  \n"
        "synth_000002,\u3000b\u200b\n"
        "synth_000003,c\n"
        "synth_000004,D\n"
    )
    csv_path.write_text(csv_content, encoding="utf-8")
    loaded = load_predictions(csv_path, case_fold=True, strip_whitespace=True, normalize_unicode=True)
    labels = [p.prediction for p in loaded.predictions]
    assert labels == ["A", "B", "C", "D"]
    # Check raw_prediction preservation
    assert loaded.predictions[0].raw_prediction == "  a  "


# ==============================================================================
# Case 12: Invalid File Formats
# ==============================================================================

def test_case_12_invalid_file_formats(tmp_path: Path):
    """Test error handling for non-existent, corrupted, or unsupported file formats."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "non_existent.json")

    # 2. Corrupted JSON
    corrupted_json = tmp_path / "corrupt.json"
    corrupted_json.write_text("{broken json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse prediction JSON"):
        load_predictions(corrupted_json)

    # 3. Unsupported format
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported prediction format"):
        load_predictions(txt_file, file_format="unsupported_xyz")


# ==============================================================================
# Case 13: Empty Datasets
# ==============================================================================

def test_case_13_empty_datasets(tmp_path: Path):
    """Test that evaluating against an empty ground truth returns null accuracy without crashing."""
    gt_empty = CanonicalGroundTruth(
        schema_version="1.0",
        dataset="ReVA",
        split="validation",
        labeled=True,
        samples=[],
    )
    preds = CanonicalPredictions(
        schema_version="1.0",
        dataset="ReVA",
        split="validation",
        model=ModelMetadata(name="TestModel"),
        predictions=[],
    )
    rep = evaluate(gt_empty, preds, save_plots=False)
    assert rep.metrics.total_samples == 0
    assert rep.metrics.accuracy is None
    assert rep.metrics.prediction_coverage == 0.0


# ==============================================================================
# Case 14: Single-Record Datasets
# ==============================================================================

def test_case_14_single_record_datasets(tmp_path: Path):
    """Test boundary condition with a dataset containing exactly one record (N=1)."""
    gt_single = make_synthetic_ground_truth(num_samples=1)
    preds_correct = make_synthetic_predictions(gt_single, prediction_mode="all_correct")

    rep = evaluate(gt_single, preds_correct, output_dir=tmp_path / "out_single", save_plots=True)
    assert rep.metrics.total_samples == 1
    assert rep.metrics.correct == 1
    assert rep.metrics.accuracy == 1.0


# ==============================================================================
# Case 15: Per-Category Evaluation
# ==============================================================================

def test_case_15_per_category_and_task_evaluation(tmp_path: Path):
    """Test grouped metric calculation across all 4 categories and tasks."""
    gt = make_synthetic_ground_truth(num_samples=8)
    preds = make_synthetic_predictions(gt, prediction_mode="mixed")

    rep = evaluate(gt, preds, save_plots=False)
    assert len(rep.category_metrics) == 4
    for cm in rep.category_metrics:
        assert cm["total"] == 2
        assert "accuracy" in cm
        assert "coverage" in cm
        assert cm["coverage"] == 1.0


# ==============================================================================
# Case 16: Report Generation
# ==============================================================================

def test_case_16_report_generation(tmp_path: Path):
    """Test that all reports, CSVs, and plots are correctly written to the output directory."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="mixed")
    out_dir = tmp_path / "report_out"

    rep = evaluate(gt, preds, output_dir=out_dir, save_plots=True)
    assert rep.status == "completed"

    expected_files = [
        "evaluation.json",
        "evaluation_report.md",
        "validation_issues.json",
        "category_metrics.csv",
        "task_metrics.csv",
        "predictions_aligned.csv",
        "incorrect_predictions.csv",
        "error_patterns.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "category_accuracy.png",
        "task_accuracy.png",
        "error_distribution.png",
    ]
    for fn in expected_files:
        p = out_dir / fn
        assert p.is_file(), f"Missing expected output artifact: {fn}"
        assert p.stat().st_size > 0


# ==============================================================================
# Case 17: Experiment Tracking
# ==============================================================================

def test_case_17_experiment_tracking(tmp_path: Path):
    """Test local experiment tracker logging, retrieval, comparison, and export."""
    tracker_dir = tmp_path / "exp_tracking"
    tracker = ExperimentTracker(registry_dir=tracker_dir)

    gt = make_synthetic_ground_truth(num_samples=4)
    preds1 = make_synthetic_predictions(gt, model_name="Model_A", prediction_mode="all_correct")
    preds2 = make_synthetic_predictions(gt, model_name="Model_B", prediction_mode="mixed")

    rep1 = evaluate(gt, preds1, track_experiment=True, experiment_tracker_dir=tracker_dir, experiment_id="EXP_001")
    rep2 = evaluate(gt, preds2, track_experiment=True, experiment_tracker_dir=tracker_dir, experiment_id="EXP_002")

    # List runs
    runs = tracker.list_runs()
    assert len(runs) == 2
    assert {r.experiment_id for r in runs} == {"EXP_001", "EXP_002"}

    # Get single run
    run1 = tracker.get_run("EXP_001")
    assert run1 is not None
    assert run1.accuracy == 1.0

    # Compare runs DataFrame
    df_comp = tracker.compare_runs()
    assert len(df_comp) == 2
    assert "accuracy" in df_comp.columns

    # Export summary
    csv_summary = tracker.export_summary(tmp_path / "summary.csv")
    assert csv_summary.is_file()
    md_summary = tracker.export_summary(tmp_path / "summary.md")
    assert md_summary.is_file()


# ==============================================================================
# Case 18: Error-Analysis Exports
# ==============================================================================

def test_case_18_error_analysis_exports(tmp_path: Path):
    """Test detailed error analysis outputs: incorrect_predictions.csv and error_patterns.json."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="mixed")  # 1 wrong sample
    out_dir = tmp_path / "error_out"

    evaluate(gt, preds, output_dir=out_dir, save_plots=False)

    # Check incorrect_predictions.csv
    csv_path = out_dir / "incorrect_predictions.csv"
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # Header + 1 error row

    # Check error_patterns.json
    json_path = out_dir / "error_patterns.json"
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        patterns = json.load(f)
    assert patterns["total_errors"] == 1
    assert patterns["error_breakdown"]["incorrect"] == 1
    assert len(patterns["frequent_confusion_pairs"]) == 1


# ==============================================================================
# Case 19: Strict Validation Mode
# ==============================================================================

def test_case_19_strict_validation_mode(tmp_path: Path):
    """Test that strict validation mode fails clearly on missing or unexpected predictions."""
    gt = make_synthetic_ground_truth(num_samples=4)

    # 1. Missing predictions fail in strict mode
    missing_preds = make_synthetic_predictions(gt, prediction_mode="missing")
    with pytest.raises(ValueError, match="Strict validation failure: Missing predictions"):
        evaluate(gt, missing_preds, strict_mode=True)

    # 2. Unexpected prediction IDs fail in strict mode
    unexpected_preds = make_synthetic_predictions(gt, prediction_mode="unexpected_id")
    with pytest.raises(ValueError, match="Strict validation failure: Found.*unexpected prediction IDs"):
        evaluate(gt, unexpected_preds, strict_mode=True)

    # 3. Invalid choice labels fail in strict mode
    invalid_preds = make_synthetic_predictions(gt, prediction_mode="invalid_label")
    with pytest.raises(ValueError, match="Strict validation failure: Found.*predictions with invalid choice labels"):
        evaluate(gt, invalid_preds, strict_mode=True)


# ==============================================================================
# Case 20: Non-Strict Validation Mode
# ==============================================================================

def test_case_20_non_strict_validation_mode(tmp_path: Path):
    """Test that non-strict mode records warnings and diagnostics without halting execution."""
    gt = make_synthetic_ground_truth(num_samples=4)
    missing_preds = make_synthetic_predictions(gt, prediction_mode="missing")

    # Non-strict mode completes successfully
    rep = evaluate(gt, missing_preds, strict_mode=False, save_plots=False)
    assert rep.status == "completed"
    assert rep.metrics.missing_predictions == 1
    assert rep.metrics.total_samples == 4  # Denominator penalty maintained


# ==============================================================================
# Bonus: Model Integration Interface & Dummy Adapter
# ==============================================================================

def test_model_interface_dummy_adapter(tmp_path: Path):
    """Test ReVAModelInterface lifecycle using DummyBaselineModelAdapter."""
    gt = make_synthetic_ground_truth(num_samples=4)
    adapter = DummyBaselineModelAdapter(model_name="TestDummyModel")

    # Simulate checkpoint load
    adapter.load_checkpoint(tmp_path / "mock_weights.pt")
    assert adapter.is_loaded

    # Generate canonical predictions directly from model adapter
    preds = adapter.generate_predictions(gt, output_path=tmp_path / "adapter_preds.json")
    assert len(preds.predictions) == 4
    assert preds.model.name == "TestDummyModel"

    # Evaluate model end-to-end through model interface
    rep = adapter.evaluate_model(gt, output_dir=tmp_path / "dummy_report", save_plots=False)
    assert rep.status == "completed"
    assert rep.metrics.provided_predictions == 4
