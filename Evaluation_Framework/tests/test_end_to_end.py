"""End-to-end integration tests for CLI commands and artifacts generation."""

import json
from pathlib import Path
import pytest

from reva_eval.adapters.generic_json import GenericJSONAdapter
from reva_eval.cli import main
from reva_eval.dataset_loader import save_canonical_ground_truth
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


def test_cli_prepare_ground_truth(tmp_path: Path):
    """Test CLI prepare-ground-truth command."""
    raw_anno = {
        "metadata": {"total_videos": 1},
        "QA": [
            {
                "qa_id": "cli_sample_001",
                "video_path": "visdrone/test.mp4",
                "category": "Factual Perception",
                "task": "General Understanding",
                "question": "CLI question?",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "B",
            }
        ],
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw_anno), encoding="utf-8")

    out_gt = tmp_path / "canonical_gt.json"
    exit_code = main([
        "prepare-ground-truth",
        "--annotation", str(raw_path),
        "--output", str(out_gt),
        "--split", "validation",
    ])
    assert exit_code == 0
    assert out_gt.is_file()

    with open(out_gt, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert len(data["samples"]) == 1
    assert data["samples"][0]["sample_id"] == "cli_sample_001"


def test_cli_evaluate_generates_all_ten_artifacts(tmp_path: Path):
    """Test CLI evaluate command produces all 10 specified output artifacts."""
    gt = make_synthetic_ground_truth(num_samples=4)
    gt_file = tmp_path / "gt.json"
    save_canonical_ground_truth(gt, gt_file)

    preds = make_synthetic_predictions(gt, prediction_mode="mixed")
    pred_file = tmp_path / "preds.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(preds.model_dump(), f, indent=2)

    report_dir = tmp_path / "reports" / "SyntheticModel" / "validation"
    exit_code = main([
        "evaluate",
        "--ground-truth", str(gt_file),
        "--predictions", str(pred_file),
        "--output-dir", str(report_dir),
    ])
    assert exit_code == 0

    # Verify all specified artifacts exist
    expected_files = [
        "evaluation.json",
        "category_metrics.csv",
        "task_metrics.csv",
        "predictions_aligned.csv",
        "incorrect_predictions.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "category_accuracy.png",
        "task_accuracy.png",
        "evaluation_report.md",
        "validation_issues.json",
    ]
    for filename in expected_files:
        artifact_path = report_dir / filename
        assert artifact_path.is_file(), f"Missing artifact: {filename}"
        assert artifact_path.stat().st_size > 0, f"Artifact is empty: {filename}"


def test_cli_compare_command(tmp_path: Path):
    """Test CLI compare command produces comparison CSV and MD."""
    gt = make_synthetic_ground_truth(num_samples=4)
    gt_file = tmp_path / "gt.json"
    save_canonical_ground_truth(gt, gt_file)

    preds1 = make_synthetic_predictions(gt, prediction_mode="all_correct", model_name="ModelA")
    pred1_file = tmp_path / "preds_a.json"
    with open(pred1_file, "w", encoding="utf-8") as f:
        json.dump(preds1.model_dump(), f, indent=2)

    preds2 = make_synthetic_predictions(gt, prediction_mode="mixed", model_name="ModelB")
    pred2_file = tmp_path / "preds_b.json"
    with open(pred2_file, "w", encoding="utf-8") as f:
        json.dump(preds2.model_dump(), f, indent=2)

    comp_dir = tmp_path / "comparison"
    exit_code = main([
        "compare",
        "--ground-truth", str(gt_file),
        "--predictions", f"ModelA={pred1_file}", f"ModelB={pred2_file}",
        "--output-dir", str(comp_dir),
    ])
    assert exit_code == 0
    assert (comp_dir / "model_comparison.csv").is_file()
    assert (comp_dir / "model_comparison.md").is_file()


def test_generic_json_adapter_conversion(tmp_path: Path):
    """Test GenericJSONAdapter converting raw lists, scores mapping, and option_order validation."""
    adapter = GenericJSONAdapter()

    raw_items = [
        {
            "qa_id": "raw_001",
            "answer": "B",
            "scores": [0.1, 0.7, 0.1, 0.1],
            "latency": 0.42,
        },
        {
            "qa_id": "raw_002",
            "answer": "C",
            "scores": {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1},
            "latency": 0.55,
        },
    ]

    canonical = adapter.convert(
        raw_data=raw_items,
        model_name="CustomTeammateModel",
        split="validation",
        score_semantics="probabilities",
        option_order=["A", "B", "C", "D"],
    )

    assert canonical.schema_version == "1.0"
    assert canonical.model.name == "CustomTeammateModel"
    assert len(canonical.predictions) == 2

    p1 = canonical.predictions[0]
    assert p1.sample_id == "raw_001"
    assert p1.prediction == "B"
    assert p1.scores == {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}
    assert p1.inference_time_seconds == 0.42

    # Verify invalid option order raises ValueError when mapping a score list
    with pytest.raises(ValueError, match="Invalid option_order"):
        adapter.convert(
            raw_data=raw_items,
            model_name="CustomTeammateModel",
            split="validation",
            option_order=["A", "B", "C"],  # missing D
        )


def test_cli_invalid_command_and_missing_file_returns_nonzero():
    """Verify non-zero exit code on CLI error."""
    assert main([]) != 0
    assert main(["evaluate", "--ground-truth", "nonexistent.json", "--predictions", "nonexistent.json", "--output-dir", "out"]) != 0

