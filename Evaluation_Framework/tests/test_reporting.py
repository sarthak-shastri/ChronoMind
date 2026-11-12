"""Tests for reporting generation and JSON schema conformance."""

import json
from pathlib import Path
import jsonschema
import pytest

from reva_eval.evaluator import evaluate
from reva_eval.reporting import (
    generate_markdown_report,
    generate_model_comparison_report,
    save_evaluation_json,
)
from tests.fixtures.synthetic_fixtures import (
    make_synthetic_ground_truth,
    make_synthetic_predictions,
)


@pytest.fixture
def eval_report_schema() -> dict:
    schema_path = Path("schemas/evaluation_report.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def predictions_schema() -> dict:
    schema_path = Path("schemas/predictions.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def ground_truth_schema() -> dict:
    schema_path = Path("schemas/ground_truth.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_validates_sample_evaluation_report(eval_report_schema):
    """Validate examples/sample_evaluation_report.json against evaluation_report.schema.json."""
    sample_path = Path("examples/sample_evaluation_report.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    jsonschema.validate(instance=data, schema=eval_report_schema)


def test_schema_validates_sample_predictions(predictions_schema):
    """Validate examples/sample_predictions.json against predictions.schema.json."""
    sample_path = Path("examples/sample_predictions.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    jsonschema.validate(instance=data, schema=predictions_schema)


def test_schema_validates_sample_ground_truth(ground_truth_schema):
    """Validate examples/sample_ground_truth.json against ground_truth.schema.json."""
    sample_path = Path("examples/sample_ground_truth.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    jsonschema.validate(instance=data, schema=ground_truth_schema)


def test_generated_evaluation_report_conforms_to_schema(eval_report_schema, tmp_path: Path):
    """Evaluate synthetic predictions and assert output evaluation.json satisfies JSON schema."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds = make_synthetic_predictions(gt, prediction_mode="mixed")

    report = evaluate(gt, preds, output_dir=tmp_path)
    eval_json_path = tmp_path / "evaluation.json"
    assert eval_json_path.is_file()

    with open(eval_json_path, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    jsonschema.validate(instance=exported_data, schema=eval_report_schema)
    assert exported_data["metrics"]["total_samples"] == 4
    assert exported_data["metrics"]["accuracy"] == 0.75


def test_model_comparison_reporting(tmp_path: Path):
    """Test generating side-by-side model comparison CSV and Markdown."""
    gt = make_synthetic_ground_truth(num_samples=4)
    preds_bimba = make_synthetic_predictions(gt, prediction_mode="all_correct", model_name="BIMBA")
    preds_adv = make_synthetic_predictions(gt, prediction_mode="mixed", model_name="Advanced")

    rep_bimba = evaluate(gt, preds_bimba)
    rep_adv = evaluate(gt, preds_adv)

    reports = {"BIMBA": rep_bimba, "Advanced": rep_adv}
    comp_dir = tmp_path / "comparison"
    csv_path, md_path = generate_model_comparison_report(reports, comp_dir)

    assert csv_path.is_file()
    assert md_path.is_file()

    md_content = md_path.read_text(encoding="utf-8")
    assert "BIMBA" in md_content
    assert "Advanced" in md_content
    assert "Overall Model Comparison" in md_content

