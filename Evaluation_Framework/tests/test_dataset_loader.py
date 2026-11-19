"""Tests for dataset loader and annotation normalization."""

import json
from pathlib import Path
import pytest

from reva_eval.dataset_loader import (
    load_canonical_ground_truth,
    load_raw_annotation_file,
    normalize_annotation,
    save_canonical_ground_truth,
)
from reva_eval.models import CanonicalGroundTruth


def test_load_current_qa_list_format(tmp_path: Path):
    """Test loading current top-level QA list annotation format."""
    data = {
        "metadata": {"total_questions": 2},
        "QA": [
            {
                "qa_id": "test_001",
                "video_path": "visdrone/vid1.mp4",
                "dataset_name": "VisDrone",
                "subdir": "Scene1",
                "category": "Factual Perception",
                "task": "Object and Land Cover Recognition",
                "question": "What is visible?",
                "options": {"A": "Car", "B": "Drone", "C": "Tree", "D": "House"},
                "correct_answer": "A",
            },
            {
                "qa_id": "test_002",
                "video_path": "visdrone/vid2.mp4",
                "dataset_name": "VisDrone",
                "subdir": "Scene2",
                "category": "Temporal Understanding",
                "task": "Temporal Grounding",
                "question": "When does it turn?",
                "options": {"A": "1s", "B": "2s", "C": "3s", "D": "4s"},
                "correct_answer": "C",
            },
        ],
    }
    p = tmp_path / "raw_qa.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    gt = normalize_annotation(p, split="validation")
    assert gt.schema_version == "1.0"
    assert gt.dataset == "ReVA"
    assert gt.split == "validation"
    assert gt.labeled is True
    assert len(gt.samples) == 2

    s1 = gt.samples[0]
    assert s1.sample_id == "test_001"
    assert s1.video_path == "visdrone/vid1.mp4"
    assert s1.category == "Factual Perception"
    assert s1.task == "Object and Land Cover Recognition"
    assert s1.ground_truth == "A"
    assert s1.options["B"] == "Drone"


def test_load_legacy_videos_mapping_format(tmp_path: Path):
    """Test loading legacy videos dictionary annotation format."""
    data = {
        "videos": {
            "vid_alpha": {
                "video_path": "uav/alpha.mp4",
                "qa_list": [
                    {
                        "qa_id": "legacy_001",
                        "category": "Spatial Reasoning",
                        "task": "Geometric Relation",
                        "question": "Where is the car?",
                        "options": {"A": "Left", "B": "Right", "C": "Up", "D": "Down"},
                        "correct_answer": "B",
                    }
                ],
            }
        }
    }
    p = tmp_path / "legacy_videos.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    gt = normalize_annotation(p, split="validation")
    assert len(gt.samples) == 1
    assert gt.samples[0].sample_id == "legacy_001"
    assert gt.samples[0].video_path == "uav/alpha.mp4"
    assert gt.samples[0].ground_truth == "B"


def test_unlabeled_test_split(tmp_path: Path):
    """Test that empty string correct_answer marks split as unlabeled."""
    data = {
        "QA": [
            {
                "qa_id": "test_unlabeled_001",
                "video_path": "vid.mp4",
                "category": "Factual Perception",
                "task": "General Understanding",
                "question": "Unlabeled test question?",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "",
            }
        ]
    }
    p = tmp_path / "test_unlabeled.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    gt = normalize_annotation(p, split="test")
    assert gt.labeled is False
    assert len(gt.samples) == 1
    assert gt.samples[0].ground_truth is None


def test_duplicate_ground_truth_id_fails(tmp_path: Path):
    """Test that duplicate sample IDs in ground truth raise ValueError."""
    data = {
        "QA": [
            {
                "qa_id": "dup_001",
                "video_path": "vid1.mp4",
                "question": "Q1",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "A",
            },
            {
                "qa_id": "dup_001",
                "video_path": "vid2.mp4",
                "question": "Q2",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "B",
            },
        ]
    }
    p = tmp_path / "dup_gt.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        normalize_annotation(p)


def test_save_and_load_canonical_ground_truth(tmp_path: Path):
    """Test round-trip serialization of CanonicalGroundTruth."""
    data = {
        "QA": [
            {
                "qa_id": "sample_001",
                "video_path": "vid1.mp4",
                "category": "Factual Perception",
                "task": "General Understanding",
                "question": "Q1",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "D",
            }
        ]
    }
    raw_p = tmp_path / "raw.json"
    raw_p.write_text(json.dumps(data), encoding="utf-8")

    gt = normalize_annotation(raw_p, split="validation")
    out_p = tmp_path / "canonical.json"
    save_canonical_ground_truth(gt, out_p)

    loaded = load_canonical_ground_truth(out_p)
    assert loaded.schema_version == "1.0"
    assert loaded.split == "validation"
    assert loaded.samples[0].sample_id == "sample_001"
    assert loaded.samples[0].ground_truth == "D"

