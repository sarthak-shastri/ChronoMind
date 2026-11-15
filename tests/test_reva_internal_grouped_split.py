import json
import sys
import types

import pytest
import numpy as np
import torch

from scripts.create_reva_internal_grouped_split import create_split


def _row(qid, path, index):
    return {
        "qa_id": qid,
        "video_path": path,
        "question": f"Question {index}",
        "options": {"D": "d", "B": "b", "A": "a", "C": "c"},
        "correct_answer": "ABCD"[index % 4],
        "category": f"category_{index % 2}",
        "task": f"task_{index % 3}",
    }


def _source(path):
    qa = []
    for video in range(10):
        # Two questions for each path must stay in the same partition.
        for question in range(2):
            idx = video * 2 + question
            qa.append(_row(f"q{idx}", f"video/{video}.mp4", idx))
    path.write_text(json.dumps({"metadata": {"source": "fixture"}, "QA": qa}), encoding="utf-8")
    return qa


def test_split_is_deterministic_grouped_and_preserves_annotation_fields(tmp_path):
    source = tmp_path / "train.json"
    original_rows = _source(source)
    first = create_split(source, tmp_path / "derived_a", seed=7)
    second = create_split(source, tmp_path / "derived_b", seed=7)
    train = json.loads((tmp_path / "derived_a" / "train_internal.json").read_text())
    val = json.loads((tmp_path / "derived_a" / "val_internal.json").read_text())
    val_again = json.loads((tmp_path / "derived_b" / "val_internal.json").read_text())
    train_paths = {row["video_path"].casefold() for row in train["QA"]}
    val_paths = {row["video_path"].casefold() for row in val["QA"]}
    assert not train_paths & val_paths
    assert first["normalized_video_path_overlap_count"] == 0
    assert first["total_questions"] == len(original_rows) == 20
    assert first["train_questions"] + first["validation_questions"] == 20
    assert first["validation_questions"] == 4
    assert [row["qa_id"] for row in val["QA"]] == [row["qa_id"] for row in val_again["QA"]]
    assert train["metadata"] == val["metadata"] == {"source": "fixture"}
    emitted = train["QA"] + val["QA"]
    assert {row["qa_id"] for row in emitted} == {row["qa_id"] for row in original_rows}
    assert all(row["options"] == original_rows[int(row["qa_id"][1:])]["options"] for row in emitted)
    assert first["category_distribution"]["train"] and first["task_distribution"]["validation"]
    assert first["answer_distribution"]["validation"]
    assert "not established" in first["source_sequence_independence"]
    assert second["normalized_video_path_overlap_count"] == 0


def test_identical_available_files_are_kept_in_one_split_and_missing_reported(tmp_path):
    source = tmp_path / "train.json"
    _source(source)
    video_root = tmp_path / "videos"
    (video_root / "video").mkdir(parents=True)
    (video_root / "video" / "0.mp4").write_bytes(b"duplicate bytes")
    (video_root / "video" / "1.mp4").write_bytes(b"duplicate bytes")
    stats = create_split(source, tmp_path / "derived", seed=7, video_root=video_root)
    train = json.loads((tmp_path / "derived" / "train_internal.json").read_text())["QA"]
    val = json.loads((tmp_path / "derived" / "val_internal.json").read_text())["QA"]
    partitions = {
        row["video_path"]: "train" if row in train else "validation"
        for row in train + val
    }
    assert partitions["video/0.mp4"] == partitions["video/1.mp4"]
    assert stats["cross_split_identical_hash_overlap"] == []
    assert len(stats["identical_file_hash_groups"]) == 1
    assert stats["hashed_unique_video_paths"] == 2
    assert stats["missing_video_file_count"] == 8
    assert stats["available_video_hashing"] is True


def test_refuses_output_overwrite_and_rejects_escaping_video_path(tmp_path):
    source = tmp_path / "train.json"
    _source(source)
    output = tmp_path / "derived"
    create_split(source, output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_split(source, output)
    bad_source = tmp_path / "bad.json"
    bad_source.write_text(json.dumps({"QA": [_row("bad", "../outside.mp4", 0)]}))
    with pytest.raises(ValueError, match="escapes video_root"):
        create_split(bad_source, tmp_path / "bad_derived", video_root=tmp_path / "videos")


def test_rejects_unlabeled_or_incomplete_training_records(tmp_path):
    source = tmp_path / "unlabeled.json"
    row = _row("bad", "video/a.mp4", 0)
    row["correct_answer"] = ""
    source.write_text(json.dumps({"QA": [row]}))
    with pytest.raises(ValueError, match="invalid/missing A/B/C/D"):
        create_split(source, tmp_path / "derived")


def test_generated_files_load_through_reva_qa_adapter_offline(tmp_path, monkeypatch):
    if "huggingface_hub" not in sys.modules:
        hub_stub = types.ModuleType("huggingface_hub")
        hub_stub.HfApi = type("HfApi", (), {})
        hub_stub.hf_hub_download = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("HF Hub is not used by this test")
        )
        sys.modules["huggingface_hub"] = hub_stub
    from preprocessing import ReVADataset
    from preprocessing import dataset as dataset_module
    from advanced_models.qa.data import ReVAQADatasetAdapter

    source = tmp_path / "train.json"
    rows = _source(source)
    video_root = tmp_path / "videos"
    for video in range(10):
        file = video_root / "video" / f"{video}.mp4"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(f"video-{video}".encode())
    stats = create_split(source, tmp_path / "derived", seed=7, video_root=video_root)
    monkeypatch.setattr(dataset_module, "get_video_metadata", lambda _: {"frame_count": 8})
    monkeypatch.setattr(dataset_module, "read_frames", lambda _path, _indices: np.zeros((4, 8, 8, 3), dtype=np.uint8))
    train = ReVAQADatasetAdapter(ReVADataset(
        tmp_path / "derived" / "train_internal.json", video_root=video_root,
        num_frames=4, split=None, missing_video="raise",
    ))
    val = ReVAQADatasetAdapter(ReVADataset(
        tmp_path / "derived" / "val_internal.json", video_root=video_root,
        num_frames=4, split=None, missing_video="raise",
    ))
    assert len(train) == stats["train_questions"]
    assert len(val) == stats["validation_questions"]
    sample = val[0]
    row_index = int(sample["sample_id"][1:])
    assert sample["question"] == rows[row_index]["question"]
    assert sample["label"] == "ABCD".index(sample["answer"])
    assert list(sample["choices"]) == ["A", "B", "C", "D"]
    assert sample["video_frames"].shape == (4, 3, 224, 224)
    assert sample["video_frames"].dtype == torch.float32
