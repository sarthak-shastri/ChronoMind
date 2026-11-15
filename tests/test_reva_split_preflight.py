import json
from scripts.reva_split_preflight import preflight_splits


def _write(path, qa):
    path.write_text(json.dumps({"metadata": {}, "QA": qa}), encoding="utf-8")
    return path


def _row(qid, video, label="A"):
    return {"qa_id": qid, "video_path": video,
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_answer": label, "source_sequence_id": f"src-{qid}"}


def test_split_preflight_reports_normalized_path_overlap(tmp_path):
    train = _write(tmp_path / "train.json", [_row("train_1", "Scene/Clip.mp4")])
    val = _write(tmp_path / "val.json", [_row("val_1", "scene\\clip.mp4", "B")])
    result = preflight_splits(train, val)
    assert result["status"] == "issues_found"
    assert any("video-path overlap" in issue for issue in result["issues"])


def test_split_preflight_passes_disjoint_paths_and_records_scope(tmp_path):
    train = _write(tmp_path / "train.json", [_row("train_1", "train/a.mp4")])
    val = _write(tmp_path / "val.json", [_row("val_1", "val/a.mp4", "C")])
    test = _write(tmp_path / "test.json", [_row("test_1", "test/a.mp4", "")])
    result = preflight_splits(train, val, test)
    assert result["status"] == "passed"
    assert result["audited_splits"] == ["train", "val", "test"]
    assert result["unverifiable_source_sequence_relationships"]["status"] == "explicit_ids_present"


def test_split_preflight_reports_identical_file_hashes(tmp_path):
    root = tmp_path / "videos"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "train" / "same.mp4").write_bytes(b"same video content")
    (root / "val" / "copy.mp4").write_bytes(b"same video content")
    train = _write(tmp_path / "train.json", [_row("t", "train/same.mp4")])
    val = _write(tmp_path / "val.json", [_row("v", "val/copy.mp4", "B")])
    result = preflight_splits(train, val, video_root=root)
    assert len(result["identical_content_hash_overlaps"]) == 1
    assert result["video_hashing"]["hashed_files"] == 2


def test_missing_source_ids_are_reported_as_unverifiable_not_overlap(tmp_path):
    row = _row("t", "a.mp4")
    row.pop("source_sequence_id")
    train = _write(tmp_path / "train.json", [row])
    val = _write(tmp_path / "val.json", [_row("v", "b.mp4", "B")])
    result = preflight_splits(train, val)
    unknown = result["unverifiable_source_sequence_relationships"]
    assert result["status"] == "passed_with_limitations"
    assert unknown["rows_without_source_sequence_id"] == {"train": 1, "val": 0}
    assert "cannot be established" in unknown["warning"]
    assert result["path_overlaps"] == []
    assert "do not alone prove answer leakage" in result["interpretation"]


def test_path_only_strict_issues_separate_from_source_id_limitation(tmp_path):
    row = _row("t", "same.mp4")
    row.pop("source_sequence_id")
    train = _write(tmp_path / "train.json", [row])
    val = _write(tmp_path / "val.json", [_row("v", "same.mp4", "B")])
    result = preflight_splits(train, val)
    assert any("video-path overlap" in issue for issue in result["path_only_issues"])
    assert result["unverifiable_source_sequence_relationships"]["status"] == "unverifiable"


def test_path_only_strict_can_pass_path_isolation_with_source_warning(tmp_path):
    row = _row("t", "train.mp4")
    row.pop("source_sequence_id")
    train = _write(tmp_path / "train.json", [row])
    val = _write(tmp_path / "val.json", [_row("v", "val.mp4", "B")])
    result = preflight_splits(train, val)
    assert result["path_only_issues"] == []
    assert result["unverifiable_source_sequence_relationships"]["warning"]
