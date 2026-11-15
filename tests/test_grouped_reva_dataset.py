"""Offline integration checks for Sarthak's grouped annotation filenames."""
import json
import sys
import types

# The local/annotation-only paths under test do not use HF Hub. Keep the
# optional import resolvable in minimal offline test environments.
if "huggingface_hub" not in sys.modules:
    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.HfApi = type("HfApi", (), {})
    hub_stub.hf_hub_download = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("HF Hub is not used by this offline test")
    )
    sys.modules["huggingface_hub"] = hub_stub

import numpy as np
import pytest
import torch

from preprocessing import ReVADataset
from preprocessing import dataset as dataset_module
from advanced_models.qa.data import ReVAQADatasetAdapter


def _record(qid, video_path="clip.mp4", answer="C"):
    return {
        "qa_id": qid,
        "video_path": video_path,
        "question": "Which event occurs?",
        "options": {"D": "fourth", "B": "second", "A": "first", "C": "third"},
        "correct_answer": answer,
        "category": "Temporal Understanding",
    }


@pytest.mark.parametrize("filename,split", [
    ("train_grouped.json", "train"), ("val_grouped.json", "val"),
    ("train_internal.json", "train"), ("val_internal.json", "val"),
])
def test_grouped_annotation_filename_loads_without_stem_filtering(tmp_path, filename, split):
    annotation = tmp_path / filename
    annotation.write_text(json.dumps({"QA": [_record(f"{split}_1"), _record(f"{split}_2", "other.mp4", "A")]}))
    dataset = ReVADataset(annotation, video_root=tmp_path, split=split, missing_video="zeros")
    assert len(dataset) == 2
    assert [row["qa_id"] for row in dataset.rows] == [f"{split}_1", f"{split}_2"]


def test_grouped_sample_order_label_and_path_adapter(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"fixture")
    annotation = tmp_path / "train_internal.json"
    annotation.write_text(json.dumps({"QA": [_record("train_1")] }))
    monkeypatch.setattr(dataset_module, "get_video_metadata", lambda _: {"frame_count": 8})
    rgb = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    monkeypatch.setattr(dataset_module, "read_frames", lambda _p, _indices: rgb)
    sample = ReVAQADatasetAdapter(ReVADataset(annotation, video_root=tmp_path, split=None, num_frames=4))[0]
    assert sample["label"] == 2
    assert sample["answer"] == "C"
    assert list(sample["choices"]) == ["A", "B", "C", "D"]
    assert sample["choices"] == {"A": "first", "B": "second", "C": "third", "D": "fourth"}
    assert sample["question"] == "Which event occurs?"
    assert sample["video_path"] == "clip.mp4"
    assert sample["video_frames"].shape == (4, 3, 224, 224)
    assert sample["video_frames"].dtype == torch.float32
    expected_rgb_pixel = torch.tensor([(1 - 0.485) / 0.229, (0 - 0.456) / 0.224, (0 - 0.406) / 0.225])
    assert torch.allclose(sample["video_frames"][0, :, 0, 0], expected_rgb_pixel, atol=1e-5)


def test_missing_video_raise_and_zero_mode_is_explicitly_marked(tmp_path):
    annotation = tmp_path / "val_grouped.json"
    annotation.write_text(json.dumps({"QA": [_record("val_1", "missing.mp4")] }))
    with pytest.raises(Exception, match="Video file not found"):
        ReVADataset(annotation, video_root=tmp_path, split=None, missing_video="raise")[0]
    sample = ReVADataset(annotation, video_root=tmp_path, split=None, missing_video="zeros")[0]
    assert sample["video_error"]
    with pytest.raises(RuntimeError, match="invalid video"):
        ReVAQADatasetAdapter(ReVADataset(annotation, video_root=tmp_path, split=None, missing_video="zeros"))[0]


def test_video_path_cannot_escape_configured_root(tmp_path):
    annotation = tmp_path / "train_grouped.json"
    annotation.write_text(json.dumps({"QA": [_record("train_1", "../escape.mp4")] }))
    with pytest.raises(ValueError, match="escapes video_root"):
        ReVADataset(annotation, video_root=tmp_path, split=None)[0]
