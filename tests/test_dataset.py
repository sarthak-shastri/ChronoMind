import json
import numpy as np
import torch
from preprocessing.dataset import read_annotations, ReVADataset
from preprocessing.frame_sampling import uniform_indices, adaptive_indices
from preprocessing.transformations import transform_frames

def test_read_qa_object(tmp_path):
    f=tmp_path/'a.json'; f.write_text(json.dumps({'metadata':{},'QA':[{'qa_id':'x'}]}),encoding='utf-8'); assert read_annotations(f)[0]['qa_id']=='x'
def test_read_jsonl(tmp_path):
    f=tmp_path/'a.jsonl'; f.write_text('{"qa_id":"x"}\n{"qa_id":"y"}\n',encoding='utf-8'); assert len(read_annotations(f))==2
def test_uniform_sampling():
    a=uniform_indices(101,8); assert len(a)==8 and np.array_equal(a,uniform_indices(101,8)) and a[0]==0 and a[-1]==100
def test_adaptive_fallback(): assert np.array_equal(adaptive_indices(10,4),uniform_indices(10,4))
def test_transform_contract():
    x=transform_frames(np.zeros((4,20,30,3),dtype=np.uint8),(16,16),normalize=False); assert tuple(x.shape)==(4,3,16,16) and x.dtype==torch.float32
def test_test_split_without_answer(tmp_path):
    f=tmp_path/'test.json'; row={'qa_id':'test_1','video_path':'missing.mp4','question':'q','options':dict(A='a',B='b',C='c',D='d'),'correct_answer':''}; f.write_text(json.dumps({'QA':[row]}),encoding='utf-8')
    s=ReVADataset(f,tmp_path,missing_video='zeros')[0]; assert s['sample_id']=='test_1' and s['answer'] is None and list(s['choices'])==list('ABCD') and tuple(s['video_frames'].shape)==(8,3,224,224)


def test_huggingface_video_is_fetched_lazily(tmp_path, monkeypatch):
    import preprocessing.dataset as module
    annotation = tmp_path / "train.json"
    row = {"qa_id": "train_1", "video_path": "VisDrone/clip.mp4", "question": "q",
           "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "A"}
    annotation.write_text(json.dumps({"QA": [row]}), encoding="utf-8")
    cached_video = tmp_path / "clip.mp4"
    cached_video.write_bytes(b"fake video bytes")
    calls = []
    def fake_fetch(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs.get("revision")))
        return cached_video
    monkeypatch.setattr(module, "fetch_dataset_file", fake_fetch)
    monkeypatch.setattr(module, "get_video_metadata", lambda _: {"frame_count": 2})
    monkeypatch.setattr(module, "read_frames", lambda _p, _i: [np.zeros((4, 4, 3), dtype=np.uint8)] * 8)
    ds = ReVADataset(annotation, num_frames=8, dataset_repo_id="ReVA-Benchmark/ReVA", revision="abc123")
    sample = ds[0]
    assert calls == [("ReVA-Benchmark/ReVA", "VisDrone/clip.mp4", "abc123")]
    assert sample["sample_id"] == "train_1"
    assert tuple(sample["video_frames"].shape) == (8, 3, 224, 224)


def test_frame_cache_miss_then_hit(tmp_path):
    from preprocessing.cache_manager import FrameCache
    cache=FrameCache(tmp_path)
    assert cache.get('sample') is None
    expected=np.zeros((2,5,6,3),dtype=np.uint8)
    cache.put('sample',expected)
    actual=cache.get('sample')
    assert np.array_equal(actual,expected)
    assert cache.misses==1 and cache.hits==1


def test_feature_cache_round_trip(tmp_path):
    from preprocessing.feature_cache import NumpyFeatureCache
    cache=NumpyFeatureCache(tmp_path)
    assert cache.get('video-a') is None
    features=np.arange(12,dtype=np.float32).reshape(3,4)
    cache.put('video-a',features)
    assert np.array_equal(cache.get('video-a'),features)
    assert cache.hits==1 and cache.misses==1


def test_split_filter_uses_split_named_annotation_file(tmp_path):
    f=tmp_path/'train.json'
    f.write_text(json.dumps({'QA':[{'qa_id':'train_1'}]}),encoding='utf-8')
    ds=ReVADataset(f,video_root=tmp_path,split='train',missing_video='zeros')
    assert len(ds)==1
    import pytest
    with pytest.raises(ValueError,match="not 'val'"):
        ReVADataset(f,video_root=tmp_path,split='val')


def test_split_filter_accepts_grouped_annotation_filename(tmp_path):
    f=tmp_path/'train_grouped.json'
    f.write_text(json.dumps({'QA':[{'qa_id':'train_1'}]}),encoding='utf-8')
    ds=ReVADataset(f,video_root=tmp_path,split='train',missing_video='zeros')
    assert len(ds)==1
