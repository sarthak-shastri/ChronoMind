from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor


class FakeBackbone(nn.Module):
    def __init__(self, text=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(projection_dim=512)


def test_vision_loading_arguments_are_separate_from_extractor_kwargs(monkeypatch, tmp_path):
    backbone = FakeBackbone()
    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return backbone

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPVisionModelWithProjection=Loader))
    ext = CLIPVisionFeatureExtractor.from_pretrained(
        model_name=str(tmp_path), revision="pinned", local_files_only=True,
        cache_dir="cache", output_loading_info=False,
        input_normalization="imagenet", frame_chunk_size=2,
    )
    assert calls[0][0] == str(tmp_path)
    assert calls[0][1] == {"local_files_only": True, "output_loading_info": False, "cache_dir": "cache"}
    assert ext.frame_chunk_size == 2 and ext.input_normalization == "imagenet"
    assert all(not p.requires_grad for p in ext.parameters())


def test_text_model_tokenizer_forward_local_loading_and_keep_kwargs_separate(monkeypatch, tmp_path):
    model, tokenizer = FakeBackbone(text=True), object()
    calls = []

    class ModelLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("model", name, kwargs))
            return model, {"missing_keys": [], "unexpected_keys": []}

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPTextModelWithProjection=ModelLoader, CLIPTokenizer=TokenizerLoader))
    ext = CLIPTextFeatureExtractor.from_pretrained(
        str(tmp_path), revision="pinned", local_files_only=True, cache_dir="cache",
        output_loading_info=True, tokenizer_kwargs={"use_fast": False}, max_length=64,
    )
    assert calls[0] == ("tokenizer", str(tmp_path), {"local_files_only": True, "cache_dir": "cache", "use_fast": False})
    assert calls[1] == ("model", str(tmp_path), {"local_files_only": True, "output_loading_info": True, "cache_dir": "cache"})
    assert ext.max_length == 64 and ext.tokenizer is tokenizer
    assert ext.loading_info == {"missing_keys": [], "unexpected_keys": []}
    assert all(not p.requires_grad for p in ext.text_model.parameters())


def test_remote_style_calling_conventions_still_forward_revision(monkeypatch):
    backbone = FakeBackbone()
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            return backbone
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPVisionModelWithProjection=Loader))
    CLIPVisionFeatureExtractor.from_pretrained(input_normalization="imagenet", revision="rev")
    assert calls == [{"local_files_only": False, "output_loading_info": False, "revision": "rev"}]


def test_unavailable_local_checkpoint_fails_without_online_fallback(monkeypatch, tmp_path):
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            assert kwargs["local_files_only"] is True
            raise OSError("local files absent")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPVisionModelWithProjection=Loader))
    with pytest.raises(OSError, match="prevents network fallback"):
        CLIPVisionFeatureExtractor.from_pretrained(
            model_name=str(tmp_path / "missing"), input_normalization="imagenet",
            local_files_only=True,
        )
    assert len(calls) == 1
