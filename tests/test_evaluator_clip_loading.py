from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
from scripts.evaluate_reva_qa import load_clip_feature_extractors


def test_evaluator_forwards_local_clip_options_to_both_extractors(monkeypatch):
    calls = []

    def capture(kind):
        def loader(cls, **kwargs):
            calls.append((kind, kwargs))
            return kind
        return classmethod(loader)

    monkeypatch.setattr(CLIPVisionFeatureExtractor, "from_pretrained", capture("vision"))
    monkeypatch.setattr(CLIPTextFeatureExtractor, "from_pretrained", capture("text"))

    vision, text = load_clip_feature_extractors(
        r"C:\local\clip", "pinned-revision", "imagenet", 4,
        local_files_only=True, cache_dir=r"C:\hf-cache",
    )

    assert (vision, text) == ("vision", "text")
    assert calls == [
        ("vision", {
            "model_name": r"C:\local\clip", "input_normalization": "imagenet",
            "frame_chunk_size": 4, "cuda_mixed_precision": False,
            "revision": "pinned-revision", "local_files_only": True,
            "cache_dir": r"C:\hf-cache",
        }),
        ("text", {
            "model_name": r"C:\local\clip", "revision": "pinned-revision",
            "local_files_only": True, "cache_dir": r"C:\hf-cache",
        }),
    ]
