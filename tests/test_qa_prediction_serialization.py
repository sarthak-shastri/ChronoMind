import torch
from torch import nn

from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor
from scripts.evaluate_reva_qa import calculate_metrics, make_prediction_records


def test_prediction_schema_preserves_metadata_order_scores_and_correctness():
    logits = torch.tensor([[0.0, 4.0, 1.0, -1.0], [3.0, 0.0, 0.0, 0.0]])
    options = [dict(A="first", B="second", C="third", D="fourth")] * 2
    predictions, rows = make_prediction_records(
        logits,
        ["qa-1", "qa-2"],
        ["B", "C"],
        ["Factual Perception"] * 2,
        0.02,
        qa_ids=["qa-1", "qa-2"],
        video_paths=["v/1.mp4", "v/2.mp4"],
        tasks=["Object Recognition", "Temporal Grounding"],
        questions=["q1", "q2"],
        choices=options,
    )
    assert [row["qa_id"] for row in predictions] == ["qa-1", "qa-2"]
    assert [list(row["options"]) for row in predictions] == [["A", "B", "C", "D"]] * 2
    assert [row["ground_truth"] for row in predictions] == ["B", "C"]
    assert [row["predicted_answer"] for row in predictions] == ["B", "A"]
    assert [row["is_correct"] for row in predictions] == [True, False]
    assert [row["correctness"] for row in predictions] == ["correct", "incorrect"]
    assert all(list(row["scores"]) == ["A", "B", "C", "D"] for row in predictions)
    assert all(abs(sum(row["scores"].values()) - 1.0) < 1e-6 for row in predictions)
    assert predictions[0]["prediction_confidence"] == max(predictions[0]["scores"].values())
    metrics = calculate_metrics(rows)
    assert metrics["task_accuracy"]["Object Recognition"]["accuracy"] == 1.0
    assert metrics["prediction_distribution"] == {"A": 1, "B": 1, "C": 0, "D": 0}


def test_unlabeled_prediction_does_not_invent_ground_truth():
    predictions, rows = make_prediction_records(
        torch.tensor([[0.0, 0.0, 3.0, 0.0]]), ["test-1"], [None], [None], 0.01,
        qa_ids=["test-1"],
    )
    assert predictions[0]["ground_truth"] is None
    assert predictions[0]["is_correct"] is None
    assert predictions[0]["correctness"] == "unlabeled"
    metrics = calculate_metrics(rows)
    assert metrics["labeled_count"] == 0
    assert metrics["accuracy"] is None


class _VisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.frames_seen = 0

    def forward(self, pixel_values):
        self.frames_seen += len(pixel_values)
        result = pixel_values.mean((1, 2, 3), keepdim=False)[:, None] * self.scale
        return type("Out", (), {"image_embeds": result.expand(-1, 512)})()


def test_visual_cache_reuses_video_embeddings_across_batches_and_deduplicates():
    frames = torch.rand(2, 4, 3, 224, 224)
    uncached_backbone = _VisionBackbone()
    uncached = CLIPVisionFeatureExtractor(uncached_backbone, input_normalization="zero_one")
    reference = uncached(frames)

    cached_backbone = _VisionBackbone()
    cached = CLIPVisionFeatureExtractor(
        cached_backbone,
        input_normalization="zero_one",
        feature_cache_namespace="dataset+preprocessing+clip-fingerprint",
    )
    one_video = frames[:1].expand(2, -1, -1, -1, -1).contiguous()
    first = cached(one_video, cache_keys=["same/video.mp4", "same/video.mp4"])
    second = cached(one_video, cache_keys=["same/video.mp4", "same/video.mp4"])
    assert torch.allclose(first, second)
    assert torch.allclose(first, reference[:1].expand_as(first))
    assert cached_backbone.frames_seen == 4
    assert cached.feature_cache_stats() == {"entries": 1, "hits": 2, "misses": 1}
    cached.clear_feature_cache()
    assert cached.feature_cache_stats() == {"entries": 0, "hits": 0, "misses": 0}


class _Tokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        values = [sum(text.encode("utf-8")) % 101 for text in texts]
        return {"input_ids": torch.tensor(values, dtype=torch.long).reshape(-1, 1)}


class _TextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, input_ids):
        return type("Out", (), {"text_embeds": (input_ids.float() * self.scale).expand(-1, 512)})()


def test_text_cache_reuses_duplicate_questions_and_choices():
    tokenizer = _Tokenizer()
    extractor = CLIPTextFeatureExtractor(_TextBackbone(), tokenizer)
    first = extractor(["question", "choice A", "question"])
    calls_after_first = tokenizer.calls
    second = extractor(["question", "choice A"])
    assert tokenizer.calls == calls_after_first
    assert torch.equal(first[0], first[2])
    assert torch.equal(first[:2], second)
    assert extractor.feature_cache_stats() == {"entries": 2, "hits": 5, "misses": 2}
