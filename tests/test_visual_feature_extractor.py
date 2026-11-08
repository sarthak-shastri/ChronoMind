from types import SimpleNamespace

import pytest
import torch
from torch import nn
import sys

from advanced_models.temporal import TransformerTemporalEncoder
from advanced_models.vision import CLIPVisionFeatureExtractor


class MockVisionBackbone(nn.Module):
    """Deterministic, parameterized stand-in for projected CLIP vision output."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 512, bias=False)
        self.batch_sizes = []

    def forward(self, pixel_values):
        self.batch_sizes.append(pixel_values.shape[0])
        pooled = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(image_embeds=self.projection(pooled))


def make_extractor(**kwargs):
    return CLIPVisionFeatureExtractor(
        MockVisionBackbone(), input_normalization="zero_one", **kwargs
    )


@pytest.mark.parametrize("batch_size,sequence_length", [(1, 1), (2, 4), (3, 8)])
def test_output_shape_for_batch_and_sequence_sizes(batch_size, sequence_length):
    extractor = make_extractor()

    result = extractor(torch.rand(batch_size, sequence_length, 3, 224, 224))

    assert result.shape == (batch_size, sequence_length, 512)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_backbone_is_frozen_and_remains_in_evaluation_mode():
    extractor = make_extractor()
    extractor.train()

    assert not extractor.training
    assert not extractor.vision_model.training
    assert all(not parameter.requires_grad for parameter in extractor.parameters())


def test_imagenet_inputs_are_inverted_then_clip_normalized():
    rgb = torch.tensor([0.2, 0.5, 0.8]).view(1, 1, 3, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(
        1, 1, 3, 1, 1
    )
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(
        1, 1, 3, 1, 1
    )
    imagenet_input = (rgb - imagenet_mean) / imagenet_std
    extractor = CLIPVisionFeatureExtractor(
        MockVisionBackbone(), input_normalization="imagenet"
    )

    converted = extractor._to_clip_normalized(
        imagenet_input.reshape(1, 3, 1, 1)
    )

    torch.testing.assert_close(
        converted.reshape(1, 1, 3, 1, 1), (rgb - clip_mean) / clip_std
    )


@pytest.mark.parametrize("mode", ["imagenet", "zero_one", "clip"])
def test_all_explicit_normalization_modes_are_supported(mode):
    extractor = CLIPVisionFeatureExtractor(
        MockVisionBackbone(), input_normalization=mode
    )
    frames = torch.rand(1, 2, 3, 224, 224)
    if mode == "imagenet":
        mean = frames.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = frames.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        frames = (frames - mean) / std
    elif mode == "clip":
        mean = frames.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(
            1, 1, 3, 1, 1
        )
        std = frames.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(
            1, 1, 3, 1, 1
        )
        frames = (frames - mean) / std

    assert extractor(frames).shape == (1, 2, 512)


def test_chunked_and_unchunked_outputs_match():
    torch.manual_seed(4)
    chunked_backbone = MockVisionBackbone()
    unchunked_backbone = MockVisionBackbone()
    unchunked_backbone.load_state_dict(chunked_backbone.state_dict())
    chunked = CLIPVisionFeatureExtractor(
        chunked_backbone, input_normalization="zero_one", frame_chunk_size=3
    )
    unchunked = CLIPVisionFeatureExtractor(
        unchunked_backbone, input_normalization="zero_one", frame_chunk_size=None
    )
    frames = torch.rand(2, 4, 3, 224, 224)

    chunked_result = chunked(frames)
    unchunked_result = unchunked(frames)

    torch.testing.assert_close(chunked_result, unchunked_result)
    assert chunked_backbone.batch_sizes == [3, 3, 2]
    assert unchunked_backbone.batch_sizes == [8]


@pytest.mark.parametrize(
    "frames",
    [
        torch.rand(2, 3, 224, 224),
        torch.rand(2, 3, 3, 3, 224, 224),
        torch.rand(2, 0, 3, 224, 224),
        torch.rand(2, 3, 1, 224, 224),
        torch.rand(2, 3, 3, 112, 112),
    ],
)
def test_invalid_shapes_are_rejected(frames):
    with pytest.raises(ValueError):
        make_extractor()(frames)


def test_invalid_normalization_mode_is_rejected():
    with pytest.raises(ValueError, match="input_normalization"):
        CLIPVisionFeatureExtractor(MockVisionBackbone(), input_normalization="auto")


def test_normalization_mode_must_be_explicit():
    with pytest.raises(TypeError):
        CLIPVisionFeatureExtractor(MockVisionBackbone())


def test_zero_one_mode_rejects_values_outside_expected_range():
    frames = torch.full((1, 1, 3, 224, 224), 1.2)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_extractor()(frames)


def test_runs_on_cpu_and_connects_to_temporal_transformer():
    extractor = make_extractor().cpu()
    temporal = TransformerTemporalEncoder(
        embedding_dim=512,
        num_heads=8,
        num_layers=1,
        dropout=0.0,
        max_seq_len=4,
    )
    features = extractor(torch.rand(2, 4, 3, 224, 224))

    assert features.device.type == "cpu"
    temporal(features).sum().backward()
    assert temporal.positional_embedding.grad is not None


def test_pretrained_loader_is_explicit_and_uses_injected_huggingface_api(
    monkeypatch,
):
    backbone = MockVisionBackbone()
    backbone.config = SimpleNamespace(projection_dim=512)
    calls = []

    class MockCLIPVisionModelWithProjection:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return backbone

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPVisionModelWithProjection=MockCLIPVisionModelWithProjection
        ),
    )

    extractor = CLIPVisionFeatureExtractor.from_pretrained(
        input_normalization="imagenet", revision="pinned-revision"
    )

    assert calls == [
        ("openai/clip-vit-base-patch32", {"revision": "pinned-revision"})
    ]
    assert extractor.vision_model is backbone


def test_missing_transformers_dependency_has_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match="optional 'transformers' package"):
        CLIPVisionFeatureExtractor.from_pretrained(input_normalization="imagenet")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_execution_and_optional_mixed_precision():
    extractor = make_extractor(
        cuda_mixed_precision=True, frame_chunk_size=2
    ).cuda()
    frames = torch.rand(1, 3, 3, 224, 224, device="cuda")

    result = extractor(frames)

    assert result.shape == (1, 3, 512)
    assert result.device.type == "cuda"
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()

