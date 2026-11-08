from types import SimpleNamespace

import pytest
import torch
from torch import nn

from advanced_models.temporal import TransformerTemporalEncoder
from advanced_models.video_temporal_pipeline import VideoTemporalPipeline
from advanced_models.vision import CLIPVisionFeatureExtractor


class MockVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 512, bias=False)

    def forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(image_embeds=self.projection(pooled))


def make_pipeline(*, temporal_dim=512, max_seq_len=8):
    extractor = CLIPVisionFeatureExtractor(
        MockVisionBackbone(),
        input_normalization="imagenet",
        frame_chunk_size=3,
    )
    temporal = TransformerTemporalEncoder(
        embedding_dim=temporal_dim,
        num_heads=8 if temporal_dim % 8 == 0 else 4,
        num_layers=1,
        dropout=0.0,
        max_seq_len=max_seq_len,
    )
    return VideoTemporalPipeline(extractor, temporal)


def make_batch(batch_size=2, sequence_length=4):
    rgb = torch.rand(batch_size, sequence_length, 3, 224, 224)
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return {"video_frames": (rgb - mean) / std}


@pytest.mark.parametrize("sequence_length", [4, 8])
def test_end_to_end_shapes_for_fixed_frame_counts(sequence_length):
    pipeline = make_pipeline()

    result = pipeline(make_batch(batch_size=2, sequence_length=sequence_length))

    assert result["frame_features"].shape == (2, sequence_length, 512)
    assert result["video_features"].shape == (2, 512)
    assert result["frame_features"].device.type == "cpu"
    assert result["video_features"].device.type == "cpu"
    assert torch.isfinite(result["frame_features"]).all()
    assert torch.isfinite(result["video_features"]).all()


def test_visual_backbone_stays_frozen_and_in_eval_when_pipeline_trains():
    pipeline = make_pipeline()

    pipeline.train()

    assert pipeline.training
    assert pipeline.temporal_encoder.training
    assert not pipeline.feature_extractor.training
    assert not pipeline.feature_extractor.vision_model.training
    assert all(
        not parameter.requires_grad
        for parameter in pipeline.feature_extractor.vision_model.parameters()
    )


def test_train_and_eval_modes_propagate_to_temporal_encoder():
    pipeline = make_pipeline()

    pipeline.train()
    assert pipeline.temporal_encoder.training
    pipeline.eval()
    assert not pipeline.temporal_encoder.training
    assert not pipeline.feature_extractor.training


def test_gradients_flow_through_temporal_encoder_only():
    pipeline = make_pipeline()
    result = pipeline(make_batch())

    result["video_features"].square().mean().backward()

    assert pipeline.temporal_encoder.positional_embedding.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in pipeline.temporal_encoder.parameters()
        if parameter.requires_grad
    )
    assert all(
        parameter.grad is None
        for parameter in pipeline.feature_extractor.vision_model.parameters()
    )


@pytest.mark.parametrize(
    "shape,dtype,exception",
    [
        ((2, 3, 224, 224), torch.float32, ValueError),
        ((2, 4, 3, 3, 224, 224), torch.float32, ValueError),
        ((2, 4, 1, 224, 224), torch.float32, ValueError),
        ((2, 4, 3, 112, 112), torch.float32, ValueError),
        ((0, 4, 3, 224, 224), torch.float32, ValueError),
        ((2, 0, 3, 224, 224), torch.float32, ValueError),
        ((2, 4, 3, 224, 224), torch.float16, TypeError),
        ((2, 4, 3, 224, 224), torch.uint8, TypeError),
    ],
)
def test_invalid_video_tensor_shape_or_dtype_is_rejected(shape, dtype, exception):
    pipeline = make_pipeline()
    video_frames = torch.zeros(shape, dtype=dtype)

    with pytest.raises(exception):
        pipeline({"video_frames": video_frames})


def test_missing_video_frames_key_is_rejected():
    with pytest.raises(KeyError, match="video_frames"):
        make_pipeline()({"sample_id": ["sample-1"]})


def test_non_tensor_video_frames_is_rejected():
    with pytest.raises(TypeError, match="torch.Tensor"):
        make_pipeline()({"video_frames": [[[[[]]]]]})


def test_mismatched_video_error_metadata_is_rejected():
    batch = make_batch()
    batch["video_error"] = [None, "video decode failed"]

    with pytest.raises(ValueError, match="video decoding errors"):
        make_pipeline()(batch)


def test_malformed_video_error_metadata_is_rejected():
    batch = make_batch()
    batch["video_error"] = "video decode failed"

    with pytest.raises(TypeError, match="list or tuple"):
        make_pipeline()(batch)


def test_incompatible_embedding_dimension_is_rejected():
    with pytest.raises(ValueError, match="embedding_dim must be 512"):
        make_pipeline(temporal_dim=256)


def test_non_imagenet_extractor_configuration_is_rejected():
    extractor = CLIPVisionFeatureExtractor(
        MockVisionBackbone(), input_normalization="zero_one"
    )
    temporal = TransformerTemporalEncoder(
        embedding_dim=512, num_heads=8, num_layers=1
    )

    with pytest.raises(ValueError, match="input_normalization='imagenet'"):
        VideoTemporalPipeline(extractor, temporal)

