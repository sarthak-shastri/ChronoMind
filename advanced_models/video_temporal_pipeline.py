"""Compose frozen CLIP frame features with temporal video reasoning."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .temporal.temporal_encoder import TransformerTemporalEncoder
from .vision.feature_extractor import (
    EMBEDDING_DIM,
    IMAGE_SIZE,
    CLIPVisionFeatureExtractor,
)


class VideoTemporalPipeline(nn.Module):
    """Map a Sarthak-collated batch to frame and video representations.

    Expected input is a mapping containing ``video_frames`` with shape
    ``[B,T,3,224,224]``, RGB channels, float32 values and ImageNet
    normalization. Each batch must have one fixed frame count. Missing-video
    error strings emitted by Sarthak's ``missing_video="zeros"`` path are
    rejected when present in the collated mapping.

    The CLIP extractor runs its own inference-mode forward pass. The temporal
    encoder remains in the normal autograd graph and is trainable.
    """

    def __init__(
        self,
        feature_extractor: CLIPVisionFeatureExtractor,
        temporal_encoder: TransformerTemporalEncoder,
    ) -> None:
        super().__init__()
        if not isinstance(feature_extractor, CLIPVisionFeatureExtractor):
            raise TypeError(
                "feature_extractor must be a CLIPVisionFeatureExtractor"
            )
        if not isinstance(temporal_encoder, TransformerTemporalEncoder):
            raise TypeError(
                "temporal_encoder must be a TransformerTemporalEncoder"
            )
        if feature_extractor.input_normalization != "imagenet":
            raise ValueError(
                "Sarthak's default preprocessing is ImageNet-normalized; "
                "configure feature_extractor with input_normalization='imagenet'"
            )
        if feature_extractor.output_dtype != torch.float32:
            raise ValueError(
                "VideoTemporalPipeline expects float32 frame features; "
                "configure feature_extractor with output_dtype=torch.float32"
            )
        if temporal_encoder.embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Temporal encoder embedding_dim must be {EMBEDDING_DIM}, "
                f"got {temporal_encoder.embedding_dim}"
            )

        self.feature_extractor = feature_extractor
        self.temporal_encoder = temporal_encoder
        self.feature_extractor.vision_model.requires_grad_(False)
        self.feature_extractor.eval()

    def train(self, mode: bool = True) -> "VideoTemporalPipeline":
        """Train the temporal encoder while keeping the visual tower frozen/eval."""
        super().train(mode)
        self.feature_extractor.vision_model.requires_grad_(False)
        self.feature_extractor.eval()
        return self

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _validate_video_error_metadata(batch: Mapping, batch_size: int) -> None:
        """Reject Sarthak's known ``video_error`` list when it marks failures."""
        if "video_error" not in batch:
            return

        errors = batch["video_error"]
        if not isinstance(errors, (list, tuple)):
            raise TypeError(
                "video_error metadata from collate_reva must be a list or tuple"
            )
        if len(errors) != batch_size:
            raise ValueError(
                "video_error metadata length must match the video batch size"
            )

        invalid_entries = [
            index
            for index, error in enumerate(errors)
            if error is not None and not isinstance(error, str)
        ]
        if invalid_entries:
            raise TypeError(
                "video_error entries must be strings or None; invalid entries "
                f"at batch indices {invalid_entries}"
            )

        failed_entries = [
            (index, error)
            for index, error in enumerate(errors)
            if isinstance(error, str) and error.strip()
        ]
        if failed_entries:
            details = "; ".join(
                f"index {index}: {error}" for index, error in failed_entries[:3]
            )
            raise ValueError(
                "batch contains samples with video decoding errors "
                f"({details})"
            )

    def forward(self, batch: Mapping) -> dict[str, Tensor]:
        """Return ``frame_features`` [B,T,512] and ``video_features`` [B,512]."""
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping containing 'video_frames'")
        if "video_frames" not in batch:
            raise KeyError("batch is missing required key 'video_frames'")

        video_frames = batch["video_frames"]
        if not isinstance(video_frames, Tensor):
            raise TypeError("batch['video_frames'] must be a torch.Tensor")
        if video_frames.ndim != 5:
            raise ValueError(
                "batch['video_frames'] must have shape [B,T,3,224,224]"
            )
        if video_frames.shape[0] < 1:
            raise ValueError("video batch must not be empty")
        if video_frames.shape[1] < 1:
            raise ValueError("video sequence must contain at least one frame")
        if tuple(video_frames.shape[2:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                "batch['video_frames'] must have 3 RGB channels and "
                f"{IMAGE_SIZE}x{IMAGE_SIZE} frames; got "
                f"{tuple(video_frames.shape[2:])}"
            )
        if video_frames.dtype != torch.float32:
            raise TypeError(
                "batch['video_frames'] must have dtype torch.float32 "
                "(Sarthak's default preprocessing output); no dtype cast is applied"
            )

        batch_size, sequence_length = video_frames.shape[:2]
        if sequence_length > self.temporal_encoder.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds temporal encoder "
                f"max_seq_len ({self.temporal_encoder.max_seq_len})"
            )
        self._validate_video_error_metadata(batch, batch_size)

        vision_device = self._module_device(self.feature_extractor.vision_model)
        temporal_device = self._module_device(self.temporal_encoder)
        if vision_device != temporal_device:
            raise ValueError(
                "feature extractor and temporal encoder must be on the same "
                f"device; got {vision_device} and {temporal_device}"
            )
        if video_frames.device != vision_device:
            raise ValueError(
                "batch['video_frames'] and pipeline modules must be on the "
                f"same device; got {video_frames.device} and {vision_device}"
            )

        frame_features = self.feature_extractor(video_frames)
        expected_frame_shape = (batch_size, sequence_length, EMBEDDING_DIM)
        if tuple(frame_features.shape) != expected_frame_shape:
            raise ValueError(
                "feature extractor returned an incompatible shape: expected "
                f"{expected_frame_shape}, got {tuple(frame_features.shape)}"
            )
        if frame_features.dtype != self.temporal_encoder.positional_embedding.dtype:
            raise TypeError(
                "frame feature dtype must match temporal encoder parameter "
                f"dtype; got {frame_features.dtype} and "
                f"{self.temporal_encoder.positional_embedding.dtype}"
            )

        video_features = self.temporal_encoder(frame_features)
        expected_video_shape = (batch_size, EMBEDDING_DIM)
        if tuple(video_features.shape) != expected_video_shape:
            raise ValueError(
                "temporal encoder returned an incompatible shape: expected "
                f"{expected_video_shape}, got {tuple(video_features.shape)}"
            )

        return {
            "frame_features": frame_features,
            "video_features": video_features,
        }
