"""Temporal encoders for sequences of precomputed video features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_video_features(video_features: Tensor) -> None:
    if not isinstance(video_features, Tensor):
        raise TypeError("video_features must be a torch.Tensor")
    if video_features.ndim != 3:
        raise ValueError(
            "video_features must have shape [batch, sequence, feature_dim]"
        )
    if video_features.shape[0] < 1:
        raise ValueError("video_features must contain at least one video")
    if video_features.shape[1] < 1:
        raise ValueError("video_features must contain at least one timestep")
    if video_features.shape[2] < 1:
        raise ValueError("feature_dim must be positive")


class MeanPoolingTemporalEncoder(nn.Module):
    """Reference baseline that averages frame or clip features over time."""

    def __init__(self, embedding_dim: int | None = None) -> None:
        super().__init__()
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim

    def forward(self, video_features: Tensor) -> Tensor:
        """Encode ``[B, T, D]`` features into ``[B, D]`` representations."""
        _validate_video_features(video_features)
        if self.embedding_dim is not None and video_features.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected feature dimension {self.embedding_dim}, got {video_features.shape[2]}"
            )
        return video_features.mean(dim=1)


class TransformerTemporalEncoder(nn.Module):
    """Transformer encoder with learnable positions and temporal mean pooling.

    This module consumes already extracted frame or clip features. It does not
    decode video or depend on any particular visual backbone.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        feedforward_dim = (
            4 * embedding_dim if dim_feedforward is None else dim_feedforward
        )
        if feedforward_dim < 1:
            raise ValueError("dim_feedforward must be positive")

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.positional_embedding = nn.Parameter(
            torch.empty(1, max_seq_len, embedding_dim)
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, video_features: Tensor) -> Tensor:
        """Encode ``[B, T, D]`` features into ``[B, D]`` representations."""
        _validate_video_features(video_features)
        if video_features.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected feature dimension {self.embedding_dim}, "
                f"got {video_features.shape[2]}"
            )
        sequence_length = video_features.shape[1]
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max_seq_len "
                f"({self.max_seq_len})"
            )

        positioned = video_features + self.positional_embedding[:, :sequence_length]
        encoded = self.transformer(positioned)
        return encoded.mean(dim=1)


def build_temporal_encoder(
    encoder_type: str,
    *,
    embedding_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_seq_len: int = 16,
) -> nn.Module:
    """Create a configured temporal encoder from checkpoint/CLI metadata."""
    if encoder_type == "mean_pooling":
        return MeanPoolingTemporalEncoder(embedding_dim=embedding_dim)
    if encoder_type == "transformer":
        return TransformerTemporalEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    raise ValueError("encoder_type must be 'mean_pooling' or 'transformer'")


def build_temporal_encoder_from_config(config: dict) -> nn.Module:
    """Reconstruct a checkpoint encoder, defaulting legacy checkpoints to Transformer."""
    return build_temporal_encoder(
        config.get("temporal_encoder_type", "transformer"),
        embedding_dim=config.get("embedding_dim", 512),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 16),
    )

