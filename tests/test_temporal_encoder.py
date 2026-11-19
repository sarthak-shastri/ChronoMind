import pytest
import torch

from advanced_models.temporal import (
    MeanPoolingTemporalEncoder,
    TransformerTemporalEncoder,
)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sequence_length", [1, 4, 8])
def test_transformer_output_shape_for_supported_batches_and_lengths(
    batch_size, sequence_length
):
    encoder = TransformerTemporalEncoder(
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        max_seq_len=8,
    )
    result = encoder(torch.randn(batch_size, sequence_length, 16))

    assert result.shape == (batch_size, 16)


def test_mean_pooling_output_shape_and_values():
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    result = MeanPoolingTemporalEncoder()(features)

    assert result.shape == (2, 4)
    torch.testing.assert_close(result, features.mean(dim=1))


@pytest.mark.parametrize(
    "features",
    [torch.randn(3, 8), torch.randn(2, 3, 4, 1), torch.randn(2, 0, 4)],
)
def test_mean_pooling_rejects_invalid_dimensions_or_empty_sequences(features):
    with pytest.raises(ValueError):
        MeanPoolingTemporalEncoder()(features)


def test_transformer_rejects_invalid_rank_feature_dimension_and_length():
    encoder = TransformerTemporalEncoder(
        embedding_dim=16, num_heads=4, max_seq_len=5
    )

    with pytest.raises(ValueError, match="shape"):
        encoder(torch.randn(2, 16))
    with pytest.raises(ValueError, match="feature dimension"):
        encoder(torch.randn(2, 3, 12))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        encoder(torch.randn(2, 6, 16))


def test_transformer_outputs_are_finite():
    encoder = TransformerTemporalEncoder(
        embedding_dim=16, num_heads=4, dropout=0.0, max_seq_len=6
    )

    result = encoder(torch.randn(2, 6, 16))

    assert torch.isfinite(result).all()


def test_transformer_supports_backward_propagation():
    encoder = TransformerTemporalEncoder(
        embedding_dim=16, num_heads=4, dropout=0.0, max_seq_len=6
    )
    features = torch.randn(2, 5, 16, requires_grad=True)

    encoder(features).square().mean().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert encoder.positional_embedding.grad is not None
    assert torch.isfinite(encoder.positional_embedding.grad).all()


def test_transformer_validates_configuration():
    with pytest.raises(ValueError, match="divisible"):
        TransformerTemporalEncoder(embedding_dim=15, num_heads=4)
    with pytest.raises(ValueError, match="max_seq_len"):
        TransformerTemporalEncoder(embedding_dim=16, max_seq_len=0)
