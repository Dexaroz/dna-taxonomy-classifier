import pytest
import torch

from taxonomy_classifier.model.encoders import (
    CnnConfig,
    CnnEncoder,
    TransformerConfig,
    TransformerEncoder,
    rotate,
)
from taxonomy_classifier.model.tokens import collate

SMALL_CNN = CnnConfig(embedding_dim=8, channels=(16, 24), kernel_size=5, dropout=0.0)

SMALL_TRANSFORMER = TransformerConfig(
    embedding_dim=8,
    model_dim=16,
    token_kernel=4,
    token_stride=2,
    layers=2,
    heads=2,
    dropout=0.0,
)

SHORT = "ACGTTGCAACGTAGCTAGCTAGGATCCA"

LONG = SHORT * 3


def _encoders() -> list[CnnEncoder | TransformerEncoder]:
    torch.manual_seed(0)

    return [CnnEncoder(SMALL_CNN).eval(), TransformerEncoder(SMALL_TRANSFORMER).eval()]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": ()}, "at least one stage"),
        ({"kernel_size": 4}, "odd kernel size"),
        ({"dilations": (1,)}, "one dilation per block"),
    ],
)
def test_cnn_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CnnConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_dim": 30, "heads": 4}, "heads of even dimension"),
        ({"model_dim": 24, "heads": 8}, "heads of even dimension"),
        ({"token_kernel": 2, "token_stride": 4}, "token_kernel >= token_stride"),
        ({"layers": 0}, "at least one layer"),
    ],
)
def test_transformer_config_rejects_invalid_values(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TransformerConfig(**kwargs)


@pytest.mark.parametrize("encoder", _encoders(), ids=["cnn", "transformer"])
def test_encoders_return_one_feature_vector_per_sequence(
    encoder: CnnEncoder | TransformerEncoder,
) -> None:
    tokens, mask = collate([SHORT, LONG])

    with torch.no_grad():
        features = encoder(tokens, mask)

    assert features.shape == (2, encoder.output_dim)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("encoder", _encoders(), ids=["cnn", "transformer"])
def test_padding_does_not_change_the_features(encoder: CnnEncoder | TransformerEncoder) -> None:
    alone_tokens, alone_mask = collate([SHORT])
    batch_tokens, batch_mask = collate([SHORT, LONG])

    with torch.no_grad():
        alone = encoder(alone_tokens, alone_mask)[0]
        batched = encoder(batch_tokens, batch_mask)[0]

    assert torch.allclose(alone, batched, atol=1e-5)


def test_transformer_accepts_sequences_shorter_than_its_kmer_window() -> None:
    encoder = _encoders()[1]
    tokens, mask = collate(["AC"])

    with torch.no_grad():
        features = encoder(tokens, mask)

    assert torch.isfinite(features).all()


def test_rotary_embedding_depends_only_on_relative_position() -> None:
    torch.manual_seed(0)
    vector = torch.randn(1, 1, 1, 8).expand(1, 1, 10, 8)

    rotated = rotate(vector)[0, 0]
    scores = rotated @ rotated.T

    assert torch.allclose(scores[0, 3], scores[4, 7], atol=1e-5)
    assert torch.allclose(rotated.norm(dim=-1), vector[0, 0].norm(dim=-1), atol=1e-5)
