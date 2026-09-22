import pytest
import torch

from taxonomy_classifier.model.classifier import build_classifier, parameter_counts
from taxonomy_classifier.model.encoders import (
    CnnConfig,
    CnnEncoder,
    TransformerConfig,
    TransformerEncoder,
)
from taxonomy_classifier.model.heads import HeadConfig, hierarchical_loss
from taxonomy_classifier.model.labels import LEVELS, LabelSpace
from taxonomy_classifier.model.tokens import collate

SPACE = LabelSpace(
    classes={
        level: tuple(f"{level}-{index}" for index in range(index + 2))
        for index, level in enumerate(LEVELS)
    },
    min_count=1,
)

CONFIGS: list[CnnConfig | TransformerConfig] = [
    CnnConfig(embedding_dim=8, channels=(16, 24), kernel_size=5),
    TransformerConfig(
        embedding_dim=8, model_dim=16, token_kernel=4, token_stride=2, layers=1, heads=2
    ),
]


@pytest.mark.parametrize(
    ("config", "encoder_type"),
    list(zip(CONFIGS, (CnnEncoder, TransformerEncoder), strict=True)),
    ids=["cnn", "transformer"],
)
def test_build_classifier_wires_the_requested_encoder(
    config: CnnConfig | TransformerConfig,
    encoder_type: type,
) -> None:
    model = build_classifier(config, SPACE, HeadConfig(conditioning_dim=4))

    assert isinstance(model.encoder, encoder_type)


@pytest.mark.parametrize("config", CONFIGS, ids=["cnn", "transformer"])
def test_classifier_trains_end_to_end(config: CnnConfig | TransformerConfig) -> None:
    torch.manual_seed(0)
    model = build_classifier(config, SPACE)
    tokens, mask = collate(["ACGTACGTAGGCTA", "TTGACCA"])
    targets = torch.zeros((2, len(LEVELS)), dtype=torch.long)

    logits = model(tokens, mask)
    torch.autograd.backward(hierarchical_loss(logits, targets))

    assert [level.shape[-1] for level in logits] == list(SPACE.sizes)
    assert model.heads.heads[-1].weight.grad is not None
    assert model.encoder.embedding.weight.grad is not None


def test_parameter_counts_add_up() -> None:
    model = build_classifier(CONFIGS[0], SPACE)

    counts = parameter_counts(model)

    assert counts["total"] == counts["encoder"] + counts["heads"]
    assert counts["total"] == sum(parameter.numel() for parameter in model.parameters())
