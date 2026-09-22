import pytest
import torch
from torch import nn
import torch.nn.functional as F

from taxonomy_classifier.model.heads import (
    HeadConfig,
    HierarchicalHeads,
    hierarchical_loss,
    predict,
)
from taxonomy_classifier.model.labels import IGNORE_INDEX


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"conditioning_dim": 0}, "conditioning_dim"), ({"dropout": 1.0}, "dropout")],
)
def test_head_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HeadConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("sizes", [(), (3, 0)])
def test_heads_need_classes_at_every_level(sizes: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="at least one class"):
        HierarchicalHeads(8, sizes, HeadConfig())


def test_heads_produce_one_logit_tensor_per_level() -> None:
    heads = HierarchicalHeads(8, (2, 5, 7), HeadConfig(conditioning_dim=4))

    logits = heads(torch.randn(3, 8))

    assert [tuple(level.shape) for level in logits] == [(3, 2), (3, 5), (3, 7)]


def test_each_level_is_conditioned_on_the_previous_prediction() -> None:
    torch.manual_seed(0)
    heads = HierarchicalHeads(8, (2, 5), HeadConfig(conditioning_dim=4, dropout=0.0)).eval()
    features = torch.randn(1, 8)

    first = heads.heads[0]

    assert isinstance(first, nn.Linear)

    with torch.no_grad():
        baseline = heads(features)[1]
        first.bias.add_(torch.tensor([10.0, -10.0]))
        shifted = heads(features)[1]

    assert not torch.allclose(baseline, shifted)


def test_loss_matches_cross_entropy_on_a_single_level() -> None:
    logits = [torch.randn(4, 3)]
    targets = torch.tensor([[0], [2], [1], [0]])

    loss = hierarchical_loss(logits, targets)

    assert torch.allclose(loss, F.cross_entropy(logits[0], targets[:, 0]))


def test_loss_ignores_missing_labels_and_averages_labeled_levels() -> None:
    logits = [torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 5)]
    targets = torch.tensor([[1, IGNORE_INDEX, IGNORE_INDEX], [2, 3, IGNORE_INDEX]])

    loss = hierarchical_loss(logits, targets)

    expected = (
        F.cross_entropy(logits[0], targets[:, 0]) + F.cross_entropy(logits[1][1:], targets[1:, 1])
    ) / 2

    assert torch.allclose(loss, expected)


def test_loss_is_zero_without_any_label() -> None:
    logits = [torch.randn(2, 3, requires_grad=True)]
    targets = torch.full((2, 1), IGNORE_INDEX)

    loss = hierarchical_loss(logits, targets)
    torch.autograd.backward(loss)

    assert loss.item() == 0.0


def test_predict_abstains_below_threshold_and_below_any_abstained_level() -> None:
    confident = torch.tensor([[10.0, 0.0]])
    unsure = torch.tensor([[0.1, 0.0]])

    prediction = predict([confident, unsure, confident], threshold=0.9)

    assert prediction.indices.tolist() == [[0, -1, -1]]
    assert prediction.confidence.shape == (1, 3)


def test_predict_threshold_zero_never_abstains() -> None:
    prediction = predict([torch.tensor([[0.0, 1.0]])], threshold=0.0)

    assert prediction.indices.tolist() == [[1]]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_predict_threshold_must_be_a_probability(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        predict([torch.zeros(1, 2)], threshold=threshold)
