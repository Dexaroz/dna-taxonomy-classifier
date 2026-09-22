import pytest

from taxonomy_classifier.training.balancing import balanced_weights, balancing_keys


def test_balancing_keys_use_the_deepest_labeled_rank_up_to_depth() -> None:
    lineages = [
        ("Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Escherichia coli"),
        ("Bacteria", "Pseudomonadota", None, None),
        ("Bacteria", None, "Clostridia", None),
    ]

    assert balancing_keys(lineages, depth=3) == [
        "Bacteria;Pseudomonadota;Gammaproteobacteria",
        "Bacteria;Pseudomonadota",
        "Bacteria;;Clostridia",
    ]


def test_balancing_keys_reject_non_positive_depth() -> None:
    with pytest.raises(ValueError, match="depth must be positive"):
        balancing_keys([], depth=0)


def test_balancing_keys_reject_unlabeled_lineages() -> None:
    with pytest.raises(ValueError, match="no labeled rank"):
        balancing_keys([(None, None)], depth=2)


def test_power_zero_keeps_natural_frequencies() -> None:
    assert balanced_weights(["a", "a", "b"], power=0.0) == [1.0, 1.0, 1.0]


def test_power_one_gives_every_class_the_same_total_weight() -> None:
    keys = ["a"] * 9 + ["b"]

    weights = balanced_weights(keys, power=1.0)
    totals = {key: sum(w for k, w in zip(keys, weights, strict=True) if k == key) for key in "ab"}

    assert totals["a"] == pytest.approx(totals["b"])
    assert sum(weights) == pytest.approx(len(keys))


def test_intermediate_power_favors_rare_classes_without_equalizing() -> None:
    weights = balanced_weights(["a"] * 4 + ["b"], power=0.5)

    assert weights[-1] == pytest.approx(2 * weights[0])


def test_empty_input_gives_no_weights() -> None:
    assert balanced_weights([]) == []


@pytest.mark.parametrize("power", [-0.1, 1.1])
def test_power_must_be_in_unit_interval(power: float) -> None:
    with pytest.raises(ValueError, match="power must be in"):
        balanced_weights(["a"], power=power)
