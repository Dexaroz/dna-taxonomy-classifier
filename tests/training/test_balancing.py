import pytest

from taxonomy_classifier.training.balancing import balancing_keys


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
