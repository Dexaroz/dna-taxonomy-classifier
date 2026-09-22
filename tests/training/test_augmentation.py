import random

import pytest

from taxonomy_classifier.training.augmentation import (
    AugmentationConfig,
    SequenceAugmenter,
    reverse_complement,
)

SEQUENCE = "ACGT" * 400

IDENTITY = AugmentationConfig(
    substitution_rate=0.0,
    insertion_rate=0.0,
    deletion_rate=0.0,
    crop_probability=0.0,
    reverse_complement_probability=0.0,
)


def _augmenter(config: AugmentationConfig, seed: int = 7) -> SequenceAugmenter:
    return SequenceAugmenter(config, random.Random(seed))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"substitution_rate": -0.1}, "substitution_rate must be in"),
        ({"insertion_rate": 1.5}, "insertion_rate must be in"),
        ({"reverse_complement_probability": 2.0}, "reverse_complement_probability must be in"),
        ({"crop_min_length": 0}, "crop lengths must satisfy"),
        ({"crop_min_length": 600, "crop_max_length": 500}, "crop lengths must satisfy"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AugmentationConfig(**kwargs)  # type: ignore[arg-type]


def test_identity_config_returns_the_sequence_unchanged() -> None:
    assert _augmenter(IDENTITY)(SEQUENCE) == SEQUENCE


def test_same_seed_gives_same_augmentation() -> None:
    config = AugmentationConfig(crop_probability=1.0, reverse_complement_probability=0.5)

    first = [_augmenter(config, seed=3)(SEQUENCE) for _ in range(3)]
    second = [_augmenter(config, seed=3)(SEQUENCE) for _ in range(3)]

    assert first == second


def test_crop_returns_a_contiguous_window_within_bounds() -> None:
    config = AugmentationConfig(crop_min_length=100, crop_max_length=200)
    augmenter = _augmenter(config)

    for _ in range(50):
        window = augmenter.crop(SEQUENCE)

        assert 100 <= len(window) <= 200
        assert window in SEQUENCE


def test_crop_keeps_sequences_shorter_than_the_window() -> None:
    config = AugmentationConfig(crop_min_length=100, crop_max_length=200)

    assert _augmenter(config).crop("ACGT") == "ACGT"


def test_substitutions_always_change_the_base() -> None:
    config = AugmentationConfig(substitution_rate=1.0, insertion_rate=0.0, deletion_rate=0.0)

    mutated = _augmenter(config).mutate("AAAACCCC")

    assert len(mutated) == 8
    assert all(new != old for new, old in zip(mutated, "AAAACCCC", strict=True))


def test_substitutions_replace_ambiguous_bases_with_canonical_ones() -> None:
    config = AugmentationConfig(substitution_rate=1.0, insertion_rate=0.0, deletion_rate=0.0)

    assert set(_augmenter(config).mutate("NNNN")) <= set("ACGT")


def test_deletions_shorten_and_insertions_lengthen() -> None:
    deleting = AugmentationConfig(substitution_rate=0.0, insertion_rate=0.0, deletion_rate=1.0)
    inserting = AugmentationConfig(substitution_rate=0.0, insertion_rate=1.0, deletion_rate=0.0)

    assert _augmenter(deleting).mutate("ACGTACGT") == ""
    assert len(_augmenter(inserting).mutate("ACGTACGT")) == 16


def test_mutation_rate_is_close_to_the_configured_one() -> None:
    config = AugmentationConfig(substitution_rate=0.01, insertion_rate=0.0, deletion_rate=0.0)
    sequence = "A" * 100_000

    mutated = _augmenter(config).mutate(sequence)
    changed = sum(base != "A" for base in mutated)

    assert 800 < changed < 1200


def test_mutating_an_empty_sequence_is_a_no_op() -> None:
    assert _augmenter(AugmentationConfig()).mutate("") == ""


def test_reverse_complement_is_applied_when_certain() -> None:
    config = AugmentationConfig(
        substitution_rate=0.0,
        insertion_rate=0.0,
        deletion_rate=0.0,
        crop_probability=0.0,
        reverse_complement_probability=1.0,
    )

    assert _augmenter(config)("AACGN") == "NCGTT"


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [("ACGT", "ACGT"), ("AAAC", "GTTT"), ("RYKM", "KMRY"), ("", "")],
)
def test_reverse_complement(sequence: str, expected: str) -> None:
    assert reverse_complement(sequence) == expected
