import torch

from taxonomy_classifier.model.tokens import AMBIGUOUS, PAD, collate, encode_sequence


def test_encode_sequence_maps_bases_and_ambiguity_codes() -> None:
    assert encode_sequence("ACGTNR").tolist() == [1, 2, 3, 4, AMBIGUOUS, AMBIGUOUS]


def test_collate_pads_and_masks_to_the_longest_sequence() -> None:
    tokens, mask = collate(["ACGT", "AC"])

    assert tokens.tolist() == [[1, 2, 3, 4], [1, 2, PAD, PAD]]
    assert mask.tolist() == [[True, True, True, True], [True, True, False, False]]
    assert tokens.dtype == torch.long


def test_collate_truncates_to_max_length() -> None:
    tokens, mask = collate(["ACGTACGT"], max_length=3)

    assert tokens.tolist() == [[1, 2, 3]]
    assert mask.all()


def test_collate_honours_a_minimum_width() -> None:
    tokens, mask = collate(["AC"], min_length=5)

    assert tokens.shape == (1, 5)
    assert mask.tolist() == [[True, True, False, False, False]]
