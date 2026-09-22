import re

import pytest

from taxonomy_classifier.data.dna import count_ambiguous, normalize_sequence
from taxonomy_classifier.exceptions import InvalidSequenceError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACGU", "ACGT"),
        ("acgt", "ACGT"),
        ("AC GT\n", "ACGT"),
        ("acgun", "ACGTN"),
        ("RYSWKMBDHVN", "RYSWKMBDHVN"),
    ],
)
def test_normalize_sequence(raw: str, expected: str) -> None:
    assert normalize_sequence(raw) == expected


@pytest.mark.parametrize("raw", ["", "  \n\t"])
def test_normalize_sequence_rejects_empty(raw: str) -> None:
    with pytest.raises(InvalidSequenceError, match="empty"):
        normalize_sequence(raw)


@pytest.mark.parametrize(("raw", "invalid"), [("ACGX", "X"), ("AC-GT", "-"), ("AC.GT", ".")])
def test_normalize_sequence_rejects_non_iupac(raw: str, invalid: str) -> None:
    with pytest.raises(InvalidSequenceError, match=re.escape(f"'{invalid}'")):
        normalize_sequence(raw)


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [("ACGT", 0), ("NNNN", 4), ("ACNRT", 2), ("", 0)],
)
def test_count_ambiguous(sequence: str, expected: int) -> None:
    assert count_ambiguous(sequence) == expected
