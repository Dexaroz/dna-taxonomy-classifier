from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import RecordFactory


def test_length_counts_every_base(make_record: RecordFactory) -> None:
    assert make_record(sequence="ACGTN").length == 5


def test_n_ambiguous_counts_non_canonical_bases(make_record: RecordFactory) -> None:
    assert make_record(sequence="ACGTNRY").n_ambiguous == 3
