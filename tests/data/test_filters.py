from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.filters import FilterConfig, build_filters, first_exclusion

if TYPE_CHECKING:
    from conftest import RecordFactory

CONFIG = FilterConfig(min_length=10, max_length=20, max_ambiguous_fraction=0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_length": 0}, "min_length must be positive"),
        ({"min_length": 30, "max_length": 20}, "exceeds max_length"),
        ({"max_ambiguous_fraction": -0.1}, r"must be in \[0, 1\]"),
        ({"max_ambiguous_fraction": 1.5}, r"must be in \[0, 1\]"),
    ],
)
def test_filter_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FilterConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (9, ExclusionReason.TOO_SHORT),
        (10, None),
        (20, None),
        (21, ExclusionReason.TOO_LONG),
    ],
)
def test_length_bounds_are_inclusive(
    length: int,
    expected: ExclusionReason | None,
    make_record: RecordFactory,
) -> None:
    record = make_record(sequence="A" * length)

    assert first_exclusion(record, build_filters(CONFIG)) is expected


@pytest.mark.parametrize(
    ("n_ambiguous", "expected"),
    [(2, None), (3, ExclusionReason.TOO_AMBIGUOUS)],
)
def test_ambiguity_threshold_is_inclusive(
    n_ambiguous: int,
    expected: ExclusionReason | None,
    make_record: RecordFactory,
) -> None:
    record = make_record(sequence="N" * n_ambiguous + "A" * (20 - n_ambiguous))

    assert first_exclusion(record, build_filters(CONFIG)) is expected


def test_first_exclusion_reports_the_first_failing_filter(make_record: RecordFactory) -> None:
    record = make_record(sequence="N" * 5)

    assert first_exclusion(record, build_filters(CONFIG)) is ExclusionReason.TOO_SHORT


def test_first_exclusion_without_filters_keeps_everything(make_record: RecordFactory) -> None:
    assert first_exclusion(make_record(sequence="N"), ()) is None
