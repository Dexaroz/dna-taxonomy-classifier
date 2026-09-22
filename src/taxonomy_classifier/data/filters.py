from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from taxonomy_classifier.data.exclusions import ExclusionReason

if TYPE_CHECKING:
    from collections.abc import Sequence

    from taxonomy_classifier.data.records import SequenceRecord

type Filter = Callable[[SequenceRecord], ExclusionReason | None]


@dataclass(frozen=True, slots=True)
class FilterConfig:
    min_length: int = 900
    max_length: int = 4000
    max_ambiguous_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.min_length < 1:
            msg = f"min_length must be positive, got {self.min_length}"
            raise ValueError(msg)

        if self.min_length > self.max_length:
            msg = f"min_length ({self.min_length}) exceeds max_length ({self.max_length})"
            raise ValueError(msg)

        if not 0.0 <= self.max_ambiguous_fraction <= 1.0:
            msg = f"max_ambiguous_fraction must be in [0, 1], got {self.max_ambiguous_fraction}"
            raise ValueError(msg)


def build_filters(config: FilterConfig) -> tuple[Filter, ...]:
    return (
        _length_filter(config.min_length, config.max_length),
        _ambiguity_filter(config.max_ambiguous_fraction),
    )


def first_exclusion(record: SequenceRecord, filters: Sequence[Filter]) -> ExclusionReason | None:
    for check in filters:
        reason = check(record)

        if reason is not None:
            return reason

    return None


def _length_filter(min_length: int, max_length: int) -> Filter:
    def check(record: SequenceRecord) -> ExclusionReason | None:
        if record.length < min_length:
            return ExclusionReason.TOO_SHORT

        if record.length > max_length:
            return ExclusionReason.TOO_LONG

        return None

    return check


def _ambiguity_filter(max_fraction: float) -> Filter:
    def check(record: SequenceRecord) -> ExclusionReason | None:
        if record.n_ambiguous > max_fraction * record.length:
            return ExclusionReason.TOO_AMBIGUOUS

        return None

    return check
