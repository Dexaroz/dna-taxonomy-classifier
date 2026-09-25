from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.markers import MARKER_LENGTHS, LengthRange, Marker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from taxonomy_classifier.data.records import SequenceRecord

type Filter = Callable[[SequenceRecord], ExclusionReason | None]


@dataclass(frozen=True, slots=True)
class FilterConfig:
    min_length: int = 500
    max_length: int = 4000
    max_ambiguous_fraction: float = 0.01
    marker_lengths: Mapping[Marker, LengthRange] = field(
        default_factory=lambda: dict(MARKER_LENGTHS)
    )

    def __post_init__(self) -> None:
        LengthRange(min_length=self.min_length, max_length=self.max_length)

        if not 0.0 <= self.max_ambiguous_fraction <= 1.0:
            msg = f"max_ambiguous_fraction must be in [0, 1], got {self.max_ambiguous_fraction}"
            raise ValueError(msg)

    def length_range(self, marker: Marker) -> LengthRange:
        default = LengthRange(min_length=self.min_length, max_length=self.max_length)

        return self.marker_lengths.get(marker, default)


def build_filters(config: FilterConfig) -> tuple[Filter, ...]:
    return (
        _length_filter({marker: config.length_range(marker) for marker in Marker}),
        _ambiguity_filter(config.max_ambiguous_fraction),
    )


def first_exclusion(record: SequenceRecord, filters: Sequence[Filter]) -> ExclusionReason | None:
    for check in filters:
        reason = check(record)

        if reason is not None:
            return reason

    return None


def _length_filter(ranges: Mapping[Marker, LengthRange]) -> Filter:
    def check(record: SequenceRecord) -> ExclusionReason | None:
        allowed = ranges[record.marker]

        if record.length < allowed.min_length:
            return ExclusionReason.TOO_SHORT

        if record.length > allowed.max_length:
            return ExclusionReason.TOO_LONG

        return None

    return check


def _ambiguity_filter(max_fraction: float) -> Filter:
    def check(record: SequenceRecord) -> ExclusionReason | None:
        if record.n_ambiguous > max_fraction * record.length:
            return ExclusionReason.TOO_AMBIGUOUS

        return None

    return check
