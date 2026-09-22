from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from taxonomy_classifier.data.records import SequenceRecord

_ORGANELLE_TAXA: Final = frozenset({"Chloroplast", "Mitochondria"})


class ExclusionReason(StrEnum):
    MALFORMED_HEADER = "malformed_header"
    INVALID_SEQUENCE = "invalid_sequence"
    UNKNOWN_TAXON_PATH = "unknown_taxon_path"
    DUPLICATE_RANK = "duplicate_rank"
    INVALID_LINEAGE = "invalid_lineage"
    ORGANELLE = "organelle"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    TOO_AMBIGUOUS = "too_ambiguous"


type Filter = Callable[[SequenceRecord], ExclusionReason | None]


@dataclass(frozen=True, slots=True)
class FilterConfig:
    min_length: int = 900
    max_length: int = 4000
    max_ambiguous_fraction: float = 0.01
    exclude_organelles: bool = True

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
    filters: list[Filter] = []

    if config.exclude_organelles:
        filters.append(_exclude_organelle)

    filters.append(_length_filter(config.min_length, config.max_length))
    filters.append(_ambiguity_filter(config.max_ambiguous_fraction))

    return tuple(filters)


def first_exclusion(record: SequenceRecord, filters: Sequence[Filter]) -> ExclusionReason | None:
    for check in filters:
        reason = check(record)

        if reason is not None:
            return reason

    return None


def _exclude_organelle(record: SequenceRecord) -> ExclusionReason | None:
    if _ORGANELLE_TAXA.isdisjoint(record.lineage.path):
        return None

    return ExclusionReason.ORGANELLE


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
