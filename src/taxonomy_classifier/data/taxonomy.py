from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.files import open_text
from taxonomy_classifier.exceptions import (
    DuplicateRankError,
    TaxonomyError,
    UnknownTaxonPathError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class Rank(StrEnum):
    DOMAIN = "domain"
    KINGDOM = "kingdom"
    PHYLUM = "phylum"
    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"


CANONICAL_RANKS: Final[tuple[Rank, ...]] = tuple(Rank)

type RankMap = Mapping[str, Rank | None]

_RANK_INDEX: Final = {rank: index for index, rank in enumerate(CANONICAL_RANKS)}

_RANK_VALUES: Final = frozenset(rank.value for rank in CANONICAL_RANKS)

_PLACEHOLDER_NAMES: Final = frozenset({"incertae sedis"})

_PLACEHOLDER_PREFIXES: Final = ("uncultured", "unidentified")

_PLACEHOLDER_SUFFIXES: Final = ("--other",)

_MIN_RANK_FIELDS: Final = 3


@dataclass(frozen=True, slots=True)
class Lineage:
    path: tuple[str, ...]
    names: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not self.path:
            msg = "Lineage path must not be empty"
            raise ValueError(msg)

        if len(self.names) != len(CANONICAL_RANKS):
            msg = f"Expected {len(CANONICAL_RANKS)} rank names, got {len(self.names)}"
            raise ValueError(msg)

    @property
    def domain(self) -> str:
        return self.path[0]

    def get(self, rank: Rank) -> str | None:
        return self.names[_RANK_INDEX[rank]]


def is_placeholder(name: str) -> bool:
    normalized = name.strip().lower()

    if normalized in _PLACEHOLDER_NAMES:
        return True

    return normalized.startswith(_PLACEHOLDER_PREFIXES) or normalized.endswith(
        _PLACEHOLDER_SUFFIXES
    )


def parse_rank_line(line: str) -> tuple[str, Rank | None]:
    fields = line.rstrip("\r\n").split("\t")

    if len(fields) < _MIN_RANK_FIELDS or not fields[0]:
        msg = f"Malformed taxonomy line: {line!r}"
        raise TaxonomyError(msg)

    path, rank_name = fields[0], fields[2]

    if rank_name not in _RANK_VALUES:
        return path, None

    return path, Rank(rank_name)


def load_rank_map(path: Path) -> RankMap:
    rank_map: dict[str, Rank | None] = {}

    with open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue

            taxon_path, rank = parse_rank_line(line)

            if taxon_path in rank_map:
                msg = f"Duplicate taxonomy path: {taxon_path!r}"
                raise TaxonomyError(msg)

            rank_map[taxon_path] = rank

    return rank_map


def parse_lineage(path: Sequence[str], rank_map: RankMap) -> Lineage:
    names: dict[Rank, str | None] = {}
    prefix = ""

    for taxon in path:
        prefix = f"{prefix}{taxon};"

        if prefix not in rank_map:
            raise UnknownTaxonPathError(prefix)

        rank = rank_map[prefix]

        if rank is None:
            continue

        if rank in names:
            raise DuplicateRankError(rank, prefix)

        names[rank] = None if is_placeholder(taxon) else taxon

    if names.get(Rank.DOMAIN) is None:
        msg = f"Lineage has no domain: {';'.join(path)!r}"
        raise TaxonomyError(msg)

    return Lineage(
        path=tuple(path),
        names=tuple(names.get(rank) for rank in CANONICAL_RANKS),
    )
