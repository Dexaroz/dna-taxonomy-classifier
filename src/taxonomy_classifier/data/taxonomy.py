from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Rank(StrEnum):
    DOMAIN = "domain"
    PHYLUM = "phylum"
    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"


CANONICAL_RANKS: Final[tuple[Rank, ...]] = tuple(Rank)

RANK_INDEX: Final = {rank: index for index, rank in enumerate(CANONICAL_RANKS)}


@dataclass(frozen=True, slots=True)
class Taxon:
    name: str
    rank: Rank | None


@dataclass(frozen=True, slots=True)
class Lineage:
    names: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(CANONICAL_RANKS):
            msg = f"Expected {len(CANONICAL_RANKS)} rank names, got {len(self.names)}"
            raise ValueError(msg)

        if not self.names[0]:
            msg = "Lineage must have a domain"
            raise ValueError(msg)

    @classmethod
    def from_ranks(cls, names: dict[Rank, str | None]) -> Lineage:
        return cls(names=tuple(names.get(rank) for rank in CANONICAL_RANKS))

    @classmethod
    def domain_only(cls, domain: str) -> Lineage:
        return cls.from_ranks({Rank.DOMAIN: domain})

    @property
    def domain(self) -> str:
        return self.names[0] or ""

    @property
    def deepest_rank(self) -> Rank:
        labeled = [rank for rank, name in zip(CANONICAL_RANKS, self.names, strict=True) if name]

        return labeled[-1]

    def get(self, rank: Rank) -> str | None:
        return self.names[RANK_INDEX[rank]]

    def truncate(self, rank: Rank) -> Lineage:
        depth = RANK_INDEX[rank] + 1
        padding = (None,) * (len(CANONICAL_RANKS) - depth)

        return Lineage(names=self.names[:depth] + padding)
