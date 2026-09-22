from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest

from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, Lineage, load_rank_map

if TYPE_CHECKING:
    from taxonomy_classifier.data.taxonomy import RankMap

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class RecordFactory(Protocol):
    def __call__(
        self,
        *,
        sequence: str = ...,
        path: tuple[str, ...] = ...,
    ) -> SequenceRecord: ...


@pytest.fixture
def silva_fasta() -> Path:
    return FIXTURES_DIR / "mini_silva.fasta"


@pytest.fixture
def silva_taxonomy() -> Path:
    return FIXTURES_DIR / "mini_tax_slv.txt"


@pytest.fixture
def rank_map(silva_taxonomy: Path) -> RankMap:
    return load_rank_map(silva_taxonomy)


@pytest.fixture
def make_record() -> RecordFactory:
    def factory(
        *,
        sequence: str = "ACGT" * 250,
        path: tuple[str, ...] = ("Bacteria", "Pseudomonadati"),
    ) -> SequenceRecord:
        names = (path[0], *(None,) * (len(CANONICAL_RANKS) - 1))

        return SequenceRecord(
            accession="AB000001",
            start=1,
            end=len(sequence),
            organism="Escherichia coli",
            sequence=sequence,
            lineage=Lineage(path=path, names=names),
        )

    return factory
