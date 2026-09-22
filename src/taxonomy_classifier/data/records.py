from dataclasses import dataclass
from typing import TYPE_CHECKING

from taxonomy_classifier.data.dna import count_ambiguous

if TYPE_CHECKING:
    from taxonomy_classifier.data.taxonomy import Lineage


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    accession: str
    start: int
    end: int
    organism: str
    sequence: str
    lineage: Lineage

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def n_ambiguous(self) -> int:
        return count_ambiguous(self.sequence)
