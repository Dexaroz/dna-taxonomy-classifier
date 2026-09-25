from dataclasses import dataclass
from typing import TYPE_CHECKING

from taxonomy_classifier.data.dna import count_ambiguous

if TYPE_CHECKING:
    from taxonomy_classifier.data.kingdom import Kingdom
    from taxonomy_classifier.data.markers import Marker
    from taxonomy_classifier.data.taxonomy import Lineage, Taxon


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    source: str
    accession: str
    sequence: str
    source_lineage: tuple[Taxon, ...]
    lineage: Lineage
    kingdom: Kingdom | None
    marker: Marker

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def n_ambiguous(self) -> int:
        return count_ambiguous(self.sequence)
