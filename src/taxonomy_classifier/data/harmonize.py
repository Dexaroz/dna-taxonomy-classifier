from collections import defaultdict
from typing import TYPE_CHECKING

from taxonomy_classifier.data.taxonomy import Rank

if TYPE_CHECKING:
    from collections.abc import Iterable

    from taxonomy_classifier.data.records import SequenceRecord
    from taxonomy_classifier.data.taxonomy import Lineage


class BackboneIndex:
    def __init__(self) -> None:
        self._taxa: defaultdict[str, set[Lineage]] = defaultdict(set)
        self._species: defaultdict[str, set[Lineage]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._taxa) + len(self._species)

    def add(self, record: SequenceRecord) -> None:
        deepest: Rank | None = None

        for taxon in record.source_lineage:
            if taxon.rank is Rank.SPECIES:
                self._add_species(record.lineage)

                continue

            if taxon.rank is not None:
                deepest = taxon.rank

                if record.lineage.get(taxon.rank) is None:
                    continue

            if deepest is not None:
                self._taxa[taxon.name].add(record.lineage.truncate(deepest))

    def add_all(self, records: Iterable[SequenceRecord]) -> None:
        for record in records:
            self.add(record)

    def lookup(self, name: str, *, domain: str) -> Lineage | None:
        return _unique_in_domain(self._taxa.get(name), domain)

    def lookup_species(self, name: str, *, domain: str) -> Lineage | None:
        return _unique_in_domain(self._species.get(name), domain)

    def map_lineage(self, names: Iterable[str], *, domain: str) -> Lineage | None:
        for name in names:
            lineage = self.lookup(name, domain=domain)

            if lineage is not None:
                return lineage

        return None

    def _add_species(self, lineage: Lineage) -> None:
        species = lineage.get(Rank.SPECIES)

        if species is None:
            return

        self._species[species].add(lineage)


def _unique_in_domain(candidates: set[Lineage] | None, domain: str) -> Lineage | None:
    if not candidates:
        return None

    matches = [lineage for lineage in candidates if lineage.domain == domain]

    if len(matches) != 1:
        return None

    return matches[0]
