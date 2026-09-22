from collections import defaultdict
from typing import TYPE_CHECKING

from taxonomy_classifier.data.taxonomy import Rank

if TYPE_CHECKING:
    from collections.abc import Iterable

    from taxonomy_classifier.data.kingdom import Kingdom
    from taxonomy_classifier.data.records import SequenceRecord
    from taxonomy_classifier.data.taxonomy import Lineage


class BackboneIndex:
    def __init__(self) -> None:
        self._taxa: defaultdict[str, set[Lineage]] = defaultdict(set)
        self._species: defaultdict[str, set[Lineage]] = defaultdict(set)
        self._kingdoms: defaultdict[Lineage, set[Kingdom]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._taxa) + len(self._species)

    def add(self, record: SequenceRecord) -> None:
        deepest: Rank | None = None

        for taxon in record.source_lineage:
            if taxon.rank is Rank.SPECIES:
                self._add_species(record)

                continue

            if taxon.rank is not None:
                deepest = taxon.rank

                if record.lineage.get(taxon.rank) is None:
                    continue

            if deepest is not None:
                truncated = record.lineage.truncate(deepest)

                self._taxa[taxon.name].add(truncated)
                self._add_kingdom(truncated, record.kingdom)

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

    def kingdom_of(self, lineage: Lineage) -> Kingdom | None:
        kingdoms = self._kingdoms.get(lineage)

        if not kingdoms or len(kingdoms) != 1:
            return None

        return next(iter(kingdoms))

    def _add_species(self, record: SequenceRecord) -> None:
        species = record.lineage.get(Rank.SPECIES)

        if species is None:
            return

        self._species[species].add(record.lineage)
        self._add_kingdom(record.lineage, record.kingdom)

    def _add_kingdom(self, lineage: Lineage, kingdom: Kingdom | None) -> None:
        if kingdom is not None:
            self._kingdoms[lineage].add(kingdom)


def _unique_in_domain(candidates: set[Lineage] | None, domain: str) -> Lineage | None:
    if not candidates:
        return None

    matches = [lineage for lineage in candidates if lineage.domain == domain]

    if len(matches) != 1:
        return None

    return matches[0]
