from dataclasses import dataclass
import io
import sys
from typing import TYPE_CHECKING, Final
import zipfile

from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.organisms import split_organism
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, Lineage, Rank
from taxonomy_classifier.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.remote import RemoteFile

SOURCE_NAME: Final = "ncbi_taxonomy"

NODES_MEMBER: Final = "nodes.dmp"

LINEAGES_MEMBER: Final = "rankedlineage.dmp"

_SEPARATOR: Final = "\t|\t"

_TERMINATOR: Final = "\t|\n"

_LINEAGE_FIELDS: Final = 10

_KINGDOMS: Final = {
    "Metazoa": Kingdom.ANIMALIA,
    "Fungi": Kingdom.FUNGI,
    "Viridiplantae": Kingdom.PLANTAE,
}

_RANKS: Final = {rank.value: rank for rank in CANONICAL_RANKS[1:]}


@dataclass(frozen=True, slots=True)
class RankedTaxon:
    lineage: Lineage
    kingdom: Kingdom


@dataclass(frozen=True, slots=True)
class NcbiTaxonomy:
    release: str
    archive: RemoteFile
    domain: str = "Eukaryota"

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_{self.release}"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return (self.archive,)

    def load(self, raw_dir: Path, backbone: BackboneIndex) -> int:
        count = 0

        for taxon in read_taxa(raw_dir / self.archive.filename, domain=self.domain):
            backbone.add_lineage(taxon.lineage, taxon.kingdom)
            count += 1

        return count


def read_taxa(path: Path, *, domain: str) -> Iterator[RankedTaxon]:
    ranks = {
        fields[0]: _RANKS[fields[2]] for fields in _rows(path, NODES_MEMBER) if fields[2] in _RANKS
    }

    for fields in _rows(path, LINEAGES_MEMBER):
        rank = ranks.get(fields[0])

        if rank is None or len(fields) != _LINEAGE_FIELDS or fields[9] != domain:
            continue

        taxon = ranked_taxon(fields, rank)

        if taxon is not None:
            yield taxon


def ranked_taxon(fields: list[str], rank: Rank) -> RankedTaxon | None:
    name = fields[1]

    if rank is Rank.SPECIES and split_organism(name)[1] != name:
        return None

    ancestors = {
        Rank.DOMAIN: fields[9],
        Rank.PHYLUM: fields[7],
        Rank.CLASS: fields[6],
        Rank.ORDER: fields[5],
        Rank.FAMILY: fields[4],
        Rank.GENUS: fields[3],
    }
    names: dict[Rank, str | None] = {
        key: sys.intern(value) for key, value in ancestors.items() if value
    }
    names[rank] = name

    return RankedTaxon(
        lineage=Lineage.from_ranks(names).truncate(rank),
        kingdom=_KINGDOMS.get(fields[8], Kingdom.PROTISTA),
    )


def _rows(path: Path, member: str) -> Iterator[list[str]]:
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            msg = f"{path} has no {member}"
            raise DataError(msg)

        with io.TextIOWrapper(archive.open(member), encoding="utf-8") as lines:
            for line in lines:
                yield line.removesuffix(_TERMINATOR).split(_SEPARATOR)
