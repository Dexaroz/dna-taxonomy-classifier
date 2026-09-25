from dataclasses import dataclass
from functools import partial
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.fasta import read_fasta_member
from taxonomy_classifier.data.markers import Marker
from taxonomy_classifier.data.organisms import split_organism
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.base import parse_fasta_records
from taxonomy_classifier.data.taxonomy import Lineage, Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.remote import RemoteFile
    from taxonomy_classifier.data.sources.base import Outcome

SOURCE_NAME: Final = "unite"

EUKARYOTA: Final = "Eukaryota"

_HEADER_FIELDS: Final = 5

_PREFIXES: Final = {
    "p": Rank.PHYLUM,
    "c": Rank.CLASS,
    "o": Rank.ORDER,
    "f": Rank.FAMILY,
    "g": Rank.GENUS,
    "s": Rank.SPECIES,
}

_PLACEHOLDER: Final = re.compile(r"_(?:kgd|phy|cls|ord|fam|gen)_Incertae_sedis$")

_UNNAMED_SPECIES: Final = " sp"


@dataclass(frozen=True, slots=True)
class UniteSource:
    release: str
    archive: RemoteFile
    member: str

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_{self.release}"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return (self.archive,)

    @property
    def is_backbone(self) -> bool:
        return False

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        records = read_fasta_member(raw_dir / self.archive.filename, self.member)
        parse = partial(parse_record, backbone=backbone)

        yield from parse_fasta_records(records, parse)


def parse_record(record: FastaRecord, *, backbone: BackboneIndex) -> SequenceRecord:
    fields = record.header.split("|")

    if len(fields) != _HEADER_FIELDS or not fields[1]:
        msg = f"UNITE header is not 'name|accession|SH|type|taxonomy': {record.header!r}"
        raise MalformedHeaderError(msg)

    names = parse_taxonomy(fields[4])
    lineage = map_names(names, backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=fields[1],
        sequence=normalize_sequence(record.sequence),
        source_lineage=(
            Taxon(name=EUKARYOTA, rank=Rank.DOMAIN),
            *(Taxon(name=name, rank=rank) for rank, name in names.items()),
        ),
        lineage=lineage,
        kingdom=backbone.kingdom_of(lineage),
        marker=Marker.ITS,
    )


def parse_taxonomy(taxonomy: str) -> dict[Rank, str]:
    names: dict[Rank, str] = {}

    for field in taxonomy.split(";"):
        prefix, _, name = field.partition("__")
        rank = _PREFIXES.get(prefix)

        if rank is not None and name and _PLACEHOLDER.search(name) is None:
            names[rank] = name.replace("_", " ") if rank is Rank.SPECIES else name

    return names


def map_names(names: Mapping[Rank, str], *, backbone: BackboneIndex) -> Lineage:
    organism = names.get(Rank.SPECIES, "")
    _, species = split_organism(organism)

    if species is not None and not organism.endswith(_UNNAMED_SPECIES):
        lineage = backbone.lookup_species(species, domain=EUKARYOTA)

        if lineage is not None:
            return lineage

    candidates = [
        names[rank]
        for rank in (Rank.GENUS, Rank.FAMILY, Rank.ORDER, Rank.CLASS, Rank.PHYLUM)
        if rank in names
    ]

    return backbone.map_lineage(candidates, domain=EUKARYOTA) or Lineage.domain_only(EUKARYOTA)
