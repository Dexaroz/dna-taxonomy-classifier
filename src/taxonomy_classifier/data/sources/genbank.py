from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.organisms import split_organism
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.base import parse_fasta_file
from taxonomy_classifier.data.sources.refseq import map_organism
from taxonomy_classifier.data.taxonomy import Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from taxonomy_classifier.data.entrez import EntrezQuery
    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.markers import Marker
    from taxonomy_classifier.data.remote import RemoteFile
    from taxonomy_classifier.data.sources.base import Outcome

SOURCE_NAME: Final = "genbank"

_UNVERIFIED: Final = "UNVERIFIED: "


@dataclass(frozen=True, slots=True)
class GenBankSource:
    name: str
    marker: Marker
    query: EntrezQuery
    domain: str = "Eukaryota"

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_{self.name}"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return ()

    @property
    def queries(self) -> tuple[EntrezQuery, ...]:
        return (self.query,)

    @property
    def is_backbone(self) -> bool:
        return False

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        parse = partial(parse_record, marker=self.marker, domain=self.domain, backbone=backbone)

        yield from parse_fasta_file(raw_dir / self.query.filename, parse)


def parse_record(
    record: FastaRecord, *, marker: Marker, domain: str, backbone: BackboneIndex
) -> SequenceRecord:
    accession, _, description = record.header.partition(" ")
    organism = description.removeprefix(_UNVERIFIED).strip()

    if not accession or not organism:
        msg = f"GenBank header is not '<accession> <organism> ...': {record.header!r}"
        raise MalformedHeaderError(msg)

    genus, species = split_organism(organism)
    lineage = map_organism(genus, species, domain=domain, backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=(Taxon(name=domain, rank=Rank.DOMAIN), Taxon(name=organism, rank=None)),
        lineage=lineage,
        kingdom=backbone.kingdom_of(lineage),
        marker=marker,
    )
