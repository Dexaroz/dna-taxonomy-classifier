from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.kingdom import prokaryote_kingdom
from taxonomy_classifier.data.organisms import split_organism
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.base import parse_fasta_file
from taxonomy_classifier.data.taxonomy import Lineage, Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.remote import RemoteFile
    from taxonomy_classifier.data.sources.base import Outcome

SOURCE_NAME: Final = "refseq"

_MARKER: Final = " 16S ribosomal RNA"


@dataclass(frozen=True, slots=True)
class RefSeqSource:
    bacteria: RemoteFile
    archaea: RemoteFile

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_16s"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return (self.bacteria, self.archaea)

    @property
    def is_backbone(self) -> bool:
        return False

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        for remote, domain in ((self.bacteria, "Bacteria"), (self.archaea, "Archaea")):
            parse = partial(parse_record, domain=domain, backbone=backbone)

            yield from parse_fasta_file(raw_dir / remote.filename, parse)


def parse_record(record: FastaRecord, *, domain: str, backbone: BackboneIndex) -> SequenceRecord:
    accession, _, description = record.header.partition(" ")
    organism, marker, _ = description.partition(_MARKER)

    if not marker or not organism.strip():
        msg = (
            f"RefSeq header is not '<accession> <organism> 16S ribosomal RNA...': {record.header!r}"
        )
        raise MalformedHeaderError(msg)

    genus, species = split_organism(organism)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=(Taxon(name=domain, rank=Rank.DOMAIN), Taxon(name=organism, rank=None)),
        lineage=map_organism(genus, species, domain=domain, backbone=backbone),
        kingdom=prokaryote_kingdom(domain),
    )


def map_organism(
    genus: str,
    species: str | None,
    *,
    domain: str,
    backbone: BackboneIndex,
) -> Lineage:
    if species is not None:
        lineage = backbone.lookup_species(species, domain=domain)

        if lineage is not None:
            return lineage

    lineage = backbone.lookup(genus, domain=domain)

    if lineage is not None:
        return lineage

    return Lineage.domain_only(domain)
