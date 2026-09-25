from dataclasses import dataclass
from functools import partial
import re
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


@dataclass(frozen=True, slots=True)
class RefSeqLocus:
    fasta: RemoteFile
    domain: str
    marker: str

    @property
    def pattern(self) -> re.Pattern[str]:
        return re.compile(rf" {re.escape(self.marker)} (?:ribosomal RNA|rRNA)\b")


@dataclass(frozen=True, slots=True)
class RefSeqSource:
    name: str
    loci: tuple[RefSeqLocus, ...]

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_{self.name}"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return tuple(locus.fasta for locus in self.loci)

    @property
    def is_backbone(self) -> bool:
        return False

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        for locus in self.loci:
            parse = partial(parse_record, locus=locus, backbone=backbone)

            yield from parse_fasta_file(raw_dir / locus.fasta.filename, parse)


def parse_record(
    record: FastaRecord, *, locus: RefSeqLocus, backbone: BackboneIndex
) -> SequenceRecord:
    accession, _, description = record.header.partition(" ")
    match = locus.pattern.search(description)
    organism = description[: match.start()] if match else ""

    if not organism.strip():
        msg = (
            f"RefSeq header is not '<accession> <organism> {locus.marker} ribosomal RNA...': "
            f"{record.header!r}"
        )
        raise MalformedHeaderError(msg)

    genus, species = split_organism(organism)
    lineage = map_organism(genus, species, domain=locus.domain, backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=(
            Taxon(name=locus.domain, rank=Rank.DOMAIN),
            Taxon(name=organism, rank=None),
        ),
        lineage=lineage,
        kingdom=prokaryote_kingdom(locus.domain) or backbone.kingdom_of(lineage),
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
