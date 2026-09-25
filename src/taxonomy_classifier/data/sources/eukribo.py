from dataclasses import dataclass
from functools import partial
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.markers import Marker
from taxonomy_classifier.data.organisms import split_organism
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.base import parse_fasta_file
from taxonomy_classifier.data.taxonomy import Lineage, Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.remote import RemoteFile
    from taxonomy_classifier.data.sources.base import Outcome

SOURCE_NAME: Final = "eukribo"

EUKARYOTA: Final = "Eukaryota"

_GENUS_PREFIX: Final = "g:"

_SPECIES_SEPARATOR: Final = "+"

_CLADE: Final = re.compile(r"[A-Za-z][\w-]*", re.ASCII)


@dataclass(frozen=True, slots=True)
class EukRiboSource:
    release: str
    fasta: RemoteFile

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_{self.release}"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return (self.fasta,)

    @property
    def is_backbone(self) -> bool:
        return False

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        parse = partial(parse_record, backbone=backbone)

        yield from parse_fasta_file(raw_dir / self.fasta.filename, parse)


def parse_record(record: FastaRecord, *, backbone: BackboneIndex) -> SequenceRecord:
    accession, separator, description = record.header.partition(" ")
    taxa = description.split("|")

    if not separator or taxa[0] != EUKARYOTA:
        msg = f"EukRibo header is not '<accession> Eukaryota|...': {record.header!r}"
        raise MalformedHeaderError(msg)

    lineage = map_taxa(taxa[1:], backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=(
            Taxon(name=EUKARYOTA, rank=Rank.DOMAIN),
            *(Taxon(name=taxon, rank=None) for taxon in taxa[1:]),
        ),
        lineage=lineage,
        kingdom=backbone.kingdom_of(lineage),
        marker=Marker.SSU,
    )


def map_taxa(taxa: Sequence[str], *, backbone: BackboneIndex) -> Lineage:
    labeled_genus = next(
        (taxon.removeprefix(_GENUS_PREFIX) for taxon in taxa if taxon.startswith(_GENUS_PREFIX)),
        None,
    )
    organism = next(
        (taxon.replace(_SPECIES_SEPARATOR, " ") for taxon in taxa if _SPECIES_SEPARATOR in taxon),
        "",
    )
    genus, species = split_organism(organism)

    if species is not None:
        lineage = backbone.lookup_species(species, domain=EUKARYOTA)

        if lineage is not None:
            return lineage

    clades = [taxon for taxon in reversed(taxa) if _CLADE.fullmatch(taxon)]
    candidates = [name for name in (labeled_genus, genus) if name] + clades

    return backbone.map_lineage(candidates, domain=EUKARYOTA) or Lineage.domain_only(EUKARYOTA)
