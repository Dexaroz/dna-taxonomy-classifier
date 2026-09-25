from dataclasses import dataclass
from functools import partial
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.kingdom import prokaryote_kingdom
from taxonomy_classifier.data.markers import Marker
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

SOURCE_NAME: Final = "silva"

_IDENTIFIER: Final = re.compile(r".+\.\d+\.\d+", re.ASCII)

_ORGANELLES: Final = frozenset({"Chloroplast", "Mitochondria"})

_PLACEHOLDER_NAMES: Final = frozenset({"incertae sedis"})

_PLACEHOLDER_PREFIXES: Final = ("uncultured", "unidentified")

_PLACEHOLDER_SUFFIXES: Final = ("--other",)


@dataclass(frozen=True, slots=True)
class SilvaHeader:
    accession: str
    taxa: tuple[str, ...]
    organism: str


@dataclass(frozen=True, slots=True)
class SilvaSource:
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


def parse_header(header: str) -> SilvaHeader:
    identifier, separator, description = header.partition(" ")

    if not separator:
        msg = f"SILVA header has no taxonomy: {header!r}"
        raise MalformedHeaderError(msg)

    if _IDENTIFIER.fullmatch(identifier) is None:
        msg = f"SILVA identifier is not '<accession>.<start>.<end>': {identifier!r}"
        raise MalformedHeaderError(msg)

    *taxa, organism = description.split(";")

    if not taxa or not all(taxon.strip() for taxon in taxa):
        msg = f"SILVA header has an empty or missing taxon: {header!r}"
        raise MalformedHeaderError(msg)

    return SilvaHeader(accession=identifier, taxa=tuple(taxa), organism=organism)


def parse_record(record: FastaRecord, *, backbone: BackboneIndex) -> Outcome:
    header = parse_header(record.header)

    if not _ORGANELLES.isdisjoint(header.taxa):
        return ExclusionReason.ORGANELLE

    domain = header.taxa[0]
    candidates = [taxon for taxon in reversed(header.taxa[1:]) if not is_placeholder(taxon)]
    mapped = backbone.map_lineage(candidates, domain=domain) or Lineage.domain_only(domain)
    lineage = with_species(mapped, header.organism, backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=header.accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=(
            Taxon(name=domain, rank=Rank.DOMAIN),
            *(Taxon(name=taxon, rank=None) for taxon in header.taxa[1:]),
        ),
        lineage=lineage,
        kingdom=prokaryote_kingdom(domain) or backbone.kingdom_of(lineage),
        marker=Marker.SSU,
    )


def with_species(lineage: Lineage, organism: str, *, backbone: BackboneIndex) -> Lineage:
    if lineage.get(Rank.GENUS) is None:
        return lineage

    _, species = split_organism(organism)

    if species is None:
        return lineage

    candidate = backbone.lookup_species(species, domain=lineage.domain)

    if candidate is None or candidate.truncate(Rank.GENUS) != lineage:
        return lineage

    return candidate


def is_placeholder(name: str) -> bool:
    normalized = name.strip().lower()

    if normalized in _PLACEHOLDER_NAMES:
        return True

    return normalized.startswith(_PLACEHOLDER_PREFIXES) or normalized.endswith(
        _PLACEHOLDER_SUFFIXES
    )
