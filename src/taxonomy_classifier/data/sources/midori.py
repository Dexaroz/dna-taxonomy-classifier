from dataclasses import dataclass
from functools import partial
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
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
    from taxonomy_classifier.data.markers import Marker
    from taxonomy_classifier.data.remote import RemoteFile
    from taxonomy_classifier.data.sources.base import Outcome

SOURCE_NAME: Final = "midori2"

_TAXONOMY_TAG: Final = ";tax="

_FIELD_SEPARATOR: Final = re.compile(r",(?=[kpcofgs]:)")

_PREFIXES: Final = {
    "k": Rank.DOMAIN,
    "p": Rank.PHYLUM,
    "c": Rank.CLASS,
    "o": Rank.ORDER,
    "f": Rank.FAMILY,
    "g": Rank.GENUS,
    "s": Rank.SPECIES,
}

_TAXID: Final = re.compile(r"_\d+$")

_PLACEHOLDER: Final = re.compile(r"^(?:phylum|class|order|family|genus)_")


@dataclass(frozen=True, slots=True)
class Midori2Source:
    release: str
    marker: Marker
    fasta: RemoteFile

    @property
    def slug(self) -> str:
        return f"{SOURCE_NAME}_{self.release}_{self.marker}"

    @property
    def files(self) -> tuple[RemoteFile, ...]:
        return (self.fasta,)

    @property
    def is_backbone(self) -> bool:
        return False

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        parse = partial(parse_record, marker=self.marker, backbone=backbone)

        yield from parse_fasta_file(raw_dir / self.fasta.filename, parse)


def parse_record(record: FastaRecord, *, marker: Marker, backbone: BackboneIndex) -> SequenceRecord:
    accession, separator, taxonomy = record.header.rstrip(";").partition(_TAXONOMY_TAG)
    names = parse_taxonomy(taxonomy) if separator else {}
    domain = names.get(Rank.DOMAIN)

    if not accession or domain is None:
        msg = f"MIDORI2 header is not '<accession>;tax=k:<domain>,...': {record.header!r}"
        raise MalformedHeaderError(msg)

    lineage = map_names(names, domain=domain, backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=tuple(Taxon(name=name, rank=rank) for rank, name in names.items()),
        lineage=lineage,
        kingdom=backbone.kingdom_of(lineage),
        marker=marker,
    )


def parse_taxonomy(taxonomy: str) -> dict[Rank, str]:
    names: dict[Rank, str] = {}

    for field in _FIELD_SEPARATOR.split(taxonomy):
        prefix, _, value = field.partition(":")
        rank = _PREFIXES.get(prefix)
        name = _TAXID.sub("", value).strip()

        if rank is not None and name and _PLACEHOLDER.match(name) is None:
            names[rank] = name

    return names


def map_names(names: dict[Rank, str], *, domain: str, backbone: BackboneIndex) -> Lineage:
    _, species = split_organism(names.get(Rank.SPECIES, ""))

    if species is not None:
        lineage = backbone.lookup_species(species, domain=domain)

        if lineage is not None:
            return lineage

    candidates = [
        names[rank]
        for rank in (Rank.GENUS, Rank.FAMILY, Rank.ORDER, Rank.CLASS, Rank.PHYLUM)
        if rank in names
    ]

    return backbone.map_lineage(candidates, domain=domain) or Lineage.domain_only(domain)
