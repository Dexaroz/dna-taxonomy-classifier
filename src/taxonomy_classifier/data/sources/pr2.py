from dataclasses import dataclass
from functools import partial
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.markers import Marker
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

SOURCE_NAME: Final = "pr2"

PR2_RANKS: Final[tuple[Rank | None, ...]] = (
    Rank.DOMAIN,
    None,
    Rank.PHYLUM,
    None,
    Rank.CLASS,
    Rank.ORDER,
    Rank.FAMILY,
    Rank.GENUS,
    Rank.SPECIES,
)

_TARGET: Final = ("18S_rRNA", "nucleus", "Eukaryota")

_METADATA_FIELDS: Final = 4

_PLACEHOLDER: Final = re.compile(r"_X+$")

_UNNAMED_SPECIES: Final = "_sp."


@dataclass(frozen=True, slots=True)
class Pr2Source:
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


def parse_record(record: FastaRecord, *, backbone: BackboneIndex) -> Outcome:
    fields = record.header.split("|")

    if len(fields) != _METADATA_FIELDS + len(PR2_RANKS):
        expected = _METADATA_FIELDS + len(PR2_RANKS)
        msg = f"PR2 header does not have {expected} fields: {record.header!r}"
        raise MalformedHeaderError(msg)

    accession, gene, organelle = fields[0], fields[1], fields[2]
    raw_names = fields[_METADATA_FIELDS:]

    if (gene, organelle, raw_names[0]) != _TARGET:
        return ExclusionReason.OFF_TARGET

    lineage = map_names(raw_names, domain=raw_names[0], backbone=backbone)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=tuple(
            Taxon(name=raw_name, rank=rank)
            for raw_name, rank in zip(raw_names, PR2_RANKS, strict=True)
            if raw_name
        ),
        lineage=lineage,
        kingdom=backbone.kingdom_of(lineage),
        marker=Marker.SSU,
    )


def map_names(raw_names: list[str], *, domain: str, backbone: BackboneIndex) -> Lineage:
    species = clean_name(raw_names[-1], Rank.SPECIES)

    if species is not None:
        lineage = backbone.lookup_species(species, domain=domain)

        if lineage is not None:
            return lineage

    candidates = [
        name
        for name in (clean_name(raw_name, Rank.GENUS) for raw_name in reversed(raw_names[1:-1]))
        if name is not None
    ]

    return backbone.map_lineage(candidates, domain=domain) or Lineage.domain_only(domain)


def clean_name(raw_name: str, rank: Rank) -> str | None:
    name = raw_name.strip()

    if not name or _PLACEHOLDER.search(name):
        return None

    if rank is not Rank.SPECIES:
        return name

    if _UNNAMED_SPECIES in name:
        return None

    return name.replace("_", " ")
