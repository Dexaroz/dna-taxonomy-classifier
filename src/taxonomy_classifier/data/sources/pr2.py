from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.kingdom import Kingdom
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

_DIVISION: Final = 2

_SUBDIVISION: Final = 3

_PLANT_DIVISIONS: Final = frozenset(
    {"Streptophyta", "Chlorophyta", "Prasinodermophyta", "Rhodophyta", "Glaucophyta"}
)

_OPISTHOKONT_KINGDOMS: Final = {"Metazoa": Kingdom.ANIMALIA, "Fungi": Kingdom.FUNGI}


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
        return True

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]:
        del backbone

        yield from parse_fasta_file(raw_dir / self.fasta.filename, parse_record)


def parse_record(record: FastaRecord) -> Outcome:
    fields = record.header.split("|")

    if len(fields) != _METADATA_FIELDS + len(PR2_RANKS):
        expected = _METADATA_FIELDS + len(PR2_RANKS)
        msg = f"PR2 header does not have {expected} fields: {record.header!r}"
        raise MalformedHeaderError(msg)

    accession, gene, organelle = fields[0], fields[1], fields[2]
    raw_names = fields[_METADATA_FIELDS:]

    if (gene, organelle, raw_names[0]) != _TARGET:
        return ExclusionReason.OFF_TARGET

    names: dict[Rank, str | None] = {}

    for raw_name, rank in zip(raw_names, PR2_RANKS, strict=True):
        if rank is not None:
            names[rank] = clean_name(raw_name, rank)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=tuple(
            Taxon(name=raw_name, rank=rank)
            for raw_name, rank in zip(raw_names, PR2_RANKS, strict=True)
            if raw_name
        ),
        lineage=Lineage.from_ranks(names),
        kingdom=pr2_kingdom(raw_names[_DIVISION], raw_names[_SUBDIVISION]),
    )


def pr2_kingdom(division: str, subdivision: str) -> Kingdom:
    if division in _PLANT_DIVISIONS:
        return Kingdom.PLANTAE

    return _OPISTHOKONT_KINGDOMS.get(subdivision, Kingdom.PROTISTA)


def clean_name(raw_name: str, rank: Rank) -> str | None:
    name = raw_name.strip()

    if not name or _PLACEHOLDER.search(name):
        return None

    if rank is not Rank.SPECIES:
        return name

    if _UNNAMED_SPECIES in name:
        return None

    return name.replace("_", " ")
