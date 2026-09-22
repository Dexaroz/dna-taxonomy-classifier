from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.base import parse_fasta_file
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, Lineage, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.remote import RemoteFile
    from taxonomy_classifier.data.sources.base import Outcome

SOURCE_NAME: Final = "gtdb"

_RANK_PREFIXES: Final = ("d__", "p__", "c__", "o__", "f__", "g__", "s__")

_PREFIX_LENGTH: Final = 3


@dataclass(frozen=True, slots=True)
class GtdbSource:
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


def parse_record(record: FastaRecord) -> SequenceRecord:
    accession, separator, description = record.header.partition(" ")

    if not separator:
        msg = f"GTDB header has no taxonomy: {record.header!r}"
        raise MalformedHeaderError(msg)

    fields = description.split(" [", 1)[0].split(";")

    if len(fields) != len(_RANK_PREFIXES) or not all(
        field.startswith(prefix) for field, prefix in zip(fields, _RANK_PREFIXES, strict=False)
    ):
        msg = f"GTDB taxonomy is not 'd__;p__;c__;o__;f__;g__;s__': {record.header!r}"
        raise MalformedHeaderError(msg)

    names = tuple(field[_PREFIX_LENGTH:].strip() or None for field in fields)

    if names[0] is None:
        msg = f"GTDB taxonomy has no domain: {record.header!r}"
        raise MalformedHeaderError(msg)

    return SequenceRecord(
        source=SOURCE_NAME,
        accession=accession,
        sequence=normalize_sequence(record.sequence),
        source_lineage=tuple(
            Taxon(name=name, rank=rank)
            for name, rank in zip(names, CANONICAL_RANKS, strict=True)
            if name is not None
        ),
        lineage=Lineage(names=names),
    )
