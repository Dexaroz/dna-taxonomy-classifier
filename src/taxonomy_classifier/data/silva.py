from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.dna import normalize_sequence
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.taxonomy import parse_lineage
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.taxonomy import RankMap

_IDENTIFIER: Final = re.compile(r"(?P<accession>.+)\.(?P<start>\d+)\.(?P<end>\d+)", re.ASCII)


@dataclass(frozen=True, slots=True)
class SilvaHeader:
    accession: str
    start: int
    end: int
    taxa: tuple[str, ...]
    organism: str


def parse_header(header: str) -> SilvaHeader:
    identifier, separator, description = header.partition(" ")

    if not separator:
        msg = f"Header has no taxonomy description: {header!r}"
        raise MalformedHeaderError(msg)

    match = _IDENTIFIER.fullmatch(identifier)

    if match is None:
        msg = f"Header identifier is not '<accession>.<start>.<end>': {identifier!r}"
        raise MalformedHeaderError(msg)

    *taxa, organism = description.split(";")

    if not taxa or not all(taxon.strip() for taxon in taxa):
        msg = f"Header has an empty or missing taxon: {header!r}"
        raise MalformedHeaderError(msg)

    return SilvaHeader(
        accession=match["accession"],
        start=int(match["start"]),
        end=int(match["end"]),
        taxa=tuple(taxa),
        organism=organism,
    )


def to_sequence_record(record: FastaRecord, rank_map: RankMap) -> SequenceRecord:
    header = parse_header(record.header)

    return SequenceRecord(
        accession=header.accession,
        start=header.start,
        end=header.end,
        organism=header.organism,
        sequence=normalize_sequence(record.sequence),
        lineage=parse_lineage(header.taxa, rank_map),
    )
