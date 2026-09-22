from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import read_fasta
from taxonomy_classifier.exceptions import InvalidSequenceError, MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.records import SequenceRecord
    from taxonomy_classifier.data.remote import RemoteFile

type Outcome = SequenceRecord | ExclusionReason

type RecordParser = Callable[[FastaRecord], Outcome]


class DataSource(Protocol):
    @property
    def slug(self) -> str: ...

    @property
    def files(self) -> tuple[RemoteFile, ...]: ...

    @property
    def is_backbone(self) -> bool: ...

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]: ...


def parse_fasta_file(path: Path, parse: RecordParser) -> Iterator[Outcome]:
    for fasta_record in read_fasta(path):
        try:
            yield parse(fasta_record)

        except MalformedHeaderError:
            yield ExclusionReason.MALFORMED_HEADER

        except InvalidSequenceError:
            yield ExclusionReason.INVALID_SEQUENCE
