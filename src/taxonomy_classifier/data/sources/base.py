from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import read_fasta
from taxonomy_classifier.exceptions import InvalidSequenceError, MalformedHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from taxonomy_classifier.data.entrez import EntrezQuery
    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.records import SequenceRecord
    from taxonomy_classifier.data.remote import RemoteFile

type Outcome = SequenceRecord | ExclusionReason

type RecordParser = Callable[[FastaRecord], Outcome]


class RemoteBundle(Protocol):
    @property
    def slug(self) -> str: ...

    @property
    def files(self) -> tuple[RemoteFile, ...]: ...


@runtime_checkable
class QueriedBundle(Protocol):
    @property
    def queries(self) -> tuple[EntrezQuery, ...]: ...


class TaxonomySource(RemoteBundle, Protocol):
    def load(self, raw_dir: Path, backbone: BackboneIndex) -> int: ...


class DataSource(RemoteBundle, Protocol):
    @property
    def is_backbone(self) -> bool: ...

    def read(self, raw_dir: Path, backbone: BackboneIndex) -> Iterator[Outcome]: ...


def expected_filenames(bundle: RemoteBundle) -> list[str]:
    names = [remote.filename for remote in bundle.files]

    if isinstance(bundle, QueriedBundle):
        names.extend(query.filename for query in bundle.queries)

    return names


def parse_fasta_file(path: Path, parse: RecordParser) -> Iterator[Outcome]:
    yield from parse_fasta_records(read_fasta(path), parse)


def parse_fasta_records(records: Iterable[FastaRecord], parse: RecordParser) -> Iterator[Outcome]:
    for fasta_record in records:
        try:
            yield parse(fasta_record)

        except MalformedHeaderError:
            yield ExclusionReason.MALFORMED_HEADER

        except InvalidSequenceError:
            yield ExclusionReason.INVALID_SEQUENCE
