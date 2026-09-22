from typing import TYPE_CHECKING

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.sources.base import parse_fasta_file
from taxonomy_classifier.exceptions import InvalidSequenceError, MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord


def _fail_on(record: FastaRecord) -> ExclusionReason:
    if record.header == "bad-header":
        msg = "bad header"
        raise MalformedHeaderError(msg)

    if record.header == "bad-sequence":
        msg = "bad sequence"
        raise InvalidSequenceError(msg)

    return ExclusionReason.OFF_TARGET


def test_parse_fasta_file_translates_record_errors(tmp_path: Path) -> None:
    path = tmp_path / "records.fasta"
    path.write_text(">bad-header\nACGT\n>bad-sequence\nACGT\n>fine\nACGT\n", encoding="utf-8")

    assert list(parse_fasta_file(path, _fail_on)) == [
        ExclusionReason.MALFORMED_HEADER,
        ExclusionReason.INVALID_SEQUENCE,
        ExclusionReason.OFF_TARGET,
    ]
