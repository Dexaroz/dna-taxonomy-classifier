import gzip
from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.fasta import FastaRecord, parse_fasta, read_fasta
from taxonomy_classifier.exceptions import MalformedFastaError

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_fasta_joins_multiline_sequences() -> None:
    lines = [">seq1 description\n", "ACGT\n", "TTAA\n"]

    assert list(parse_fasta(lines)) == [FastaRecord(header="seq1 description", sequence="ACGTTTAA")]


def test_parse_fasta_yields_every_record_and_skips_blank_lines() -> None:
    lines = ["\n", ">a\r\n", "AC\r\n", "\n", ">b\n", "GT\n", "\n"]

    assert list(parse_fasta(lines)) == [
        FastaRecord(header="a", sequence="AC"),
        FastaRecord(header="b", sequence="GT"),
    ]


def test_parse_fasta_accepts_empty_input() -> None:
    assert list(parse_fasta([])) == []


def test_parse_fasta_rejects_sequence_before_header() -> None:
    with pytest.raises(MalformedFastaError, match="before the first header at line 1"):
        list(parse_fasta(["ACGT\n", ">a\n", "ACGT\n"]))


@pytest.mark.parametrize("header_line", [">\n", ">   \n"])
def test_parse_fasta_rejects_empty_headers(header_line: str) -> None:
    with pytest.raises(MalformedFastaError, match="Empty header at line 1"):
        list(parse_fasta([header_line, "ACGT\n"]))


@pytest.mark.parametrize(
    "lines",
    [
        [">a\n", ">b\n", "ACGT\n"],
        [">a\n", "ACGT\n", ">b\n"],
    ],
    ids=["middle", "last"],
)
def test_parse_fasta_rejects_records_without_sequence(lines: list[str]) -> None:
    with pytest.raises(MalformedFastaError, match="has no sequence"):
        list(parse_fasta(lines))


def test_read_fasta_reads_plain_files(silva_fasta: Path) -> None:
    records = list(read_fasta(silva_fasta))

    assert len(records) == 13
    assert records[0].sequence == "AUUGAACGCUGGCGGCAGGCCUAA"


def test_read_fasta_reads_gzip_files(tmp_path: Path) -> None:
    path = tmp_path / "records.fasta.gz"

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(">a\nACGT\n")

    assert list(read_fasta(path)) == [FastaRecord(header="a", sequence="ACGT")]
