from dataclasses import dataclass
from typing import TYPE_CHECKING

from taxonomy_classifier.data.files import open_text
from taxonomy_classifier.exceptions import MalformedFastaError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class FastaRecord:
    header: str
    sequence: str


def parse_fasta(lines: Iterable[str]) -> Iterator[FastaRecord]:
    header: str | None = None
    chunks: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                yield _build_record(header, chunks)

            header = _parse_header_line(line, line_number)
            chunks = []

            continue

        if header is None:
            msg = f"Sequence data before the first header at line {line_number}"
            raise MalformedFastaError(msg)

        chunks.append(line)

    if header is not None:
        yield _build_record(header, chunks)


def read_fasta(path: Path) -> Iterator[FastaRecord]:
    with open_text(path) as handle:
        yield from parse_fasta(handle)


def _parse_header_line(line: str, line_number: int) -> str:
    header = line[1:].strip()

    if not header:
        msg = f"Empty header at line {line_number}"
        raise MalformedFastaError(msg)

    return header


def _build_record(header: str, chunks: list[str]) -> FastaRecord:
    if not chunks:
        msg = f"Record {header!r} has no sequence"
        raise MalformedFastaError(msg)

    return FastaRecord(header=header, sequence="".join(chunks))
