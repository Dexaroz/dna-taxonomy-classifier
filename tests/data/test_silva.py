from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.silva import SilvaHeader, parse_header, to_sequence_record
from taxonomy_classifier.data.taxonomy import Rank
from taxonomy_classifier.exceptions import (
    InvalidSequenceError,
    MalformedHeaderError,
    UnknownTaxonPathError,
)

if TYPE_CHECKING:
    from taxonomy_classifier.data.taxonomy import RankMap

ESCHERICHIA_HEADER = (
    "AB000001.1.24 Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;"
    "Enterobacterales;Enterobacteriaceae;Escherichia-Shigella;Escherichia coli"
)


def test_parse_header_splits_identifier_taxa_and_organism() -> None:
    header = parse_header(
        "HG374866.9.1343 Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;"
        "Pseudomonadales;Pseudomonadaceae;Pseudomonas;Rhizoctonia solani"
    )

    assert header == SilvaHeader(
        accession="HG374866",
        start=9,
        end=1343,
        taxa=(
            "Bacteria",
            "Pseudomonadati",
            "Pseudomonadota",
            "Gammaproteobacteria",
            "Pseudomonadales",
            "Pseudomonadaceae",
            "Pseudomonas",
        ),
        organism="Rhizoctonia solani",
    )


def test_parse_header_keeps_dots_inside_the_accession() -> None:
    assert parse_header("AB.123.1.1343 Bacteria;Escherichia coli").accession == "AB.123"


def test_parse_header_keeps_organism_verbatim() -> None:
    header = parse_header("HO778519.8.1309 Eukaryota;Phaseolus acutifolius (tepary bean)")

    assert header.organism == "Phaseolus acutifolius (tepary bean)"


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("AB000001.1.24", "no taxonomy description"),
        ("AB000001 Bacteria;Escherichia coli", "is not '<accession>.<start>.<end>'"),
        ("AB000001.x.24 Bacteria;Escherichia coli", "is not '<accession>.<start>.<end>'"),
        (".1.24 Bacteria;Escherichia coli", "is not '<accession>.<start>.<end>'"),
        ("AB000001.1.24 Escherichia coli", "empty or missing taxon"),
        ("AB000001.1.24 Bacteria;;Escherichia coli", "empty or missing taxon"),
    ],
)
def test_parse_header_rejects_malformed_headers(header: str, message: str) -> None:
    with pytest.raises(MalformedHeaderError, match=message):
        parse_header(header)


def test_to_sequence_record_normalizes_sequence_and_lineage(rank_map: RankMap) -> None:
    record = to_sequence_record(
        FastaRecord(header=ESCHERICHIA_HEADER, sequence="auugaacgcuggcggcaggccuaa"),
        rank_map,
    )

    assert record.accession == "AB000001"
    assert (record.start, record.end) == (1, 24)
    assert record.organism == "Escherichia coli"
    assert record.sequence == "ATTGAACGCTGGCGGCAGGCCTAA"
    assert record.lineage.get(Rank.GENUS) == "Escherichia-Shigella"


def test_to_sequence_record_propagates_sequence_errors(rank_map: RankMap) -> None:
    with pytest.raises(InvalidSequenceError):
        to_sequence_record(FastaRecord(header=ESCHERICHIA_HEADER, sequence="ACGX"), rank_map)


def test_to_sequence_record_propagates_taxonomy_errors(rank_map: RankMap) -> None:
    header = "AB000011.1.4 Bacteria;Fakeota;Fakeus fakeus"

    with pytest.raises(UnknownTaxonPathError):
        to_sequence_record(FastaRecord(header=header, sequence="ACGT"), rank_map)
