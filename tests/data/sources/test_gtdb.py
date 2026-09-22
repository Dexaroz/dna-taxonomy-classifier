from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.gtdb import parse_record
from taxonomy_classifier.data.taxonomy import Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources.gtdb import GtdbSource

ECOLI = (
    "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;"
    "f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
)


def _parse(header: str, sequence: str = "acgu") -> SequenceRecord:
    return parse_record(FastaRecord(header=header, sequence=sequence))


def test_parse_record_reads_the_full_lineage() -> None:
    record = _parse(f"RS_GCF_1.1~NC_1.1 {ECOLI} [location=3..1530] [ssu_len=1528]")

    assert record.source == "gtdb"
    assert record.accession == "RS_GCF_1.1~NC_1.1"
    assert record.sequence == "ACGT"
    assert record.lineage.names == (
        "Bacteria",
        "Pseudomonadota",
        "Gammaproteobacteria",
        "Enterobacterales",
        "Enterobacteriaceae",
        "Escherichia",
        "Escherichia coli",
    )
    assert record.source_lineage[-1] == Taxon(name="Escherichia coli", rank=Rank.SPECIES)
    assert record.kingdom is Kingdom.BACTERIA


def test_parse_record_treats_empty_ranks_as_missing() -> None:
    record = _parse(f"GB_GCA_1.1~C1 {ECOLI.removesuffix('Escherichia coli')}")

    assert record.lineage.get(Rank.SPECIES) is None
    assert record.lineage.deepest_rank is Rank.GENUS
    assert len(record.source_lineage) == 6


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("RS_GCF_1.1~NC_1.1", "has no taxonomy"),
        ("RS_GCF_1.1~NC_1.1 d__Bacteria;p__Pseudomonadota", "is not 'd__;p__"),
        (f"RS_GCF_1.1~NC_1.1 {ECOLI.replace('p__', 'x__')}", "is not 'd__;p__"),
        (f"RS_GCF_1.1~NC_1.1 {ECOLI.replace('d__Bacteria', 'd__')}", "has no domain"),
    ],
)
def test_parse_record_rejects_malformed_headers(header: str, message: str) -> None:
    with pytest.raises(MalformedHeaderError, match=message):
        _parse(header)


def test_source_metadata(gtdb_source: GtdbSource) -> None:
    assert gtdb_source.slug == "gtdb_test"
    assert gtdb_source.files == (gtdb_source.fasta,)
    assert gtdb_source.is_backbone


def test_read_yields_records_and_exclusions(gtdb_source: GtdbSource, fixtures_dir: Path) -> None:
    outcomes = list(gtdb_source.read(fixtures_dir, BackboneIndex()))

    assert len(outcomes) == 7
    assert outcomes[-1] is ExclusionReason.MALFORMED_HEADER
    assert all(isinstance(outcome, SequenceRecord) for outcome in outcomes[:-1])
