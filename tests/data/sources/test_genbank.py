from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.markers import Marker
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.base import expected_filenames
from taxonomy_classifier.data.sources.genbank import parse_record
from taxonomy_classifier.data.taxonomy import Lineage, Rank
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.genbank import GenBankSource


def _parse(header: str, backbone: BackboneIndex) -> SequenceRecord:
    return parse_record(
        FastaRecord(header=header, sequence="acgt"),
        marker=Marker.RBCL,
        domain="Eukaryota",
        backbone=backbone,
    )


def test_named_species_get_the_reference_lineage(backbone: BackboneIndex) -> None:
    record = _parse("OR1.1 Micromonas pusilla voucher X rbcL gene, partial cds", backbone)

    assert record.source == "genbank"
    assert record.accession == "OR1.1"
    assert record.marker is Marker.RBCL
    assert record.lineage.get(Rank.SPECIES) == "Micromonas pusilla"
    assert record.kingdom is Kingdom.PLANTAE


def test_unverified_records_are_mapped_through_their_organism(backbone: BackboneIndex) -> None:
    record = _parse("OR2.1 UNVERIFIED: Micromonas fictitia rbcL gene", backbone)

    assert record.lineage.deepest_rank is Rank.GENUS


def test_unknown_organisms_keep_only_the_domain(backbone: BackboneIndex) -> None:
    record = _parse("OR3.1 Nowhere nothing rbcL gene", backbone)

    assert record.lineage == Lineage.domain_only("Eukaryota")
    assert record.kingdom is None


@pytest.mark.parametrize("header", ["OR4.1", "OR5.1 UNVERIFIED: "])
def test_parse_record_rejects_headers_without_organism(
    header: str, backbone: BackboneIndex
) -> None:
    with pytest.raises(MalformedHeaderError, match="GenBank header"):
        _parse(header, backbone)


def test_read_parses_the_downloaded_query(
    genbank_source: GenBankSource, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(genbank_source.read(fixtures_dir, backbone))
    records = [outcome for outcome in outcomes if isinstance(outcome, SequenceRecord)]

    assert genbank_source.slug == "genbank_rbcl_test"
    assert genbank_source.files == ()
    assert not genbank_source.is_backbone
    assert expected_filenames(genbank_source) == ["genbank_rbcl.fasta.gz"]
    assert outcomes[-1] is ExclusionReason.MALFORMED_HEADER
    assert [record.lineage.deepest_rank for record in records] == [Rank.SPECIES, Rank.GENUS]
