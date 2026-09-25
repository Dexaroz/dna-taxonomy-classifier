from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.markers import Marker
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.midori import parse_record, parse_taxonomy
from taxonomy_classifier.data.taxonomy import Lineage, Rank
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.midori import Midori2Source

PODOSPORA = (
    "AB1.1.<1.>658;tax=k:Eukaryota_2759,p:Ascomycota_4890,c:Sordariomycetes_147550,"
    "o:Sordariales_5139,f:Podosporaceae_2822337,g:Podospora_5144,s:Podospora anserina_5145;"
)


def _parse(header: str, backbone: BackboneIndex) -> SequenceRecord:
    return parse_record(
        FastaRecord(header=header, sequence="acgt"), marker=Marker.COI, backbone=backbone
    )


def test_parse_taxonomy_strips_taxids_and_placeholders() -> None:
    names = parse_taxonomy(
        "k:Eukaryota_2759,p:Arthropoda_6656,c:class_order_family_genus_X sp._1,"
        "g:Culex_7174,s:Culex pipiens, form molestus_7175"
    )

    assert names == {
        Rank.DOMAIN: "Eukaryota",
        Rank.PHYLUM: "Arthropoda",
        Rank.GENUS: "Culex",
        Rank.SPECIES: "Culex pipiens, form molestus",
    }


def test_named_species_get_the_reference_lineage(backbone: BackboneIndex) -> None:
    record = _parse(PODOSPORA, backbone)

    assert record.source == "midori2"
    assert record.accession == "AB1.1.<1.>658"
    assert record.sequence == "ACGT"
    assert record.marker is Marker.COI
    assert record.lineage.get(Rank.SPECIES) == "Podospora anserina"
    assert record.lineage.get(Rank.FAMILY) == "Podosporaceae"
    assert record.kingdom is Kingdom.FUNGI
    assert record.source_lineage[0].name == "Eukaryota"


def test_unnamed_species_fall_back_to_the_genus(backbone: BackboneIndex) -> None:
    record = _parse(PODOSPORA.replace("Podospora anserina_5145", "Podospora sp. X_9"), backbone)

    assert record.lineage.deepest_rank is Rank.GENUS


def test_unknown_species_fall_back_to_the_deepest_known_rank(backbone: BackboneIndex) -> None:
    header = PODOSPORA.replace(
        "g:Podospora_5144,s:Podospora anserina_5145", "g:Nova_1,s:Nova xenia_2"
    )

    assert _parse(header, backbone).lineage.deepest_rank is Rank.FAMILY


def test_placeholder_lineages_keep_only_the_domain(backbone: BackboneIndex) -> None:
    record = _parse("AB2.1;tax=k:Eukaryota_2759,p:phylum_class_X sp._1,s:X sp._1;", backbone)

    assert record.lineage == Lineage.domain_only("Eukaryota")
    assert record.kingdom is None


@pytest.mark.parametrize("header", ["AB3.1", ";tax=k:Eukaryota_2759", "AB4.1;tax=p:Chordata_1"])
def test_parse_record_rejects_headers_without_accession_or_domain(
    header: str, backbone: BackboneIndex
) -> None:
    with pytest.raises(MalformedHeaderError, match="MIDORI2 header"):
        _parse(header, backbone)


def test_read_parses_the_fixture(
    midori_source: Midori2Source, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(midori_source.read(fixtures_dir, backbone))
    records = [outcome for outcome in outcomes if isinstance(outcome, SequenceRecord)]

    assert midori_source.slug == "midori2_test_coi"
    assert midori_source.files == (midori_source.fasta,)
    assert not midori_source.is_backbone
    assert outcomes[-1] is ExclusionReason.MALFORMED_HEADER
    assert [record.lineage.deepest_rank for record in records] == [
        Rank.SPECIES,
        Rank.GENUS,
        Rank.DOMAIN,
    ]
