from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.eukribo import parse_record
from taxonomy_classifier.data.taxonomy import Rank
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.eukribo import EukRiboSource


def _parse(header: str, backbone: BackboneIndex) -> SequenceRecord:
    return parse_record(FastaRecord(header=header, sequence="acgu"), backbone=backbone)


def test_known_species_gets_the_backbone_lineage(backbone: BackboneIndex) -> None:
    record = _parse("AB1 Eukaryota|Opisthokonta|Fungi|g:Podospora|Podospora+anserina", backbone)

    assert record.source == "eukribo"
    assert record.accession == "AB1"
    assert record.sequence == "ACGT"
    assert record.lineage.get(Rank.SPECIES) == "Podospora anserina"
    assert record.lineage.get(Rank.CLASS) == "Sordariomycetes"
    assert record.kingdom is Kingdom.FUNGI
    assert record.source_lineage[0].rank is Rank.DOMAIN
    assert [taxon.name for taxon in record.source_lineage[1:]] == [
        "Opisthokonta",
        "Fungi",
        "g:Podospora",
        "Podospora+anserina",
    ]


def test_unknown_species_falls_back_to_the_labeled_genus(backbone: BackboneIndex) -> None:
    record = _parse("AB2 Eukaryota|Fungi|g:Podospora|Podospora+fictitia|strain=X", backbone)

    assert record.lineage.deepest_rank is Rank.GENUS
    assert record.lineage.get(Rank.GENUS) == "Podospora"


def test_species_without_a_genus_label_uses_its_first_word(backbone: BackboneIndex) -> None:
    record = _parse("AB3 Eukaryota|Opisthokonta|Podospora+fictitia", backbone)

    assert record.lineage.get(Rank.GENUS) == "Podospora"


def test_records_without_genus_map_through_their_deepest_known_clade(
    backbone: BackboneIndex,
) -> None:
    record = _parse("AB4 Eukaryota|Sar|Alveolata|Dinophyceae|core-dinos|strain=PSP1", backbone)

    assert record.lineage.deepest_rank is Rank.CLASS
    assert record.lineage.get(Rank.CLASS) == "Dinophyceae"
    assert record.kingdom is Kingdom.PROTISTA


def test_unknown_taxa_keep_only_the_domain(backbone: BackboneIndex) -> None:
    record = _parse("AB5 Eukaryota|Obscurozoa|Unknownia+novus", backbone)

    assert record.lineage.deepest_rank is Rank.DOMAIN
    assert record.lineage.domain == "Eukaryota"
    assert record.kingdom is None


@pytest.mark.parametrize("header", ["AB6", "AB7 Bacteria|Pseudomonadota"])
def test_parse_record_rejects_non_eukaryote_headers(header: str, backbone: BackboneIndex) -> None:
    with pytest.raises(MalformedHeaderError, match="Eukaryota"):
        _parse(header, backbone)


def test_read_parses_the_fixture(
    eukribo_source: EukRiboSource, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(eukribo_source.read(fixtures_dir, backbone))
    records = [outcome for outcome in outcomes if isinstance(outcome, SequenceRecord)]

    assert eukribo_source.slug == "eukribo_test"
    assert eukribo_source.files == (eukribo_source.fasta,)
    assert not eukribo_source.is_backbone
    assert outcomes.count(ExclusionReason.MALFORMED_HEADER) == 2
    assert [record.lineage.deepest_rank for record in records] == [
        Rank.SPECIES,
        Rank.GENUS,
        Rank.CLASS,
        Rank.DOMAIN,
    ]
