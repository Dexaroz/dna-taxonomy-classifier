from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.pr2 import clean_name, parse_record
from taxonomy_classifier.data.taxonomy import Lineage, Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.pr2 import Pr2Source

PODOSPORA = (
    "AB1.1.24_U|18S_rRNA|nucleus||Eukaryota|Obazoa|Opisthokonta|Fungi|Ascomycota|"
    "Pezizomycotina|Sordariomycetes|Podospora|Podospora_anserina"
)


def _parse(header: str, backbone: BackboneIndex) -> SequenceRecord | ExclusionReason:
    return parse_record(FastaRecord(header=header, sequence="ACGT"), backbone=backbone)


def _record(header: str, backbone: BackboneIndex) -> SequenceRecord:
    record = _parse(header, backbone)

    assert isinstance(record, SequenceRecord)

    return record


def test_parse_record_maps_species_to_the_reference_taxonomy(backbone: BackboneIndex) -> None:
    record = _record(PODOSPORA, backbone)

    assert record.accession == "AB1.1.24_U"
    assert record.lineage.names == (
        "Eukaryota",
        "Ascomycota",
        "Sordariomycetes",
        "Sordariales",
        "Podosporaceae",
        "Podospora",
        "Podospora anserina",
    )
    assert record.kingdom is Kingdom.FUNGI
    assert record.source_lineage[1] == Taxon(name="Obazoa", rank=None)
    assert record.source_lineage[3] == Taxon(name="Fungi", rank=None)
    assert len(record.source_lineage) == 9


@pytest.mark.parametrize(
    "header",
    [
        PODOSPORA.replace("18S_rRNA", "16S_rRNA"),
        PODOSPORA.replace("nucleus", "plastid"),
        PODOSPORA.replace("|Eukaryota|", "|Eukaryota:plas|"),
    ],
    ids=["gene", "organelle", "domain"],
)
def test_parse_record_excludes_off_target_records(header: str, backbone: BackboneIndex) -> None:
    assert _parse(header, backbone) is ExclusionReason.OFF_TARGET


def test_parse_record_rejects_headers_with_wrong_field_count(backbone: BackboneIndex) -> None:
    with pytest.raises(MalformedHeaderError, match="does not have 13 fields"):
        _parse("AB1.1.24_U|18S_rRNA|nucleus", backbone)


def test_unknown_species_falls_back_to_the_deepest_known_name(backbone: BackboneIndex) -> None:
    record = _record(PODOSPORA.replace("Podospora_anserina", "Podospora_fictitia"), backbone)

    assert record.lineage.deepest_rank is Rank.GENUS
    assert record.lineage.get(Rank.GENUS) == "Podospora"


def test_placeholders_are_skipped_when_mapping(backbone: BackboneIndex) -> None:
    record = _record(
        "AB2.1.24_U|18S_rRNA|nucleus||Eukaryota|TSAR|Alveolata|Dinoflagellata|Dinophyceae|"
        "Dinophyceae_X|Dinophyceae_XX|Dinophyceae_XXX|Dinophyceae_XXX_sp.",
        backbone,
    )

    assert record.lineage.names == ("Eukaryota", None, "Dinophyceae", None, None, None, None)
    assert record.kingdom is Kingdom.PROTISTA


def test_unknown_names_keep_only_the_domain(backbone: BackboneIndex) -> None:
    record = _record(
        "AB3.1.24_U|18S_rRNA|nucleus||Eukaryota|Nowhere|Nothing|Nil|Null|None|Void|Zero|Zero_nulla",
        backbone,
    )

    assert record.lineage == Lineage.domain_only("Eukaryota")
    assert record.kingdom is None


@pytest.mark.parametrize(
    ("raw_name", "rank", "expected"),
    [
        ("Dinophyceae", Rank.CLASS, "Dinophyceae"),
        ("Dinophyceae_X", Rank.ORDER, None),
        ("Dinophyceae_XXX", Rank.GENUS, None),
        ("MAST-1A", Rank.GENUS, "MAST-1A"),
        ("Podospora_anserina", Rank.SPECIES, "Podospora anserina"),
        ("Podospora_sp.", Rank.SPECIES, None),
        ("Dinophyceae_XXX_sp.", Rank.SPECIES, None),
        ("  ", Rank.GENUS, None),
    ],
)
def test_clean_name(raw_name: str, rank: Rank, expected: str | None) -> None:
    assert clean_name(raw_name, rank) == expected


def test_source_metadata(pr2_source: Pr2Source) -> None:
    assert pr2_source.slug == "pr2_test"
    assert pr2_source.files == (pr2_source.fasta,)
    assert not pr2_source.is_backbone


def test_read_yields_records_and_exclusions(
    pr2_source: Pr2Source, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(pr2_source.read(fixtures_dir, backbone))

    assert [type(outcome).__name__ for outcome in outcomes[:2]] == ["SequenceRecord"] * 2
    assert outcomes[2:] == [ExclusionReason.OFF_TARGET, ExclusionReason.MALFORMED_HEADER]
