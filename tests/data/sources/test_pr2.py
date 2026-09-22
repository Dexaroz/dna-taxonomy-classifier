from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.pr2 import clean_name, parse_record, pr2_kingdom
from taxonomy_classifier.data.taxonomy import Rank, Taxon
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources.pr2 import Pr2Source

PODOSPORA = (
    "AB1.1.24_U|18S_rRNA|nucleus||Eukaryota|Obazoa|Opisthokonta|Fungi|Ascomycota|"
    "Pezizomycotina|Sordariomycetes|Podospora|Podospora_anserina"
)


def _parse(header: str) -> SequenceRecord | ExclusionReason:
    return parse_record(FastaRecord(header=header, sequence="ACGT"))


def test_parse_record_maps_pr2_ranks_to_canonical_ones() -> None:
    record = _parse(PODOSPORA)

    assert isinstance(record, SequenceRecord)
    assert record.accession == "AB1.1.24_U"
    assert record.lineage.names == (
        "Eukaryota",
        "Opisthokonta",
        "Ascomycota",
        "Pezizomycotina",
        "Sordariomycetes",
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
def test_parse_record_excludes_off_target_records(header: str) -> None:
    assert _parse(header) is ExclusionReason.OFF_TARGET


def test_parse_record_rejects_headers_with_wrong_field_count() -> None:
    with pytest.raises(MalformedHeaderError, match="does not have 13 fields"):
        _parse("AB1.1.24_U|18S_rRNA|nucleus")


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
    assert pr2_source.is_backbone


def test_read_yields_records_and_exclusions(pr2_source: Pr2Source, fixtures_dir: Path) -> None:
    outcomes = list(pr2_source.read(fixtures_dir, BackboneIndex()))

    assert [type(outcome).__name__ for outcome in outcomes[:2]] == ["SequenceRecord"] * 2
    assert outcomes[2:] == [ExclusionReason.OFF_TARGET, ExclusionReason.MALFORMED_HEADER]


@pytest.mark.parametrize(
    ("division", "subdivision", "expected"),
    [
        ("Opisthokonta", "Metazoa", Kingdom.ANIMALIA),
        ("Opisthokonta", "Fungi", Kingdom.FUNGI),
        ("Opisthokonta", "Choanoflagellata", Kingdom.PROTISTA),
        ("Streptophyta", "Streptophyta_X", Kingdom.PLANTAE),
        ("Chlorophyta", "Chlorophyta_X", Kingdom.PLANTAE),
        ("Rhodophyta", "Eurhodophytina", Kingdom.PLANTAE),
        ("Alveolata", "Dinoflagellata", Kingdom.PROTISTA),
        ("Picozoa", "Picozoa_X", Kingdom.PROTISTA),
    ],
)
def test_pr2_kingdom(division: str, subdivision: str, expected: Kingdom) -> None:
    assert pr2_kingdom(division, subdivision) is expected
