from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.markers import Marker
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.unite import parse_record, parse_taxonomy
from taxonomy_classifier.data.taxonomy import Lineage, Rank
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.unite import UniteSource

PODOSPORA = (
    "Podospora_anserina|AB1|SH1.10FU|refs|k__Fungi;p__Ascomycota;c__Sordariomycetes;"
    "o__Sordariales;f__Podosporaceae;g__Podospora;s__Podospora_anserina"
)


def _parse(header: str, backbone: BackboneIndex) -> SequenceRecord:
    return parse_record(FastaRecord(header=header, sequence="acgt"), backbone=backbone)


def test_parse_taxonomy_drops_placeholders_and_the_kingdom() -> None:
    names = parse_taxonomy(
        "k__Fungi;p__Ascomycota;c__Ascomycota_cls_Incertae_sedis;g__Podospora;s__Podospora_sp"
    )

    assert names == {
        Rank.PHYLUM: "Ascomycota",
        Rank.GENUS: "Podospora",
        Rank.SPECIES: "Podospora sp",
    }


def test_named_species_get_the_reference_lineage(backbone: BackboneIndex) -> None:
    record = _parse(PODOSPORA, backbone)

    assert record.source == "unite"
    assert record.accession == "AB1"
    assert record.marker is Marker.ITS
    assert record.lineage.get(Rank.SPECIES) == "Podospora anserina"
    assert record.kingdom is Kingdom.FUNGI
    assert record.source_lineage[0].rank is Rank.DOMAIN


def test_unnamed_species_fall_back_to_the_genus(backbone: BackboneIndex) -> None:
    record = _parse(PODOSPORA.replace("s__Podospora_anserina", "s__Podospora_sp"), backbone)

    assert record.lineage.deepest_rank is Rank.GENUS


def test_unknown_species_fall_back_to_the_deepest_known_rank(backbone: BackboneIndex) -> None:
    header = PODOSPORA.replace("g__Podospora;s__Podospora_anserina", "g__Nova;s__Nova_xenia")

    assert _parse(header, backbone).lineage.deepest_rank is Rank.FAMILY


def test_unknown_taxa_keep_only_the_domain(backbone: BackboneIndex) -> None:
    record = _parse("X_sp|AB2|SH2|reps|k__Eukaryota_kgd_Incertae_sedis;s__X_sp", backbone)

    assert record.lineage == Lineage.domain_only("Eukaryota")


@pytest.mark.parametrize("header", ["name|AB3", "name||SH|refs|k__Fungi"])
def test_parse_record_rejects_malformed_headers(header: str, backbone: BackboneIndex) -> None:
    with pytest.raises(MalformedHeaderError, match="UNITE header"):
        _parse(header, backbone)


def test_read_parses_the_archived_fasta(
    unite_source: UniteSource, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(unite_source.read(fixtures_dir, backbone))
    records = [outcome for outcome in outcomes if isinstance(outcome, SequenceRecord)]

    assert unite_source.slug == "unite_test"
    assert unite_source.files == (unite_source.archive,)
    assert not unite_source.is_backbone
    assert outcomes[-1] is ExclusionReason.MALFORMED_HEADER
    assert [record.lineage.deepest_rank for record in records] == [
        Rank.SPECIES,
        Rank.GENUS,
        Rank.DOMAIN,
    ]
