from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.silva import (
    SilvaHeader,
    is_placeholder,
    parse_header,
    parse_record,
)
from taxonomy_classifier.data.taxonomy import Rank
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.silva import SilvaSource

PSEUDOMONAS = (
    "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Pseudomonadales;"
    "Pseudomonadaceae;Pseudomonas;Pseudomonas putida"
)


def _parse(header: str, backbone: BackboneIndex) -> SequenceRecord | ExclusionReason:
    return parse_record(FastaRecord(header=header, sequence="acgu"), backbone=backbone)


def test_parse_header_splits_identifier_taxa_and_organism() -> None:
    header = parse_header("AB.12.1.1343 Eukaryota;Phaseolus acutifolius (tepary bean)")

    assert header == SilvaHeader(
        accession="AB.12.1.1343",
        taxa=("Eukaryota",),
        organism="Phaseolus acutifolius (tepary bean)",
    )


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("AB1.1.24", "has no taxonomy"),
        ("AB1 Bacteria;Escherichia coli", "is not '<accession>.<start>.<end>'"),
        ("AB1.x.24 Bacteria;Escherichia coli", "is not '<accession>.<start>.<end>'"),
        ("AB1.1.24 Escherichia coli", "empty or missing taxon"),
        ("AB1.1.24 Bacteria;;Escherichia coli", "empty or missing taxon"),
    ],
)
def test_parse_header_rejects_malformed_headers(header: str, message: str) -> None:
    with pytest.raises(MalformedHeaderError, match=message):
        parse_header(header)


def test_parse_record_maps_the_deepest_known_name(backbone: BackboneIndex) -> None:
    record = _parse(f"AB1.1.24 {PSEUDOMONAS}", backbone)

    assert isinstance(record, SequenceRecord)
    assert record.source == "silva"
    assert record.sequence == "ACGT"
    assert record.lineage.get(Rank.GENUS) == "Pseudomonas"
    assert record.lineage.get(Rank.SPECIES) is None


def test_parse_record_climbs_past_unknown_names(backbone: BackboneIndex) -> None:
    header = PSEUDOMONAS.replace("Pseudomonas;", "Pseudomonas-Novel;")

    record = _parse(f"AB1.1.24 {header}", backbone)

    assert isinstance(record, SequenceRecord)
    assert record.lineage.deepest_rank is Rank.FAMILY
    assert record.lineage.get(Rank.FAMILY) == "Pseudomonadaceae"


def test_parse_record_maps_eukaryotes_onto_pr2(backbone: BackboneIndex) -> None:
    header = "AB1.1.24 Eukaryota;Amorphea;Obazoa;Fungi;Ascomycota;Sordariomycetes;uncultured"

    record = _parse(header, backbone)

    assert isinstance(record, SequenceRecord)
    assert record.lineage.names == (
        "Eukaryota",
        "Opisthokonta",
        "Ascomycota",
        "Pezizomycotina",
        "Sordariomycetes",
        None,
        None,
    )


def test_parse_record_skips_placeholders_and_falls_back_to_domain(
    backbone: BackboneIndex,
) -> None:
    record = _parse("AB1.1.24 Archaea;Unknownarchaeota;Incertae Sedis;uncultured", backbone)

    assert isinstance(record, SequenceRecord)
    assert record.lineage.deepest_rank is Rank.DOMAIN
    assert record.lineage.domain == "Archaea"


@pytest.mark.parametrize("organelle", ["Chloroplast", "Mitochondria"])
def test_parse_record_excludes_organelles(organelle: str, backbone: BackboneIndex) -> None:
    header = f"AB1.1.24 Bacteria;Bacillati;Cyanobacteriota;{organelle};Incertae Sedis;Zea mays"

    assert _parse(header, backbone) is ExclusionReason.ORGANELLE


@pytest.mark.parametrize(
    "name",
    ["Incertae Sedis", "incertae sedis ", "uncultured", "unidentified", "Chloroflexota--other"],
)
def test_placeholder_names_are_detected(name: str) -> None:
    assert is_placeholder(name)


@pytest.mark.parametrize("name", ["Escherichia-Shigella", "Sva0996 marine group", "Subgroup 9"])
def test_real_taxa_are_not_placeholders(name: str) -> None:
    assert not is_placeholder(name)


def test_source_metadata(silva_source: SilvaSource) -> None:
    assert silva_source.slug == "silva_test"
    assert silva_source.files == (silva_source.fasta,)
    assert not silva_source.is_backbone


def test_read_yields_records_and_exclusions(
    silva_source: SilvaSource, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(silva_source.read(fixtures_dir, backbone))

    assert outcomes.count(ExclusionReason.ORGANELLE) == 1
    assert outcomes.count(ExclusionReason.MALFORMED_HEADER) == 1
    assert sum(isinstance(outcome, SequenceRecord) for outcome in outcomes) == 5
