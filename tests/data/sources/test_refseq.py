from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.fasta import FastaRecord
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.sources.refseq import parse_record, split_organism
from taxonomy_classifier.data.taxonomy import Rank
from taxonomy_classifier.exceptions import MalformedHeaderError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.sources.refseq import RefSeqSource


def _parse(header: str, backbone: BackboneIndex, domain: str = "Bacteria") -> SequenceRecord:
    return parse_record(
        FastaRecord(header=header, sequence="ACGT"),
        domain=domain,
        backbone=backbone,
    )


@pytest.mark.parametrize(
    ("organism", "expected"),
    [
        ("Escherichia coli strain K-12", ("Escherichia", "Escherichia coli")),
        ("Candidatus Novelbacter mysteriosus", ("Novelbacter", "Novelbacter mysteriosus")),
        ("Pseudomonas sp. 12", ("Pseudomonas", None)),
        ("Clostridiales bacterium", ("Clostridiales", None)),
        ("Pseudomonas", ("Pseudomonas", None)),
    ],
)
def test_split_organism(organism: str, expected: tuple[str, str | None]) -> None:
    assert split_organism(organism) == expected


def test_species_known_to_the_backbone_gets_its_full_lineage(backbone: BackboneIndex) -> None:
    record = _parse("NR_1.1 Escherichia coli strain K-12 16S ribosomal RNA, partial", backbone)

    assert record.source == "refseq"
    assert record.accession == "NR_1.1"
    assert record.lineage.get(Rank.SPECIES) == "Escherichia coli"
    assert record.lineage.get(Rank.FAMILY) == "Enterobacteriaceae"


def test_unknown_species_falls_back_to_its_genus(backbone: BackboneIndex) -> None:
    record = _parse("NR_2.1 Pseudomonas fictitia 16S ribosomal RNA, complete", backbone)

    assert record.lineage.deepest_rank is Rank.GENUS
    assert record.lineage.get(Rank.GENUS) == "Pseudomonas"


def test_unknown_genus_keeps_only_the_domain(backbone: BackboneIndex) -> None:
    record = _parse("NR_3.1 Candidatus Novelbacter mysteriosus 16S ribosomal RNA", backbone)

    assert record.lineage.deepest_rank is Rank.DOMAIN
    assert record.lineage.domain == "Bacteria"


def test_lookup_respects_the_file_domain(backbone: BackboneIndex) -> None:
    record = _parse("NR_4.1 Escherichia coli 16S ribosomal RNA", backbone, domain="Archaea")

    assert record.lineage.deepest_rank is Rank.DOMAIN
    assert record.lineage.domain == "Archaea"


@pytest.mark.parametrize(
    "header",
    ["NR_5.1 Escherichia coli 23S ribosomal RNA", "NR_6.1  16S ribosomal RNA", "NR_7.1"],
)
def test_parse_record_rejects_non_16s_headers(header: str, backbone: BackboneIndex) -> None:
    with pytest.raises(MalformedHeaderError, match="16S ribosomal RNA"):
        _parse(header, backbone)


def test_source_metadata(refseq_source: RefSeqSource) -> None:
    assert refseq_source.slug == "refseq_16s"
    assert refseq_source.files == (refseq_source.bacteria, refseq_source.archaea)
    assert not refseq_source.is_backbone


def test_read_parses_both_domains(
    refseq_source: RefSeqSource, fixtures_dir: Path, backbone: BackboneIndex
) -> None:
    outcomes = list(refseq_source.read(fixtures_dir, backbone))
    records = [outcome for outcome in outcomes if isinstance(outcome, SequenceRecord)]

    assert outcomes.count(ExclusionReason.MALFORMED_HEADER) == 1
    assert [record.lineage.domain for record in records] == ["Bacteria"] * 3 + ["Archaea"]
    assert records[-1].lineage.get(Rank.SPECIES) == "Methanobrevibacter smithii"


def test_unnamed_species_are_mapped_through_their_genus(backbone: BackboneIndex) -> None:
    record = _parse("NR_8.1 Pseudomonas sp. 12 16S ribosomal RNA, partial", backbone)

    assert record.lineage.deepest_rank is Rank.GENUS
    assert record.lineage.get(Rank.GENUS) == "Pseudomonas"
