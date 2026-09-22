from typing import TYPE_CHECKING

from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, Lineage, Rank, Taxon

if TYPE_CHECKING:
    from conftest import RecordFactory
    from taxonomy_classifier.data.records import SequenceRecord


def _native_record(make_record: RecordFactory, *names: str | None) -> SequenceRecord:
    taxa = tuple(
        Taxon(name=name, rank=rank)
        for name, rank in zip(names, CANONICAL_RANKS, strict=True)
        if name is not None
    )

    return make_record(lineage=Lineage(names=names), source_lineage=taxa)


def _native(make_record: RecordFactory, *names: str | None) -> BackboneIndex:
    index = BackboneIndex()

    index.add(_native_record(make_record, *names))

    return index


def test_lookup_returns_the_lineage_truncated_at_the_name(make_record: RecordFactory) -> None:
    index = _native(make_record, "Bacteria", "P", "C", "O", "F", "G", "G s")

    assert index.lookup("F", domain="Bacteria") == Lineage(
        names=("Bacteria", "P", "C", "O", "F", None, None)
    )
    assert index.lookup_species("G s", domain="Bacteria") == Lineage(
        names=("Bacteria", "P", "C", "O", "F", "G", "G s")
    )


def test_species_names_are_not_indexed_as_taxa(make_record: RecordFactory) -> None:
    index = _native(make_record, "Bacteria", "P", "C", "O", "F", "G", "G s")

    assert index.lookup("G s", domain="Bacteria") is None


def test_lookup_is_restricted_to_the_domain(make_record: RecordFactory) -> None:
    index = _native(make_record, "Bacteria", "P", "C", "O", "F", "G", "G s")

    assert index.lookup("F", domain="Archaea") is None
    assert index.lookup_species("G s", domain="Archaea") is None


def test_ambiguous_names_are_not_resolved(make_record: RecordFactory) -> None:
    index = BackboneIndex()

    index.add_all(
        [
            _native_record(make_record, "Bacteria", "P", "C", "O", "F1", "Clostridium", None),
            _native_record(make_record, "Bacteria", "P", "C", "O", "F2", "Clostridium", None),
        ]
    )

    assert index.lookup("Clostridium", domain="Bacteria") is None
    assert index.lookup("F2", domain="Bacteria") is not None


def test_non_canonical_taxa_map_to_the_deepest_canonical_rank_above(
    make_record: RecordFactory,
) -> None:
    index = BackboneIndex()
    lineage = Lineage.from_ranks({Rank.DOMAIN: "Eukaryota", Rank.PHYLUM: "Opisthokonta"})
    taxa = (
        Taxon(name="Eukaryota", rank=Rank.DOMAIN),
        Taxon(name="Obazoa", rank=None),
        Taxon(name="Opisthokonta", rank=Rank.PHYLUM),
        Taxon(name="Fungi", rank=None),
    )

    index.add(make_record(lineage=lineage, source_lineage=taxa))

    assert index.lookup("Obazoa", domain="Eukaryota") == Lineage.domain_only("Eukaryota")
    assert index.lookup("Fungi", domain="Eukaryota") == lineage


def test_masked_placeholder_taxa_are_not_indexed(make_record: RecordFactory) -> None:
    index = BackboneIndex()
    lineage = Lineage.from_ranks({Rank.DOMAIN: "Eukaryota", Rank.CLASS: "Dinophyceae"})
    taxa = (
        Taxon(name="Eukaryota", rank=Rank.DOMAIN),
        Taxon(name="Dinophyceae", rank=Rank.CLASS),
        Taxon(name="Dinophyceae_X", rank=Rank.ORDER),
        Taxon(name="Dinophyceae_XXX_sp.", rank=Rank.SPECIES),
    )

    index.add(make_record(lineage=lineage, source_lineage=taxa))

    assert index.lookup("Dinophyceae_X", domain="Eukaryota") is None
    assert index.lookup_species("Dinophyceae_XXX_sp.", domain="Eukaryota") is None
    assert len(index) == 2


def test_map_lineage_uses_the_first_resolvable_name(make_record: RecordFactory) -> None:
    index = _native(make_record, "Bacteria", "P", "C", "O", "F", "G", None)

    mapped = index.map_lineage(["Unknown", "F", "G"], domain="Bacteria")

    assert mapped is not None
    assert mapped.deepest_rank is Rank.FAMILY
    assert index.map_lineage(["Unknown"], domain="Bacteria") is None


def test_non_canonical_taxa_before_any_rank_are_ignored(make_record: RecordFactory) -> None:
    index = BackboneIndex()
    taxa = (Taxon(name="Root", rank=None), Taxon(name="Bacteria", rank=Rank.DOMAIN))

    index.add(make_record(lineage=Lineage.domain_only("Bacteria"), source_lineage=taxa))

    assert index.lookup("Root", domain="Bacteria") is None
    assert index.lookup("Bacteria", domain="Bacteria") == Lineage.domain_only("Bacteria")
