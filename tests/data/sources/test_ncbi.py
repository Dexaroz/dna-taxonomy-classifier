from typing import TYPE_CHECKING
import zipfile

import pytest

from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.sources.ncbi import read_taxa
from taxonomy_classifier.data.taxonomy import Lineage
from taxonomy_classifier.exceptions import DataError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources.ncbi import NcbiTaxonomy


def test_read_taxa_yields_named_eukaryote_taxa_at_canonical_ranks(fixtures_dir: Path) -> None:
    taxa = {
        taxon.lineage.get(taxon.lineage.deepest_rank) or "": taxon
        for taxon in read_taxa(fixtures_dir / "ncbi_taxdump.zip", domain="Eukaryota")
    }

    assert sorted(taxa) == [
        "Ascomycota",
        "Chlorophyta",
        "Dinophyceae",
        "Micromonas",
        "Micromonas pusilla",
        "Podospora",
        "Podospora anserina",
        "Podosporaceae",
        "Sordariales",
        "Sordariomycetes",
    ]
    assert taxa["Podospora anserina"].lineage.names == (
        "Eukaryota",
        "Ascomycota",
        "Sordariomycetes",
        "Sordariales",
        "Podosporaceae",
        "Podospora",
        "Podospora anserina",
    )
    assert taxa["Micromonas"].kingdom is Kingdom.PLANTAE
    assert taxa["Dinophyceae"].kingdom is Kingdom.PROTISTA
    assert taxa["Dinophyceae"].lineage.names[1] is None


def test_load_fills_the_backbone(ncbi_source: NcbiTaxonomy, fixtures_dir: Path) -> None:
    backbone = BackboneIndex()

    assert ncbi_source.slug == "ncbi_taxonomy_test"
    assert ncbi_source.files == (ncbi_source.archive,)
    assert ncbi_source.load(fixtures_dir, backbone) == 10

    podospora = backbone.lookup("Podospora", domain="Eukaryota")

    assert podospora is not None
    assert backbone.kingdom_of(podospora) is Kingdom.FUNGI
    assert backbone.lookup("Podospora", domain="Bacteria") is None
    assert backbone.lookup_species("Podospora sp. 1", domain="Eukaryota") is None


def test_domain_only_lineages_are_not_indexed() -> None:
    backbone = BackboneIndex()

    backbone.add_lineage(Lineage.domain_only("Eukaryota"), Kingdom.PROTISTA)

    assert len(backbone) == 0
    assert backbone.lookup("Eukaryota", domain="Eukaryota") is None


def test_read_taxa_requires_both_dump_files(tmp_path: Path) -> None:
    archive = tmp_path / "taxdump.zip"

    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nodes.dmp", "")

    with pytest.raises(DataError, match="has no rankedlineage"):
        list(read_taxa(archive, domain="Eukaryota"))
