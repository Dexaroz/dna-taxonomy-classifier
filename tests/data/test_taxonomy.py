from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.data.taxonomy import (
    CANONICAL_RANKS,
    Lineage,
    Rank,
    is_placeholder,
    load_rank_map,
    parse_lineage,
    parse_rank_line,
)
from taxonomy_classifier.exceptions import (
    DuplicateRankError,
    TaxonomyError,
    UnknownTaxonPathError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.taxonomy import RankMap

ESCHERICHIA = (
    "Bacteria",
    "Pseudomonadati",
    "Pseudomonadota",
    "Gammaproteobacteria",
    "Enterobacterales",
    "Enterobacteriaceae",
    "Escherichia-Shigella",
)

SORDARIOMYCETES = (
    "Eukaryota",
    "Amorphea",
    "Obazoa",
    "Opisthokonta",
    "Nucletmycea",
    "Fungi",
    "Dikarya",
    "Ascomycota",
    "Pezizomycotina",
    "Sordariomycetes",
)


def test_canonical_ranks_follow_hierarchy() -> None:
    assert CANONICAL_RANKS == (
        Rank.DOMAIN,
        Rank.KINGDOM,
        Rank.PHYLUM,
        Rank.CLASS,
        Rank.ORDER,
        Rank.FAMILY,
        Rank.GENUS,
    )


def test_lineage_exposes_domain_and_ranks() -> None:
    lineage = Lineage(path=("Bacteria",), names=("Bacteria", None, None, None, None, None, "E"))

    assert lineage.domain == "Bacteria"
    assert lineage.get(Rank.GENUS) == "E"
    assert lineage.get(Rank.PHYLUM) is None


def test_lineage_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Lineage(path=(), names=(None,) * len(CANONICAL_RANKS))


def test_lineage_rejects_wrong_number_of_names() -> None:
    with pytest.raises(ValueError, match="Expected 7 rank names, got 2"):
        Lineage(path=("Bacteria",), names=("Bacteria", None))


@pytest.mark.parametrize(
    "name",
    [
        "Incertae Sedis",
        "incertae sedis ",
        "uncultured",
        "unidentified marine bacterium",
        "Chloroflexota--other",
    ],
)
def test_placeholder_names_are_detected(name: str) -> None:
    assert is_placeholder(name)


@pytest.mark.parametrize("name", ["Escherichia-Shigella", "Sva0996 marine group", "Subgroup 9"])
def test_real_taxa_are_not_placeholders(name: str) -> None:
    assert not is_placeholder(name)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Archaea;\t2\tdomain\t\t\n", ("Archaea;", Rank.DOMAIN)),
        ("Archaea;Foo;\t3\tgenus\t\t144\r\n", ("Archaea;Foo;", Rank.GENUS)),
        ("Eukaryota;Amorphea;\t4\tmajor_clade\t\t144\n", ("Eukaryota;Amorphea;", None)),
        ("Bacteria;\t2\tdomain", ("Bacteria;", Rank.DOMAIN)),
    ],
)
def test_parse_rank_line(line: str, expected: tuple[str, Rank | None]) -> None:
    assert parse_rank_line(line) == expected


@pytest.mark.parametrize("line", ["Bacteria;\t2\n", "\t2\tdomain\n", ""])
def test_parse_rank_line_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(TaxonomyError, match="Malformed taxonomy line"):
        parse_rank_line(line)


def test_load_rank_map_reads_every_path(rank_map: RankMap) -> None:
    assert len(rank_map) == 33
    assert rank_map["Bacteria;"] is Rank.DOMAIN
    assert rank_map["Eukaryota;Amorphea;"] is None


def test_load_rank_map_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tax.txt"
    path.write_text("\nBacteria;\t2\tdomain\t\t144\n\n", encoding="utf-8")

    assert load_rank_map(path) == {"Bacteria;": Rank.DOMAIN}


def test_load_rank_map_rejects_duplicate_paths(tmp_path: Path) -> None:
    path = tmp_path / "tax.txt"
    path.write_text("Bacteria;\t2\tdomain\t\t\nBacteria;\t3\tdomain\t\t\n", encoding="utf-8")

    with pytest.raises(TaxonomyError, match="Duplicate taxonomy path"):
        load_rank_map(path)


def test_parse_lineage_maps_every_canonical_rank(rank_map: RankMap) -> None:
    lineage = parse_lineage(ESCHERICHIA, rank_map)

    assert lineage.path == ESCHERICHIA
    assert lineage.names == ESCHERICHIA


def test_parse_lineage_skips_non_canonical_ranks_and_leaves_gaps(rank_map: RankMap) -> None:
    lineage = parse_lineage(SORDARIOMYCETES, rank_map)

    assert lineage.names == (
        "Eukaryota",
        "Fungi",
        "Ascomycota",
        "Sordariomycetes",
        None,
        None,
        None,
    )


def test_parse_lineage_masks_placeholders(rank_map: RankMap) -> None:
    path = (*ESCHERICHIA[:-1], "Incertae Sedis")

    lineage = parse_lineage(path, rank_map)

    assert lineage.get(Rank.FAMILY) == "Enterobacteriaceae"
    assert lineage.get(Rank.GENUS) is None


def test_parse_lineage_rejects_unknown_paths(rank_map: RankMap) -> None:
    with pytest.raises(UnknownTaxonPathError) as caught:
        parse_lineage(("Bacteria", "Pseudomonadati", "Fakeota"), rank_map)

    assert caught.value.path == "Bacteria;Pseudomonadati;Fakeota;"


def test_parse_lineage_rejects_duplicate_ranks(rank_map: RankMap) -> None:
    path = ("Bacteria", "Pseudomonadati", "Pseudomonadota", "Pseudomonadota-duplicate")

    with pytest.raises(DuplicateRankError) as caught:
        parse_lineage(path, rank_map)

    assert caught.value.rank == Rank.PHYLUM
    assert caught.value.path == "Bacteria;Pseudomonadati;Pseudomonadota;Pseudomonadota-duplicate;"


@pytest.mark.parametrize("path", [("Unclassified",), ()])
def test_parse_lineage_requires_a_domain(path: tuple[str, ...], rank_map: RankMap) -> None:
    with pytest.raises(TaxonomyError, match="no domain"):
        parse_lineage(path, rank_map)
