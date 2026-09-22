import pytest

from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, Lineage, Rank

ESCHERICHIA = (
    "Bacteria",
    "Pseudomonadota",
    "Gammaproteobacteria",
    "Enterobacterales",
    "Enterobacteriaceae",
    "Escherichia",
    "Escherichia coli",
)


def test_canonical_ranks_follow_hierarchy() -> None:
    assert CANONICAL_RANKS == (
        Rank.DOMAIN,
        Rank.PHYLUM,
        Rank.CLASS,
        Rank.ORDER,
        Rank.FAMILY,
        Rank.GENUS,
        Rank.SPECIES,
    )


def test_lineage_exposes_ranks() -> None:
    lineage = Lineage(names=ESCHERICHIA)

    assert lineage.domain == "Bacteria"
    assert lineage.get(Rank.GENUS) == "Escherichia"
    assert lineage.deepest_rank is Rank.SPECIES


def test_lineage_rejects_wrong_number_of_names() -> None:
    with pytest.raises(ValueError, match="Expected 7 rank names, got 2"):
        Lineage(names=("Bacteria", None))


@pytest.mark.parametrize("domain", [None, ""])
def test_lineage_requires_a_domain(domain: str | None) -> None:
    with pytest.raises(ValueError, match="must have a domain"):
        Lineage(names=(domain, *ESCHERICHIA[1:]))


def test_from_ranks_fills_missing_ranks_with_none() -> None:
    lineage = Lineage.from_ranks({Rank.DOMAIN: "Eukaryota", Rank.CLASS: "Ascomycota"})

    assert lineage.names == ("Eukaryota", None, "Ascomycota", None, None, None, None)
    assert lineage.deepest_rank is Rank.CLASS


def test_domain_only() -> None:
    lineage = Lineage.domain_only("Archaea")

    assert lineage.names == ("Archaea", None, None, None, None, None, None)
    assert lineage.deepest_rank is Rank.DOMAIN


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (Rank.DOMAIN, ("Bacteria", None, None, None, None, None, None)),
        (Rank.FAMILY, (*ESCHERICHIA[:5], None, None)),
        (Rank.SPECIES, ESCHERICHIA),
    ],
)
def test_truncate_keeps_ranks_down_to_the_given_one(
    rank: Rank,
    expected: tuple[str | None, ...],
) -> None:
    assert Lineage(names=ESCHERICHIA).truncate(rank).names == expected
