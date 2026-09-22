import pytest

from taxonomy_classifier.data.organisms import split_organism


@pytest.mark.parametrize(
    ("organism", "expected"),
    [
        ("Escherichia coli", ("Escherichia", "Escherichia coli")),
        ("Escherichia coli strain K-12", ("Escherichia", "Escherichia coli")),
        ("Candidatus Novelbacter mysteriosus", ("Novelbacter", "Novelbacter mysteriosus")),
        (
            "Methanobrevibacter smithii-like",
            ("Methanobrevibacter", "Methanobrevibacter smithii-like"),
        ),
        ("Pseudomonas sp. 12", ("Pseudomonas", None)),
        ("Pseudomonas cf. putida", ("Pseudomonas", None)),
        ("Bacillus C3", ("Bacillus", None)),
        ("Clostridiales bacterium", ("Clostridiales", None)),
        ("uncultured Bacteroides sp.", ("uncultured", None)),
        ("marine metagenome", ("marine", None)),
        ("Pseudomonas", ("Pseudomonas", None)),
        ("   ", ("", None)),
    ],
)
def test_split_organism(organism: str, expected: tuple[str, str | None]) -> None:
    assert split_organism(organism) == expected
