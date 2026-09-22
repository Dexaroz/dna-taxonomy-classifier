import pytest

from taxonomy_classifier.data.kingdom import Kingdom, prokaryote_kingdom


@pytest.mark.parametrize(
    ("domain", "expected"),
    [("Bacteria", Kingdom.BACTERIA), ("Archaea", Kingdom.ARCHAEA), ("Eukaryota", None)],
)
def test_prokaryote_kingdom(domain: str, expected: Kingdom | None) -> None:
    assert prokaryote_kingdom(domain) is expected
