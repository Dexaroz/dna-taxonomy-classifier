from pathlib import Path
import pickle

import pytest

from taxonomy_classifier.exceptions import (
    ChecksumMismatchError,
    DataError,
    DownloadError,
    DuplicateRankError,
    GeneflowError,
    InvalidSequenceError,
    MalformedFastaError,
    MalformedHeaderError,
    TaxonomyError,
    UnknownTaxonPathError,
)


@pytest.mark.parametrize(
    ("error_type", "parent"),
    [
        (DataError, GeneflowError),
        (DownloadError, DataError),
        (ChecksumMismatchError, DownloadError),
        (MalformedFastaError, DataError),
        (MalformedHeaderError, DataError),
        (InvalidSequenceError, DataError),
        (TaxonomyError, DataError),
        (UnknownTaxonPathError, TaxonomyError),
        (DuplicateRankError, TaxonomyError),
    ],
)
def test_hierarchy(error_type: type[Exception], parent: type[Exception]) -> None:
    assert issubclass(error_type, parent)


def test_checksum_mismatch_exposes_attributes_and_message() -> None:
    path = Path("data/raw/file.gz")

    error = ChecksumMismatchError(path, expected="abc", actual="def")

    assert (error.path, error.expected, error.actual) == (path, "abc", "def")
    assert "abc" in str(error)
    assert "def" in str(error)
    assert "file.gz" in str(error)


def test_unknown_taxon_path_exposes_attributes_and_message() -> None:
    error = UnknownTaxonPathError("Bacteria;Foo;")

    assert error.path == "Bacteria;Foo;"
    assert "Bacteria;Foo;" in str(error)


def test_duplicate_rank_exposes_attributes_and_message() -> None:
    error = DuplicateRankError("genus", "Bacteria;A;B;")

    assert (error.rank, error.path) == ("genus", "Bacteria;A;B;")
    assert "genus" in str(error)
    assert "Bacteria;A;B;" in str(error)


@pytest.mark.parametrize(
    "error",
    [
        ChecksumMismatchError(Path("f.gz"), expected="abc", actual="def"),
        UnknownTaxonPathError("Bacteria;Foo;"),
        DuplicateRankError("genus", "Bacteria;A;B;"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_structured_errors_survive_pickling(error: GeneflowError) -> None:
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)
