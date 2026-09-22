from pathlib import Path
import pickle

import pytest

from taxonomy_classifier.exceptions import (
    ChecksumMismatchError,
    DataError,
    DownloadError,
    GeneflowError,
    InvalidSequenceError,
    MalformedFastaError,
    MalformedHeaderError,
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


def test_checksum_mismatch_survives_pickling() -> None:
    error = ChecksumMismatchError(Path("f.gz"), expected="abc", actual="def")

    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert type(restored) is ChecksumMismatchError
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)
