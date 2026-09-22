import pytest

from taxonomy_classifier.data.remote import RemoteFile

VALID_SHA256 = "a" * 64


def test_filename_ignores_query_string() -> None:
    remote = RemoteFile(url="https://example.org/a/b/file.gz?token=1", sha256=VALID_SHA256)

    assert remote.filename == "file.gz"


@pytest.mark.parametrize(("sha256", "pinned"), [(VALID_SHA256, True), (None, False)])
def test_pinned_reflects_whether_a_checksum_is_known(sha256: str | None, *, pinned: bool) -> None:
    assert RemoteFile(url="https://example.org/file.gz", sha256=sha256).pinned is pinned


def test_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteFile(url="http://example.org/file.gz", sha256=VALID_SHA256)


@pytest.mark.parametrize("sha256", ["abc", "A" * 64, "g" * 64])
def test_requires_lowercase_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        RemoteFile(url="https://example.org/file.gz", sha256=sha256)
