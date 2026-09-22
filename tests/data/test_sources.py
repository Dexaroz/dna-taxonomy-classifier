import pytest

from taxonomy_classifier.data.sources import LATEST_RELEASE, RELEASES, SILVA_144, RemoteFile

VALID_SHA256 = "a" * 64


def test_silva_144_points_to_frozen_release_files() -> None:
    assert SILVA_144.slug == "silva_144"
    assert "/release_144/" in SILVA_144.fasta.url
    assert "/release_144/" in SILVA_144.taxonomy.url
    assert SILVA_144.fasta.filename == "SILVA_144_SSURef_NR99_tax_silva_trunc.fasta.gz"
    assert SILVA_144.taxonomy.filename == "tax_slv_ssu_144.txt.gz"


def test_releases_registry_is_read_only() -> None:
    assert RELEASES[LATEST_RELEASE] is SILVA_144

    with pytest.raises(TypeError):
        RELEASES["999"] = SILVA_144  # type: ignore[index]


def test_remote_file_filename_ignores_query_string() -> None:
    remote = RemoteFile(url="https://example.org/a/b/file.gz?token=1", sha256=VALID_SHA256)

    assert remote.filename == "file.gz"


def test_remote_file_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteFile(url="http://example.org/file.gz", sha256=VALID_SHA256)


@pytest.mark.parametrize("sha256", ["abc", "A" * 64, "g" * 64])
def test_remote_file_requires_lowercase_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        RemoteFile(url="https://example.org/file.gz", sha256=sha256)
