import hashlib
from typing import TYPE_CHECKING

import httpx
import pytest

from taxonomy_classifier.data.download import download_all, download_file, sha256_of
from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.data.remote import RemoteFile
from taxonomy_classifier.exceptions import ChecksumMismatchError, DownloadError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

CONTENT = b"ACGT" * 1024

REMOTE = RemoteFile(
    url="https://example.org/files/data.gz",
    sha256=hashlib.sha256(CONTENT).hexdigest(),
)


class FakeServer:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        return self._handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _serve(content: bytes) -> FakeServer:
    return FakeServer(lambda _: httpx.Response(200, content=content))


def test_sha256_of_hashes_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "data.gz"
    path.write_bytes(CONTENT)

    assert sha256_of(path) == REMOTE.sha256


def test_download_file_writes_verified_file(tmp_path: Path) -> None:
    server = _serve(CONTENT)

    with server.client() as client:
        path = download_file(REMOTE, tmp_path / "raw", client=client)

    assert path == tmp_path / "raw" / "data.gz"
    assert path.read_bytes() == CONTENT
    assert not partial_path(path).exists()
    assert [str(request.url) for request in server.requests] == [REMOTE.url]


def test_download_file_skips_verified_existing_file(tmp_path: Path) -> None:
    (tmp_path / "data.gz").write_bytes(CONTENT)
    server = _serve(CONTENT)

    with server.client() as client:
        download_file(REMOTE, tmp_path, client=client)

    assert server.requests == []


def test_download_file_refuses_to_overwrite_corrupted_file(tmp_path: Path) -> None:
    target = tmp_path / "data.gz"
    target.write_bytes(b"corrupted")
    server = _serve(CONTENT)

    with server.client() as client, pytest.raises(ChecksumMismatchError) as caught:
        download_file(REMOTE, tmp_path, client=client)

    assert caught.value.path == target
    assert target.read_bytes() == b"corrupted"
    assert server.requests == []


def test_download_file_discards_content_with_wrong_checksum(tmp_path: Path) -> None:
    server = _serve(b"tampered")

    with server.client() as client, pytest.raises(ChecksumMismatchError) as caught:
        download_file(REMOTE, tmp_path, client=client)

    assert caught.value.expected == REMOTE.sha256
    assert caught.value.actual == hashlib.sha256(b"tampered").hexdigest()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", [404, 500])
def test_download_file_wraps_http_errors(tmp_path: Path, status: int) -> None:
    server = FakeServer(lambda _: httpx.Response(status))

    with server.client() as client, pytest.raises(DownloadError, match=str(status)):
        download_file(REMOTE, tmp_path, client=client)

    assert list(tmp_path.iterdir()) == []


def test_download_file_wraps_network_errors(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        msg = "connection refused"
        raise httpx.ConnectError(msg, request=request)

    server = FakeServer(refuse)

    with server.client() as client, pytest.raises(DownloadError, match="connection refused"):
        download_file(REMOTE, tmp_path, client=client)

    assert list(tmp_path.iterdir()) == []


def test_unpinned_files_are_downloaded_without_verification(tmp_path: Path) -> None:
    remote = RemoteFile(url=REMOTE.url, sha256=None)
    server = _serve(b"rolling release")

    with server.client() as client:
        path = download_file(remote, tmp_path, client=client)

    assert path.read_bytes() == b"rolling release"


def test_existing_unpinned_files_are_kept_as_they_are(tmp_path: Path) -> None:
    (tmp_path / "data.gz").write_bytes(b"previous snapshot")
    server = _serve(CONTENT)

    with server.client() as client:
        download_file(RemoteFile(url=REMOTE.url, sha256=None), tmp_path, client=client)

    assert (tmp_path / "data.gz").read_bytes() == b"previous snapshot"
    assert server.requests == []


def test_download_all_fetches_every_file(tmp_path: Path) -> None:
    other_content = b"taxonomy"
    other = RemoteFile(
        url="https://example.org/files/tax.txt.gz",
        sha256=hashlib.sha256(other_content).hexdigest(),
    )
    contents = {REMOTE.url: CONTENT, other.url: other_content}
    server = FakeServer(lambda request: httpx.Response(200, content=contents[str(request.url)]))

    with server.client() as client:
        paths = download_all([REMOTE, other], tmp_path, client=client)

    assert paths == (tmp_path / "data.gz", tmp_path / "tax.txt.gz")
    assert paths[1].read_bytes() == other_content
