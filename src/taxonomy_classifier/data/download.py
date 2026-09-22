import hashlib
import logging
from typing import TYPE_CHECKING, Final

import httpx

from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.exceptions import ChecksumMismatchError, DownloadError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from taxonomy_classifier.data.remote import RemoteFile

_LOGGER: Final = logging.getLogger(__name__)

_CHUNK_SIZE: Final = 1 << 20


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_file(remote: RemoteFile, dest_dir: Path, *, client: httpx.Client) -> Path:
    target = dest_dir / remote.filename

    if target.exists():
        _verify(target, remote.sha256)
        _LOGGER.info("Already downloaded: %s", target)

        return target

    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = partial_path(target)

    _LOGGER.info("Downloading %s", remote.url)

    try:
        actual = _stream_to_file(client, remote.url, partial)

    except httpx.HTTPError as error:
        partial.unlink(missing_ok=True)

        msg = f"Failed to download {remote.url}: {error}"
        raise DownloadError(msg) from error

    if remote.sha256 is not None and actual != remote.sha256:
        partial.unlink()

        raise ChecksumMismatchError(target, remote.sha256, actual)

    partial.replace(target)
    _LOGGER.info("Downloaded %s (sha256 %s)", target, actual)

    return target


def download_all(
    remotes: Iterable[RemoteFile],
    dest_dir: Path,
    *,
    client: httpx.Client,
) -> tuple[Path, ...]:
    return tuple(download_file(remote, dest_dir, client=client) for remote in remotes)


def _verify(path: Path, expected: str | None) -> None:
    if expected is None:
        return

    actual = sha256_of(path)

    if actual != expected:
        raise ChecksumMismatchError(path, expected, actual)


def _stream_to_file(client: httpx.Client, url: str, destination: Path) -> str:
    digest = hashlib.sha256()

    with client.stream("GET", url) as response:
        response.raise_for_status()

        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK_SIZE):
                digest.update(chunk)
                handle.write(chunk)

    return digest.hexdigest()
