from dataclasses import dataclass
import hashlib
import logging
from typing import TYPE_CHECKING, Final

import httpx

from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.exceptions import ChecksumMismatchError, DownloadError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources import RemoteFile, SilvaRelease

_LOGGER: Final = logging.getLogger(__name__)

_CHUNK_SIZE: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class ReleaseFiles:
    fasta: Path
    taxonomy: Path


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_file(remote: RemoteFile, dest_dir: Path, *, client: httpx.Client) -> Path:
    target = dest_dir / remote.filename

    if target.exists():
        _verify(target, remote.sha256)
        _LOGGER.info("Already downloaded and verified: %s", target)

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

    if actual != remote.sha256:
        partial.unlink()

        raise ChecksumMismatchError(target, remote.sha256, actual)

    partial.replace(target)
    _LOGGER.info("Downloaded and verified: %s", target)

    return target


def download_release(
    release: SilvaRelease, dest_dir: Path, *, client: httpx.Client
) -> ReleaseFiles:
    return ReleaseFiles(
        fasta=download_file(release.fasta, dest_dir, client=client),
        taxonomy=download_file(release.taxonomy, dest_dir, client=client),
    )


def _verify(path: Path, expected: str) -> None:
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
