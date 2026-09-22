import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from taxonomy_classifier.data.build import BuildConfig, build_dataset
from taxonomy_classifier.data.download import ReleaseFiles, download_release
from taxonomy_classifier.data.sources import LATEST_RELEASE, RELEASES
from taxonomy_classifier.exceptions import DataError, GeneflowError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from taxonomy_classifier.data.sources import SilvaRelease

_LOGGER: Final = logging.getLogger(__name__)

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_TIMEOUT: Final = httpx.Timeout(30.0, read=300.0)

_COMMANDS: Final = {
    "download": "Download and verify the raw SILVA files",
    "build": "Build the Parquet dataset from already downloaded files",
    "prepare": "Download and build in one step",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geneflow",
        description="Hierarchical taxonomic classification of DNA sequences.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in _COMMANDS.items():
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--release", choices=sorted(RELEASES), default=LATEST_RELEASE)
        command.add_argument("--data-dir", type=Path, default=Path("data"))

    return parser


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    args = build_parser().parse_args(argv)

    command: str = args.command
    release = RELEASES[args.release]
    data_dir: Path = args.data_dir

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=_LOG_FORMAT)

    try:
        _run(command, release, data_dir, transport=transport)

    except GeneflowError:
        _LOGGER.exception("Command %r failed", command)

        return 1

    return 0


def _run(
    command: str,
    release: SilvaRelease,
    data_dir: Path,
    *,
    transport: httpx.BaseTransport | None,
) -> None:
    raw_dir = data_dir / "raw" / release.slug

    if command in {"download", "prepare"}:
        with httpx.Client(transport=transport, timeout=_TIMEOUT, follow_redirects=True) as client:
            download_release(release, raw_dir, client=client)

    if command in {"build", "prepare"}:
        files = _existing_release_files(release, raw_dir)

        build_dataset(
            files.fasta,
            files.taxonomy,
            data_dir / "processed" / release.slug,
            release=release.version,
            config=BuildConfig(),
        )


def _existing_release_files(release: SilvaRelease, raw_dir: Path) -> ReleaseFiles:
    files = ReleaseFiles(
        fasta=raw_dir / release.fasta.filename,
        taxonomy=raw_dir / release.taxonomy.filename,
    )

    missing = [path for path in (files.fasta, files.taxonomy) if not path.exists()]

    if missing:
        msg = f"Missing raw files {[str(path) for path in missing]}; run 'geneflow download' first"
        raise DataError(msg)

    return files
