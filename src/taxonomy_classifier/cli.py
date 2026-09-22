import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from taxonomy_classifier.data.build import BuildConfig, DataLayout, build_dataset
from taxonomy_classifier.data.download import download_all
from taxonomy_classifier.data.sources.registry import DEFAULT_SOURCES
from taxonomy_classifier.exceptions import DataError, GeneflowError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from taxonomy_classifier.data.sources.base import DataSource

_LOGGER: Final = logging.getLogger(__name__)

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_TIMEOUT: Final = httpx.Timeout(30.0, read=300.0)

_COMMANDS: Final = {
    "download": "Download and verify the raw files of every source",
    "build": "Build the harmonized dataset from already downloaded files",
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
        command.add_argument("--data-dir", type=Path, default=Path("data"))

    return parser


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    args = build_parser().parse_args(argv)

    command: str = args.command
    layout = DataLayout(root=args.data_dir)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=_LOG_FORMAT)

    try:
        _run(command, DEFAULT_SOURCES, layout, transport=transport)

    except GeneflowError:
        _LOGGER.exception("Command %r failed", command)

        return 1

    return 0


def _run(
    command: str,
    sources: Sequence[DataSource],
    layout: DataLayout,
    *,
    transport: httpx.BaseTransport | None,
) -> None:
    if command in {"download", "prepare"}:
        with httpx.Client(transport=transport, timeout=_TIMEOUT, follow_redirects=True) as client:
            for source in sources:
                download_all(source.files, layout.raw_dir(source), client=client)

    if command in {"build", "prepare"}:
        _require_raw_files(sources, layout)

        build_dataset(sources, layout, config=BuildConfig())


def _require_raw_files(sources: Sequence[DataSource], layout: DataLayout) -> None:
    missing = [
        str(layout.raw_dir(source) / remote.filename)
        for source in sources
        for remote in source.files
        if not (layout.raw_dir(source) / remote.filename).exists()
    ]

    if missing:
        msg = f"Missing raw files {missing}; run 'geneflow download' first"
        raise DataError(msg)
