import gzip
import hashlib
import json
import runpy
import sys
from typing import TYPE_CHECKING

import httpx
import pytest

from taxonomy_classifier import cli
from taxonomy_classifier.data.build import DATASET_FILENAME, REPORT_FILENAME
from taxonomy_classifier.data.sources import LATEST_RELEASE, RemoteFile, SilvaRelease

if TYPE_CHECKING:
    from pathlib import Path


def _remote(url: str, content: bytes) -> RemoteFile:
    return RemoteFile(url=url, sha256=hashlib.sha256(content).hexdigest())


@pytest.fixture
def fake_release(
    silva_fasta: Path,
    silva_taxonomy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SilvaRelease, httpx.MockTransport]:
    contents = {
        "https://example.org/test/silva.fasta.gz": gzip.compress(silva_fasta.read_bytes()),
        "https://example.org/test/tax.txt.gz": gzip.compress(silva_taxonomy.read_bytes()),
    }
    fasta_url, taxonomy_url = contents
    release = SilvaRelease(
        version="test",
        fasta=_remote(fasta_url, contents[fasta_url]),
        taxonomy=_remote(taxonomy_url, contents[taxonomy_url]),
    )

    monkeypatch.setattr(cli, "RELEASES", {"test": release})

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=contents[str(request.url)])
    )

    return release, transport


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["build"])

    assert args.command == "build"
    assert args.release == LATEST_RELEASE
    assert str(args.data_dir) == "data"
    assert args.verbose is False


@pytest.mark.parametrize("argv", [[], ["build", "--release", "999"], ["unknown"]])
def test_invalid_arguments_exit_with_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)

    assert caught.value.code == 2


def test_build_without_raw_files_fails_cleanly(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exit_code = cli.main(["build", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'geneflow download' first" in caplog.text


def test_download_only_fetches_raw_files(
    fake_release: tuple[SilvaRelease, httpx.MockTransport],
    tmp_path: Path,
) -> None:
    release, transport = fake_release

    exit_code = cli.main(
        ["download", "--release", "test", "--data-dir", str(tmp_path)],
        transport=transport,
    )

    raw_dir = tmp_path / "raw" / release.slug

    assert exit_code == 0
    assert sorted(path.name for path in raw_dir.iterdir()) == ["silva.fasta.gz", "tax.txt.gz"]
    assert not (tmp_path / "processed").exists()


def test_prepare_downloads_and_builds(
    fake_release: tuple[SilvaRelease, httpx.MockTransport],
    tmp_path: Path,
) -> None:
    release, transport = fake_release

    exit_code = cli.main(
        ["-v", "prepare", "--release", "test", "--data-dir", str(tmp_path)],
        transport=transport,
    )

    processed_dir = tmp_path / "processed" / release.slug
    report = json.loads((processed_dir / REPORT_FILENAME).read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (processed_dir / DATASET_FILENAME).exists()
    assert report["release"] == "test"
    assert report["total"] == 13


def test_module_entrypoint_exits_with_command_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["geneflow", "build", "--data-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("taxonomy_classifier", run_name="__main__")

    assert caught.value.code == 1
