import json
import runpy
import sys
from typing import TYPE_CHECKING

import httpx
import optuna
import pytest

from taxonomy_classifier import cli
from taxonomy_classifier.data.build import DataLayout

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources.base import DataSource


@pytest.fixture
def transport(
    sources: tuple[DataSource, ...], fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> httpx.MockTransport:
    monkeypatch.setattr(cli, "DEFAULT_SOURCES", sources)

    def serve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=(fixtures_dir / request.url.path.rsplit("/")[-1]).read_bytes()
        )

    return httpx.MockTransport(serve)


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["build"])

    assert args.command == "build"
    assert str(args.data_dir) == "data"
    assert args.verbose is False


@pytest.mark.parametrize("argv", [[], ["unknown"], ["build", "--release", "144"]])
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


def test_download_fetches_every_source_file(
    sources: tuple[DataSource, ...],
    transport: httpx.MockTransport,
    tmp_path: Path,
) -> None:
    exit_code = cli.main(["download", "--data-dir", str(tmp_path)], transport=transport)

    layout = DataLayout(root=tmp_path)

    assert exit_code == 0

    for source in sources:
        for remote in source.files:
            assert (layout.raw_dir(source) / remote.filename).exists()

    assert not layout.processed_dir.exists()


def test_prepare_downloads_and_builds(transport: httpx.MockTransport, tmp_path: Path) -> None:
    exit_code = cli.main(["-v", "prepare", "--data-dir", str(tmp_path)], transport=transport)

    layout = DataLayout(root=tmp_path)
    report = json.loads(layout.report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert layout.dataset_path.exists()
    assert layout.synthetic_path.exists()
    assert layout.augment_report_path.exists()
    assert [source["read"] for source in report["sources"]] == [7, 4, 7, 5]


def test_module_entrypoint_exits_with_command_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["geneflow", "build", "--data-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("taxonomy_classifier", run_name="__main__")

    assert caught.value.code == 1


def test_augment_without_dataset_fails_cleanly(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exit_code = cli.main(["augment", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'geneflow build' first" in caplog.text


def test_cache_writes_the_label_space_and_token_caches(built_layout: DataLayout) -> None:
    exit_code = cli.main(["cache", "--data-dir", str(built_layout.root), "--min-class-count", "1"])

    assert exit_code == 0
    assert (built_layout.processed_dir / "label_space.json").exists()
    assert (built_layout.interim_dir / "train_tokens.bin").exists()


def test_cache_requires_the_synthetic_data(
    built_layout: DataLayout, caplog: pytest.LogCaptureFixture
) -> None:
    built_layout.synthetic_path.unlink()

    exit_code = cli.main(["cache", "--data-dir", str(built_layout.root)])

    assert exit_code == 1
    assert "run 'geneflow augment' first" in caplog.text


def test_search_runs_and_writes_its_results(built_layout: DataLayout, tmp_path: Path) -> None:
    storage = tmp_path / "search" / "cnn.log"

    exit_code = cli.main(
        [
            "search",
            "--data-dir",
            str(built_layout.root),
            "--min-class-count",
            "1",
            "--architecture",
            "CNN",
            "--storage",
            str(storage),
            "--trials",
            "1",
            "--epochs",
            "1",
            "--samples-per-epoch",
            "4",
            "--batch-size",
            "2",
            "--validation-samples",
            "2",
            "--precision",
            "fp32",
            "--gpu-memory-fraction",
            "0.9",
        ]
    )

    assert exit_code == 0
    assert storage.exists()
    assert storage.with_name("cnn-trials.json").exists()


def test_search_without_any_budget_fails_cleanly(
    built_layout: DataLayout, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = cli.main(["search", "--data-dir", str(built_layout.root), "--architecture", "CNN"])

    assert exit_code == 1
    assert "needs a number of trials" in caplog.text


def test_search_warns_when_no_trial_completes(
    built_layout: DataLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cli, "run_search", lambda *_, **__: optuna.create_study())

    exit_code = cli.main(
        ["search", "--data-dir", str(built_layout.root), "--architecture", "CNN", "--trials", "1"]
    )

    assert exit_code == 0
    assert "finished without completed trials" in caplog.text


def test_search_reports_the_best_trial(
    built_layout: DataLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=0.42, params={}, distributions={}))
    monkeypatch.setattr(cli, "run_search", lambda *_, **__: study)

    exit_code = cli.main(
        ["search", "--data-dir", str(built_layout.root), "--architecture", "CNN", "--trials", "1"]
    )

    assert exit_code == 0
    assert "best trial 0: 0.4200" in caplog.text
