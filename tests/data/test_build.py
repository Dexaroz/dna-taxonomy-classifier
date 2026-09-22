import json
import shutil
from typing import TYPE_CHECKING, Literal

import polars as pl
import pytest

from taxonomy_classifier.data.build import BuildConfig, DataLayout, build_dataset
from taxonomy_classifier.data.filters import FilterConfig
from taxonomy_classifier.data.split import Split, SplitConfig

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.build import BuildReport
    from taxonomy_classifier.data.sources.base import DataSource

CONFIG = BuildConfig(
    filters=FilterConfig(min_length=10, max_length=100),
    split=SplitConfig(val_fraction=0.2, test_fraction=0.2),
    batch_size=2,
)


@pytest.fixture
def layout(sources: tuple[DataSource, ...], fixtures_dir: Path, tmp_path: Path) -> DataLayout:
    layout = DataLayout(root=tmp_path / "data")

    for source in sources:
        raw_dir = layout.raw_dir(source)
        raw_dir.mkdir(parents=True)

        for remote in source.files:
            shutil.copy(fixtures_dir / remote.filename, raw_dir / remote.filename)

    return layout


@pytest.fixture
def report(sources: tuple[DataSource, ...], layout: DataLayout) -> BuildReport:
    return build_dataset(sources, layout, config=CONFIG)


def test_backbone_sources_are_staged_first(report: BuildReport) -> None:
    assert [source.source for source in report.sources] == [
        "gtdb_test",
        "pr2_test",
        "silva_test",
        "refseq_16s",
    ]


def test_report_counts_every_stage(report: BuildReport) -> None:
    assert sum(source.kept for source in report.sources) == 16
    assert report.merge.to_dict() == {
        "staged": 16,
        "unique": 14,
        "kept": 14,
        "merged": 1,
        "conflicts": 1,
        "domain_conflicts": 0,
    }
    assert sum(report.splits.values()) == 14
    assert set(report.splits) == {split.value for split in Split}
    assert report.rank_coverage["domain"] == 14
    assert report.kingdoms == {
        "Bacteria": 8,
        "Archaea": 3,
        "Animalia": 0,
        "Fungi": 2,
        "Plantae": 0,
        "Protista": 1,
        "unknown": 0,
    }


def test_dataset_merges_duplicates_across_sources(report: BuildReport, layout: DataLayout) -> None:
    del report

    dataset = pl.read_parquet(layout.dataset_path)
    [ecoli] = dataset.filter(pl.col("n_records") == 3).to_dicts()

    assert dataset.height == 14
    assert ecoli["sources"] == ["gtdb", "silva"]
    assert ecoli["label_conflict"] is True
    assert (ecoli["class"], ecoli["order"]) == ("Gammaproteobacteria", None)
    assert set(dataset.get_column("split")) <= {split.value for split in Split}


def test_report_is_written_as_json(report: BuildReport, layout: DataLayout) -> None:
    written = json.loads(layout.report_path.read_text(encoding="utf-8"))

    assert written == json.loads(report.to_json())
    assert written["config"]["filters"]["min_length"] == 10
    assert written["sources"][0]["source"] == "gtdb_test"


def test_build_leaves_no_partial_files(
    sources: tuple[DataSource, ...],
    report: BuildReport,
    layout: DataLayout,
) -> None:
    del report

    produced = [*layout.interim_dir.iterdir(), *layout.processed_dir.iterdir()]

    assert not [path for path in produced if path.suffix == ".part"]
    assert layout.staged_path(sources[0]).exists()


def test_build_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BuildConfig(batch_size=0)


def test_build_cleans_up_when_writing_the_dataset_fails(
    sources: tuple[DataSource, ...],
    layout: DataLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pl.LazyFrame.sink_parquet

    def fail_on_dataset(frame: pl.LazyFrame, path: Path, *, compression: Literal["zstd"]) -> None:
        if path.name.startswith("dataset"):
            path.write_bytes(b"partial")
            msg = "disk full"
            raise OSError(msg)

        original(frame, path, compression=compression)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_on_dataset)

    with pytest.raises(OSError, match="disk full"):
        build_dataset(sources, layout, config=CONFIG)

    assert list(layout.processed_dir.iterdir()) == []
