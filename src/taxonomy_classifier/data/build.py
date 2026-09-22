from dataclasses import asdict, dataclass, field
import json
import logging
from typing import TYPE_CHECKING, Final

import polars as pl

from taxonomy_classifier.data.columns import RANK_COLUMNS
from taxonomy_classifier.data.files import partial_path, write_text_atomic
from taxonomy_classifier.data.filters import FilterConfig
from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.merge import merge_staged
from taxonomy_classifier.data.split import Split, SplitConfig, assign_splits
from taxonomy_classifier.data.staging import stage_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from taxonomy_classifier.data.merge import MergeReport
    from taxonomy_classifier.data.sources.base import DataSource
    from taxonomy_classifier.data.staging import SourceReport

_LOGGER: Final = logging.getLogger(__name__)

DATASET_NAME: Final = "geneflow"

DATASET_FILENAME: Final = "dataset.parquet"

REPORT_FILENAME: Final = "build_report.json"

MERGED_FILENAME: Final = "merged.parquet"

UNKNOWN_KINGDOM: Final = "unknown"


@dataclass(frozen=True, slots=True)
class DataLayout:
    root: Path

    def raw_dir(self, source: DataSource) -> Path:
        return self.root / "raw" / source.slug

    @property
    def interim_dir(self) -> Path:
        return self.root / "interim" / DATASET_NAME

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed" / DATASET_NAME

    @property
    def dataset_path(self) -> Path:
        return self.processed_dir / DATASET_FILENAME

    @property
    def report_path(self) -> Path:
        return self.processed_dir / REPORT_FILENAME

    def staged_path(self, source: DataSource) -> Path:
        return self.interim_dir / f"{source.slug}.parquet"


@dataclass(frozen=True, slots=True)
class BuildConfig:
    filters: FilterConfig = field(default_factory=FilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    batch_size: int = 50_000

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            msg = f"batch_size must be positive, got {self.batch_size}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BuildReport:
    config: BuildConfig
    sources: Sequence[SourceReport]
    merge: MergeReport
    splits: Mapping[str, int]
    kingdoms: Mapping[str, int]
    rank_coverage: Mapping[str, int]

    def to_json(self) -> str:
        payload = {
            "config": asdict(self.config),
            "sources": [report.to_dict() for report in self.sources],
            "merge": self.merge.to_dict(),
            "splits": dict(self.splits),
            "kingdoms": dict(self.kingdoms),
            "rank_coverage": dict(self.rank_coverage),
        }

        return json.dumps(payload, indent=2)


def build_dataset(
    sources: Sequence[DataSource],
    layout: DataLayout,
    *,
    config: BuildConfig,
) -> BuildReport:
    backbone = BackboneIndex()

    source_reports = [
        stage_source(
            source,
            layout.raw_dir(source),
            layout.staged_path(source),
            filters=config.filters,
            backbone=backbone,
            batch_size=config.batch_size,
        )
        for source in _backbones_first(sources)
    ]

    merged_path = layout.interim_dir / MERGED_FILENAME
    merge_report = merge_staged([layout.staged_path(source) for source in sources], merged_path)
    _LOGGER.info(
        "Merged %d staged records into %d sequences", merge_report.staged, merge_report.kept
    )

    _write_dataset(merged_path, layout, config.split)

    report = BuildReport(
        config=config,
        sources=source_reports,
        merge=merge_report,
        splits=_split_counts(layout.dataset_path),
        kingdoms=_kingdom_counts(layout.dataset_path),
        rank_coverage=_rank_coverage(layout.dataset_path),
    )

    write_text_atomic(layout.report_path, f"{report.to_json()}\n")
    _LOGGER.info("Wrote %s", layout.dataset_path)

    return report


def _backbones_first(sources: Sequence[DataSource]) -> list[DataSource]:
    return sorted(sources, key=lambda source: not source.is_backbone)


def _write_dataset(merged_path: Path, layout: DataLayout, config: SplitConfig) -> None:
    labels = pl.read_parquet(merged_path, columns=["seq_hash", *RANK_COLUMNS])
    splits = assign_splits(labels, config)

    layout.processed_dir.mkdir(parents=True, exist_ok=True)
    partial = partial_path(layout.dataset_path)

    try:
        (
            pl.scan_parquet(merged_path)
            .join(splits.lazy(), on="seq_hash", how="left", maintain_order="left")
            .sink_parquet(partial, compression="zstd")
        )

    except BaseException:
        partial.unlink(missing_ok=True)

        raise

    partial.replace(layout.dataset_path)


def _split_counts(dataset_path: Path) -> dict[str, int]:
    counts = pl.scan_parquet(dataset_path).group_by("split").len().collect()
    observed = dict(counts.iter_rows())

    return {split.value: observed.get(split.value, 0) for split in Split}


def _kingdom_counts(dataset_path: Path) -> dict[str, int]:
    counts = pl.scan_parquet(dataset_path).group_by("kingdom").len().collect()
    observed = dict(counts.iter_rows())

    return {
        **{kingdom.value: observed.get(kingdom.value, 0) for kingdom in Kingdom},
        UNKNOWN_KINGDOM: observed.get(None, 0),
    }


def _rank_coverage(dataset_path: Path) -> dict[str, int]:
    counts = pl.scan_parquet(dataset_path).select(pl.col(RANK_COLUMNS).count()).collect()

    return counts.row(0, named=True)
