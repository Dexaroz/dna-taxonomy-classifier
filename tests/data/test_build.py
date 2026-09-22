import json
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from taxonomy_classifier.data.build import (
    DATASET_FILENAME,
    PARQUET_SCHEMA,
    REPORT_FILENAME,
    BuildConfig,
    build_dataset,
)
from taxonomy_classifier.data.filters import ExclusionReason, FilterConfig
from taxonomy_classifier.exceptions import MalformedFastaError

if TYPE_CHECKING:
    from pathlib import Path

CONFIG = BuildConfig(
    filters=FilterConfig(min_length=10, max_length=40, max_ambiguous_fraction=0.1),
    batch_size=2,
)


def _build(fasta: Path, taxonomy: Path, out_dir: Path, config: BuildConfig = CONFIG) -> None:
    build_dataset(fasta, taxonomy, out_dir, release="test", config=config)


def test_build_dataset_reports_every_outcome(
    silva_fasta: Path,
    silva_taxonomy: Path,
    tmp_path: Path,
) -> None:
    report = build_dataset(silva_fasta, silva_taxonomy, tmp_path, release="test", config=CONFIG)

    assert report.total == 13
    assert report.kept == 4
    assert report.excluded == dict.fromkeys(ExclusionReason, 1)


def test_build_dataset_writes_expected_rows(
    silva_fasta: Path,
    silva_taxonomy: Path,
    tmp_path: Path,
) -> None:
    _build(silva_fasta, silva_taxonomy, tmp_path)

    table = pq.read_table(tmp_path / DATASET_FILENAME)
    rows = table.to_pylist()

    assert table.schema.equals(PARQUET_SCHEMA)
    assert [row["accession"] for row in rows] == ["AB000001", "AB000002", "AB000003", "AB000004"]

    assert rows[0]["sequence"] == "ATTGAACGCTGGCGGCAGGCCTAA"
    assert rows[0]["length"] == 24
    assert rows[0]["genus"] == "Escherichia-Shigella"
    assert rows[0]["organism"] == "Escherichia coli"

    assert rows[2]["class"] == "Sordariomycetes"
    assert rows[2]["order"] is None
    assert rows[2]["lineage"][-1] == "Sordariomycetes"

    assert rows[3]["family"] == "Enterobacteriaceae"
    assert rows[3]["genus"] is None


def test_build_dataset_writes_report_json(
    silva_fasta: Path,
    silva_taxonomy: Path,
    tmp_path: Path,
) -> None:
    report = build_dataset(silva_fasta, silva_taxonomy, tmp_path, release="test", config=CONFIG)

    written = json.loads((tmp_path / REPORT_FILENAME).read_text(encoding="utf-8"))

    assert written == json.loads(report.to_json())
    assert written["release"] == "test"
    assert written["config"]["filters"]["min_length"] == 10
    assert set(written["excluded"]) == {reason.value for reason in ExclusionReason}


def test_build_dataset_output_does_not_depend_on_batch_size(
    silva_fasta: Path,
    silva_taxonomy: Path,
    tmp_path: Path,
) -> None:
    single = BuildConfig(filters=CONFIG.filters, batch_size=1)
    large = BuildConfig(filters=CONFIG.filters, batch_size=1000)

    _build(silva_fasta, silva_taxonomy, tmp_path / "single", single)
    _build(silva_fasta, silva_taxonomy, tmp_path / "large", large)

    single_table = pq.read_table(tmp_path / "single" / DATASET_FILENAME)
    large_table = pq.read_table(tmp_path / "large" / DATASET_FILENAME)

    assert single_table.equals(large_table)


def test_build_dataset_leaves_no_partial_files(
    silva_fasta: Path,
    silva_taxonomy: Path,
    tmp_path: Path,
) -> None:
    _build(silva_fasta, silva_taxonomy, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [REPORT_FILENAME, DATASET_FILENAME]


def test_build_dataset_cleans_up_when_the_input_is_corrupt(
    silva_taxonomy: Path,
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "corrupt.fasta"
    fasta.write_text("ACGT\n>a\nACGT\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    with pytest.raises(MalformedFastaError):
        _build(fasta, silva_taxonomy, out_dir)

    assert list(out_dir.iterdir()) == []


def test_build_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BuildConfig(batch_size=0)
