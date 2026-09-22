from typing import TYPE_CHECKING

import polars as pl
import pytest

from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.filters import FilterConfig
from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.staging import SEQ_HASH_BYTES, sequence_hash, stage_source
from taxonomy_classifier.data.taxonomy import Rank
from taxonomy_classifier.exceptions import MalformedFastaError

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources.gtdb import GtdbSource
    from taxonomy_classifier.data.sources.silva import SilvaSource

FILTERS = FilterConfig(min_length=10, max_length=100)


def test_sequence_hash_is_a_stable_fixed_size_digest() -> None:
    assert len(sequence_hash("ACGT")) == SEQ_HASH_BYTES
    assert sequence_hash("ACGT") == sequence_hash("ACGT")
    assert sequence_hash("ACGT") != sequence_hash("ACGA")


def test_stage_source_reports_every_outcome(
    gtdb_source: GtdbSource,
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    report = stage_source(
        gtdb_source,
        fixtures_dir,
        tmp_path / "gtdb.parquet",
        filters=FILTERS,
        backbone=BackboneIndex(),
        batch_size=2,
    )

    assert (report.source, report.read, report.kept) == ("gtdb_test", 7, 5)
    assert report.excluded == {ExclusionReason.MALFORMED_HEADER: 1, ExclusionReason.TOO_SHORT: 1}
    assert report.deepest_rank == {Rank.SPECIES: 4, Rank.GENUS: 1}

    serialized = report.to_dict()
    excluded = serialized["excluded"]
    depths = serialized["deepest_rank"]

    assert isinstance(excluded, dict)
    assert isinstance(depths, dict)
    assert set(excluded) == {reason.value for reason in ExclusionReason}
    assert depths["domain"] == 0


def test_stage_source_writes_one_row_per_kept_record(
    gtdb_source: GtdbSource,
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "gtdb.parquet"

    stage_source(
        gtdb_source,
        fixtures_dir,
        out_path,
        filters=FILTERS,
        backbone=BackboneIndex(),
        batch_size=2,
    )

    frame = pl.read_parquet(out_path)
    first = frame.row(0, named=True)

    assert frame.height == 5
    assert first["source"] == "gtdb"
    assert first["seq_hash"] == sequence_hash(first["sequence"])
    assert first["source_lineage"][-1] == "Escherichia coli"
    assert first["species"] == "Escherichia coli"
    assert not (tmp_path / "gtdb.parquet.part").exists()


def test_only_backbone_sources_feed_the_index(
    gtdb_source: GtdbSource,
    silva_source: SilvaSource,
    fixtures_dir: Path,
    tmp_path: Path,
    backbone: BackboneIndex,
) -> None:
    size_before = len(backbone)

    stage_source(
        silva_source,
        fixtures_dir,
        tmp_path / "silva.parquet",
        filters=FILTERS,
        backbone=backbone,
        batch_size=100,
    )

    assert len(backbone) == size_before

    empty = BackboneIndex()

    stage_source(
        gtdb_source,
        fixtures_dir,
        tmp_path / "gtdb.parquet",
        filters=FILTERS,
        backbone=empty,
        batch_size=100,
    )

    assert empty.lookup("Escherichia", domain="Bacteria") is not None


def test_stage_source_cleans_up_on_failure(gtdb_source: GtdbSource, tmp_path: Path) -> None:
    (tmp_path / "gtdb_ssu.fna").write_text("ACGT\n>a\nACGT\n", encoding="utf-8")
    out_path = tmp_path / "out" / "gtdb.parquet"

    with pytest.raises(MalformedFastaError, match="before the first header"):
        stage_source(
            gtdb_source,
            tmp_path,
            out_path,
            filters=FILTERS,
            backbone=BackboneIndex(),
            batch_size=2,
        )

    assert list(out_path.parent.iterdir()) == []
