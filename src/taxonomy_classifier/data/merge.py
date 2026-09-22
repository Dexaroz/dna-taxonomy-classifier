from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from taxonomy_classifier.data.columns import RANK_COLUMNS, taxon_key
from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_PREFIX: Final = "_prefix_"

_DISTINCT: Final = "_distinct_"

_KINGDOM_CONFLICT: Final = "_kingdom_conflict"

MERGED_COLUMNS: Final = (
    "seq_hash",
    "sequence",
    "length",
    "n_ambiguous",
    RANK_COLUMNS[0],
    "kingdom",
    *RANK_COLUMNS[1:],
    "sources",
    "accessions",
    "n_records",
    "label_conflict",
)


@dataclass(frozen=True, slots=True)
class MergeReport:
    staged: int
    unique: int
    kept: int
    merged: int
    conflicts: int

    @property
    def domain_conflicts(self) -> int:
        return self.unique - self.kept

    def to_dict(self) -> dict[str, object]:
        return {
            "staged": self.staged,
            "unique": self.unique,
            "kept": self.kept,
            "merged": self.merged,
            "conflicts": self.conflicts,
            "domain_conflicts": self.domain_conflicts,
        }


def merge_staged(staged: Sequence[Path], out_path: Path) -> MergeReport:
    partial = partial_path(out_path)

    try:
        merged_frame(pl.scan_parquet(list(staged))).sink_parquet(partial, compression="zstd")

    except BaseException:
        partial.unlink(missing_ok=True)

        raise

    partial.replace(out_path)

    return _report(staged, out_path)


def merged_frame(staged: pl.LazyFrame) -> pl.LazyFrame:
    prefixes = [taxon_key(rank).alias(f"{_PREFIX}{rank.value}") for rank in CANONICAL_RANKS]

    grouped = (
        staged.with_columns(prefixes)
        .group_by("seq_hash")
        .agg(
            pl.col("sequence").first(),
            pl.col("length").first(),
            pl.col("n_ambiguous").first(),
            *(pl.col(column).drop_nulls().first() for column in RANK_COLUMNS),
            pl.col("kingdom").drop_nulls().first(),
            (pl.col("kingdom").drop_nulls().n_unique() > 1).alias(_KINGDOM_CONFLICT),
            *(
                pl.col(f"{_PREFIX}{column}").drop_nulls().n_unique().alias(f"{_DISTINCT}{column}")
                for column in RANK_COLUMNS
            ),
            pl.col("source").unique().sort().alias("sources"),
            pl.concat_str("source", "accession", separator=":").sort().alias("accessions"),
            pl.len().cast(pl.Int32).alias("n_records"),
        )
    )

    conflicts = [pl.col(f"{_DISTINCT}{column}") > 1 for column in RANK_COLUMNS]

    resolved = [
        pl.when(pl.any_horizontal(conflicts[: depth + 1]))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for depth, column in enumerate(RANK_COLUMNS)
    ]

    kingdom = (
        pl.when(pl.col(_KINGDOM_CONFLICT)).then(None).otherwise(pl.col("kingdom")).alias("kingdom")
    )

    label_conflict = pl.any_horizontal(*conflicts, pl.col(_KINGDOM_CONFLICT)).alias(
        "label_conflict"
    )

    return (
        grouped.with_columns(*resolved, kingdom, label_conflict)
        .filter(pl.col("domain").is_not_null())
        .select(MERGED_COLUMNS)
        .sort("seq_hash")
    )


def _report(staged: Sequence[Path], merged: Path) -> MergeReport:
    totals = (
        pl.scan_parquet(list(staged))
        .select(pl.len().alias("staged"), pl.col("seq_hash").n_unique().alias("unique"))
        .collect()
        .row(0, named=True)
    )

    stats = (
        pl.scan_parquet(merged)
        .select(
            pl.len().alias("kept"),
            (pl.col("n_records") > 1).sum().alias("merged"),
            pl.col("label_conflict").sum().alias("conflicts"),
        )
        .collect()
        .row(0, named=True)
    )

    return MergeReport(
        staged=totals["staged"],
        unique=totals["unique"],
        kept=stats["kept"],
        merged=stats["merged"],
        conflicts=stats["conflicts"],
    )
