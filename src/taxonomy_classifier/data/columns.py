from typing import Final

import polars as pl

from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, RANK_INDEX, Rank

RANK_COLUMNS: Final = tuple(rank.value for rank in CANONICAL_RANKS)


def taxon_key(rank: Rank) -> pl.Expr:
    depth = RANK_INDEX[rank]
    parents = [pl.col(column).fill_null("") for column in RANK_COLUMNS[: depth + 1]]

    return pl.when(pl.col(rank.value).is_not_null()).then(pl.concat_str(parents, separator=";"))
