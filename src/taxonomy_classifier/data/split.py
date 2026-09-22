from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import TYPE_CHECKING, Final

import polars as pl

from taxonomy_classifier.data.columns import taxon_key
from taxonomy_classifier.data.taxonomy import Rank

if TYPE_CHECKING:
    from collections.abc import Iterable

_UNIT_DIGEST_BYTES: Final = 8

_UNIT_SCALE: Final = float(1 << (8 * _UNIT_DIGEST_BYTES))

_GENUS_KEY: Final = "_genus_key"

_FAMILY_KEY: Final = "_family_key"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TEST_NOVEL_GENUS = "test_novel_genus"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    novel_genus_fraction: float = 0.05
    min_novel_genus_size: int = 3
    seed: int = 20260922

    def __post_init__(self) -> None:
        fractions = (self.val_fraction, self.test_fraction, self.novel_genus_fraction)

        if any(not 0.0 <= fraction < 1.0 for fraction in fractions):
            msg = f"Split fractions must be in [0, 1), got {fractions}"
            raise ValueError(msg)

        if self.val_fraction + self.test_fraction >= 1.0:
            msg = "val_fraction + test_fraction must leave room for training data"
            raise ValueError(msg)

        if self.min_novel_genus_size < 1:
            msg = f"min_novel_genus_size must be positive, got {self.min_novel_genus_size}"
            raise ValueError(msg)

        if not 0 <= self.seed < 1 << 64:
            msg = f"seed must fit in 64 unsigned bits, got {self.seed}"
            raise ValueError(msg)


def unit_interval(key: bytes, *, seed: int, purpose: bytes) -> float:
    digest = hashlib.blake2b(
        key,
        digest_size=_UNIT_DIGEST_BYTES,
        key=seed.to_bytes(8, "little"),
        person=purpose,
    )

    return int.from_bytes(digest.digest(), "little") / _UNIT_SCALE


def assign_splits(labels: pl.DataFrame, config: SplitConfig) -> pl.DataFrame:
    keyed = labels.with_columns(
        taxon_key(Rank.GENUS).alias(_GENUS_KEY),
        taxon_key(Rank.FAMILY).alias(_FAMILY_KEY),
    )

    held_out = novel_genera(keyed, config)

    splits = [
        _split_for(seq_hash, genus_key, held_out, config)
        for seq_hash, genus_key in keyed.select("seq_hash", _GENUS_KEY).iter_rows()
    ]

    return keyed.select("seq_hash").with_columns(pl.Series("split", splits, dtype=pl.String))


def novel_genera(keyed: pl.DataFrame, config: SplitConfig) -> frozenset[str]:
    genera = keyed.filter(pl.col(_GENUS_KEY).is_not_null()).group_by(_GENUS_KEY, _FAMILY_KEY).len()

    siblings = genera.group_by(_FAMILY_KEY).agg(pl.len().alias("_siblings"))

    eligible = (
        genera.join(siblings, on=_FAMILY_KEY, how="left", nulls_equal=True)
        .filter(
            pl.col(_FAMILY_KEY).is_not_null(),
            pl.col("_siblings") > 1,
            pl.col("len") >= config.min_novel_genus_size,
        )
        .get_column(_GENUS_KEY)
    )

    return frozenset(_sample(eligible, config))


def _sample(genus_keys: Iterable[str], config: SplitConfig) -> Iterable[str]:
    for genus_key in genus_keys:
        draw = unit_interval(genus_key.encode(), seed=config.seed, purpose=b"novel-genus")

        if draw < config.novel_genus_fraction:
            yield genus_key


def _split_for(
    seq_hash: bytes,
    genus_key: str | None,
    held_out: frozenset[str],
    config: SplitConfig,
) -> str:
    if genus_key is not None and genus_key in held_out:
        return Split.TEST_NOVEL_GENUS.value

    draw = unit_interval(seq_hash, seed=config.seed, purpose=b"sequence")

    if draw < config.val_fraction:
        return Split.VAL.value

    if draw < config.val_fraction + config.test_fraction:
        return Split.TEST.value

    return Split.TRAIN.value
