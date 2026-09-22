from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Final

import polars as pl

from taxonomy_classifier.data.columns import taxon_key
from taxonomy_classifier.data.taxonomy import Rank

if TYPE_CHECKING:
    from collections.abc import Mapping

LEVELS: Final = ("domain", "kingdom", "phylum", "class", "order", "family", "genus", "species")

IGNORE_INDEX: Final = -100


def level_key(level: str) -> pl.Expr:
    if level == "kingdom":
        return pl.col("kingdom")

    return taxon_key(Rank(level))


@dataclass(frozen=True, slots=True)
class LabelSpace:
    classes: Mapping[str, tuple[str, ...]]
    min_count: int

    def __post_init__(self) -> None:
        if tuple(self.classes) != LEVELS:
            msg = f"LabelSpace levels must be {LEVELS}, got {tuple(self.classes)}"
            raise ValueError(msg)

    @classmethod
    def from_frame(cls, frame: pl.LazyFrame, *, min_count: int) -> LabelSpace:
        if min_count < 1:
            msg = f"min_count must be positive, got {min_count}"
            raise ValueError(msg)

        classes = {level: _frequent_keys(frame, level, min_count) for level in LEVELS}

        return cls(classes=classes, min_count=min_count)

    @classmethod
    def from_json(cls, text: str) -> LabelSpace:
        payload = json.loads(text)
        classes = {level: tuple(payload["classes"][level]) for level in LEVELS}

        return cls(classes=classes, min_count=payload["min_count"])

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(self.classes[level]) for level in LEVELS)

    def encode(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(
            level_key(level)
            .replace_strict(self._index(level), default=IGNORE_INDEX, return_dtype=pl.Int64)
            .alias(level)
            for level in LEVELS
        )

    def decode(self, level: str, index: int) -> str:
        return self.classes[level][index]

    def to_json(self) -> str:
        payload = {
            "min_count": self.min_count,
            "classes": {level: list(self.classes[level]) for level in LEVELS},
        }

        return json.dumps(payload, indent=2)

    def _index(self, level: str) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.classes[level])}


def _frequent_keys(frame: pl.LazyFrame, level: str, min_count: int) -> tuple[str, ...]:
    counts = (
        frame.select(level_key(level).alias("key"))
        .drop_nulls()
        .group_by("key")
        .len()
        .filter(pl.col("len") >= min_count)
        .sort("key")
        .collect()
    )

    return tuple(counts.get_column("key").to_list())
