import random

import polars as pl
import pytest

from taxonomy_classifier.data.columns import RANK_COLUMNS
from taxonomy_classifier.data.split import Split, SplitConfig, assign_splits, unit_interval


def _labels(rows: list[tuple[bytes, tuple[str | None, ...]]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"seq_hash": seq_hash, **dict(zip(RANK_COLUMNS, names, strict=True))}
            for seq_hash, names in rows
        ],
        schema={"seq_hash": pl.Binary, **dict.fromkeys(RANK_COLUMNS, pl.String)},
    )


def _genus(family: str, genus: str | None) -> tuple[str | None, ...]:
    return ("Bacteria", "P", "C", "O", family, genus, None)


def _hashes(count: int, seed: int = 0) -> list[bytes]:
    rng = random.Random(seed)

    return [rng.randbytes(16) for _ in range(count)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"val_fraction": -0.1}, "must be in"),
        ({"test_fraction": 1.0}, "must be in"),
        ({"val_fraction": 0.6, "test_fraction": 0.5}, "must leave room"),
        ({"min_novel_genus_size": 0}, "min_novel_genus_size must be positive"),
        ({"seed": -1}, "seed must fit"),
        ({"seed": 1 << 64}, "seed must fit"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SplitConfig(**kwargs)  # type: ignore[arg-type]


def test_unit_interval_is_deterministic_and_seed_dependent() -> None:
    draw = unit_interval(b"key", seed=1, purpose=b"sequence")

    assert 0.0 <= draw < 1.0
    assert draw == unit_interval(b"key", seed=1, purpose=b"sequence")
    assert draw != unit_interval(b"key", seed=2, purpose=b"sequence")
    assert draw != unit_interval(b"key", seed=1, purpose=b"novel-genus")


def test_splits_follow_the_configured_fractions() -> None:
    config = SplitConfig(val_fraction=0.1, test_fraction=0.2, novel_genus_fraction=0.0)
    labels = _labels([(seq_hash, _genus("F", None)) for seq_hash in _hashes(20_000)])

    counts = dict(assign_splits(labels, config).group_by("split").len().iter_rows())

    assert 1_800 < counts[Split.VAL] < 2_200
    assert 3_700 < counts[Split.TEST] < 4_300
    assert Split.TEST_NOVEL_GENUS not in counts


def test_splits_do_not_depend_on_row_order() -> None:
    rows = [(seq_hash, _genus("F", "G")) for seq_hash in _hashes(500)]

    forward = assign_splits(_labels(rows), SplitConfig())
    backward = assign_splits(_labels(rows[::-1]), SplitConfig())

    assert forward.sort("seq_hash").equals(backward.sort("seq_hash"))


def test_novel_genera_require_a_sibling_genus_and_enough_sequences() -> None:
    config = SplitConfig(novel_genus_fraction=0.999_999, min_novel_genus_size=3)
    hashes = iter(_hashes(40))
    rows = [
        *((next(hashes), _genus("F", "G1")) for _ in range(3)),
        *((next(hashes), _genus("F", "G2")) for _ in range(3)),
        *((next(hashes), _genus("F", "Rare")) for _ in range(2)),
        *((next(hashes), _genus("Lonely", "Only")) for _ in range(5)),
        *((next(hashes), _genus("F", None)) for _ in range(5)),
    ]

    splits = assign_splits(_labels(rows), config).get_column("split").to_list()

    assert splits[:6] == [Split.TEST_NOVEL_GENUS] * 6
    assert Split.TEST_NOVEL_GENUS not in splits[6:]


def test_assign_splits_handles_an_empty_table() -> None:
    assert assign_splits(_labels([]), SplitConfig()).is_empty()


def test_eligible_genera_are_only_held_out_when_drawn() -> None:
    config = SplitConfig(novel_genus_fraction=0.0, min_novel_genus_size=1)
    hashes = iter(_hashes(4))
    rows = [(next(hashes), _genus("F", genus)) for genus in ("G1", "G1", "G2", "G2")]

    splits = assign_splits(_labels(rows), config).get_column("split").to_list()

    assert Split.TEST_NOVEL_GENUS not in splits
