import polars as pl
import pytest

from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS, LabelSpace

RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")


def _frame(rows: list[tuple[str | None, ...]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=[*RANKS, "kingdom"], orient="row")


ECOLI = ("Bacteria", "P", "C", "O", "F", "Escherichia", "Escherichia coli", "Bacteria")

RARE = ("Bacteria", "P", "C", "O", "F", "Rarus", "Rarus unicus", "Bacteria")

SHALLOW = ("Bacteria", "P", None, None, None, None, None, None)

TRAIN = _frame([ECOLI, ECOLI, RARE, SHALLOW])


def test_from_frame_keeps_classes_seen_at_least_min_count_times() -> None:
    space = LabelSpace.from_frame(TRAIN.lazy(), min_count=2)

    assert space.sizes == (1, 1, 1, 1, 1, 1, 1, 1)
    assert space.classes["genus"] == ("Bacteria;P;C;O;F;Escherichia",)
    assert space.classes["kingdom"] == ("Bacteria",)


def test_encode_marks_missing_and_rare_labels_as_ignored() -> None:
    space = LabelSpace.from_frame(TRAIN.lazy(), min_count=2)

    encoded = space.encode(_frame([ECOLI, RARE, SHALLOW]))

    assert encoded.columns == list(LEVELS)
    assert encoded.row(0) == (0,) * len(LEVELS)
    assert encoded.row(1)[-2:] == (IGNORE_INDEX, IGNORE_INDEX)
    assert encoded.row(2) == (0, IGNORE_INDEX, 0, *(IGNORE_INDEX,) * 5)


def test_decode_returns_the_class_key() -> None:
    space = LabelSpace.from_frame(TRAIN.lazy(), min_count=1)

    assert space.decode("species", 0).endswith("Escherichia coli")


def test_json_round_trip() -> None:
    space = LabelSpace.from_frame(TRAIN.lazy(), min_count=1)

    assert LabelSpace.from_json(space.to_json()) == space


def test_min_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_count must be positive"):
        LabelSpace.from_frame(TRAIN.lazy(), min_count=0)


def test_levels_must_match_the_hierarchy() -> None:
    with pytest.raises(ValueError, match="levels must be"):
        LabelSpace(classes={"domain": ()}, min_count=1)
