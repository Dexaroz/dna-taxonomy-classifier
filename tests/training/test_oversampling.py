import json
from typing import TYPE_CHECKING

import polars as pl
import pytest

from taxonomy_classifier.data.staging import sequence_hash
from taxonomy_classifier.training import oversampling
from taxonomy_classifier.training.augmentation import AugmentationConfig
from taxonomy_classifier.training.oversampling import (
    OversamplingConfig,
    cap_by_group,
    distribute,
    oversample_train,
    plan_copies,
    synthesize,
    training_frame,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = pl.Schema(
    {
        "seq_hash": pl.Binary(),
        "sequence": pl.String(),
        "length": pl.Int32(),
        "n_ambiguous": pl.Int32(),
        "domain": pl.String(),
        "kingdom": pl.String(),
        "phylum": pl.String(),
        "class": pl.String(),
        "order": pl.String(),
        "family": pl.String(),
        "genus": pl.String(),
        "species": pl.String(),
        "sources": pl.List(pl.String()),
        "accessions": pl.List(pl.String()),
        "n_records": pl.Int32(),
        "label_conflict": pl.Boolean(),
        "split": pl.String(),
    }
)

NO_MUTATIONS = AugmentationConfig(
    substitution_rate=0.0,
    insertion_rate=0.0,
    deletion_rate=0.0,
    crop_probability=0.0,
)


def _sequence(index: int) -> str:
    bases = "ACGT"

    return "".join(bases[(index >> shift) & 3] for shift in range(0, 40, 2)) * 50


def _row(
    index: int,
    genus: str,
    *,
    kingdom: str | None = "Bacteria",
    split: str = "train",
) -> dict[str, object]:
    sequence = _sequence(index)

    return {
        "seq_hash": sequence_hash(sequence),
        "sequence": sequence,
        "length": len(sequence),
        "n_ambiguous": 0,
        "domain": "Eukaryota" if kingdom == "Animalia" else "Bacteria",
        "kingdom": kingdom,
        "phylum": "P",
        "class": "C",
        "order": "O",
        "family": "F",
        "genus": genus,
        "species": None,
        "sources": ["gtdb"],
        "accessions": [f"gtdb:{index}"],
        "n_records": 1,
        "label_conflict": False,
        "split": split,
    }


def _dataset(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "dataset.parquet"
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(path)

    return path


def _run(
    tmp_path: Path, dataset: Path, config: OversamplingConfig
) -> oversampling.OversamplingReport:
    return oversample_train(
        dataset,
        tmp_path / "synthetic.parquet",
        tmp_path / "report.json",
        config=config,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"floor": 0}, "floor must be positive"),
        ({"max_copies_per_sequence": -1}, "must not be negative"),
        ({"max_synthetic_ratio": 0.0}, r"must be in \(0, 1\]"),
        ({"max_synthetic_ratio": 1.5}, r"must be in \(0, 1\]"),
        ({"seed": -1}, "seed must fit"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OversamplingConfig(**kwargs)  # type: ignore[arg-type]


def test_plan_copies_fills_up_to_the_floor_within_the_per_sequence_cap() -> None:
    plan = plan_copies({"tiny": 1, "small": 8, "large": 50}, floor=20, max_copies_per_sequence=5)

    assert plan == {"tiny": 5, "small": 12}


def test_cap_by_group_keeps_plans_within_budget() -> None:
    plan = {"a1": 3, "a2": 3, "a3": 3, "b1": 4}
    groups = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}

    capped = cap_by_group(plan, groups, {"A": 7, "B": 10}, max_ratio=1.0)

    assert capped == {"a1": 3, "a2": 2, "a3": 2, "b1": 4}


def test_cap_by_group_drops_classes_left_without_copies() -> None:
    capped = cap_by_group({"a": 1, "b": 1}, {"a": "G", "b": "G"}, {"G": 1}, max_ratio=1.0)

    assert capped == {"a": 1}


def test_distribute_spreads_copies_evenly_and_deterministically() -> None:
    assert distribute([b"c", b"a", b"b"], 7) == {b"a": 3, b"b": 2, b"c": 2}
    assert distribute([b"b", b"a"], 1) == {b"a": 1}


def test_synthesize_is_deterministic_and_always_changes_the_sequence() -> None:
    config = OversamplingConfig()
    parent = _sequence(1)

    first = synthesize(parent, b"parent", 0, config)

    assert first != parent
    assert first == synthesize(parent, b"parent", 0, config)
    assert first != synthesize(parent, b"parent", 1, config)


@pytest.mark.parametrize(("sequence", "expected"), [("ACGT", "CCGT"), ("TCGT", "ACGT")])
def test_synthesize_forces_a_change_when_no_mutation_is_drawn(sequence: str, expected: str) -> None:
    config = OversamplingConfig(mutations=NO_MUTATIONS)

    assert synthesize(sequence, b"parent", 0, config) == expected


def test_oversample_train_fills_small_classes_from_training_data_only(tmp_path: Path) -> None:
    rows = [
        _row(0, "Rare"),
        _row(1, "Rare", split="test"),
        *(_row(index, "Common") for index in range(10, 30)),
    ]
    config = OversamplingConfig(floor=5, max_copies_per_sequence=3)

    report = _run(tmp_path, _dataset(tmp_path, rows), config)
    synthetic = pl.read_parquet(tmp_path / "synthetic.parquet")

    assert (report.real, report.synthetic, report.classes) == (21, 3, 2)
    assert (report.below_floor_before, report.below_floor_after) == (1, 1)
    assert synthetic.get_column("parent_seq_hash").to_list() == [rows[0]["seq_hash"]] * 3
    assert synthetic.get_column("genus").to_list() == ["Rare"] * 3
    assert synthetic.get_column("split").unique().to_list() == ["train"]
    assert synthetic.get_column("synthetic").all()
    assert synthetic.get_column("length").to_list() == [
        len(sequence) for sequence in synthetic.get_column("sequence")
    ]


def test_no_kingdom_ever_has_more_synthetic_than_real_sequences(tmp_path: Path) -> None:
    rows = [
        *(_row(index, f"Animal{index}", kingdom="Animalia") for index in range(4)),
        *(_row(index, f"Bug{index}", kingdom="Bacteria") for index in range(10, 14)),
    ]
    config = OversamplingConfig(floor=20, max_copies_per_sequence=5, max_synthetic_ratio=0.5)

    report = _run(tmp_path, _dataset(tmp_path, rows), config)

    assert report.real_by_kingdom == {"Animalia": 4, "Bacteria": 4}
    assert report.synthetic_by_kingdom == {"Animalia": 2, "Bacteria": 2}
    assert report.synthetic_ratio == 0.5


def test_report_is_written_as_json(tmp_path: Path) -> None:
    rows = [_row(0, "Rare"), _row(1, "Rare", kingdom=None)]

    report = _run(tmp_path, _dataset(tmp_path, rows), OversamplingConfig(floor=2))
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert written == json.loads(report.to_json())
    assert written["by_kingdom"]["unknown"]["real"] == 1
    assert written["config"]["floor"] == 2


def test_empty_training_split_produces_an_empty_synthetic_table(tmp_path: Path) -> None:
    report = _run(tmp_path, _dataset(tmp_path, [_row(0, "G", split="val")]), OversamplingConfig())

    assert (report.real, report.synthetic, report.synthetic_ratio) == (0, 0, 0.0)
    assert pl.read_parquet(tmp_path / "synthetic.parquet").is_empty()


def test_synthetic_sequences_never_duplicate_real_ones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(0, "Rare"), _row(1, "Other", split="test")]
    dataset = _dataset(tmp_path, rows)

    monkeypatch.setattr(oversampling, "synthesize", lambda *_: rows[1]["sequence"])

    report = _run(tmp_path, dataset, OversamplingConfig(floor=3, max_copies_per_sequence=2))

    assert (report.synthetic, report.collisions) == (0, 1)


def test_synthetic_file_is_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, [_row(0, "Rare")])

    def explode(_: pl.DataFrame, path: Path, **__: object) -> None:
        path.write_bytes(b"partial")
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", explode)

    with pytest.raises(OSError, match="disk full"):
        _run(tmp_path, dataset, OversamplingConfig(floor=2))

    assert sorted(path.name for path in tmp_path.iterdir()) == ["dataset.parquet"]


def test_training_frame_combines_real_and_synthetic_rows(tmp_path: Path) -> None:
    rows = [_row(0, "Rare"), _row(1, "Rare", split="val")]
    dataset = _dataset(tmp_path, rows)
    _run(tmp_path, dataset, OversamplingConfig(floor=3, max_copies_per_sequence=2))

    frame = training_frame(dataset, tmp_path / "synthetic.parquet").collect()

    assert frame.height == 2
    assert frame.get_column("synthetic").to_list() == [False, True]
    assert frame.get_column("parent_seq_hash").null_count() == 1
    assert set(frame.get_column("split")) == {"train"}
