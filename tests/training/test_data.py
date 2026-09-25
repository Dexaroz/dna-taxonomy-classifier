import random
from typing import TYPE_CHECKING

import polars as pl
import pytest
import torch

from taxonomy_classifier.data.staging import sequence_hash
from taxonomy_classifier.data.taxonomy import Rank
from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS, LabelSpace
from taxonomy_classifier.model.tokens import PAD
from taxonomy_classifier.training import data
from taxonomy_classifier.training.data import (
    CropConfig,
    LabeledSet,
    SequenceBank,
    TrainingData,
    class_frequencies,
    iterate_batches,
    make_batch,
    sampling_weights,
)

if TYPE_CHECKING:
    from pathlib import Path

ECOLI = ("Bacteria", "Bacteria", "P", "C", "O", "F", "Escherichia", "Escherichia coli")

RARE = ("Bacteria", "Bacteria", "P", "C", "O", "F", "Rarus", None)


def _frame(rows: list[tuple[str, tuple[str | None, ...]]]) -> pl.LazyFrame:
    records = [
        {
            "seq_hash": sequence_hash(sequence),
            "sequence": sequence,
            "length": len(sequence),
            "domain": names[0],
            "kingdom": names[1],
            **dict(
                zip(
                    ("phylum", "class", "order", "family", "genus", "species"),
                    names[2:],
                    strict=True,
                )
            ),
        }
        for sequence, names in rows
    ]

    return pl.LazyFrame(records)


FRAME = _frame([("ACGTACGT", ECOLI), ("CCCC", ECOLI), ("GGGGGGTT", RARE)])

SPACE = LabelSpace.from_frame(FRAME, min_count=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"probability": 1.5}, "probability must be in"),
        ({"min_length": 0}, "crop lengths"),
        ({"min_length": 10, "max_length": 5}, "crop lengths"),
    ],
)
def test_crop_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CropConfig(**kwargs)  # type: ignore[arg-type]


def test_sequence_bank_stores_every_sequence_contiguously() -> None:
    sequences = pl.Series(["ACGT", "", "NNA"])
    bank = SequenceBank.from_batches(
        [sequences.slice(0, 2), sequences.slice(2)], sequences.str.len_bytes()
    )

    assert len(bank) == 3
    assert bank.get(0).tolist() == [1, 2, 3, 4]
    assert bank.get(1).tolist() == []
    assert bank.get(2).tolist() == [5, 5, 1]


def test_sequence_bank_rejects_lengths_that_do_not_match() -> None:
    with pytest.raises(ValueError, match="lengths add up to"):
        SequenceBank.from_batches([pl.Series(["ACGT"])], pl.Series([3]))


def test_labeled_set_streams_sequences_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "_CHUNK_ROWS", 2)

    labeled = LabeledSet.from_frame(FRAME, SPACE)

    assert len(labeled) == 3
    assert labeled.bank.get(2).tolist() == [3, 3, 3, 3, 3, 3, 4, 4]
    assert labeled.targets.shape == (3, len(LEVELS))
    assert labeled.targets[2, -2:].tolist() == [IGNORE_INDEX, IGNORE_INDEX]


def test_training_data_needs_one_weight_per_sequence() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    with pytest.raises(ValueError, match="one class frequency per training sequence"):
        TrainingData(train=labeled, frequencies=torch.ones(2), validation=labeled)


def test_training_data_needs_one_marker_per_validation_sequence() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    with pytest.raises(ValueError, match="one marker per validation sequence"):
        TrainingData(
            train=labeled,
            frequencies=torch.ones(len(labeled)),
            validation=labeled,
            validation_markers=("ssu",),
        )


def test_training_data_rejects_frequencies_below_one() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    with pytest.raises(ValueError, match="at least 1"):
        TrainingData(train=labeled, frequencies=torch.tensor([1.0, 0.0, 1.0]), validation=labeled)


def test_class_frequencies_count_each_sequence_genus() -> None:
    frequencies = class_frequencies(FRAME, rank=Rank.GENUS)

    assert frequencies.dtype == torch.double
    assert frequencies.tolist() == [2.0, 2.0, 1.0]


def test_power_zero_keeps_natural_frequencies() -> None:
    assert sampling_weights(torch.tensor([2.0, 2.0, 1.0]), power=0.0).tolist() == [1.0, 1.0, 1.0]


def test_power_one_gives_every_class_the_same_total_weight() -> None:
    frequencies = torch.tensor([9.0] * 9 + [1.0])

    weights = sampling_weights(frequencies, power=1.0)

    assert float(weights[:9].sum()) == pytest.approx(float(weights[9]))
    assert float(weights.sum()) == pytest.approx(10.0)


def test_intermediate_power_favors_rare_classes_without_equalizing() -> None:
    weights = sampling_weights(torch.tensor([4.0, 4.0, 4.0, 4.0, 1.0]), power=0.5)

    assert float(weights[-1]) == pytest.approx(2 * float(weights[0]))


@pytest.mark.parametrize("power", [-0.1, 1.1])
def test_power_must_be_in_unit_interval(power: float) -> None:
    with pytest.raises(ValueError, match="power must be in"):
        sampling_weights(torch.ones(2), power=power)


def test_make_batch_pads_masks_and_gathers_targets() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    batch = make_batch(labeled, [1, 0], crop=None, rng=random.Random(0))

    assert batch.tokens.tolist() == [[2, 2, 2, 2, PAD, PAD, PAD, PAD], [1, 2, 3, 4, 1, 2, 3, 4]]
    assert batch.mask.sum(dim=-1).tolist() == [4, 8]
    assert torch.equal(batch.targets, labeled.targets[[1, 0]])
    assert batch.tokens.dtype == torch.long


def test_make_batch_crops_contiguous_windows() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)
    crop = CropConfig(probability=1.0, min_length=3, max_length=3)

    batch = make_batch(labeled, [0, 1], crop=crop, rng=random.Random(1))
    full = labeled.bank.get(0).tolist()
    window = batch.tokens[0, :3].tolist()

    assert batch.mask.sum(dim=-1).tolist() == [3, 3]
    assert any(full[start : start + 3] == window for start in range(len(full) - 2))


def test_make_batch_keeps_sequences_shorter_than_the_crop() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)
    crop = CropConfig(probability=1.0, min_length=6, max_length=6)

    batch = make_batch(labeled, [1], crop=crop, rng=random.Random(0))

    assert batch.tokens.tolist() == [[2, 2, 2, 2]]


def test_iterate_batches_covers_every_index_in_order() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    batches = list(
        iterate_batches(
            labeled, torch.tensor([2, 0, 1]), batch_size=2, crop=None, rng=random.Random(0)
        )
    )

    assert [len(batch.targets) for batch in batches] == [2, 1]
    assert torch.equal(batches[0].targets, labeled.targets[[2, 0]])


def test_batch_moves_to_a_device() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    batch = make_batch(labeled, [0], crop=None, rng=random.Random(0)).to(torch.device("cpu"))

    assert batch.tokens.device.type == "cpu"


def test_sequence_bank_round_trips_through_a_memory_mapped_file(tmp_path: Path) -> None:
    sequences = pl.Series(["ACGT", "", "NNA"])
    path = tmp_path / "bank.bin"

    written = SequenceBank.write([sequences], sequences.str.len_bytes(), path)
    reopened = SequenceBank.open(path)

    assert [reopened.get(index).tolist() for index in range(3)] == [[1, 2, 3, 4], [], [5, 5, 1]]
    assert torch.equal(written.offsets, reopened.offsets)


def test_empty_banks_can_be_written_and_opened(tmp_path: Path) -> None:
    bank = SequenceBank.write(
        [pl.Series([], dtype=pl.String)], pl.Series([], dtype=pl.Int64), tmp_path / "empty.bin"
    )

    assert len(bank) == 0


def test_failed_writes_leave_no_partial_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lengths add up to"):
        SequenceBank.write([pl.Series(["ACGT"])], pl.Series([3]), tmp_path / "bank.bin")

    assert list(tmp_path.iterdir()) == []


def test_labeled_set_reuses_a_matching_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "tokens.bin"
    first = LabeledSet.from_frame(FRAME, SPACE, cache=cache)

    def fail(*_: object) -> SequenceBank:
        msg = "the cache should have been reused"
        raise AssertionError(msg)

    monkeypatch.setattr(SequenceBank, "write", fail)

    second = LabeledSet.from_frame(FRAME, SPACE, cache=cache)

    assert torch.equal(first.bank.tokens, second.bank.tokens)
    assert torch.equal(first.targets, second.targets)


def test_labeled_set_rebuilds_a_stale_cache(tmp_path: Path) -> None:
    cache = tmp_path / "tokens.bin"
    LabeledSet.from_frame(FRAME, SPACE, cache=cache)

    changed = _frame([("TTTT", ECOLI)])
    rebuilt = LabeledSet.from_frame(changed, SPACE, cache=cache)

    assert len(rebuilt) == 1
    assert rebuilt.bank.get(0).tolist() == [4, 4, 4, 4]


def test_make_batch_truncates_sequences_to_max_length() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    batch = make_batch(labeled, [0, 1], crop=None, rng=random.Random(0), max_length=3)

    assert batch.tokens.tolist() == [[1, 2, 3], [2, 2, 2]]
    assert batch.mask.all()


def test_take_builds_a_compact_subset() -> None:
    labeled = LabeledSet.from_frame(FRAME, SPACE)

    subset = labeled.take([2, 0])

    assert len(subset) == 2
    assert subset.bank.get(0).tolist() == labeled.bank.get(2).tolist()
    assert subset.bank.get(1).tolist() == labeled.bank.get(0).tolist()
    assert torch.equal(subset.targets, labeled.targets[[2, 0]])


def test_take_accepts_an_empty_selection() -> None:
    assert len(LabeledSet.from_frame(FRAME, SPACE).take([])) == 0
