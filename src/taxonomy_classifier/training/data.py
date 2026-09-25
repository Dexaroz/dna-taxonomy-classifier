from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Final

import polars as pl
import torch

from taxonomy_classifier.data.columns import RANK_COLUMNS
from taxonomy_classifier.data.files import partial_path, write_text_atomic
from taxonomy_classifier.data.taxonomy import RANK_INDEX, Rank
from taxonomy_classifier.model.tokens import PAD, encode_bytes
from taxonomy_classifier.training.balancing import balancing_keys

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path
    import random

    from taxonomy_classifier.model.labels import LabelSpace

_CHUNK_ROWS: Final = 100_000

LABEL_COLUMNS: Final = ("kingdom", *RANK_COLUMNS)


@dataclass(frozen=True, slots=True)
class CropConfig:
    probability: float = 0.5
    min_length: int = 150
    max_length: int = 500

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            msg = f"probability must be in [0, 1], got {self.probability}"
            raise ValueError(msg)

        if not 1 <= self.min_length <= self.max_length:
            msg = (
                "crop lengths must satisfy 1 <= min <= max, "
                f"got {self.min_length} and {self.max_length}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SequenceBank:
    tokens: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_batches(cls, batches: Iterable[pl.Series], lengths: pl.Series) -> SequenceBank:
        offsets = _offsets(lengths)
        encoded = bytearray(b"".join(_encode(batch) for batch in batches))

        _check_size(len(encoded), int(offsets[-1]))

        tokens = (
            torch.frombuffer(encoded, dtype=torch.uint8)
            if encoded
            else torch.empty(0, dtype=torch.uint8)
        )

        return cls(tokens=tokens, offsets=offsets)

    @classmethod
    def write(cls, batches: Iterable[pl.Series], lengths: pl.Series, path: Path) -> SequenceBank:
        offsets = _offsets(lengths)
        partial = partial_path(path)
        written = 0

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with partial.open("wb") as handle:
                for batch in batches:
                    written += handle.write(_encode(batch))

            _check_size(written, int(offsets[-1]))

        except BaseException:
            partial.unlink(missing_ok=True)

            raise

        partial.replace(path)
        torch.save(offsets, _offsets_path(path))

        return cls.open(path)

    @classmethod
    def open(cls, path: Path) -> SequenceBank:
        offsets: torch.Tensor = torch.load(_offsets_path(path), weights_only=True)
        size = int(offsets[-1])

        if size == 0:
            return cls(tokens=torch.empty(0, dtype=torch.uint8), offsets=offsets)

        tokens = torch.from_file(str(path), shared=True, size=size, dtype=torch.uint8)

        return cls(tokens=tokens, offsets=offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, index: int) -> torch.Tensor:
        return self.tokens[int(self.offsets[index]) : int(self.offsets[index + 1])]


@dataclass(frozen=True, slots=True)
class LabeledSet:
    bank: SequenceBank
    targets: torch.Tensor

    @classmethod
    def from_frame(
        cls,
        frame: pl.LazyFrame,
        label_space: LabelSpace,
        *,
        cache: Path | None = None,
    ) -> LabeledSet:
        labels = frame.select("seq_hash", "length", *LABEL_COLUMNS).collect()

        return cls(
            bank=_bank(frame, labels, cache),
            targets=label_space.encode(labels).to_torch(),
        )

    def __len__(self) -> int:
        return len(self.bank)

    def take(self, indices: Sequence[int]) -> LabeledSet:
        pieces = [self.bank.get(index) for index in indices]
        offsets = torch.zeros(len(pieces) + 1, dtype=torch.long)
        offsets[1:] = torch.tensor([len(piece) for piece in pieces], dtype=torch.long).cumsum(dim=0)

        tokens = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.uint8)

        return LabeledSet(
            bank=SequenceBank(tokens=tokens, offsets=offsets),
            targets=self.targets[list(indices)],
        )


@dataclass(frozen=True, slots=True)
class TrainingData:
    train: LabeledSet
    frequencies: torch.Tensor
    validation: LabeledSet
    validation_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.validation_markers and len(self.validation_markers) != len(self.validation):
            msg = "Expected one marker per validation sequence"
            raise ValueError(msg)

        if len(self.frequencies) != len(self.train):
            msg = f"Expected one class frequency per training sequence, got {len(self.frequencies)}"
            raise ValueError(msg)

        if len(self.frequencies) and float(self.frequencies.min()) < 1:
            msg = "Class frequencies must be at least 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Batch:
    tokens: torch.Tensor
    mask: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            tokens=self.tokens.to(device, non_blocking=True),
            mask=self.mask.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
        )


def class_frequencies(frame: pl.LazyFrame, *, rank: Rank = Rank.GENUS) -> torch.Tensor:
    depth = RANK_INDEX[rank] + 1
    lineages = frame.select(RANK_COLUMNS[:depth]).collect()
    keys = balancing_keys(lineages.iter_rows(), depth=depth)
    counts = Counter(keys)

    return torch.tensor([counts[key] for key in keys], dtype=torch.double)


def sampling_weights(frequencies: torch.Tensor, *, power: float) -> torch.Tensor:
    if not 0.0 <= power <= 1.0:
        msg = f"power must be in [0, 1], got {power}"
        raise ValueError(msg)

    weights = frequencies.double().pow(-power)

    return weights * len(weights) / weights.sum()


def make_batch(
    data: LabeledSet,
    indices: Sequence[int],
    *,
    crop: CropConfig | None,
    rng: random.Random,
    max_length: int | None = None,
) -> Batch:
    pieces = [_maybe_crop(data.bank.get(index), crop, rng)[:max_length] for index in indices]
    lengths = torch.tensor([len(piece) for piece in pieces], dtype=torch.long)

    tokens = torch.full((len(pieces), max(1, int(lengths.max()))), PAD, dtype=torch.uint8)

    for row, piece in enumerate(pieces):
        tokens[row, : len(piece)] = piece

    mask = torch.arange(tokens.shape[1]) < lengths.unsqueeze(-1)

    return Batch(tokens=tokens.long(), mask=mask, targets=data.targets[list(indices)])


def iterate_batches(
    data: LabeledSet,
    indices: torch.Tensor,
    *,
    batch_size: int,
    crop: CropConfig | None,
    rng: random.Random,
    max_length: int | None = None,
) -> Iterator[Batch]:
    order = indices.tolist()

    for start in range(0, len(order), batch_size):
        yield make_batch(
            data, order[start : start + batch_size], crop=crop, rng=rng, max_length=max_length
        )


def _bank(frame: pl.LazyFrame, labels: pl.DataFrame, cache: Path | None) -> SequenceBank:
    lengths = labels.get_column("length")

    def sequences() -> Iterator[pl.Series]:
        for batch in frame.select("sequence").collect_batches(chunk_size=_CHUNK_ROWS):
            yield batch.get_column("sequence")

    if cache is None:
        return SequenceBank.from_batches(sequences(), lengths)

    manifest = cache.with_suffix(".json")
    fingerprint = _fingerprint(labels.get_column("seq_hash"))

    if cache.exists() and manifest.exists():
        stored = json.loads(manifest.read_text(encoding="utf-8"))

        if stored.get("fingerprint") == fingerprint:
            return SequenceBank.open(cache)

    bank = SequenceBank.write(sequences(), lengths, cache)
    write_text_atomic(manifest, json.dumps({"fingerprint": fingerprint, "sequences": len(lengths)}))

    return bank


def _fingerprint(seq_hashes: pl.Series) -> str:
    digest = hashlib.blake2b(digest_size=16)

    for seq_hash in seq_hashes:
        digest.update(seq_hash)

    return digest.hexdigest()


def _offsets(lengths: pl.Series) -> torch.Tensor:
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = lengths.cast(pl.Int64).to_torch().cumsum(dim=0)

    return offsets


def _offsets_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.offsets.pt")


def _encode(batch: pl.Series) -> bytes:
    return encode_bytes("".join(batch.to_list()).encode("ascii"))


def _check_size(written: int, expected: int) -> None:
    if written != expected:
        msg = f"Sequences hold {written} bases but their lengths add up to {expected}"
        raise ValueError(msg)


def _maybe_crop(tokens: torch.Tensor, crop: CropConfig | None, rng: random.Random) -> torch.Tensor:
    if crop is None or rng.random() >= crop.probability:
        return tokens

    length = rng.randint(crop.min_length, crop.max_length)

    if len(tokens) <= length:
        return tokens

    start = rng.randint(0, len(tokens) - length)

    return tokens[start : start + length]
