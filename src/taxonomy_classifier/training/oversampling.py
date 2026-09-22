from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import random
from typing import TYPE_CHECKING, Final

import polars as pl

from taxonomy_classifier.data.build import UNKNOWN_KINGDOM
from taxonomy_classifier.data.columns import RANK_COLUMNS
from taxonomy_classifier.data.dna import count_ambiguous
from taxonomy_classifier.data.files import partial_path, write_text_atomic
from taxonomy_classifier.data.split import Split
from taxonomy_classifier.data.staging import sequence_hash
from taxonomy_classifier.data.taxonomy import RANK_INDEX, Rank
from taxonomy_classifier.training.augmentation import AugmentationConfig, SequenceAugmenter
from taxonomy_classifier.training.balancing import balancing_keys

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_LOGGER: Final = logging.getLogger(__name__)

_SEED_BYTES: Final = 8

MUTATIONS_ONLY: Final = AugmentationConfig(
    substitution_rate=0.005,
    insertion_rate=0.001,
    deletion_rate=0.001,
    crop_probability=0.0,
    reverse_complement_probability=0.0,
)


@dataclass(frozen=True, slots=True)
class OversamplingConfig:
    rank: Rank = Rank.GENUS
    floor: int = 20
    max_copies_per_sequence: int = 5
    max_synthetic_ratio: float = 1.0
    seed: int = 20260922
    mutations: AugmentationConfig = field(default_factory=lambda: MUTATIONS_ONLY)

    def __post_init__(self) -> None:
        if self.floor < 1:
            msg = f"floor must be positive, got {self.floor}"
            raise ValueError(msg)

        if self.max_copies_per_sequence < 0:
            msg = (
                f"max_copies_per_sequence must not be negative, got {self.max_copies_per_sequence}"
            )
            raise ValueError(msg)

        if not 0.0 < self.max_synthetic_ratio <= 1.0:
            msg = f"max_synthetic_ratio must be in (0, 1], got {self.max_synthetic_ratio}"
            raise ValueError(msg)

        if not 0 <= self.seed < 1 << 64:
            msg = f"seed must fit in 64 unsigned bits, got {self.seed}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class OversamplingReport:
    config: OversamplingConfig
    real: int
    synthetic: int
    classes: int
    below_floor_before: int
    below_floor_after: int
    collisions: int
    real_by_kingdom: Mapping[str, int]
    synthetic_by_kingdom: Mapping[str, int]

    @property
    def synthetic_ratio(self) -> float:
        return self.synthetic / self.real if self.real else 0.0

    def to_json(self) -> str:
        payload = {
            "config": asdict(self.config),
            "real": self.real,
            "synthetic": self.synthetic,
            "synthetic_ratio": round(self.synthetic_ratio, 4),
            "classes": self.classes,
            "below_floor_before": self.below_floor_before,
            "below_floor_after": self.below_floor_after,
            "collisions": self.collisions,
            "by_kingdom": {
                kingdom: {
                    "real": real,
                    "synthetic": self.synthetic_by_kingdom.get(kingdom, 0),
                }
                for kingdom, real in self.real_by_kingdom.items()
            },
        }

        return json.dumps(payload, indent=2)


def plan_copies(
    class_sizes: Mapping[str, int],
    *,
    floor: int,
    max_copies_per_sequence: int,
) -> dict[str, int]:
    plan = {
        key: min(max(floor - size, 0), max_copies_per_sequence * size)
        for key, size in class_sizes.items()
    }

    return {key: copies for key, copies in plan.items() if copies > 0}


def cap_by_group(
    plan: Mapping[str, int],
    groups: Mapping[str, str],
    group_sizes: Mapping[str, int],
    *,
    max_ratio: float,
) -> dict[str, int]:
    members: defaultdict[str, dict[str, int]] = defaultdict(dict)

    for key, copies in plan.items():
        members[groups[key]][key] = copies

    capped: dict[str, int] = {}

    for group, group_plan in members.items():
        budget = int(max_ratio * group_sizes[group])
        capped.update(_largest_remainder(group_plan, budget))

    return {key: copies for key, copies in capped.items() if copies > 0}


def _largest_remainder(plan: Mapping[str, int], budget: int) -> dict[str, int]:
    total = sum(plan.values())

    if total <= budget:
        return dict(plan)

    exact = {key: copies * budget / total for key, copies in plan.items()}
    allocated = {key: int(share) for key, share in exact.items()}

    leftover = budget - sum(allocated.values())
    ranked = sorted(exact, key=lambda key: (allocated[key] - exact[key], key))

    for key in ranked[:leftover]:
        allocated[key] += 1

    return allocated


def distribute(members: Sequence[bytes], total: int) -> dict[bytes, int]:
    ordered = sorted(members)
    base, extra = divmod(total, len(ordered))

    shares = {member: base + (1 if index < extra else 0) for index, member in enumerate(ordered)}

    return {member: share for member, share in shares.items() if share > 0}


def synthesize(sequence: str, parent: bytes, copy: int, config: OversamplingConfig) -> str:
    rng = random.Random(_copy_seed(parent, copy, config.seed))
    mutated = SequenceAugmenter(config.mutations, rng)(sequence)

    if mutated != sequence:
        return mutated

    replacement = "C" if sequence[0] == "A" else "A"

    return replacement + sequence[1:]


def oversample_train(
    dataset_path: Path,
    out_path: Path,
    report_path: Path,
    *,
    config: OversamplingConfig,
) -> OversamplingReport:
    depth = RANK_INDEX[config.rank] + 1
    train = pl.scan_parquet(dataset_path).filter(pl.col("split") == Split.TRAIN.value)
    labels = train.select("seq_hash", "kingdom", *RANK_COLUMNS[:depth]).collect()

    members, kingdoms = _classes(labels, depth)
    sizes = {key: len(hashes) for key, hashes in members.items()}
    real_by_kingdom = _count_by_kingdom(labels.get_column("kingdom"))

    uncapped = plan_copies(
        sizes,
        floor=config.floor,
        max_copies_per_sequence=config.max_copies_per_sequence,
    )
    plan = cap_by_group(
        uncapped,
        kingdoms,
        real_by_kingdom,
        max_ratio=config.max_synthetic_ratio,
    )

    copies: dict[bytes, int] = {}

    for key, total in plan.items():
        copies.update(distribute(members[key], total))

    frame, collisions = _synthesize_rows(dataset_path, copies, config)
    _write(frame, out_path)

    report = OversamplingReport(
        config=config,
        real=labels.height,
        synthetic=frame.height,
        classes=len(sizes),
        below_floor_before=sum(size < config.floor for size in sizes.values()),
        below_floor_after=sum(
            size + plan.get(key, 0) < config.floor for key, size in sizes.items()
        ),
        collisions=collisions,
        real_by_kingdom=real_by_kingdom,
        synthetic_by_kingdom=_count_by_kingdom(frame.get_column("kingdom")),
    )

    write_text_atomic(report_path, f"{report.to_json()}\n")
    _LOGGER.info("Wrote %d synthetic sequences for %d real ones", report.synthetic, report.real)

    return report


def training_frame(dataset_path: Path, synthetic_path: Path) -> pl.LazyFrame:
    real = (
        pl.scan_parquet(dataset_path)
        .filter(pl.col("split") == Split.TRAIN.value)
        .with_columns(
            pl.lit(None, dtype=pl.Binary).alias("parent_seq_hash"),
            pl.lit(value=False).alias("synthetic"),
        )
    )

    return pl.concat([real, pl.scan_parquet(synthetic_path)], how="vertical")


def _classes(labels: pl.DataFrame, depth: int) -> tuple[dict[str, list[bytes]], dict[str, str]]:
    lineage_keys = balancing_keys(labels.select(RANK_COLUMNS[:depth]).iter_rows(), depth=depth)
    rows = zip(
        labels.get_column("seq_hash"), labels.get_column("kingdom"), lineage_keys, strict=True
    )

    members: defaultdict[str, list[bytes]] = defaultdict(list)
    kingdoms: dict[str, str] = {}

    for seq_hash, kingdom, lineage_key in rows:
        group = kingdom or UNKNOWN_KINGDOM
        key = f"{group}|{lineage_key}"

        members[key].append(seq_hash)
        kingdoms[key] = group

    return dict(members), kingdoms


def _synthesize_rows(
    dataset_path: Path,
    copies: Mapping[bytes, int],
    config: OversamplingConfig,
) -> tuple[pl.DataFrame, int]:
    dataset = pl.scan_parquet(dataset_path)
    schema = _synthetic_schema(dataset.collect_schema())

    parents = dataset.filter(pl.col("seq_hash").is_in(list(copies))).collect()
    taken = set(dataset.select("seq_hash").collect().get_column("seq_hash"))
    rows: list[dict[str, object]] = []
    collisions = 0

    for parent in parents.iter_rows(named=True):
        for copy in range(copies[parent["seq_hash"]]):
            sequence = synthesize(parent["sequence"], parent["seq_hash"], copy, config)
            seq_hash = sequence_hash(sequence)

            if seq_hash in taken:
                collisions += 1

                continue

            taken.add(seq_hash)
            rows.append(_synthetic_row(parent, sequence, seq_hash))

    frame = pl.DataFrame(rows, schema=schema)

    return frame.sort("seq_hash"), collisions


def _synthetic_row(parent: dict[str, object], sequence: str, seq_hash: bytes) -> dict[str, object]:
    return {
        **parent,
        "seq_hash": seq_hash,
        "sequence": sequence,
        "length": len(sequence),
        "n_ambiguous": count_ambiguous(sequence),
        "n_records": 1,
        "parent_seq_hash": parent["seq_hash"],
        "synthetic": True,
    }


def _synthetic_schema(dataset_schema: pl.Schema) -> pl.Schema:
    extra = [("parent_seq_hash", pl.Binary()), ("synthetic", pl.Boolean())]

    return pl.Schema([*dataset_schema.items(), *extra])


def _write(frame: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(out_path)

    try:
        frame.write_parquet(partial, compression="zstd")

    except BaseException:
        partial.unlink(missing_ok=True)

        raise

    partial.replace(out_path)


def _count_by_kingdom(kingdoms: pl.Series) -> dict[str, int]:
    counts = Counter(kingdom or UNKNOWN_KINGDOM for kingdom in kingdoms)

    return dict(counts.most_common())


def _copy_seed(parent: bytes, copy: int, seed: int) -> int:
    digest = hashlib.blake2b(
        parent + copy.to_bytes(4, "little"),
        digest_size=_SEED_BYTES,
        key=seed.to_bytes(_SEED_BYTES, "little"),
        person=b"oversampling",
    )

    return int.from_bytes(digest.digest(), "little")
