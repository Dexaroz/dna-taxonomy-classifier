from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import batched
import logging
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from taxonomy_classifier.data.columns import RANK_COLUMNS
from taxonomy_classifier.data.exclusions import ExclusionReason
from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.data.filters import build_filters, first_exclusion
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, Rank

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    from taxonomy_classifier.data.filters import Filter, FilterConfig
    from taxonomy_classifier.data.harmonize import BackboneIndex
    from taxonomy_classifier.data.records import SequenceRecord
    from taxonomy_classifier.data.sources.base import DataSource, Outcome

_LOGGER: Final = logging.getLogger(__name__)

SEQ_HASH_BYTES: Final = 16

_FIELDS: Final[list[pa.Field[pa.DataType]]] = [
    pa.field("source", pa.string(), nullable=False),
    pa.field("accession", pa.string(), nullable=False),
    pa.field("seq_hash", pa.binary(), nullable=False),
    pa.field("sequence", pa.string(), nullable=False),
    pa.field("length", pa.int32(), nullable=False),
    pa.field("n_ambiguous", pa.int32(), nullable=False),
    pa.field("source_lineage", pa.list_(pa.string()), nullable=False),
    *(pa.field(column, pa.string()) for column in RANK_COLUMNS),
]

STAGING_SCHEMA: Final = pa.schema(_FIELDS)


@dataclass(frozen=True, slots=True)
class SourceReport:
    source: str
    read: int
    kept: int
    excluded: Mapping[ExclusionReason, int]
    deepest_rank: Mapping[Rank, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "read": self.read,
            "kept": self.kept,
            "excluded": {reason.value: self.excluded.get(reason, 0) for reason in ExclusionReason},
            "deepest_rank": {rank.value: self.deepest_rank.get(rank, 0) for rank in Rank},
        }


def sequence_hash(sequence: str) -> bytes:
    return hashlib.blake2b(sequence.encode("ascii"), digest_size=SEQ_HASH_BYTES).digest()


def stage_source(
    source: DataSource,
    raw_dir: Path,
    out_path: Path,
    *,
    filters: FilterConfig,
    backbone: BackboneIndex,
    batch_size: int,
) -> SourceReport:
    excluded: Counter[ExclusionReason] = Counter()
    depths: Counter[Rank] = Counter()

    outcomes = source.read(raw_dir, backbone)
    kept = _select(outcomes, build_filters(filters), excluded)
    tracked = _track(kept, depths, backbone if source.is_backbone else None)

    written = _write(tracked, out_path, batch_size=batch_size)

    _LOGGER.info("Staged %d sequences from %s", written, source.slug)

    return SourceReport(
        source=source.slug,
        read=written + excluded.total(),
        kept=written,
        excluded=dict(excluded),
        deepest_rank=dict(depths),
    )


def _select(
    outcomes: Iterable[Outcome],
    filters: Sequence[Filter],
    excluded: Counter[ExclusionReason],
) -> Iterator[SequenceRecord]:
    for outcome in outcomes:
        if isinstance(outcome, ExclusionReason):
            excluded[outcome] += 1

            continue

        reason = first_exclusion(outcome, filters)

        if reason is not None:
            excluded[reason] += 1

            continue

        yield outcome


def _track(
    records: Iterable[SequenceRecord],
    depths: Counter[Rank],
    backbone: BackboneIndex | None,
) -> Iterator[SequenceRecord]:
    for record in records:
        depths[record.lineage.deepest_rank] += 1

        if backbone is not None:
            backbone.add(record)

        yield record


def _write(records: Iterable[SequenceRecord], out_path: Path, *, batch_size: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(out_path)
    written = 0

    try:
        with pq.ParquetWriter(partial, STAGING_SCHEMA, compression="zstd") as writer:
            for batch in batched(records, batch_size, strict=False):
                writer.write_batch(_to_record_batch(batch))
                written += len(batch)

    except BaseException:
        partial.unlink(missing_ok=True)

        raise

    partial.replace(out_path)

    return written


def _to_record_batch(records: Sequence[SequenceRecord]) -> pa.RecordBatch:
    columns: dict[str, list[object]] = {
        "source": [record.source for record in records],
        "accession": [record.accession for record in records],
        "seq_hash": [sequence_hash(record.sequence) for record in records],
        "sequence": [record.sequence for record in records],
        "length": [record.length for record in records],
        "n_ambiguous": [record.n_ambiguous for record in records],
        "source_lineage": [[taxon.name for taxon in record.source_lineage] for record in records],
    }

    for rank, column in zip(CANONICAL_RANKS, RANK_COLUMNS, strict=True):
        columns[column] = [record.lineage.get(rank) for record in records]

    return pa.RecordBatch.from_pydict(columns, schema=STAGING_SCHEMA)
