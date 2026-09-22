from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import batched
import json
import logging
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from taxonomy_classifier.data.fasta import read_fasta
from taxonomy_classifier.data.files import partial_path, write_text_atomic
from taxonomy_classifier.data.filters import (
    ExclusionReason,
    FilterConfig,
    build_filters,
    first_exclusion,
)
from taxonomy_classifier.data.silva import to_sequence_record
from taxonomy_classifier.data.taxonomy import CANONICAL_RANKS, load_rank_map
from taxonomy_classifier.exceptions import (
    DuplicateRankError,
    InvalidSequenceError,
    MalformedHeaderError,
    TaxonomyError,
    UnknownTaxonPathError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    from taxonomy_classifier.data.fasta import FastaRecord
    from taxonomy_classifier.data.filters import Filter
    from taxonomy_classifier.data.records import SequenceRecord
    from taxonomy_classifier.data.taxonomy import RankMap

_LOGGER: Final = logging.getLogger(__name__)

DATASET_FILENAME: Final = "dataset.parquet"

REPORT_FILENAME: Final = "build_report.json"

_FIELDS: Final[list[pa.Field[pa.DataType]]] = [
    pa.field("accession", pa.string(), nullable=False),
    pa.field("start", pa.int64(), nullable=False),
    pa.field("end", pa.int64(), nullable=False),
    pa.field("organism", pa.string(), nullable=False),
    pa.field("sequence", pa.string(), nullable=False),
    pa.field("length", pa.int32(), nullable=False),
    pa.field("n_ambiguous", pa.int32(), nullable=False),
    pa.field("lineage", pa.list_(pa.string()), nullable=False),
    *(pa.field(rank.value, pa.string()) for rank in CANONICAL_RANKS),
]

PARQUET_SCHEMA: Final = pa.schema(_FIELDS)


@dataclass(frozen=True, slots=True)
class BuildConfig:
    filters: FilterConfig = field(default_factory=FilterConfig)
    batch_size: int = 50_000

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            msg = f"batch_size must be positive, got {self.batch_size}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BuildReport:
    release: str
    config: BuildConfig
    total: int
    kept: int
    excluded: Mapping[ExclusionReason, int]

    def to_json(self) -> str:
        payload = {
            "release": self.release,
            "config": asdict(self.config),
            "total": self.total,
            "kept": self.kept,
            "excluded": {reason.value: self.excluded.get(reason, 0) for reason in ExclusionReason},
        }

        return json.dumps(payload, indent=2)


def build_dataset(
    fasta_path: Path,
    taxonomy_path: Path,
    out_dir: Path,
    *,
    release: str,
    config: BuildConfig,
) -> BuildReport:
    rank_map = load_rank_map(taxonomy_path)
    filters = build_filters(config.filters)
    excluded: Counter[ExclusionReason] = Counter()

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / DATASET_FILENAME

    records = _select(read_fasta(fasta_path), rank_map, filters, excluded)
    kept = _write_parquet(records, target, batch_size=config.batch_size)

    report = BuildReport(
        release=release,
        config=config,
        total=kept + excluded.total(),
        kept=kept,
        excluded=dict(excluded),
    )

    write_text_atomic(out_dir / REPORT_FILENAME, f"{report.to_json()}\n")
    _LOGGER.info("Kept %d of %d sequences in %s", report.kept, report.total, target)

    return report


def _select(
    fasta_records: Iterable[FastaRecord],
    rank_map: RankMap,
    filters: Sequence[Filter],
    excluded: Counter[ExclusionReason],
) -> Iterator[SequenceRecord]:
    for fasta_record in fasta_records:
        outcome = _to_record(fasta_record, rank_map)

        if isinstance(outcome, ExclusionReason):
            excluded[outcome] += 1

            continue

        reason = first_exclusion(outcome, filters)

        if reason is not None:
            excluded[reason] += 1

            continue

        yield outcome


def _to_record(fasta_record: FastaRecord, rank_map: RankMap) -> SequenceRecord | ExclusionReason:
    try:
        return to_sequence_record(fasta_record, rank_map)

    except MalformedHeaderError:
        return ExclusionReason.MALFORMED_HEADER

    except InvalidSequenceError:
        return ExclusionReason.INVALID_SEQUENCE

    except UnknownTaxonPathError:
        return ExclusionReason.UNKNOWN_TAXON_PATH

    except DuplicateRankError:
        return ExclusionReason.DUPLICATE_RANK

    except TaxonomyError:
        return ExclusionReason.INVALID_LINEAGE


def _write_parquet(records: Iterable[SequenceRecord], target: Path, *, batch_size: int) -> int:
    partial = partial_path(target)
    written = 0

    try:
        with pq.ParquetWriter(partial, PARQUET_SCHEMA, compression="zstd") as writer:
            for batch in batched(records, batch_size, strict=False):
                writer.write_batch(_to_record_batch(batch))
                written += len(batch)

                _LOGGER.info("Wrote %d sequences", written)

    except BaseException:
        partial.unlink(missing_ok=True)

        raise

    partial.replace(target)

    return written


def _to_record_batch(records: Sequence[SequenceRecord]) -> pa.RecordBatch:
    columns: dict[str, list[object]] = {
        "accession": [record.accession for record in records],
        "start": [record.start for record in records],
        "end": [record.end for record in records],
        "organism": [record.organism for record in records],
        "sequence": [record.sequence for record in records],
        "length": [record.length for record in records],
        "n_ambiguous": [record.n_ambiguous for record in records],
        "lineage": [list(record.lineage.path) for record in records],
    }

    for rank in CANONICAL_RANKS:
        columns[rank.value] = [record.lineage.get(rank) for record in records]

    return pa.RecordBatch.from_pydict(columns, schema=PARQUET_SCHEMA)
