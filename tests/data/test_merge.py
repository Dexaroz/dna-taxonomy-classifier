from typing import TYPE_CHECKING

import polars as pl
import pytest

from taxonomy_classifier.data.columns import RANK_COLUMNS
from taxonomy_classifier.data.merge import merge_staged, merged_frame
from taxonomy_classifier.data.staging import STAGING_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

ECOLI = (
    "Bacteria",
    "Pseudomonadota",
    "Gamma",
    "Enterobacterales",
    "Entero",
    "Escherichia",
    "E coli",
)

PSEUDO = ("Bacteria", "Pseudomonadota", "Gamma", "Pseudomonadales", "Pseudo", "Pseudomonas", None)

SCHEMA = pl.DataFrame(STAGING_SCHEMA.empty_table()).schema


def _row(
    seq_hash: bytes,
    source: str,
    accession: str,
    names: tuple[str | None, ...],
    kingdom: str | None = "Bacteria",
) -> dict[str, object]:
    return {
        "source": source,
        "accession": accession,
        "seq_hash": seq_hash,
        "sequence": "ACGT",
        "length": 4,
        "n_ambiguous": 0,
        "source_lineage": [name for name in names if name],
        **dict(zip(RANK_COLUMNS, names, strict=True)),
        "kingdom": kingdom,
    }


def _merge(*rows: dict[str, object]) -> list[dict[str, object]]:
    staged = pl.LazyFrame(list(rows), schema=SCHEMA)

    return merged_frame(staged).collect().to_dicts()


def _names(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[column] for column in RANK_COLUMNS)


def test_identical_sequences_are_merged() -> None:
    rows = _merge(
        _row(b"h1", "gtdb", "B", ECOLI),
        _row(b"h1", "gtdb", "A", ECOLI),
        _row(b"h2", "pr2", "C", PSEUDO),
    )

    assert len(rows) == 2

    ecoli = rows[0]

    assert _names(ecoli) == ECOLI
    assert ecoli["n_records"] == 2
    assert ecoli["sources"] == ["gtdb"]
    assert ecoli["accessions"] == ["gtdb:A", "gtdb:B"]
    assert ecoli["label_conflict"] is False


def test_missing_ranks_do_not_count_as_conflicts() -> None:
    shallow = (*ECOLI[:5], None, None)

    [row] = _merge(_row(b"h1", "silva", "S", shallow), _row(b"h1", "gtdb", "G", ECOLI))

    assert _names(row) == ECOLI
    assert row["sources"] == ["gtdb", "silva"]
    assert row["label_conflict"] is False


def test_conflicting_labels_resolve_to_their_common_ancestor() -> None:
    [row] = _merge(_row(b"h1", "gtdb", "G", ECOLI), _row(b"h1", "silva", "S", PSEUDO))

    assert _names(row) == ("Bacteria", "Pseudomonadota", "Gamma", None, None, None, None)
    assert row["label_conflict"] is True


def test_a_conflict_invalidates_every_rank_below_it() -> None:
    other_phylum = ("Bacteria", "Bacillota", None, None, None, None, None)

    [row] = _merge(_row(b"h1", "gtdb", "G", ECOLI), _row(b"h1", "silva", "S", other_phylum))

    assert _names(row) == ("Bacteria", None, None, None, None, None, None)


def test_sequences_with_conflicting_domains_are_dropped() -> None:
    archaea = ("Archaea", None, None, None, None, None, None)

    assert _merge(_row(b"h1", "gtdb", "G", ECOLI), _row(b"h1", "silva", "S", archaea)) == []


def test_merge_staged_writes_the_merged_table_and_reports(tmp_path: Path) -> None:
    staged = tmp_path / "staged.parquet"
    archaea = ("Archaea", None, None, None, None, None, None)
    rows = [
        _row(b"h1", "gtdb", "A", ECOLI),
        _row(b"h1", "silva", "B", PSEUDO),
        _row(b"h2", "gtdb", "C", ECOLI),
        _row(b"h3", "gtdb", "D", ECOLI),
        _row(b"h3", "silva", "E", archaea),
    ]
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(staged)

    report = merge_staged([staged], tmp_path / "merged.parquet")

    assert report.to_dict() == {
        "staged": 5,
        "unique": 3,
        "kept": 2,
        "merged": 1,
        "conflicts": 1,
        "domain_conflicts": 1,
    }
    assert pl.read_parquet(tmp_path / "merged.parquet").height == 2
    assert not (tmp_path / "merged.parquet.part").exists()


def test_merge_staged_cleans_up_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = tmp_path / "staged.parquet"
    pl.DataFrame([_row(b"h1", "gtdb", "A", ECOLI)], schema=SCHEMA).write_parquet(staged)

    def explode(*_: object, **__: object) -> None:
        (tmp_path / "merged.parquet.part").write_bytes(b"partial")
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", explode)

    with pytest.raises(OSError, match="disk full"):
        merge_staged([staged], tmp_path / "merged.parquet")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["staged.parquet"]


def test_kingdom_is_placed_after_the_domain() -> None:
    [row] = _merge(_row(b"h1", "gtdb", "G", ECOLI))

    assert list(row)[3:6] == ["n_ambiguous", "domain", "kingdom"]
    assert row["kingdom"] == "Bacteria"


def test_missing_kingdoms_do_not_count_as_conflicts() -> None:
    [row] = _merge(
        _row(b"h1", "gtdb", "G", ECOLI),
        _row(b"h1", "silva", "S", ECOLI, kingdom=None),
    )

    assert row["kingdom"] == "Bacteria"
    assert row["label_conflict"] is False


def test_conflicting_kingdoms_are_cleared_and_flagged() -> None:
    fungus = ("Eukaryota", "Opisthokonta", None, None, None, None, None)

    [row] = _merge(
        _row(b"h1", "pr2", "P", fungus, kingdom="Fungi"),
        _row(b"h1", "silva", "S", fungus, kingdom="Animalia"),
    )

    assert row["phylum"] == "Opisthokonta"
    assert row["kingdom"] is None
    assert row["label_conflict"] is True
