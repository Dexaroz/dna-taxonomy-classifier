import gzip
from typing import TYPE_CHECKING

from taxonomy_classifier.data.files import open_text, partial_path, write_text_atomic

if TYPE_CHECKING:
    from pathlib import Path


def test_open_text_reads_plain_files(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("línea\n", encoding="utf-8")

    with open_text(path) as handle:
        assert handle.read() == "línea\n"


def test_open_text_reads_gzip_files(tmp_path: Path) -> None:
    path = tmp_path / "compressed.txt.gz"

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("línea\n")

    with open_text(path) as handle:
        assert handle.read() == "línea\n"


def test_partial_path_appends_suffix(tmp_path: Path) -> None:
    assert partial_path(tmp_path / "report.json") == tmp_path / "report.json.part"


def test_write_text_atomic_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    write_text_atomic(path, "{}\n")

    assert path.read_text(encoding="utf-8") == "{}\n"
    assert not partial_path(path).exists()
