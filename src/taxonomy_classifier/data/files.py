from contextlib import contextmanager
import gzip
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import TextIO


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle

        return

    with path.open(encoding="utf-8") as handle:
        yield handle


def write_text_atomic(path: Path, content: str) -> None:
    partial = partial_path(path)

    partial.write_text(content, encoding="utf-8")

    partial.replace(path)


def partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.part")
