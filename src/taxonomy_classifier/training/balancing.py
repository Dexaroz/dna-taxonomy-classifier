from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def balancing_keys(lineages: Iterable[Sequence[str | None]], *, depth: int) -> list[str]:
    if depth < 1:
        msg = f"depth must be positive, got {depth}"
        raise ValueError(msg)

    return [_deepest_key(names[:depth]) for names in lineages]


def _deepest_key(names: Sequence[str | None]) -> str:
    labeled = [index for index, name in enumerate(names) if name is not None]

    if not labeled:
        msg = "Lineage has no labeled rank"
        raise ValueError(msg)

    return ";".join(name or "" for name in names[: labeled[-1] + 1])
