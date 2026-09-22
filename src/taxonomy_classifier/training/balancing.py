from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def balancing_keys(lineages: Iterable[Sequence[str | None]], *, depth: int) -> list[str]:
    if depth < 1:
        msg = f"depth must be positive, got {depth}"
        raise ValueError(msg)

    return [_deepest_key(names[:depth]) for names in lineages]


def balanced_weights(keys: Sequence[str], *, power: float = 0.5) -> list[float]:
    if not 0.0 <= power <= 1.0:
        msg = f"power must be in [0, 1], got {power}"
        raise ValueError(msg)

    if not keys:
        return []

    counts = Counter(keys)
    raw = [counts[key] ** -power for key in keys]
    scale = len(raw) / sum(raw)

    return [weight * scale for weight in raw]


def _deepest_key(names: Sequence[str | None]) -> str:
    labeled = [index for index, name in enumerate(names) if name is not None]

    if not labeled:
        msg = "Lineage has no labeled rank"
        raise ValueError(msg)

    return ";".join(name or "" for name in names[: labeled[-1] + 1])
