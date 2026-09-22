from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import random

_BASES: Final = "ACGT"

_ALTERNATIVES: Final = {base: _BASES.replace(base, "") for base in _BASES}

_COMPLEMENT: Final = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    substitution_rate: float = 0.005
    insertion_rate: float = 0.001
    deletion_rate: float = 0.001
    crop_probability: float = 0.5
    crop_min_length: int = 150
    crop_max_length: int = 500
    reverse_complement_probability: float = 0.0

    def __post_init__(self) -> None:
        probabilities = {
            "substitution_rate": self.substitution_rate,
            "insertion_rate": self.insertion_rate,
            "deletion_rate": self.deletion_rate,
            "crop_probability": self.crop_probability,
            "reverse_complement_probability": self.reverse_complement_probability,
        }

        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)

        if not 1 <= self.crop_min_length <= self.crop_max_length:
            msg = (
                "crop lengths must satisfy 1 <= crop_min_length <= crop_max_length, "
                f"got {self.crop_min_length} and {self.crop_max_length}"
            )
            raise ValueError(msg)


class SequenceAugmenter:
    def __init__(self, config: AugmentationConfig, rng: random.Random) -> None:
        self._config = config
        self._rng = rng

    def __call__(self, sequence: str) -> str:
        augmented = sequence

        if self._rng.random() < self._config.crop_probability:
            augmented = self.crop(augmented)

        augmented = self.mutate(augmented)

        if self._rng.random() < self._config.reverse_complement_probability:
            augmented = reverse_complement(augmented)

        return augmented

    def crop(self, sequence: str) -> str:
        length = self._rng.randint(self._config.crop_min_length, self._config.crop_max_length)

        if len(sequence) <= length:
            return sequence

        start = self._rng.randint(0, len(sequence) - length)

        return sequence[start : start + length]

    def mutate(self, sequence: str) -> str:
        bases = list(sequence)

        for position in self._positions(len(bases), self._config.substitution_rate):
            bases[position] = self._rng.choice(_ALTERNATIVES.get(bases[position], _BASES))

        deletions = set(self._positions(len(bases), self._config.deletion_rate))
        insertions = set(self._positions(len(bases), self._config.insertion_rate))

        for position in sorted(deletions | insertions, reverse=True):
            if position in deletions:
                del bases[position]

            if position in insertions:
                bases.insert(position, self._rng.choice(_BASES))

        return "".join(bases)

    def _positions(self, length: int, rate: float) -> list[int]:
        if length == 0 or rate == 0.0:
            return []

        count = self._rng.binomialvariate(length, rate)

        return self._rng.sample(range(length), count)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]
