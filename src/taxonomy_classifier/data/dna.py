from typing import Final

from taxonomy_classifier.exceptions import InvalidSequenceError

CANONICAL_BASES: Final = frozenset("ACGT")

AMBIGUOUS_BASES: Final = frozenset("RYSWKMBDHVN")

IUPAC_DNA: Final = CANONICAL_BASES | AMBIGUOUS_BASES

_RNA_TO_DNA: Final = str.maketrans("U", "T")


def normalize_sequence(raw: str) -> str:
    sequence = "".join(raw.split()).upper().translate(_RNA_TO_DNA)

    if not sequence:
        msg = "Sequence is empty"
        raise InvalidSequenceError(msg)

    invalid = set(sequence) - IUPAC_DNA

    if invalid:
        msg = f"Sequence contains non-IUPAC characters: {sorted(invalid)}"
        raise InvalidSequenceError(msg)

    return sequence


def count_ambiguous(sequence: str) -> int:
    canonical = sum(sequence.count(base) for base in CANONICAL_BASES)

    return len(sequence) - canonical
