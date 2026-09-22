from typing import TYPE_CHECKING, Final

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

PAD: Final = 0

AMBIGUOUS: Final = 5

VOCAB_SIZE: Final = 6

_TABLE: Final = bytes(
    {ord("A"): 1, ord("C"): 2, ord("G"): 3, ord("T"): 4}.get(code, AMBIGUOUS) for code in range(256)
)


def encode_bytes(data: bytes) -> bytes:
    return data.translate(_TABLE)


def encode_sequence(sequence: str) -> torch.Tensor:
    translated = bytearray(encode_bytes(sequence.encode("ascii")))

    return torch.frombuffer(translated, dtype=torch.uint8).long()


def collate(
    sequences: Sequence[str],
    *,
    max_length: int | None = None,
    min_length: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_sequence(sequence[:max_length]) for sequence in sequences]
    width = max([min_length, *(len(tokens) for tokens in encoded)])

    tokens = torch.full((len(encoded), width), PAD, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool)

    for row, sequence_tokens in enumerate(encoded):
        tokens[row, : len(sequence_tokens)] = sequence_tokens
        mask[row, : len(sequence_tokens)] = True

    return tokens, mask
