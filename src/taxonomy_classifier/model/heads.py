from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import torch
from torch import nn
import torch.nn.functional as F

from taxonomy_classifier.model.labels import IGNORE_INDEX

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class HeadConfig:
    conditioning_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.conditioning_dim < 1:
            msg = f"conditioning_dim must be positive, got {self.conditioning_dim}"
            raise ValueError(msg)

        if not 0.0 <= self.dropout < 1.0:
            msg = f"dropout must be in [0, 1), got {self.dropout}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Prediction:
    indices: torch.Tensor
    confidence: torch.Tensor


class HierarchicalHeads(nn.Module):
    def __init__(self, in_features: int, sizes: Sequence[int], config: HeadConfig) -> None:
        super().__init__()

        if not sizes or min(sizes) < 1:
            msg = f"Every level needs at least one class, got sizes {tuple(sizes)}"
            raise ValueError(msg)

        self.dropout = nn.Dropout(config.dropout)

        self.heads = nn.ModuleList(
            nn.Linear(in_features + (config.conditioning_dim if level else 0), size)
            for level, size in enumerate(sizes)
        )

        self.parents = nn.ModuleList(
            nn.Linear(size, config.conditioning_dim, bias=False) for size in sizes[:-1]
        )

    @override
    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        features = self.dropout(features)
        logits = [self.heads[0](features)]

        for head, parent in zip(self.heads[1:], self.parents, strict=True):
            context = parent(logits[-1].softmax(dim=-1))

            logits.append(head(torch.cat([features, context], dim=-1)))

        return logits


def hierarchical_loss(logits: Sequence[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    level_losses = []
    labeled_levels = []

    for level, level_logits in enumerate(logits):
        target = targets[:, level]
        labeled = (target != IGNORE_INDEX).sum()

        total = F.cross_entropy(level_logits, target, ignore_index=IGNORE_INDEX, reduction="sum")

        level_losses.append(total / labeled.clamp(min=1))
        labeled_levels.append((labeled > 0).float())

    present = torch.stack(labeled_levels)

    return (torch.stack(level_losses) * present).sum() / present.sum().clamp(min=1)


def predict(logits: Sequence[torch.Tensor], *, threshold: float) -> Prediction:
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be in [0, 1], got {threshold}"
        raise ValueError(msg)

    best = [level_logits.softmax(dim=-1).max(dim=-1) for level_logits in logits]

    confidence = torch.stack([values for values, _ in best], dim=-1)
    indices = torch.stack([positions for _, positions in best], dim=-1)

    accepted = (confidence >= threshold).long().cumprod(dim=-1).bool()

    return Prediction(indices=indices.masked_fill(~accepted, -1), confidence=confidence)
