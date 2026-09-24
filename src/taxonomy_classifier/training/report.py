from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING, Final

import torch

from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS
from taxonomy_classifier.training.trainer import Precision, predict_batches

if TYPE_CHECKING:
    from collections.abc import Sequence

    from taxonomy_classifier.model.classifier import TaxonomyClassifier
    from taxonomy_classifier.model.labels import LabelSpace
    from taxonomy_classifier.training.data import LabeledSet

_COLUMNS: Final = (
    ("accuracy", "accuracy"),
    ("macro_precision", "macro-p"),
    ("macro_recall", "macro-r"),
    ("macro_f1", "macro-f1"),
    ("weighted_precision", "weighted-p"),
    ("weighted_recall", "weighted-r"),
    ("weighted_f1", "weighted-f1"),
)


@dataclass(frozen=True, slots=True)
class LevelReport:
    level: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    support: int
    classes: int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Counts:
    hits: torch.Tensor
    predicted: torch.Tensor
    actual: torch.Tensor


def classification_report(
    model: TaxonomyClassifier,
    data: LabeledSet,
    label_space: LabelSpace,
    *,
    batch_size: int,
    device: torch.device,
    precision: Precision = Precision.FP32,
    max_length: int | None = None,
) -> list[LevelReport]:
    counts = [
        _Counts(
            hits=torch.zeros(size, dtype=torch.long, device=device),
            predicted=torch.zeros(size, dtype=torch.long, device=device),
            actual=torch.zeros(size, dtype=torch.long, device=device),
        )
        for size in label_space.sizes
    ]

    for logits, targets in predict_batches(
        model,
        data,
        batch_size=batch_size,
        device=device,
        precision=precision,
        max_length=max_length,
    ):
        for index, (level_logits, level_counts) in enumerate(zip(logits, counts, strict=True)):
            target = targets[:, index]
            known = target != IGNORE_INDEX

            actual = target[known]
            predicted = level_logits.argmax(dim=-1)[known]
            size = level_counts.actual.numel()

            level_counts.hits.add_(torch.bincount(actual[predicted == actual], minlength=size))
            level_counts.predicted.add_(torch.bincount(predicted, minlength=size))
            level_counts.actual.add_(torch.bincount(actual, minlength=size))

    return [
        _level_report(level, level_counts)
        for level, level_counts in zip(LEVELS, counts, strict=True)
    ]


def format_report(reports: Sequence[LevelReport]) -> str:
    header = (
        f"{'':>10}"
        + "".join(f"{title:>12}" for _, title in _COLUMNS)
        + f"{'support':>10}{'classes':>9}"
    )

    rows = [
        f"{report.level:>10}"
        + "".join(f"{getattr(report, name):>12.4f}" for name, _ in _COLUMNS)
        + f"{report.support:>10}{report.classes:>9}"
        for report in reports
    ]

    return "\n".join([header, "", *rows])


def _level_report(level: str, counts: _Counts) -> LevelReport:
    hits = counts.hits.double().cpu()
    predicted = counts.predicted.double().cpu()
    actual = counts.actual.double().cpu()

    support = int(actual.sum())
    present = actual > 0

    if support == 0:
        return LevelReport(
            level=level,
            accuracy=math.nan,
            macro_precision=math.nan,
            macro_recall=math.nan,
            macro_f1=math.nan,
            weighted_precision=math.nan,
            weighted_recall=math.nan,
            weighted_f1=math.nan,
            support=0,
            classes=0,
        )

    precision = hits / predicted.clamp(min=1)
    recall = hits / actual.clamp(min=1)
    total = precision + recall
    f1 = torch.where(total > 0, 2 * precision * recall / total.clamp(min=1e-12), 0.0)

    def macro(values: torch.Tensor) -> float:
        return float(values[present].mean())

    def weighted(values: torch.Tensor) -> float:
        return float((values * actual).sum() / support)

    return LevelReport(
        level=level,
        accuracy=float(hits.sum()) / support,
        macro_precision=macro(precision),
        macro_recall=macro(recall),
        macro_f1=macro(f1),
        weighted_precision=weighted(precision),
        weighted_recall=weighted(recall),
        weighted_f1=weighted(f1),
        support=support,
        classes=int(present.sum()),
    )
