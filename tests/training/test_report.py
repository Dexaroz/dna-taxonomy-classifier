import math
from typing import TYPE_CHECKING, cast, override

import polars as pl
import pytest
import torch

from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS, LabelSpace
from taxonomy_classifier.training.data import LabeledSet, SequenceBank
from taxonomy_classifier.training.report import LevelReport, classification_report, format_report

if TYPE_CHECKING:
    from taxonomy_classifier.model.classifier import TaxonomyClassifier

CPU = torch.device("cpu")

SPACE = LabelSpace(classes=dict.fromkeys(LEVELS, ("a", "b", "c")), min_count=1)

TARGETS = [0, 0, 1, 2]

PREDICTED = [0, 1, 1, 1]


class FixedModel(torch.nn.Module):
    @override
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        del mask
        logits = torch.nn.functional.one_hot(torch.tensor(PREDICTED[: len(tokens)]), 3).float()

        return [logits] * len(LEVELS)


def _data() -> LabeledSet:
    series = pl.Series(["ACGT" * 5] * len(TARGETS))
    known = [[target] * (len(LEVELS) - 1) + [IGNORE_INDEX] for target in TARGETS]

    return LabeledSet(
        bank=SequenceBank.from_batches([series], series.str.len_bytes()),
        targets=torch.tensor(known),
    )


def _reports() -> list[LevelReport]:
    return classification_report(
        cast("TaxonomyClassifier", FixedModel()), _data(), SPACE, batch_size=8, device=CPU
    )


def test_report_matches_hand_computed_metrics() -> None:
    genus = _reports()[LEVELS.index("genus")]

    assert genus.accuracy == pytest.approx(0.5)
    assert genus.macro_precision == pytest.approx((1 + 1 / 3) / 3)
    assert genus.macro_recall == pytest.approx(0.5)
    assert genus.macro_f1 == pytest.approx((2 / 3 + 0.5) / 3)
    assert genus.weighted_precision == pytest.approx((2 + 1 / 3) / 4)
    assert genus.weighted_recall == pytest.approx(0.5)
    assert genus.weighted_f1 == pytest.approx((4 / 3 + 0.5) / 4)
    assert (genus.support, genus.classes) == (4, 3)


def test_levels_without_labels_report_no_metrics() -> None:
    species = _reports()[LEVELS.index("species")]

    assert math.isnan(species.accuracy)
    assert math.isnan(species.weighted_f1)
    assert (species.support, species.classes) == (0, 0)
    assert species.to_dict()["level"] == "species"


def test_format_report_lists_every_level() -> None:
    lines = format_report(_reports()).splitlines()

    assert "macro-f1" in lines[0]
    assert "support" in lines[0]
    assert [line.split()[0] for line in lines[2:]] == list(LEVELS)
    assert lines[2 + LEVELS.index("genus")].split()[1:2] == ["0.5000"]
