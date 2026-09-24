import io
import json
from typing import TYPE_CHECKING

import pytest

from taxonomy_classifier.model.labels import LEVELS
from taxonomy_classifier.training.progress import (
    ProgressBarMonitor,
    accuracy_table,
    format_duration,
    progress_bar,
)
from taxonomy_classifier.training.trainer import EpochRecord, Evaluation

if TYPE_CHECKING:
    from pathlib import Path


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _record() -> EpochRecord:
    return EpochRecord(
        epoch=1,
        train_loss=1.25,
        validation=Evaluation(loss=0.75, accuracy=dict.fromkeys(LEVELS, 0.5)),
        learning_rate=1e-3,
        seconds=130.0,
        train_accuracy=dict.fromkeys(LEVELS, 0.625),
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(-3.0, "0s"), (5.4, "5s"), (125.0, "2m 05s"), (3725.0, "1h 02m")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


@pytest.mark.parametrize(
    ("done", "expected"),
    [(0, ">........."), (5, "=====>...."), (10, "==========")],
)
def test_progress_bar(done: int, expected: str) -> None:
    assert progress_bar(done, 10, 10) == expected


def test_accuracy_table_skips_empty_rows() -> None:
    lines = accuracy_table({"train": {}, "val": dict.fromkeys(LEVELS, 0.5)}).splitlines()

    assert lines[0].split() == list(LEVELS)
    assert lines[1].split() == ["val", *["0.5000"] * len(LEVELS)]
    assert len(lines) == 2


def test_monitor_draws_a_keras_style_epoch(tmp_path: Path) -> None:
    stream = io.StringIO()
    clock = FakeClock()
    steps_path = tmp_path / "steps.json"
    monitor = ProgressBarMonitor(
        stream, refresh_seconds=10.0, steps_path=steps_path, width=10, clock=clock
    )

    monitor.epoch_started(1, 2, 4)
    clock.now += 1.0
    monitor.step_finished(1)
    monitor.metrics_updated(2, 1.5, 800.0, 1e-3)
    clock.now += 20.0
    monitor.step_finished(2)
    monitor.step_finished(4)
    monitor.evaluation_started(2)
    monitor.evaluation_step()
    monitor.epoch_finished(_record())

    frames = stream.getvalue().split("\r")

    assert frames[0] == "Epoch 1/2\n"
    assert frames[1].rstrip() == "0/4 [>.........] - ETA: ?"
    assert frames[2].rstrip() == "2/4 [=====>....] - ETA: 21s - loss: 1.5000"
    assert frames[3].startswith("4/4 [==========] - ETA: 0s")
    assert frames[4].rstrip() == "4/4 [==========] - 21s - loss: 1.5000 - validating 0/2"
    assert frames[5].startswith(
        "4/4 [==========] - 2m 10s 5250ms/step - loss: 1.2500 - val_loss: 0.7500 - lr: 1.00e-03"
    )
    table = frames[5].splitlines()[1:4]

    assert table[0].split() == list(LEVELS)
    assert table[1].split() == ["train", *["0.6250"] * len(LEVELS)]
    assert table[2].split() == ["val", *["0.5000"] * len(LEVELS)]
    assert len(frames) == 6
    assert monitor.records == json.loads(steps_path.read_text(encoding="utf-8"))
    assert monitor.records[0]["loss"] == 1.5


def test_monitor_pads_shorter_lines_and_works_without_a_steps_file() -> None:
    stream = io.StringIO()
    monitor = ProgressBarMonitor(stream, refresh_seconds=0.0, width=4, clock=FakeClock())

    monitor.epoch_started(1, 1, 10)
    monitor.metrics_updated(1, 2.0, 10.0, 1e-3)
    monitor.step_finished(1)
    monitor.evaluation_started(1)
    monitor.epoch_finished(_record())

    frames = stream.getvalue().split("\r")

    assert len(frames[2]) >= len(frames[1])
    assert "val_loss: 0.7500" in frames[-1]
