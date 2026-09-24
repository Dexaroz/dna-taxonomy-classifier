import json
import math
import time
from typing import TYPE_CHECKING, Any, Final, TextIO

from taxonomy_classifier.data.files import write_text_atomic
from taxonomy_classifier.model.labels import LEVELS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from taxonomy_classifier.training.trainer import EpochRecord

_SECONDS_PER_MINUTE: Final = 60

_SECONDS_PER_HOUR: Final = 3600

_COLUMN_WIDTH: Final = 9


def format_duration(seconds: float) -> str:
    whole = max(0, round(seconds))

    if whole < _SECONDS_PER_MINUTE:
        return f"{whole}s"

    if whole < _SECONDS_PER_HOUR:
        return f"{whole // _SECONDS_PER_MINUTE}m {whole % _SECONDS_PER_MINUTE:02d}s"

    minutes = whole % _SECONDS_PER_HOUR // _SECONDS_PER_MINUTE

    return f"{whole // _SECONDS_PER_HOUR}h {minutes:02d}m"


def progress_bar(done: int, total: int, width: int) -> str:
    filled = min(width, width * done // max(1, total))

    if filled == width:
        return "=" * width

    return "=" * filled + ">" + "." * (width - filled - 1)


def accuracy_table(rows: Mapping[str, Mapping[str, float]]) -> str:
    header = " " * _COLUMN_WIDTH + "".join(f"{level:>{_COLUMN_WIDTH}}" for level in LEVELS)
    lines = [
        f"{name:<{_COLUMN_WIDTH}}"
        + "".join(f"{accuracy.get(level, math.nan):>{_COLUMN_WIDTH}.4f}" for level in LEVELS)
        for name, accuracy in rows.items()
        if accuracy
    ]

    return "\n".join([header, *lines])


class ProgressBarMonitor:
    def __init__(
        self,
        stream: TextIO,
        *,
        refresh_seconds: float = 0.5,
        steps_path: Path | None = None,
        width: int = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream
        self._refresh_seconds = refresh_seconds
        self._steps_path = steps_path
        self._width = width
        self._clock = clock

        self._steps = 1
        self._step = 0
        self._epoch = 0
        self._started = 0.0
        self._training_seconds = 0.0
        self._last_draw = -math.inf
        self._loss = math.nan
        self._line_length = 0
        self._evaluation_batches = 0
        self._evaluated = 0
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        self._epoch = epoch
        self._steps = steps
        self._step = 0
        self._loss = math.nan
        self._evaluation_batches = 0
        self._evaluated = 0
        self._line_length = 0
        self._started = self._clock()

        self._stream.write(f"Epoch {epoch}/{epochs}\n")
        self._draw(self._training_suffix(), force=True)

    def step_finished(self, step: int) -> None:
        self._step = step

        self._draw(self._training_suffix(), force=step == self._steps)

    def metrics_updated(
        self, step: int, loss: float, sequences_per_second: float, learning_rate: float
    ) -> None:
        self._loss = loss
        self._records.append(
            {
                "epoch": self._epoch,
                "step": step,
                "loss": loss,
                "sequences_per_second": sequences_per_second,
                "learning_rate": learning_rate,
            }
        )

    def evaluation_started(self, batches: int) -> None:
        self._training_seconds = self._clock() - self._started
        self._evaluation_batches = batches
        self._evaluated = 0

        self._draw(self._evaluation_suffix(), force=True)

    def evaluation_step(self) -> None:
        self._evaluated += 1

        self._draw(self._evaluation_suffix(), force=False)

    def epoch_finished(self, record: EpochRecord) -> None:
        self._step = self._steps
        milliseconds = 1000 * self._training_seconds / max(1, self._steps)

        self._draw(
            f" - {format_duration(record.seconds)} {milliseconds:.0f}ms/step"
            f" - loss: {record.train_loss:.4f} - val_loss: {record.validation.loss:.4f}"
            f" - lr: {record.learning_rate:.2e}",
            force=True,
        )

        table = accuracy_table({"train": record.train_accuracy, "val": record.validation.accuracy})
        self._stream.write(f"\n{table}\n\n")
        self._stream.flush()

        if self._steps_path is not None:
            write_text_atomic(self._steps_path, json.dumps(self._records, indent=2))

    def _training_suffix(self) -> str:
        elapsed = self._clock() - self._started
        remaining = elapsed / self._step * (self._steps - self._step) if self._step else math.nan
        eta = format_duration(remaining) if math.isfinite(remaining) else "?"
        loss = "" if math.isnan(self._loss) else f" - loss: {self._loss:.4f}"

        return f" - ETA: {eta}{loss}"

    def _evaluation_suffix(self) -> str:
        loss = "" if math.isnan(self._loss) else f" - loss: {self._loss:.4f}"

        return (
            f" - {format_duration(self._training_seconds)}{loss}"
            f" - validating {self._evaluated}/{self._evaluation_batches}"
        )

    def _draw(self, suffix: str, *, force: bool) -> None:
        now = self._clock()

        if not force and now - self._last_draw < self._refresh_seconds:
            return

        digits = len(str(self._steps))
        bar = progress_bar(self._step, self._steps, self._width)
        line = f"{self._step:>{digits}}/{self._steps} [{bar}]{suffix}"
        padding = " " * max(0, self._line_length - len(line))

        self._stream.write(f"\r{line}{padding}")
        self._stream.flush()

        self._line_length = len(line)
        self._last_draw = now
