from dataclasses import asdict, dataclass, replace
import json
import time
from typing import TYPE_CHECKING, Any, Final

import optuna

from taxonomy_classifier.data.files import write_text_atomic
from taxonomy_classifier.exceptions import DataError, TrainingDeadlineError
from taxonomy_classifier.model.classifier import build_classifier, parameter_counts
from taxonomy_classifier.training.report import classification_report
from taxonomy_classifier.training.search import SearchConfig, TrialSetup, suggest_setup
from taxonomy_classifier.training.setup import LABEL_SPACE_FILENAME
from taxonomy_classifier.training.trainer import (
    BEST_CHECKPOINT,
    Precision,
    SilentMonitor,
    load_weights,
    train_classifier,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from taxonomy_classifier.model.labels import LabelSpace
    from taxonomy_classifier.training.data import TrainingData
    from taxonomy_classifier.training.report import LevelReport
    from taxonomy_classifier.training.trainer import EpochRecord, TrainingMonitor

RUN_FILENAME: Final = "run.json"

REPORT_FILENAME: Final = "report.json"

_ROW_FIELDS: Final = frozenset({"trial", "state", "value", "epochs"})

_ARCHITECTURE_PREFIXES: Final = {"cnn_": "CNN", "transformer_": "Transformer"}


@dataclass(frozen=True, slots=True)
class FinalConfig:
    epochs: int = 10
    samples_per_epoch: int | None = None
    batch_size: int = 128
    precision: Precision = Precision.BF16
    time_budget_hours: float | None = None
    log_every: int = 500
    seed: int = 20260923

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.log_every) < 1:
            msg = "epochs, batch_size and log_every must be positive"
            raise ValueError(msg)

        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            msg = f"samples_per_epoch must be positive, got {self.samples_per_epoch}"
            raise ValueError(msg)

        if self.time_budget_hours is not None and self.time_budget_hours <= 0:
            msg = f"time_budget_hours must be positive, got {self.time_budget_hours}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FinalResult:
    history: list[EpochRecord]
    reports: list[LevelReport]


@dataclass(frozen=True, slots=True)
class SearchChoice:
    trial: int
    params: dict[str, Any]

    @property
    def architecture(self) -> str:
        for prefix, architecture in _ARCHITECTURE_PREFIXES.items():
            if any(name.startswith(prefix) for name in self.params):
                return architecture

        msg = f"Trial {self.trial} has no architecture parameters: {sorted(self.params)}"
        raise DataError(msg)


def load_search_choice(path: Path, *, trial: int | None = None) -> SearchChoice:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        if trial is not None and payload["trial"] != trial:
            msg = f"{path} holds trial {payload['trial']}, not trial {trial}"
            raise DataError(msg)

        return SearchChoice(trial=payload["trial"], params=dict(payload["params"]))

    if trial is None:
        msg = f"{path} lists several trials; choose one"
        raise DataError(msg)

    for row in payload:
        if row["trial"] == trial:
            params = {key: value for key, value in row.items() if key not in _ROW_FIELDS}

            return SearchChoice(trial=trial, params=params)

    msg = f"Trial {trial} is not in {path}"
    raise DataError(msg)


def final_setup(choice: SearchChoice, config: FinalConfig) -> TrialSetup:
    search_config = SearchConfig(
        batch_size=config.batch_size,
        precision=config.precision,
        seed=config.seed,
    )
    setup = suggest_setup(
        optuna.trial.FixedTrial(choice.params), choice.architecture, search_config
    )
    training = replace(
        setup.training,
        epochs=config.epochs,
        samples_per_epoch=config.samples_per_epoch,
        log_every=config.log_every,
    )

    return replace(setup, training=training)


class DeadlineMonitor:
    def __init__(self, inner: TrainingMonitor, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        self._inner.epoch_started(epoch, epochs, steps)

    def step_finished(self, step: int) -> None:
        self._inner.step_finished(step)

        if time.monotonic() >= self._deadline:
            msg = "The time budget ran out before training finished"
            raise TrainingDeadlineError(msg)

    def metrics_updated(
        self, step: int, loss: float, sequences_per_second: float, learning_rate: float
    ) -> None:
        self._inner.metrics_updated(step, loss, sequences_per_second, learning_rate)

    def evaluation_started(self, batches: int) -> None:
        self._inner.evaluation_started(batches)

    def evaluation_step(self) -> None:
        self._inner.evaluation_step()

    def epoch_finished(self, record: EpochRecord) -> None:
        self._inner.epoch_finished(record)


def train_final(
    choice: SearchChoice,
    data: TrainingData,
    label_space: LabelSpace,
    *,
    config: FinalConfig,
    output_dir: Path,
    device: torch.device,
    monitor: TrainingMonitor | None = None,
) -> FinalResult:
    setup = final_setup(choice, config)
    model = build_classifier(setup.encoder, label_space, setup.heads)

    description = _describe(choice, setup, parameter_counts(model))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / LABEL_SPACE_FILENAME, label_space.to_json())
    write_text_atomic(output_dir / RUN_FILENAME, json.dumps(description, indent=2, default=str))

    inner = monitor or SilentMonitor()
    budget = config.time_budget_hours

    watcher: TrainingMonitor = (
        inner if budget is None else DeadlineMonitor(inner, time.monotonic() + budget * 3600)
    )

    history = train_classifier(
        model,
        data,
        config=setup.training,
        checkpoint_dir=output_dir,
        device=device,
        metadata=description,
        monitor=watcher,
    )

    load_weights(model, output_dir / BEST_CHECKPOINT, device=device)

    reports = classification_report(
        model,
        data.validation,
        label_space,
        batch_size=setup.training.evaluation_batch_size,
        device=device,
        precision=setup.training.precision,
        max_length=setup.training.max_length,
    )
    write_text_atomic(
        output_dir / REPORT_FILENAME,
        json.dumps([report.to_dict() for report in reports], indent=2),
    )

    return FinalResult(history=history, reports=reports)


def _describe(
    choice: SearchChoice, setup: TrialSetup, parameters: dict[str, int]
) -> dict[str, Any]:
    description = {
        "architecture": choice.architecture,
        "search_trial": choice.trial,
        "params": choice.params,
        "encoder": asdict(setup.encoder),
        "heads": asdict(setup.heads),
        "training": asdict(setup.training),
        "parameters": parameters,
    }
    plain: dict[str, Any] = json.loads(json.dumps(description, default=str))

    return plain
