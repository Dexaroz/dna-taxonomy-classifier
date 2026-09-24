import json
import time
from typing import TYPE_CHECKING

import polars as pl
import pytest
import torch

from taxonomy_classifier.exceptions import DataError, TrainingDeadlineError
from taxonomy_classifier.model.encoders import CnnConfig, TransformerConfig
from taxonomy_classifier.model.labels import LEVELS, LabelSpace
from taxonomy_classifier.training.data import LabeledSet, SequenceBank, TrainingData
from taxonomy_classifier.training.final import (
    RUN_FILENAME,
    DeadlineMonitor,
    FinalConfig,
    SearchChoice,
    final_setup,
    load_search_choice,
    train_final,
)
from taxonomy_classifier.training.setup import LABEL_SPACE_FILENAME
from taxonomy_classifier.training.trainer import (
    BEST_CHECKPOINT,
    HISTORY_FILENAME,
    EpochRecord,
    Evaluation,
    Precision,
)

if TYPE_CHECKING:
    from pathlib import Path

CPU = torch.device("cpu")

SPACE = LabelSpace(classes=dict.fromkeys(LEVELS, ("a", "b")), min_count=1)

COMMON = {
    "dropout": 0.05,
    "learning_rate": 1e-3,
    "weight_decay": 0.05,
    "warmup_fraction": 0.05,
    "sampling_power": 0.25,
    "crop_probability": 0.2,
}

CNN_PARAMS = COMMON | {"cnn_width": "small", "cnn_blocks": "two", "cnn_kernel": 5}

TRANSFORMER_PARAMS = COMMON | {
    "transformer_tokenizer": "k6s3",
    "transformer_dim": 192,
    "transformer_layers": 4,
}

TINY_FINAL = FinalConfig(epochs=2, samples_per_epoch=8, batch_size=4, precision=Precision.FP32)


def _labeled(sequences: list[str], labels: list[int]) -> LabeledSet:
    series = pl.Series(sequences)
    targets = torch.tensor([[label] * len(LEVELS) for label in labels])

    return LabeledSet(
        bank=SequenceBank.from_batches([series], series.str.len_bytes()), targets=targets
    )


TRAIN = _labeled(["A" * 40, "C" * 40] * 4, [0, 1] * 4)

DATA = TrainingData(
    train=TRAIN,
    frequencies=torch.ones(len(TRAIN), dtype=torch.double),
    validation=_labeled(["A" * 30, "C" * 30], [0, 1]),
)


class RecordingMonitor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        self.calls.append(f"epoch_started {epoch}/{epochs} {steps}")

    def step_finished(self, step: int) -> None:
        self.calls.append(f"step_finished {step}")

    def metrics_updated(
        self, step: int, loss: float, sequences_per_second: float, learning_rate: float
    ) -> None:
        del loss, sequences_per_second, learning_rate
        self.calls.append(f"metrics_updated {step}")

    def evaluation_started(self, batches: int) -> None:
        self.calls.append(f"evaluation_started {batches}")

    def evaluation_step(self) -> None:
        self.calls.append("evaluation_step")

    def epoch_finished(self, record: EpochRecord) -> None:
        self.calls.append(f"epoch_finished {record.epoch}")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs, batch_size and log_every must be positive"),
        ({"batch_size": 0}, "epochs, batch_size and log_every must be positive"),
        ({"log_every": 0}, "epochs, batch_size and log_every must be positive"),
        ({"samples_per_epoch": 0}, "samples_per_epoch must be positive"),
        ({"time_budget_hours": 0.0}, "time_budget_hours must be positive"),
    ],
)
def test_final_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FinalConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("params", "architecture"),
    [(CNN_PARAMS, "CNN"), (TRANSFORMER_PARAMS, "Transformer")],
)
def test_choice_infers_its_architecture(params: dict[str, object], architecture: str) -> None:
    assert SearchChoice(trial=3, params=params).architecture == architecture


def test_choice_without_architecture_parameters_is_rejected() -> None:
    with pytest.raises(DataError, match="no architecture parameters"):
        _ = SearchChoice(trial=3, params=COMMON).architecture


def test_best_summary_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "cnn-best.json"
    path.write_text(json.dumps({"trial": 12, "value": 0.6, "params": CNN_PARAMS}), encoding="utf-8")

    assert load_search_choice(path) == SearchChoice(trial=12, params=CNN_PARAMS)
    assert load_search_choice(path, trial=12).trial == 12


def test_best_summary_rejects_another_trial(tmp_path: Path) -> None:
    path = tmp_path / "cnn-best.json"
    path.write_text(json.dumps({"trial": 12, "value": 0.6, "params": CNN_PARAMS}), encoding="utf-8")

    with pytest.raises(DataError, match="holds trial 12, not trial 20"):
        load_search_choice(path, trial=20)


def test_trial_rows_are_loaded_without_their_bookkeeping(tmp_path: Path) -> None:
    rows = [
        {"trial": 0, "state": "PRUNED", "value": 0.1, "epochs": 2} | TRANSFORMER_PARAMS,
        {"trial": 1, "state": "COMPLETE", "value": 0.3, "epochs": 6} | CNN_PARAMS,
    ]
    path = tmp_path / "cnn-trials.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    assert load_search_choice(path, trial=1) == SearchChoice(trial=1, params=CNN_PARAMS)


@pytest.mark.parametrize(
    ("trial", "message"),
    [(None, "lists several trials"), (7, "Trial 7 is not in")],
)
def test_trial_rows_need_an_existing_trial(tmp_path: Path, trial: int | None, message: str) -> None:
    path = tmp_path / "cnn-trials.json"
    path.write_text(json.dumps([{"trial": 1} | CNN_PARAMS]), encoding="utf-8")

    with pytest.raises(DataError, match=message):
        load_search_choice(path, trial=trial)


def test_final_setup_rebuilds_the_searched_configuration() -> None:
    config = FinalConfig(epochs=12, batch_size=64, precision=Precision.FP16, log_every=20, seed=5)

    setup = final_setup(SearchChoice(trial=1, params=CNN_PARAMS), config)

    assert isinstance(setup.encoder, CnnConfig)
    assert setup.encoder.kernel_size == 5
    assert setup.encoder.dilations == (1, 2)
    assert setup.heads.dropout == 0.05
    assert setup.training.epochs == 12
    assert setup.training.samples_per_epoch is None
    assert setup.training.batch_size == 64
    assert setup.training.learning_rate == 1e-3
    assert setup.training.crop.probability == 0.2
    assert setup.training.precision is Precision.FP16
    assert setup.training.log_every == 20
    assert setup.training.seed == 5


def test_final_setup_supports_transformers() -> None:
    setup = final_setup(SearchChoice(trial=1, params=TRANSFORMER_PARAMS), FinalConfig())

    assert isinstance(setup.encoder, TransformerConfig)
    assert (setup.encoder.token_kernel, setup.encoder.token_stride) == (6, 3)
    assert setup.encoder.model_dim == 192


def test_deadline_monitor_forwards_every_event() -> None:
    inner = RecordingMonitor()
    monitor = DeadlineMonitor(inner, time.monotonic() + 3600)
    record = EpochRecord(
        epoch=1,
        train_loss=1.0,
        validation=Evaluation(loss=1.0, accuracy=dict.fromkeys(LEVELS, 0.5)),
        learning_rate=1e-3,
        seconds=1.0,
    )

    monitor.epoch_started(1, 2, 3)
    monitor.step_finished(1)
    monitor.metrics_updated(1, 1.0, 10.0, 1e-3)
    monitor.evaluation_started(2)
    monitor.evaluation_step()
    monitor.epoch_finished(record)

    assert inner.calls == [
        "epoch_started 1/2 3",
        "step_finished 1",
        "metrics_updated 1",
        "evaluation_started 2",
        "evaluation_step",
        "epoch_finished 1",
    ]


def test_deadline_monitor_stops_training_once_the_deadline_passes() -> None:
    monitor = DeadlineMonitor(RecordingMonitor(), time.monotonic() - 1)

    with pytest.raises(TrainingDeadlineError, match="time budget ran out"):
        monitor.step_finished(1)


def test_train_final_writes_checkpoints_and_a_run_description(tmp_path: Path) -> None:
    monitor = RecordingMonitor()

    history = train_final(
        SearchChoice(trial=12, params=CNN_PARAMS),
        DATA,
        SPACE,
        config=TINY_FINAL,
        output_dir=tmp_path,
        device=CPU,
        monitor=monitor,
    )

    run = json.loads((tmp_path / RUN_FILENAME).read_text(encoding="utf-8"))
    state = torch.load(tmp_path / BEST_CHECKPOINT, weights_only=True)

    assert len(history) == 2
    assert "epoch_finished 2" in monitor.calls
    assert run["architecture"] == "CNN"
    assert run["search_trial"] == 12
    assert run["training"]["epochs"] == 2
    assert run["parameters"]["total"] > 0
    assert state["metadata"]["encoder"]["kernel_size"] == 5
    assert (tmp_path / HISTORY_FILENAME).exists()
    assert (
        LabelSpace.from_json((tmp_path / LABEL_SPACE_FILENAME).read_text(encoding="utf-8")) == SPACE
    )


def test_train_final_stops_at_its_time_budget(tmp_path: Path) -> None:
    config = FinalConfig(
        epochs=2,
        samples_per_epoch=8,
        batch_size=4,
        precision=Precision.FP32,
        time_budget_hours=1e-9,
    )

    with pytest.raises(TrainingDeadlineError):
        train_final(
            SearchChoice(trial=12, params=CNN_PARAMS),
            DATA,
            SPACE,
            config=config,
            output_dir=tmp_path,
            device=CPU,
        )

    assert (tmp_path / RUN_FILENAME).exists()
