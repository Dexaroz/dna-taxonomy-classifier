import math
from typing import TYPE_CHECKING

import optuna
import polars as pl
import pytest
import torch

from taxonomy_classifier.model.encoders import CnnConfig, TransformerConfig
from taxonomy_classifier.model.labels import LEVELS, LabelSpace
from taxonomy_classifier.training import search
from taxonomy_classifier.training.data import LabeledSet, SequenceBank, TrainingData
from taxonomy_classifier.training.search import (
    PruningMonitor,
    SearchConfig,
    objective_value,
    run_search,
    suggest_setup,
    trial_rows,
)
from taxonomy_classifier.training.trainer import EpochRecord, Evaluation, SilentMonitor

if TYPE_CHECKING:
    from pathlib import Path

CPU = torch.device("cpu")

SPACE = LabelSpace(classes=dict.fromkeys(LEVELS, ("a", "b")), min_count=1)

COMMON = {
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "warmup_fraction": 0.05,
    "sampling_power": 0.5,
    "crop_probability": 0.5,
}

TINY_SEARCH = SearchConfig(trials=2, epochs=2, samples_per_epoch=8, batch_size=4, startup_trials=0)


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


def _record(genus: float, species: float) -> EpochRecord:
    accuracy = dict.fromkeys(LEVELS, 0.0) | {"genus": genus, "species": species}

    return EpochRecord(
        epoch=1,
        train_loss=1.0,
        validation=Evaluation(loss=1.0, accuracy=accuracy),
        learning_rate=0.0,
        seconds=1.0,
    )


def _storage(tmp_path: Path) -> Path:
    return tmp_path / "search" / "journal.log"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trials": 0}, "must be positive"),
        ({"startup_trials": -1}, "startup_trials"),
        ({"objective_levels": ()}, "objective_levels"),
    ],
)
def test_search_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SearchConfig(**kwargs)  # type: ignore[arg-type]


def test_cnn_setup_uses_the_suggested_values() -> None:
    params = COMMON | {"cnn_width": "large", "cnn_blocks": "three", "cnn_kernel": 7}

    setup = suggest_setup(optuna.trial.FixedTrial(params), "CNN", TINY_SEARCH)

    assert setup.encoder == CnnConfig(
        channels=(160, 256, 320, 384),
        blocks_per_stage=3,
        dilations=(1, 2, 4),
        kernel_size=7,
        dropout=0.1,
    )
    assert setup.heads.dropout == 0.1
    assert setup.training.learning_rate == 1e-3
    assert setup.training.crop.probability == 0.5
    assert (setup.training.epochs, setup.training.samples_per_epoch) == (2, 8)


def test_transformer_setup_uses_the_suggested_values() -> None:
    params = COMMON | {
        "transformer_tokenizer": "k12s6",
        "transformer_dim": 192,
        "transformer_layers": 8,
    }

    setup = suggest_setup(optuna.trial.FixedTrial(params), "Transformer", TINY_SEARCH)

    assert setup.encoder == TransformerConfig(
        model_dim=192,
        layers=8,
        token_kernel=12,
        token_stride=6,
        dropout=0.1,
    )


def test_objective_is_the_mean_accuracy_of_the_chosen_levels() -> None:
    assert objective_value(_record(0.8, 0.4).validation, ("genus", "species")) == pytest.approx(0.6)


def test_pruning_monitor_reports_and_prunes() -> None:
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=0.9)
    )
    trial = study.ask()
    monitor = PruningMonitor(trial, ("genus",), SilentMonitor())

    monitor.epoch_started(1, 1, 1)
    monitor.step_finished(1)
    monitor.metrics_updated(1, 1.0, 1.0, 1e-3)
    monitor.evaluation_started(1)
    monitor.evaluation_step()

    with pytest.raises(optuna.TrialPruned):
        monitor.epoch_finished(_record(0.5, 0.5))

    assert trial.storage.get_trial(trial._trial_id).intermediate_values == {1: 0.5}  # noqa: SLF001


def test_pruning_monitor_prunes_undefined_objectives() -> None:
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    monitor = PruningMonitor(study.ask(), ("genus",), SilentMonitor())

    with pytest.raises(optuna.TrialPruned):
        monitor.epoch_finished(_record(math.nan, 0.5))


@pytest.mark.parametrize("architecture", ["CNN", "Transformer"])
def test_run_search_completes_trials(architecture: str, tmp_path: Path) -> None:
    started: list[int] = []

    def factory(trial: optuna.Trial) -> SilentMonitor:
        started.append(trial.number)

        return SilentMonitor()

    study = run_search(
        architecture,
        DATA,
        SPACE,
        config=TINY_SEARCH,
        storage=_storage(tmp_path),
        device=CPU,
        monitor_factory=factory,
    )

    assert started == [0, 1]
    assert study.study_name == f"{architecture.lower()}-search"
    assert all(trial.state.is_finished() for trial in study.trials)
    assert study.best_value is not None
    assert "encoder" in study.best_trial.user_attrs


def test_run_search_resumes_from_its_storage(tmp_path: Path) -> None:
    run_search("CNN", DATA, SPACE, config=TINY_SEARCH, storage=_storage(tmp_path), device=CPU)

    more = SearchConfig(trials=3, epochs=2, samples_per_epoch=8, batch_size=4, startup_trials=0)
    study = run_search("CNN", DATA, SPACE, config=more, storage=_storage(tmp_path), device=CPU)

    assert len(study.trials) == 3
    assert study.trials[2].params != study.trials[0].params


def test_out_of_memory_trials_fail_without_stopping_the_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exhausted(*_: object, **__: object) -> list[EpochRecord]:
        msg = "CUDA out of memory"
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(search, "train_classifier", exhausted)

    study = run_search(
        "CNN", DATA, SPACE, config=TINY_SEARCH, storage=_storage(tmp_path), device=CPU
    )

    assert [trial.state for trial in study.trials] == [optuna.trial.TrialState.FAIL] * 2


def test_unknown_architectures_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        run_search("RNN", DATA, SPACE, config=TINY_SEARCH, storage=_storage(tmp_path), device=CPU)


def test_trial_rows_summarize_each_trial(tmp_path: Path) -> None:
    study = run_search(
        "CNN", DATA, SPACE, config=TINY_SEARCH, storage=_storage(tmp_path), device=CPU
    )

    rows = trial_rows(study)

    assert [row["trial"] for row in rows] == [0, 1]
    assert {"state", "value", "epochs", "learning_rate", "cnn_width"} <= set(rows[0])
