from dataclasses import asdict, dataclass
import gc
import json
import math
import time
from typing import TYPE_CHECKING, Any, Final

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
import torch

from taxonomy_classifier.data.files import write_text_atomic
from taxonomy_classifier.model.classifier import build_classifier
from taxonomy_classifier.model.encoders import CnnConfig, TransformerConfig
from taxonomy_classifier.model.heads import HeadConfig
from taxonomy_classifier.training.data import CropConfig, TrainingData
from taxonomy_classifier.training.trainer import (
    Precision,
    SilentMonitor,
    TrainingConfig,
    train_classifier,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from optuna.trial import BaseTrial

    from taxonomy_classifier.model.classifier import EncoderConfig
    from taxonomy_classifier.model.labels import LabelSpace
    from taxonomy_classifier.training.trainer import EpochRecord, Evaluation, TrainingMonitor

CNN_WIDTHS: Final = {
    "small": (96, 128, 192, 256),
    "base": (128, 192, 256, 320),
    "large": (160, 256, 320, 384),
}

CNN_DEPTHS: Final = {"two": (1, 2), "three": (1, 2, 4)}

TRANSFORMER_TOKENIZERS: Final = {"k6s3": (6, 3), "k8s4": (8, 4), "k12s6": (12, 6)}


@dataclass(frozen=True, slots=True)
class SearchConfig:
    trials: int | None = 30
    time_budget_hours: float | None = None
    epochs: int = 3
    samples_per_epoch: int = 150_000
    batch_size: int = 128
    validation_samples: int | None = None
    pruning_start_epoch: int = 2
    precision: Precision = Precision.BF16
    startup_trials: int = 5
    objective_levels: tuple[str, ...] = ("genus", "species")
    seed: int = 20260923

    def __post_init__(self) -> None:
        if min(self.epochs, self.samples_per_epoch, self.batch_size) < 1:
            msg = "epochs, samples_per_epoch and batch_size must be positive"
            raise ValueError(msg)

        if self.trials is None and self.time_budget_hours is None:
            msg = "A search needs a number of trials, a time budget or both"
            raise ValueError(msg)

        if (self.trials is not None and self.trials < 1) or (
            self.time_budget_hours is not None and self.time_budget_hours <= 0
        ):
            msg = "trials and time_budget_hours must be positive when given"
            raise ValueError(msg)

        if self.pruning_start_epoch < 1:
            msg = f"pruning_start_epoch must be positive, got {self.pruning_start_epoch}"
            raise ValueError(msg)

        if self.validation_samples is not None and self.validation_samples < 1:
            msg = f"validation_samples must be positive, got {self.validation_samples}"
            raise ValueError(msg)

        if self.startup_trials < 0 or not self.objective_levels:
            msg = "startup_trials must be non-negative and objective_levels not empty"
            raise ValueError(msg)

    @property
    def time_budget_seconds(self) -> float | None:
        return None if self.time_budget_hours is None else self.time_budget_hours * 3600


@dataclass(frozen=True, slots=True)
class TrialSetup:
    encoder: EncoderConfig
    heads: HeadConfig
    training: TrainingConfig


def suggest_cnn(trial: BaseTrial, dropout: float) -> CnnConfig:
    width = trial.suggest_categorical("cnn_width", sorted(CNN_WIDTHS))
    depth = trial.suggest_categorical("cnn_blocks", sorted(CNN_DEPTHS))

    return CnnConfig(
        channels=CNN_WIDTHS[width],
        blocks_per_stage=len(CNN_DEPTHS[depth]),
        dilations=CNN_DEPTHS[depth],
        kernel_size=trial.suggest_categorical("cnn_kernel", [5, 7, 9, 11]),
        dropout=dropout,
    )


def suggest_transformer(trial: BaseTrial, dropout: float) -> TransformerConfig:
    kernel, stride = TRANSFORMER_TOKENIZERS[
        trial.suggest_categorical("transformer_tokenizer", sorted(TRANSFORMER_TOKENIZERS))
    ]

    return TransformerConfig(
        model_dim=trial.suggest_categorical("transformer_dim", [192, 256, 320]),
        layers=trial.suggest_int("transformer_layers", 4, 8, step=2),
        token_kernel=kernel,
        token_stride=stride,
        dropout=dropout,
    )


SUGGESTERS: Final[Mapping[str, Callable[[BaseTrial, float], EncoderConfig]]] = {
    "CNN": suggest_cnn,
    "Transformer": suggest_transformer,
}


def suggest_setup(trial: BaseTrial, architecture: str, config: SearchConfig) -> TrialSetup:
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    training = TrainingConfig(
        epochs=config.epochs,
        samples_per_epoch=config.samples_per_epoch,
        batch_size=config.batch_size,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-3, 0.2, log=True),
        warmup_fraction=trial.suggest_float("warmup_fraction", 0.02, 0.1),
        sampling_power=trial.suggest_float("sampling_power", 0.0, 1.0),
        crop=CropConfig(probability=trial.suggest_float("crop_probability", 0.2, 0.8)),
        precision=config.precision,
        seed=config.seed,
    )

    return TrialSetup(
        encoder=SUGGESTERS[architecture](trial, dropout),
        heads=HeadConfig(dropout=dropout),
        training=training,
    )


def objective_value(evaluation: Evaluation, levels: Sequence[str]) -> float:
    return sum(evaluation.accuracy[level] for level in levels) / len(levels)


class PruningMonitor:
    def __init__(
        self,
        trial: optuna.Trial,
        levels: Sequence[str],
        inner: TrainingMonitor,
        *,
        deadline: float | None = None,
    ) -> None:
        self._trial = trial
        self._levels = levels
        self._inner = inner
        self._deadline = deadline

    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        self._inner.epoch_started(epoch, epochs, steps)

    def step_finished(self, step: int) -> None:
        self._inner.step_finished(step)

        if self._deadline is not None and time.monotonic() >= self._deadline:
            self._trial.set_user_attr("stopped_by_deadline", value=True)

            raise optuna.TrialPruned

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

        value = objective_value(record.validation, self._levels)
        self._trial.report(value, record.epoch)

        if math.isnan(value) or self._trial.should_prune():
            raise optuna.TrialPruned


def create_study(architecture: str, *, storage: Path, config: SearchConfig) -> optuna.Study:
    storage.parent.mkdir(parents=True, exist_ok=True)

    name = f"{architecture.lower()}-search"
    journal = JournalStorage(
        JournalFileBackend(str(storage), lock_obj=JournalFileOpenLock(str(storage)))
    )
    existing = _existing_trials(name, journal)

    return optuna.create_study(
        study_name=name,
        storage=journal,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed + existing),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min(config.pruning_start_epoch, config.epochs),
            max_resource=config.epochs,
            reduction_factor=3,
        ),
    )


def run_search(
    architecture: str,
    data: TrainingData,
    label_space: LabelSpace,
    *,
    config: SearchConfig,
    storage: Path,
    device: torch.device,
    monitor_factory: Callable[[optuna.Trial], TrainingMonitor] | None = None,
) -> optuna.Study:
    if architecture not in SUGGESTERS:
        msg = f"Unknown architecture {architecture!r}; expected one of {sorted(SUGGESTERS)}"
        raise ValueError(msg)

    study = create_study(architecture, storage=storage, config=config)
    finished = sum(trial.state.is_finished() for trial in study.trials)
    budget = config.time_budget_seconds
    deadline = None if budget is None else time.monotonic() + budget
    search_data = _with_validation_subset(data, config)

    def objective(trial: optuna.Trial) -> float:
        setup = suggest_setup(trial, architecture, config)
        trial.set_user_attr("encoder", asdict(setup.encoder))

        inner = monitor_factory(trial) if monitor_factory is not None else SilentMonitor()
        model = build_classifier(setup.encoder, label_space, setup.heads)

        try:
            history = train_classifier(
                model,
                search_data,
                config=setup.training,
                checkpoint_dir=None,
                device=device,
                monitor=PruningMonitor(trial, config.objective_levels, inner, deadline=deadline),
            )

        finally:
            del model
            _release_memory()

        return objective_value(history[-1].validation, config.objective_levels)

    study.optimize(
        objective,
        n_trials=None if config.trials is None else max(0, config.trials - finished),
        timeout=budget,
        catch=(torch.OutOfMemoryError,),
        callbacks=[ResultsWriter(storage)],
        gc_after_trial=True,
    )

    return study


class ResultsWriter:
    def __init__(self, storage: Path) -> None:
        self._trials_path = storage.with_name(f"{storage.stem}-trials.json")
        self._best_path = storage.with_name(f"{storage.stem}-best.json")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        del trial

        write_text_atomic(self._trials_path, json.dumps(trial_rows(study), indent=2, default=str))

        completed = [
            item for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE
        ]

        if completed:
            best = max(completed, key=lambda item: item.value or -math.inf)
            summary = {
                "trial": best.number,
                "value": best.value,
                "params": best.params,
                "encoder": best.user_attrs.get("encoder"),
            }
            write_text_atomic(self._best_path, json.dumps(summary, indent=2, default=str))


def trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    return [
        {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "epochs": len(trial.intermediate_values),
            **trial.params,
        }
        for trial in study.trials
    ]


def _with_validation_subset(data: TrainingData, config: SearchConfig) -> TrainingData:
    samples = config.validation_samples

    if samples is None or samples >= len(data.validation):
        return data

    generator = torch.Generator().manual_seed(config.seed)
    chosen = torch.randperm(len(data.validation), generator=generator)[:samples].sort().values

    return TrainingData(
        train=data.train,
        frequencies=data.frequencies,
        validation=data.validation.take(chosen.tolist()),
    )


def _existing_trials(name: str, storage: JournalStorage) -> int:
    if name not in optuna.get_all_study_names(storage):
        return 0

    return len(optuna.load_study(study_name=name, storage=storage).trials)


def _release_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
