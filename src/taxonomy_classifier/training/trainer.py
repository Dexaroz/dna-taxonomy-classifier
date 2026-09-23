from dataclasses import asdict, dataclass, field
import json
import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any, Final, Protocol

import torch

from taxonomy_classifier.data.files import write_text_atomic
from taxonomy_classifier.model.heads import hierarchical_loss
from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS
from taxonomy_classifier.training.data import CropConfig, iterate_batches, sampling_weights

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from taxonomy_classifier.model.classifier import TaxonomyClassifier
    from taxonomy_classifier.training.data import LabeledSet, TrainingData

_LOGGER: Final = logging.getLogger(__name__)

BEST_CHECKPOINT: Final = "best.pt"

LAST_CHECKPOINT: Final = "last.pt"

HISTORY_FILENAME: Final = "history.json"

_MATRIX_NDIM: Final = 2


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 8
    samples_per_epoch: int | None = None
    batch_size: int = 128
    max_length: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    sampling_power: float = 0.5
    crop: CropConfig = field(default_factory=CropConfig)
    evaluation_batch_size: int = 256
    mixed_precision: bool = True
    log_every: int = 500
    seed: int = 20260923

    def __post_init__(self) -> None:
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "evaluation_batch_size": self.evaluation_batch_size,
            "log_every": self.log_every,
        }

        for name, value in positive.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)

        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            msg = f"samples_per_epoch must be positive, got {self.samples_per_epoch}"
            raise ValueError(msg)

        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0 or self.weight_decay < 0.0:
            msg = "learning_rate and gradient_clip must be positive and weight_decay non-negative"
            raise ValueError(msg)

        if not 0.0 <= self.warmup_fraction < 1.0:
            msg = f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Evaluation:
    loss: float
    accuracy: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation: Evaluation
    learning_rate: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.validation.loss,
            "val_accuracy": dict(self.validation.accuracy),
            "learning_rate": self.learning_rate,
            "seconds": self.seconds,
        }


class TrainingMonitor(Protocol):
    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None: ...

    def step_finished(self, step: int) -> None: ...

    def metrics_updated(
        self, step: int, loss: float, sequences_per_second: float, learning_rate: float
    ) -> None: ...

    def evaluation_started(self, batches: int) -> None: ...

    def evaluation_step(self) -> None: ...

    def epoch_finished(self, record: EpochRecord) -> None: ...


class SilentMonitor:
    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        pass

    def step_finished(self, step: int) -> None:
        pass

    def metrics_updated(
        self, step: int, loss: float, sequences_per_second: float, learning_rate: float
    ) -> None:
        pass

    def evaluation_started(self, batches: int) -> None:
        pass

    def evaluation_step(self) -> None:
        pass

    def epoch_finished(self, record: EpochRecord) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _Optimization:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    @property
    def learning_rate(self) -> float:
        return float(self.scheduler.get_last_lr()[0])


def learning_rate_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def evaluate(
    model: TaxonomyClassifier,
    data: LabeledSet,
    *,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool = False,
    max_length: int | None = None,
    on_batch: Callable[[], None] | None = None,
) -> Evaluation:
    model.eval()

    correct = torch.zeros(len(LEVELS), dtype=torch.long)
    labeled = torch.zeros(len(LEVELS), dtype=torch.long)
    loss_sum = 0.0
    batches = 0

    with torch.inference_mode(), _autocast(device, enabled=mixed_precision):
        for batch in iterate_batches(
            data,
            torch.arange(len(data)),
            batch_size=batch_size,
            crop=None,
            rng=random.Random(0),
            max_length=max_length,
        ):
            on_device = batch.to(device)
            logits = model(on_device.tokens, on_device.mask)

            loss_sum += float(hierarchical_loss(logits, on_device.targets))
            batches += 1

            predictions = torch.stack([level.argmax(dim=-1) for level in logits], dim=-1)
            known = on_device.targets != IGNORE_INDEX

            correct += ((predictions == on_device.targets) & known).sum(dim=0).cpu()
            labeled += known.sum(dim=0).cpu()

            if on_batch is not None:
                on_batch()

    accuracy = {
        level: float(hits) / float(total) if total else math.nan
        for level, hits, total in zip(LEVELS, correct, labeled, strict=True)
    }

    return Evaluation(loss=loss_sum / max(1, batches), accuracy=accuracy)


def train_classifier(
    model: TaxonomyClassifier,
    data: TrainingData,
    *,
    config: TrainingConfig,
    checkpoint_dir: Path | None,
    device: torch.device,
    metadata: Mapping[str, Any] | None = None,
    monitor: TrainingMonitor | None = None,
) -> list[EpochRecord]:
    watcher = monitor or SilentMonitor()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    model.to(device)

    samples = config.samples_per_epoch or len(data.train)
    steps_per_epoch = math.ceil(samples / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))

    optimizer = _optimizer(model, config)
    optimization = _Optimization(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: learning_rate_factor(
                step, total_steps=total_steps, warmup_steps=warmup_steps
            ),
        ),
    )

    weights = sampling_weights(data.frequencies, power=config.sampling_power)

    history: list[EpochRecord] = []
    best_loss = math.inf

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        indices = torch.multinomial(weights, samples, replacement=True, generator=generator)

        watcher.epoch_started(epoch, config.epochs, steps_per_epoch)

        train_loss = _train_epoch(
            model,
            data.train,
            indices,
            optimization,
            config=config,
            device=device,
            rng=rng,
            monitor=watcher,
        )

        watcher.evaluation_started(math.ceil(len(data.validation) / config.evaluation_batch_size))

        validation_result = evaluate(
            model,
            data.validation,
            batch_size=config.evaluation_batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
            max_length=config.max_length,
            on_batch=watcher.evaluation_step,
        )

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            validation=validation_result,
            learning_rate=optimization.learning_rate,
            seconds=time.perf_counter() - started,
        )
        history.append(record)

        improved = validation_result.loss < best_loss
        best_loss = min(best_loss, validation_result.loss)

        if checkpoint_dir is not None:
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": asdict(config),
                "metadata": dict(metadata or {}),
            }

            _save_checkpoint(checkpoint_dir, state, history, improved=improved)

        _LOGGER.info(
            "Epoch %d/%d: train %.4f, val %.4f, genus %.3f, %.0f s",
            epoch,
            config.epochs,
            train_loss,
            validation_result.loss,
            validation_result.accuracy["genus"],
            record.seconds,
        )

        watcher.epoch_finished(record)

    return history


def load_history(checkpoint_dir: Path) -> list[dict[str, Any]]:
    path = checkpoint_dir / HISTORY_FILENAME

    if not path.exists():
        return []

    history: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    return history


def load_weights(
    model: TaxonomyClassifier, checkpoint: Path, *, device: torch.device
) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])

    return state


def _train_epoch(
    model: TaxonomyClassifier,
    train: LabeledSet,
    indices: torch.Tensor,
    optimization: _Optimization,
    *,
    config: TrainingConfig,
    device: torch.device,
    rng: random.Random,
    monitor: TrainingMonitor,
) -> float:
    model.train()

    loss_sum = torch.zeros((), device=device)
    steps = 0
    started = time.perf_counter()

    for batch in iterate_batches(
        train,
        indices,
        batch_size=config.batch_size,
        crop=config.crop,
        rng=rng,
        max_length=config.max_length,
    ):
        on_device = batch.to(device)

        with _autocast(device, enabled=config.mixed_precision):
            loss = hierarchical_loss(model(on_device.tokens, on_device.mask), on_device.targets)

        optimization.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimization.optimizer.step()
        optimization.scheduler.step()

        loss_sum += loss.detach()
        steps += 1

        monitor.step_finished(steps)

        if steps % config.log_every == 0:
            running_loss = loss_sum.item() / steps
            rate = steps * config.batch_size / (time.perf_counter() - started)

            monitor.metrics_updated(steps, running_loss, rate, optimization.learning_rate)
            _LOGGER.debug("step %d: loss %.4f, %.0f sequences/s", steps, running_loss, rate)

    return loss_sum.item() / max(1, steps)


def _save_checkpoint(
    checkpoint_dir: Path,
    state: dict[str, Any],
    history: list[EpochRecord],
    *,
    improved: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / LAST_CHECKPOINT)

    if improved:
        torch.save(state, checkpoint_dir / BEST_CHECKPOINT)

    write_text_atomic(
        checkpoint_dir / HISTORY_FILENAME,
        json.dumps([item.to_dict() for item in history], indent=2),
    )


def _optimizer(model: TaxonomyClassifier, config: TrainingConfig) -> torch.optim.AdamW:
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= _MATRIX_NDIM]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < _MATRIX_NDIM]

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def _autocast(device: torch.device, *, enabled: bool) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
