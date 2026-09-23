from typing import TYPE_CHECKING

from taxonomy_classifier.model.labels import LabelSpace
from taxonomy_classifier.training.setup import (
    TRAIN_TOKENS_FILENAME,
    VALIDATION_TOKENS_FILENAME,
    label_space_path,
    load_training_data,
)

if TYPE_CHECKING:
    from taxonomy_classifier.data.build import DataLayout


def test_load_training_data_combines_real_and_synthetic_training_sequences(
    built_layout: DataLayout,
) -> None:
    label_space, data = load_training_data(built_layout, min_class_count=1)

    assert len(data.train) == 16
    assert len(data.validation) == 3
    assert len(data.frequencies) == len(data.train)
    assert data.train.targets.shape[1] == len(label_space.sizes)


def test_load_training_data_writes_the_label_space_and_caches(built_layout: DataLayout) -> None:
    label_space, _ = load_training_data(built_layout, min_class_count=1)

    saved = LabelSpace.from_json(label_space_path(built_layout).read_text(encoding="utf-8"))

    assert saved == label_space
    assert (built_layout.interim_dir / TRAIN_TOKENS_FILENAME).exists()
    assert (built_layout.interim_dir / VALIDATION_TOKENS_FILENAME).exists()
