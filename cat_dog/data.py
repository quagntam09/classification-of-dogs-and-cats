from __future__ import annotations

import tensorflow as tf

from .config import TrainingConfig


class DatasetBuilder:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def build_train_and_val(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        train_dir = self.config.dataset_root / "train"
        common_args = {
            "directory": str(train_dir),
            "validation_split": self.config.validation_split,
            "seed": self.config.seed,
            "image_size": self.config.image_size,
            "batch_size": self.config.batch_size,
            "label_mode": "binary",
        }

        train_ds = tf.keras.utils.image_dataset_from_directory(
            subset="training",
            **common_args,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            subset="validation",
            **common_args,
        )

        return self._optimize(train_ds), self._optimize(val_ds)

    @staticmethod
    def _optimize(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
