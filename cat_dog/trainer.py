from __future__ import annotations

import tensorflow as tf

from .config import TrainingConfig
from .training_monitor import EarlyStoppingTracker, EpochMetrics


class ModelTrainer:
    def __init__(self, model: tf.keras.Model, config: TrainingConfig) -> None:
        self.model = model
        self.config = config

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name="val_accuracy")

    @staticmethod
    def _normalize_labels(labels: tf.Tensor) -> tf.Tensor:
        labels = tf.cast(labels, tf.float32)
        return tf.reshape(labels, [-1, 1])

    @tf.function(jit_compile=False)
    def train_step(self, images: tf.Tensor, labels: tf.Tensor) -> None:
        labels = self._normalize_labels(labels)
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

    @tf.function(jit_compile=False)
    def val_step(self, images: tf.Tensor, labels: tf.Tensor) -> None:
        labels = self._normalize_labels(labels)
        predictions = self.model(images, training=False)
        loss = self.loss_fn(labels, predictions)

        self.val_loss.update_state(loss)
        self.val_accuracy.update_state(labels, predictions)

    def _reset_metrics(self) -> None:
        self.train_loss.reset_state()
        self.train_accuracy.reset_state()
        self.val_loss.reset_state()
        self.val_accuracy.reset_state()

    def _collect_metrics(self) -> EpochMetrics:
        return EpochMetrics(
            train_loss=float(self.train_loss.result().numpy()),
            train_accuracy=float(self.train_accuracy.result().numpy()),
            val_loss=float(self.val_loss.result().numpy()),
            val_accuracy=float(self.val_accuracy.result().numpy()),
        )

    @staticmethod
    def _print_epoch_metrics(epoch: int, total_epochs: int, metrics: EpochMetrics) -> None:
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"  - Train Loss: {metrics.train_loss:.4f} - Train Acc: {metrics.train_accuracy:.4f}")
        print(f"  - Val Loss:   {metrics.val_loss:.4f} - Val Acc:   {metrics.val_accuracy:.4f}")

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> None:
        print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
        early_stopping = EarlyStoppingTracker(patience=self.config.patience)

        for epoch in range(self.config.epochs):
            self._reset_metrics()

            for images, labels in train_ds:
                self.train_step(images, labels)

            for images, labels in val_ds:
                self.val_step(images, labels)

            metrics = self._collect_metrics()
            self._print_epoch_metrics(epoch=epoch + 1, total_epochs=self.config.epochs, metrics=metrics)

            improved, should_stop, wait_count = early_stopping.update(metrics.val_loss)
            if improved:
                self.model.save(self.config.best_model_path)
                print("  => Đã lưu mô hình tốt nhất (val_loss cải thiện).")
            else:
                print(
                    f"  => val_loss không cải thiện. "
                    f"Early stopping đếm: {wait_count}/{self.config.patience}"
                )
                if should_stop:
                    print(f"\n--- KÍCH HOẠT EARLY STOPPING TẠI EPOCH {epoch + 1} ---")
                    break

        self.model.save(self.config.final_model_path)
        print("\nHuấn luyện hoàn tất!")
        print(f"Đã lưu phiên bản cuối cùng tại: {self.config.final_model_path}")
