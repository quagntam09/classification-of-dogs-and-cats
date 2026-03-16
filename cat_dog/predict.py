from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from .layers import CustomConvBlock, CustomDenseBlock
from .model import CatDogCNN


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    dog_probability: float


class ImagePredictor:
    def __init__(self, model_path: Path, image_size: tuple[int, int]) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: Path) -> tf.keras.Model:
        if not model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        custom_objects = {
            "CatDogCNN": CatDogCNN,
            "CustomConvBlock": CustomConvBlock,
            "CustomDenseBlock": CustomDenseBlock,
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    def predict(self, image_path: Path) -> PredictionResult:
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        image = tf.keras.utils.load_img(image_path, target_size=self.image_size)
        image_array = tf.keras.utils.img_to_array(image)
        image_batch = np.expand_dims(image_array, axis=0)

        dog_probability = float(self.model.predict(image_batch, verbose=0)[0][0])
        label = "dog" if dog_probability >= 0.5 else "cat"
        confidence = dog_probability if label == "dog" else 1.0 - dog_probability

        return PredictionResult(
            label=label,
            confidence=confidence,
            dog_probability=dog_probability,
        )
