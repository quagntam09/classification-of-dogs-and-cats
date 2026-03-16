from __future__ import annotations

import tensorflow as tf

from .layers import CustomConvBlock, CustomDenseBlock


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CatDogCNN(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "cat_dog_cnn")
        super().__init__(**kwargs)
        self.rescale = tf.keras.layers.Rescaling(1.0 / 255)
        self.conv1 = CustomConvBlock(filters=32)
        self.conv2 = CustomConvBlock(filters=64)
        self.dense1 = CustomDenseBlock(units=128, activation="relu")
        self.output_layer = CustomDenseBlock(units=1, activation="sigmoid")

    def get_config(self) -> dict:
        return super().get_config()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        x = self.rescale(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = self.dense1(x)
        return self.output_layer(x)
