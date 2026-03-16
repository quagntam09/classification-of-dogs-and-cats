from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CustomConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        in_channels = int(input_shape[-1])
        self.weight = self.add_weight(
            name="conv_weight",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="conv_bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.nn.conv2d(inputs, self.weight, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.bias)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(
            x,
            ksize=[1, self.pool_size, self.pool_size, 1],
            strides=[1, self.pool_size, self.pool_size, 1],
            padding="VALID",
        )
        return x


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CustomDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units: int, activation: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation_name,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        in_features = int(input_shape[-1])
        self.weight = self.add_weight(
            name="dense_weight",
            shape=(in_features, self.units),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="dense_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.matmul(inputs, self.weight) + self.bias
        return self.activation(x) if self.activation is not None else x
