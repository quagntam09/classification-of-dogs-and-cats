from __future__ import annotations

import tensorflow as tf

from .layers import CustomConvBlock, CustomDenseBlock, CustomDropout


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CatDogCNN(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "cat_dog_cnn")
        super().__init__(**kwargs)
        
        # Rescaling layer
        self.rescale = tf.keras.layers.Rescaling(1.0 / 255)
        
        # Deep convolutional layers with batch normalization
        self.conv1 = CustomConvBlock(filters=32, use_batch_norm=True)
        self.conv2 = CustomConvBlock(filters=64, use_batch_norm=True)
        self.conv3 = CustomConvBlock(filters=128, use_batch_norm=True)
        self.conv4 = CustomConvBlock(filters=256, use_batch_norm=True)
        
        # Dropout layer after convolutions
        self.dropout_conv = CustomDropout(rate=0.25)
        
        # Dense layers with dropout
        self.dense1 = CustomDenseBlock(units=512, activation="relu", dropout_rate=0.5)
        self.dense2 = CustomDenseBlock(units=256, activation="relu", dropout_rate=0.4)
        self.dense3 = CustomDenseBlock(units=128, activation="relu", dropout_rate=0.3)
        
        # Output layer
        self.output_layer = CustomDenseBlock(units=1, activation="sigmoid")

    def get_config(self) -> dict:
        return super().get_config()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Rescaling
        x = self.rescale(inputs)
        
        # Convolutional blocks
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        # Dropout after convolutions
        x = self.dropout_conv(x, training=training)
        
        # Flatten
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        
        # Dense layers
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        
        # Output
        return self.output_layer(x, training=training)
