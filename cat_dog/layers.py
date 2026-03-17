from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CustomConvBlock(tf.keras.layers.Layer):
    """Custom Convolutional Block với Batch Normalization tùy chỉnh."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        use_batch_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batch_norm = use_batch_norm

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        in_channels = int(input_shape[-1])
        
        # Convolutional weights - HeNormal initialization
        self.weight = self.add_weight(
            name="conv_weight",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="conv_bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
        )
        
        # Batch Normalization weights
        if self.use_batch_norm:
            self.bn_weight = self.add_weight(
                name="bn_weight",
                shape=(self.filters,),
                initializer="ones",
                trainable=True,
            )
            self.bn_bias = self.add_weight(
                name="bn_bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
            self.bn_mean = self.add_weight(
                name="bn_mean",
                shape=(self.filters,),
                initializer="zeros",
                trainable=False,
            )
            self.bn_var = self.add_weight(
                name="bn_var",
                shape=(self.filters,),
                initializer="ones",
                trainable=False,
            )
        
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Convolution
        x = tf.nn.conv2d(inputs, self.weight, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.bias)
        
        # Batch Normalization
        if self.use_batch_norm:
            x = self._batch_norm(x, training)
        
        # ReLU activation
        x = tf.nn.relu(x)
        
        # Max pooling
        x = tf.nn.max_pool2d(
            x,
            ksize=[1, self.pool_size, self.pool_size, 1],
            strides=[1, self.pool_size, self.pool_size, 1],
            padding="VALID",
        )
        return x
    
    def _batch_norm(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Batch Normalization implementation tùy chỉnh."""
        if training:
            # Tính mean và var trên batch hiện tại
            mean, var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
            # Update running mean/var với exponential moving average
            self.bn_mean.assign(0.99 * self.bn_mean + 0.01 * mean)
            self.bn_var.assign(0.99 * self.bn_var + 0.01 * var)
        else:
            # Dùng accumulated mean/var
            mean = self.bn_mean
            var = self.bn_var
        
        # Normalize
        epsilon = 1e-5
        x_norm = (x - mean) / tf.sqrt(var + epsilon)
        # Scale and shift
        x = self.bn_weight * x_norm + self.bn_bias
        return x


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CustomDenseBlock(tf.keras.layers.Layer):
    """Custom Dense Block với Dropout tùy chỉnh."""
    
    def __init__(
        self,
        units: int,
        activation: str | None = None,
        dropout_rate: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation_name,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        in_features = int(input_shape[-1])
        self.weight = self.add_weight(
            name="dense_weight",
            shape=(in_features, self.units),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="dense_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Fully connected layer
        x = tf.matmul(inputs, self.weight) + self.bias
        
        # Activation
        if self.activation is not None:
            x = self.activation(x)
        
        # Dropout
        if self.dropout_rate > 0.0 and training:
            keep_prob = 1.0 - self.dropout_rate
            mask = tf.cast(
                tf.random.uniform(tf.shape(x)) < keep_prob,
                x.dtype
            )
            x = x * mask / (keep_prob + 1e-7)
        
        return x


@tf.keras.utils.register_keras_serializable(package="cat_dog")
class CustomDropout(tf.keras.layers.Layer):
    """Custom Dropout Layer - Inverted Dropout implementation."""
    
    def __init__(self, rate: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if rate < 0.0 or rate >= 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1). Got: {rate}")
        self.rate = rate

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Áp dụng Dropout nếu training=True."""
        if training and self.rate > 0.0:
            # Keep probability
            keep_prob = 1.0 - self.rate
            # Tạo random mask từ uniform distribution
            # Bernoulli(p) = (Uniform < p)
            mask = tf.cast(
                tf.random.uniform(tf.shape(inputs)) < keep_prob,
                inputs.dtype
            )
            # Inverted Dropout: scale inputs để tổng expected value không đổi
            return inputs * mask / (keep_prob + 1e-7)
        # Inference mode: không thay đổi
        return inputs
