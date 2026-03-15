import tensorflow as tf
import os
from setup_gpu import TensorFlowConfig

# ========================== CẤU HÌNH ==========================
PATH = "Datasets"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 10

# ========================== TẢI DỮ LIỆU ==========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{PATH}/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'          # Quan trọng cho binary_crossentropy
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{PATH}/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)
# Tối ưu hóa hiệu suất GPU/CPU
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ========================== CUSTOM LAYERS (từ scratch) ==========================
class CustomConvBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = self.add_weight(
            shape=[3, 3, in_channels, out_channels],
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
            name="Conv_Weights"
        )
        self.b = self.add_weight(
            shape=[out_channels],
            initializer='zeros',
            trainable=True,
            name="Conv_Bias"
        )

    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return x

class CustomDenseBlock(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        self.activation = activation
        self.W = self.add_weight(
            shape=[in_features, out_features],
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
            name="Dense_Weights"
        )
        self.b = self.add_weight(
            shape=[out_features],
            initializer='zeros',
            trainable=True,
            name="Dense_Bias"
        )

    def call(self, inputs):
        Z = tf.matmul(inputs, self.W) + self.b
        if self.activation == 'relu':
            return tf.nn.relu(Z)
        elif self.activation == 'sigmoid':
            return tf.math.sigmoid(Z)
        return Z


# ========================== MÔ HÌNH CHÍNH ==========================
class MyCustomCNN(tf.keras.Model):
    def __init__(self,input_shape=(180, 180, 3)):
        super().__init__()
        self.rescale = tf.keras.layers.Rescaling(1./255)
        
        self.conv1 = CustomConvBlock(in_channels=3, out_channels=32)
        self.conv2 = CustomConvBlock(in_channels=32, out_channels=64)
        
        # TÍNH TỰ ĐỘNG flatten_size (an toàn tuyệt đối)
        dummy = tf.zeros([1] + list(input_shape))
        x = self.rescale(dummy)
        x = self.conv1(x)
        x = self.conv2(x)
        self.flatten_size = int(tf.shape(input=x)[1] * tf.shape(x)[2] * tf.shape(x)[3])
        
        self.dense1 = CustomDenseBlock(self.flatten_size, 128, activation='relu')
        self.dense2 = CustomDenseBlock(128, 1, activation='sigmoid')

    def call(self, inputs):
        x = self.rescale(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.reshape(x, [-1, self.flatten_size])
        x = self.dense1(x)
        return self.dense2(x)

# ========================== KHỞI TẠO & HUẤN LUYỆN ==========================
model = MyCustomCNN()
model.build(input_shape=(None, 180, 180, 3))
model.summary()

# 1. Khởi tạo Loss, Optimizer và Metrics
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Các bộ đếm để tính trung bình Loss và Accuracy trong 1 epoch
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

# 2. Hàm huấn luyện (Train)
@tf.function 
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Cập nhật thông số vào metrics
    train_loss(loss)
    train_accuracy(labels, predictions)

# 3. Hàm đánh giá (Validation - Không tính đạo hàm)
@tf.function
def val_step(images, labels):
    # training=False để các lớp như Dropout/BatchNorm (nếu có sau này) hoạt động đúng
    predictions = model(images, training=False) 
    v_loss = loss_fn(labels, predictions)
    
    # Cập nhật thông số vào metrics
    val_loss(v_loss)
    val_accuracy(labels, predictions)

# 4. Vòng lặp huấn luyện chính
print("\n--- BẮT ĐẦU HUẤN LUYỆN THỦ CÔNG ---")

# Biến lưu trữ cho Early Stopping và Checkpoint
best_val_loss = float('inf')
patience = 3
wait_count = 0

for epoch in range(EPOCHS):
    # Reset metrics ở đầu mỗi epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    val_loss.reset_state()
    val_accuracy.reset_state()

    # Quá trình Training
    for batch_images, batch_labels in train_ds:
        train_step(batch_images, batch_labels)

    # Quá trình Validation
    for batch_images, batch_labels in val_ds:
        val_step(batch_images, batch_labels)

    # In kết quả của Epoch
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  - Train Loss: {train_loss.result():.4f} - Train Acc: {train_accuracy.result():.4f}")
    print(f"  - Val Loss:   {val_loss.result():.4f} - Val Acc:   {val_accuracy.result():.4f}")

    # Logic Model Checkpoint: Lưu mô hình nếu val_loss giảm
    current_val_loss = val_loss.result()
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        wait_count = 0 # Reset bộ đếm Early Stopping
        model.save("best_cat_dog_model.keras")
        print("  => Đã lưu mô hình tốt nhất (val_loss cải thiện).")
    else:
        wait_count += 1
        print(f"  => val_loss không cải thiện. Early stopping đếm: {wait_count}/{patience}")
        
        # Logic Early Stopping: Dừng nếu val_loss không giảm sau số epoch = patience
        if wait_count >= patience:
            print(f"\n--- KÍCH HOẠT EARLY STOPPING TẠI EPOCH {epoch + 1} ---")
            break

print("\nHuấn luyện hoàn tất!")

# Tùy chọn: Lưu lại phiên bản cuối cùng của mô hình sau khi thoát vòng lặp
model.save("cat_dog_cnn_model_final.keras")
print("Đã lưu phiên bản cuối cùng tại: cat_dog_cnn_model_final.keras")