# Cat-Dog Classification - CNN Model

Mô hình deep learning để phân loại ảnh chó và mèo.

## Cấu Trúc Project

```
.
├── cat_dog/                    # Core package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Hyperparameters
│   ├── data.py                # Dataset loading
│   ├── layers.py              # Custom layers
│   ├── model.py               # CNN model
│   ├── predict.py             # Prediction utilities
│   ├── trainer.py             # Training loop
│   └── training_monitor.py    # Metrics tracking
│
├── Datasets/                  # Dataset directory
│   ├── train/                 # Training data
│   │   ├── cats/
│   │   └── dogs/
│   └── test/                  # Test data
│       ├── cats/
│       └── dogs/
│
├── train.py                   # Training script
├── app_gui.py                 # GUI application
├── predict_image.py           # CLI prediction
├── best_cat_dog_model.keras   # Trained model (best)
├── cat_dog_cnn_model_final.keras # Final model
└── README.md                  # This file
```

## Model Architecture

Deep convolutional neural network với 4 tầng convolutional và 3 tầng dense:

- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization sau mỗi layer
- Dropout regularization (0.25-0.5)
- 3 Dense layers (512, 256, 128 units)
- Binary output (Cat/Dog)

Tổng parameters: 16.4M

## Cài Đặt

```bash
pip install tensorflow pillow numpy
```

## Sử Dụng

### 1. Training

```bash
python train.py
```

Model sẽ được lưu:
- `best_cat_dog_model.keras` - Model tốt nhất
- `cat_dog_cnn_model_final.keras` - Model cuối cùng

### 2. GUI Application

```bash
python app_gui.py
```

Giao diện để tải ảnh và dự đoán kết quả.

### 3. Command Line Prediction

```bash
python predict_image.py path/to/image.jpg
```

## Hyperparameters

- Image size: 180x180
- Batch size: 32
- Learning rate: 5e-4
- Epochs: 100
- Early stopping patience: 10

## Model Features

- Custom layers (không dùng pre-built)
- Batch Normalization tùy chỉnh
- Dropout regularization
- HeNormal initialization
- Training/inference mode support

## Performance

Expected accuracy: 92-95% on test set

## File Descriptions

| File | Mô Tả |
|------|-------|
| `train.py` | Script training model |
| `app_gui.py` | GUI tkinter để dự đoán |
| `predict_image.py` | CLI dự đoán |
| `cat_dog/layers.py` | Custom layers |
| `cat_dog/model.py` | Model definition |
| `cat_dog/config.py` | Configuration |
| `cat_dog/trainer.py` | Training logic |
