# Usage Guide

## Workflow

### 1. Training Model

```bash
python train.py
```

Output:
- Logs training progress
- Saves best model to: `best_cat_dog_model.keras`
- Saves final model to: `cat_dog_cnn_model_final.keras`

Expected:
- Training time: 15-20 minutes
- Final accuracy: 92-95%

### 2. Using GUI Application

```bash
python app_gui.py
```

Features:
- Loads trained model automatically (best_cat_dog_model.keras)
- Select image from disk
- View prediction result
- Shows confidence score

### 3. Command Line Prediction

```bash
python predict_image.py path/to/image.jpg
```

Output: Prediction result and confidence

### 4. Sample Prediction

```bash
python predict_sample.py
```

Uses sample image for quick testing

## Directory Structure

```
.
├── cat_dog/              # Model package
├── Datasets/             # Training data
├── train.py             # Training script
├── app_gui.py           # GUI application
├── predict_image.py     # CLI prediction
├── README.md            # Project overview
├── ARCHITECTURE.md      # Model details
└── USAGE.md             # This file
```

## Configuration

Edit `cat_dog/config.py` to change:
- Image size
- Batch size
- Learning rate
- Number of epochs
- Early stopping patience

## Model Files

- `best_cat_dog_model.keras` - Best model (used for prediction)
- `cat_dog_cnn_model_final.keras` - Final model after all epochs

## Troubleshooting

**Model not found error:**
- Run `python train.py` first to create model

**GPU not detected:**
- Edit `setup_gpu.py` or set environment variables

**Out of memory:**
- Reduce batch size in `config.py`
- Reduce image size

## Performance

Typical results:
- Accuracy: 92-95%
- Training time: 15-20 minutes
- Prediction time: < 1 second per image
