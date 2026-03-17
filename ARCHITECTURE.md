# Model Architecture

## Overview

Deep Convolutional Neural Network để phân loại ảnh chó và mèo.

## Layers

### Input
- Shape: (180, 180, 3)
- Rescaling: 1/255

### Convolutional Blocks

```
ConvBlock 1: 32 filters
  - Conv2D (3x3, HeNormal init)
  - Batch Normalization
  - ReLU
  - Max Pooling (2x2)
  
ConvBlock 2: 64 filters
  - Conv2D (3x3, HeNormal init)
  - Batch Normalization
  - ReLU
  - Max Pooling (2x2)
  
ConvBlock 3: 128 filters
  - Conv2D (3x3, HeNormal init)
  - Batch Normalization
  - ReLU
  - Max Pooling (2x2)
  
ConvBlock 4: 256 filters
  - Conv2D (3x3, HeNormal init)
  - Batch Normalization
  - ReLU
  - Max Pooling (2x2)

After Conv: Dropout (rate=0.25)
```

### Dense Layers

```
Flatten (30,976 features)

DenseBlock 1:
  - 512 units
  - ReLU
  - Dropout (rate=0.5)

DenseBlock 2:
  - 256 units
  - ReLU
  - Dropout (rate=0.4)

DenseBlock 3:
  - 128 units
  - ReLU
  - Dropout (rate=0.3)

Output:
  - 1 unit
  - Sigmoid (binary classification)
```

## Parameters

| Component | Count | Percentage |
|-----------|-------|-----------|
| Conv layers | 389,376 | 2.4% |
| Dense layers | 16,024,577 | 97.6% |
| **Total** | **16,413,953** | **100%** |

## Key Features

1. **Batch Normalization (Custom)**
   - Exponential moving average for running statistics
   - Different behavior for training vs inference
   - Stabilizes training and accelerates convergence

2. **Dropout Regularization**
   - Inverted Dropout implementation
   - Integrated into Dense layers
   - Reduces overfitting

3. **HeNormal Initialization**
   - Variance-preserving for ReLU networks
   - Faster convergence than random normal
   - Critical for deep networks

4. **Training Mode Support**
   - `model(x, training=True)` - Batch Norm active, Dropout enabled
   - `model(x, training=False)` - Batch Norm uses running stats, Dropout off

## Model Summary

- Total parameters: 16.4M
- Input shape: (None, 180, 180, 3)
- Output shape: (None, 1)
- Output range: [0, 1]
  - < 0.5: Cat
  - > 0.5: Dog

## Training Configuration

- Optimizer: Adam (learning_rate=5e-4)
- Loss: Binary Crossentropy
- Metrics: Binary Accuracy
- Batch size: 32
- Epochs: 100 (with early stopping)
- Early stopping patience: 10

## Expected Performance

- Accuracy: 92-95%
- Training time: 15-20 minutes
