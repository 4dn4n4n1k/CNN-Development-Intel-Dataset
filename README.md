# CNN Development on Custom Dataset

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) using PyTorch to classify images from the Intel Natural Scenes dataset.

The model classifies images into six categories:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Dataset

Intel Natural Scenes Dataset

- Training images: 14,034
- Test images: 3,000
- Number of classes: 6
- Image size: 150 × 150 pixels

## Model Architecture

The custom CNN contains:
- Three convolutional blocks
- Batch normalization and ReLU activation
- Max pooling
- Adaptive average pooling
- Fully connected classification layers
- Dropout regularization

Total parameters: 110,982

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 15 |
| Input size | 150 × 150 |

## Results

- Test accuracy: **82.93%**
- Weighted F1-score: 82.76%
- Model size: 0.43 MB

## Repository Contents

- `CNN_23_50641_1.ipynb`: Complete implementation, execution outputs, and visualizations.
- `intel_custom_cnn.pth`: Trained model checkpoint.

## How to Load the Model

```python
checkpoint = torch.load(
    "intel_custom_cnn.pth",
    map_location=device,
    weights_only=True
)

model = CustomCNN(num_classes=6).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```
