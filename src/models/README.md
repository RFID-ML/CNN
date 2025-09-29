# CNN Model for RFID Tag Identification and Localization

This folder contains the Python implementation of the Convolutional Neural Network (CNN) model for RFID tag identification and localization described in our paper. The CNN architecture is designed to extract patterns from RFID spectral data and classify or estimate various parameters.

## Model Architecture

The CNN model architecture consists of:

1. 1D convolutional layers for feature extraction from RFID spectra
2. Batch normalization layers for training stability
3. MaxPooling layers for downsampling
4. Dense layers for classification or regression
5. Task-specific output layers based on the recognition mode

## Recognition Modes

The model supports multiple recognition modes:
- `id`: Tag identification (classification)
- `distance`: Distance estimation (regression or classification)
- `height`: Height estimation (regression or classification)
- `angle`: Angle detection (classification)
- `multi`: Multi-task learning (ID + distance)

## Code Structure

- `cnn_model.py`: Core model architecture definition
- `train.py`: Training functions and utilities
- `evaluate.py`: Evaluation metrics and analysis

## Usage

To instantiate and train the model:

```python
from src.models.cnn_model import build_cnn_model
from src.training.train import train_model

# Build model
model = build_cnn_model(
    input_shape=(2, 1000),  # (channels, frequency points)
    num_classes=10,  # Number of tag classes
    config=CNN_CONFIG,
    recognition_mode='id'
)

# Train model
history, trained_model = train_model(
    X_train, y_train,
    model,
    TRAIN_CONFIG,
    model_save_path='models/saved/cnn_model.h5',
    recognition_mode='id'
)
```

