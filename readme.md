# RFID CNN Classification Model

This repository contains the CNN implementation for RFID tag identification and localization described in our research paper. It provides a comprehensive solution for identifying RFID tags and determining their spatial characteristics (distance, height, and angle) based on spectral data.

## Project Structure

```
rfid_cnn/
├── data/
│   ├── raw/              # Raw CSV data
│   ├── processed/        # Preprocessed data
│   └── mapping/
│       └── rfid_mapping.json  # Mapping file
├── models/
│   └── saved/            # Trained models
├── src/
│   ├── data/
│   │   ├── data_loader.py     # Data loading functions
│   │   └── preprocessing.py   # Data preprocessing
│   ├── models/
│   │   └── cnn_model.py       # CNN model architecture
│   ├── training/
│   │   ├── train.py           # Model training functions
│   │   ├── evaluate.py        # Model evaluation
│   │   └── srresnet_train.py  # SRResNet training functions
│   └── visualization/
│       └── visualize.py       # Visualization functions
├── scripts/
│   └── predict.py             # Prediction script
├── config.py                  # Global configuration
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rfid-cnn.git
cd rfid-cnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place your RFID spectral CSV files in the `data/raw/` directory
2. Create a mapping file at `data/mapping/rfid_mapping.json` to map tag IDs to class labels

The data will be automatically preprocessed when running the training script.

## Training Models

This implementation supports multiple recognition modes:

### Tag ID Classification
```bash
python main.py --recognition_mode id --visualize
```

### Distance Localization
```bash
python main.py --recognition_mode distance --visualize
```

### Height Estimation
```bash
python main.py --recognition_mode height --visualize
```

### Angle Classification
```bash
python main.py --recognition_mode angle --visualize
```

### Multi-task Recognition
```bash
python main.py --recognition_mode multi --visualize
```

## Model Architecture

The CNN model architecture uses 1D convolutional layers to process RFID spectral data. The model architecture can be configured in `config.py`:

```python
CNN_CONFIG = {
    'input_shape': (2, 1000),  # (channels, frequency points)
    'conv_filters': [32, 64, 128],
    'conv_kernel_sizes': [7, 5, 3],
    'pool_sizes': [4, 4, 2],
    'dropout_rate': 0.3,
    'dense_layers': [128, 64],
}
```

## Prediction

To make predictions with a trained model:

```bash
python scripts/predict.py --file path/to/new_spectrum.csv --model models/saved/rfid_cnn_model.h5 --labels models/saved/class_labels.json
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Ce code implémente un CNN robuste spécifiquement adapté aux données spectrales RFID, avec des fonctionnalités avancées pour gérer la variabilité de distance et l'augmentation de données. L'architecture tire parti des convolutions 1D pour capturer les motifs dans les spectres d'amplitude et de phase.
