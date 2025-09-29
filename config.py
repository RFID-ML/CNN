"""
Global configuration for the CNN-RFID project
"""

import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
MAPPING_FILE = os.path.join(DATA_DIR, 'mapping', 'rfid_mapping.json')

# Recognition mode
RECOGNITION_MODE = 'id'  # Options: 'id', 'distance', 'multi', 'height', 'height_class', 'srresnet'
VALID_RECOGNITION_MODES = ['id', 'distance', 'height', 'height_class', 'multi', 'angle']  # Includes all supported modes

# Height classification (academically valid approach for 2 discrete heights)
HEIGHT_CLASSES = {
    'h1': 0,  # Class 0: 4.0 cm
    'h2': 1   # Class 1: 11.5 cm
}
HEIGHT_CLASS_NAMES = ['h1 (4.0cm)', 'h2 (11.5cm)']

# Add angle class names
ANGLE_CLASS_NAMES = ['-30', '0', '30']  # The three angles in degrees as strings

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 os.path.dirname(MAPPING_FILE)]:
    os.makedirs(directory, exist_ok=True)

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
    'test_size': 0.20,
    'random_state': 42
}

# CNN model parameters
CNN_CONFIG = {
    'input_shape': (2, 1000),  # (channels, frequency points)
    'conv_filters': [32, 64, 128],
    'conv_kernel_sizes': [7, 5, 3],
    'pool_sizes': [4, 4, 2],
    'dropout_rate': 0.3,
    'dense_layers': [128, 64],
}

# Distance range for RFID measurements (hardcoded based on naming convention)
DISTANCE_RANGE = (1.0, 49.0)  # Valid distance range in cm

# Height range for RFID measurements (hardcoded based on naming convention)
HEIGHT_RANGE = (4.0, 11.5)  # Valid height range in cm (h1=4cm, h2=11.5cm)

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    'num_frequency_points': 1000,
    'normalize': True,
    'window_size': 5,
    'data_augmentation': False,
    'augmentation_factor': 1,  # Augmentation factor
    'freq_shift_range': 0.003,  # Frequency shift range for augmentation
    'noise_level': 0.015,       # Noise level for augmentation
    'apply_post_split': True    # Apply augmentation after train/test split
}

# SRRESNET Integration Configuration
SRRESNET_CONFIG = {
    'base_dir': os.path.join(DATA_DIR, 'pre_processed', 'SRRESNET_ID'),
    'train_environments': ['ENV1', 'ENV2', 'ENV3'],
    'test_environments': ['ENV1', 'ENV2', 'ENV3'],
    'use_for_training': True,
    'use_separate_test_set': True
}

# Angle-specific paths
ANGLE_DATA_DIR = os.path.join(RAW_DATA_DIR, 'angle_data')  # Directory for angle data