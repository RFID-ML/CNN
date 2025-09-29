"""
Script to predict RFID tag ID from a spectrum
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config import PREPROCESSING_CONFIG
from src.data.data_loader import load_csv_spectrum
from src.data.preprocessing import resample_spectrum, smooth_spectrum, normalize_spectrum
import json

def load_model_and_labels(model_path, labels_path):
    """
    Load the model and class labels
    
    Args:
        model_path (str): Path to the saved model
        labels_path (str): Path to the labels file
        
    Returns:
        tuple: (model, labels)
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the labels
    with open(labels_path, 'r') as f:
        class_labels = json.load(f)
    
    return model, class_labels

def preprocess_single_spectrum(spectrum_data, config):
    """
    Preprocess a spectrum for prediction
    
    Args:
        spectrum_data (dict): Spectral data
        config (dict): Preprocessing configuration
        
    Returns:
        np.array: Preprocessed spectrum
    """
    target_size = config['num_frequency_points']
    window_size = config['window_size']
    
    # Extract data
    frequencies = spectrum_data['frequencies']
    amplitude = spectrum_data['amplitude']
    phase = spectrum_data['phase']
    
    # Resampling
    amplitude_resampled = resample_spectrum(frequencies, amplitude, target_size)
    phase_resampled = resample_spectrum(frequencies, phase, target_size)
    
    # Smoothing
    amplitude_smoothed = smooth_spectrum(amplitude_resampled, window_size)
    phase_smoothed = smooth_spectrum(phase_resampled, window_size)
    
    # Normalization
    amplitude_normalized = normalize_spectrum(amplitude_smoothed)
    phase_normalized = normalize_spectrum(phase_smoothed)
    
    # Format for CNN: [channels, frequency]
    spectrum_processed = np.stack([amplitude_normalized, phase_normalized], axis=0)
    
    # Add batch dimension
    spectrum_processed = np.expand_dims(spectrum_processed, axis=0)
    
    # Reshape for TensorFlow (batch, points, channels)
    spectrum_processed = np.transpose(spectrum_processed, (0, 2, 1))
    
    return spectrum_processed

def predict_tag_id(file_path, model_path, labels_path):
    """
    Predict tag ID from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        model_path (str): Path to the saved model
        labels_path (str): Path to the labels file
        
    Returns:
        tuple: (Predicted ID, probabilities)
    """
    # Load the model and labels
    model, class_labels = load_model_and_labels(model_path, labels_path)
    
    # Load and preprocess the spectrum
    spectrum_data = load_csv_spectrum(file_path)
    if spectrum_data is None:
        return None, None
    
    processed_spectrum = preprocess_single_spectrum(spectrum_data, PREPROCESSING_CONFIG)
    
    # Make prediction
    predictions = model.predict(processed_spectrum)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_id = class_labels[predicted_class_idx]
    
    return predicted_id, predictions

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Prédire l'ID d'un tag RFID")
    parser.add_argument('--file', type=str, required=True, help='Chemin du fichier CSV')
    parser.add_argument('--model', type=str, required=True, help='Chemin du modèle')
    parser.add_argument('--labels', type=str, required=True, help='Chemin du fichier d\'étiquettes')
    parser.add_argument('--visualize', action='store_true', help='Visualiser le spectre')
    
    args = parser.parse_args()
    
    # Prédire l'ID
    predicted_id, probabilities = predict_tag_id(args.file, args.model, args.labels)
    
    if predicted_id is None:
        print(f"Erreur lors du chargement du fichier {args.file}")
        return
    
    print(f"ID prédit: {predicted_id}")
    
    # Afficher les probabilités pour chaque classe
    model, class_labels = load_model_and_labels(args.model, args.labels)
    for i, cls in enumerate(class_labels):
        print(f"  {cls}: {probabilities[i]:.4f}")
    
    # Visualiser le spectre si demandé
    if args.visualize:
        from src.visualization.visualize import plot_spectrum
        
        spectrum_data = load_csv_spectrum(args.file)
        plot_spectrum(
            spectrum_data['frequencies'],
            spectrum_data['amplitude'],
            spectrum_data['phase'],
            title=f"Spectre prédit comme: {predicted_id}"
        )

if __name__ == "__main__":
    main()