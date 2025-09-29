"""
Main script for training and evaluating the RFID CNN model
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import sys
import logging

# Set matplotlib to non-interactive mode to prevent windows from appearing
plt.switch_backend('Agg')

# Importer les modules du projet
from sklearn.metrics import confusion_matrix
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MAPPING_FILE
from config import TRAIN_CONFIG, CNN_CONFIG, PREPROCESSING_CONFIG, RECOGNITION_MODE

from src.data.data_loader import load_dataset
from src.data.preprocessing import prepare_data_for_training
from src.models.cnn_model import build_cnn_model
from src.training.train import train_model, plot_training_history, train_model_with_presplit
from src.training.evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    analyze_errors,
    evaluate_by_distance
)
from src.visualization.visualize import (
    plot_spectrum, 
    plot_processed_data, 
    plot_performance_by_distance
)

print("Importation des modules terminée")

# Define preprocessed directory
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'pre_processed', 'CNN_filtering')

def create_results_directory(timestamp=None):
    """
    Create a results directory structure for saving outputs
    
    Args:
        timestamp: Optional timestamp string for the directory name
        
    Returns:
        str: Path to the created directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base results directory
    base_results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Create timestamped subdirectory
    results_dir = os.path.join(base_results_dir, f"training_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    subdirs = ['models', 'plots', 'reports', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    
    return results_dir

def setup_logging(results_dir, recognition_mode):
    """
    Set up comprehensive logging to capture all terminal output
    
    Args:
        results_dir: Directory where logs will be saved
        recognition_mode: Current recognition mode for filename
    """
    # Create log filename with timestamp and mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{recognition_mode}_{timestamp}.txt"
    log_path = os.path.join(results_dir, "logs", log_filename)
    
    # Create a custom logger
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

class LoggingPrintWrapper:
    """
    Wrapper to capture all print statements and log them
    """
    def __init__(self, logger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout
        
    def write(self, message):
        if message.strip():  # Only log non-empty messages
            self.logger.info(message.rstrip())
        self.original_stdout.write(message)
        
    def flush(self):
        self.original_stdout.flush()

def ensure_preprocessed_data_exists(raw_dir, preproc_dir=PREPROCESSED_DIR):
    """
    Check if preprocessed files exist, if not run the preprocessing script
    
    Args:
        raw_dir: Path to raw data directory
        preproc_dir: Path to preprocessed data directory
        
    Returns:
        bool: True if preprocessing was performed, False if files already existed
    """
    # Check if the preprocessed directory exists and has CSV files
    if os.path.exists(preproc_dir) and any(f.endswith('.csv') for f in os.listdir(preproc_dir)):
        print(f"Preprocessed files found in {preproc_dir}. Using existing files.")
        return False
    else:
        print(f"No preprocessed files found in {preproc_dir}. Running preprocessing...")
        os.makedirs(preproc_dir, exist_ok=True)
        
        # Run the filtering script with minimal output
        filter_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    'src', 'data', 'filtering_export.py')
        
        try:
            subprocess.run([
                'python', filter_script, 
                '--input', raw_dir,
                '--output', preproc_dir,
                '--no-plots',  # Only generate one sample plot
                '--quiet'  # Use new quiet mode
            ], check=True)
            print(f"Preprocessing complete. Files saved to {preproc_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running preprocessing: {e}")
            raise
            
def analyze_angle_distance_balance(labels, metadata):
    """
    Analyze the balance of angle-distance combinations
    """
    print("\nAnalyzing angle-distance balance:")
    
    # Create distribution matrix
    angle_distance_matrix = {}
    for i, label in enumerate(labels):
        angle = label
        distance = metadata[i].get('distance_cm', 'unknown')
        
        if angle not in angle_distance_matrix:
            angle_distance_matrix[angle] = {}
        if distance not in angle_distance_matrix[angle]:
            angle_distance_matrix[angle][distance] = 0
        angle_distance_matrix[angle][distance] += 1
    
    # Print matrix
    print("Angle-Distance Distribution Matrix:")
    all_distances = sorted(set(meta.get('distance_cm', 'unknown') for meta in metadata))
    
    # Header
    print(f"{'Angle':<8}", end="")
    for dist in all_distances:
        print(f"{dist:<8}", end="")
    print()
    
    # Data rows
    for angle in sorted(angle_distance_matrix.keys()):
        print(f"{angle:<8}", end="")
        for dist in all_distances:
            count = angle_distance_matrix[angle].get(dist, 0)
            print(f"{count:<8}", end="")
        print()
    
    return angle_distance_matrix

def save_training_report(report_path, args, recognition_mode, metrics, class_labels, data_shape, num_samples):
    """
    Save a training report with key information about the training run
    
    Args:
        report_path: Path to save the report
        args: Command line arguments
        recognition_mode: The recognition mode used
        metrics: The evaluation metrics
        class_labels: The class labels
        data_shape: Shape of the processed data
        num_samples: Number of samples in the dataset
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("CNN RFID Training Report\n")
        f.write("=======================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Recognition Mode: {recognition_mode}\n")
        f.write(f"Dataset: {num_samples} samples\n")
        f.write(f"Processed Data Shape: {data_shape}\n\n")
        
        # Write class information
        if recognition_mode == 'angle':
            f.write(f"Angle Classes: {class_labels}\n\n")
        
        # Write metrics
        f.write("Performance Metrics:\n")
        if 'accuracy' in metrics:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        if 'classification_report' in metrics:
            f.write("Classification Report:\n")
            for cls in class_labels:
                if cls in metrics['classification_report']:
                    report = metrics['classification_report'][cls]
                    f.write(f"  Class {cls}:\n")
                    f.write(f"    Precision: {report['precision']:.4f}\n")
                    f.write(f"    Recall: {report['recall']:.4f}\n")
                    f.write(f"    F1-score: {report['f1-score']:.4f}\n")
        
        # Command line arguments
        f.write("\nCommand Line Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
    
    print(f"Training report saved to {report_path}")

def main(args):
    """Fonction principale"""
    # Add this line to declare RECOGNITION_MODE as a global variable
    global RECOGNITION_MODE
    
    print("CNN pour la classification de tags RFID")
    print("======================================\n")
    
    # Create results directory right at the beginning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = create_results_directory(timestamp)
    
    # Setup logging
    logger, log_path = setup_logging(results_dir, args.recognition_mode or RECOGNITION_MODE)
    
    # Redirect print statements to logger
    original_stdout = sys.stdout
    sys.stdout = LoggingPrintWrapper(logger, original_stdout)
    
    # Determine data directory to use
    if args.data_dir:
        # User specified a custom data directory - use it directly
        print(f"Using custom data directory: {args.data_dir}")
        data_dir_to_use = args.data_dir
        did_preprocessing = False
    else:
        # For angle recognition, always use raw data
        if (args.recognition_mode == 'angle'):
            print("Angle recognition: Using raw data directory")
            data_dir_to_use = RAW_DATA_DIR
            did_preprocessing = False
        else:
            # Default behavior for other modes: ensure preprocessed data exists
            data_dir_to_use = RAW_DATA_DIR
            did_preprocessing = ensure_preprocessed_data_exists(data_dir_to_use)
            
            # Use preprocessed data if it exists
            if did_preprocessing or os.path.exists(PREPROCESSED_DIR):
                print("Using preprocessed data for training")
                data_dir_to_use = PREPROCESSED_DIR
      # 1. Charger les données
    print("Chargement des données...")
    data, labels, metadata = load_dataset(
        data_dir=data_dir_to_use,
        mapping_file=args.mapping_file or MAPPING_FILE,
        limit=args.limit,
        recognition_mode=args.recognition_mode or RECOGNITION_MODE
    )
    
    print(f"Nombre d'échantillons chargés: {len(data)}")
    if args.recognition_mode == 'id' or RECOGNITION_MODE == 'id':
        print(f"Classes trouvées: {sorted(list(set(labels)))}")
    elif args.recognition_mode == 'distance' or RECOGNITION_MODE == 'distance':
        print(f"Distances trouvées: {sorted(list(set(labels)))}")
    elif args.recognition_mode == 'height' or RECOGNITION_MODE == 'height':
        print(f"Hauteurs trouvées: {sorted(list(set(labels)))}")
        print("FORCING CLASSIFICATION MODE FOR HEIGHT RECOGNITION")
        
        # ALWAYS force height to use classification mode - no regression allowed
        recognition_mode = 'height_class'
        args.recognition_mode = 'height_class'
        RECOGNITION_MODE = 'height_class'
        
        # Convert heights to classification labels using LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        original_heights = sorted(list(set(labels)))
        labels = label_encoder.fit_transform(labels)
        
        print(f"Height values converted to class indices:")
        for i, h in enumerate(original_heights):
            print(f"  Height {h} → Class {i}")
        
        print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # Import HEIGHT_CLASS_NAMES for consistency
        from config import HEIGHT_CLASS_NAMES
    elif args.recognition_mode == 'height_class':
        from config import HEIGHT_CLASS_NAMES
        unique_labels = sorted(list(set([l for l in labels if l is not None])))
        print(f"Classes de hauteur: 2 classes (h1, h2)")
        print(f"Classes trouvées: {unique_labels}")
        print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"Noms des classes: {HEIGHT_CLASS_NAMES}")
    elif args.recognition_mode == 'multi' or RECOGNITION_MODE == 'multi':
        id_values = [label[0] for label in labels]
        distance_values = [label[1] for label in labels]
        print(f"Classes trouvées: {sorted(list(set(id_values)))}")
        print(f"Distances trouvées: {sorted(list(set(distance_values)))}")
    elif args.recognition_mode == 'angle' or RECOGNITION_MODE == 'angle':
        print(f"Angles trouvés: {sorted(list(set(labels)))}")
        # Show distribution
        angle_counts = {}
        for label in labels:
            angle_counts[label] = angle_counts.get(label, 0) + 1
        print(f"Distribution des angles: {angle_counts}")
        
        # Add detailed balance analysis for angle mode
        analyze_angle_distance_balance(labels, metadata)
    print()
      # 2. Prétraiter les données
    print("Prétraitement des données...")
    
    # Special handling for angle recognition
    if (args.recognition_mode == 'angle' or RECOGNITION_MODE == 'angle'):
        print("Mode reconnaissance d'angle activé")
        print("Utilisation des fichiers bruts sans prétraitement")
        
        # Override preprocessing config for angle mode
        angle_config = {
            'normalize': True,
            'filter_enabled': False,  # No filtering for angle data
            'target_points': None,    # Use original point count
            'use_raw_files': True
        }
        
        # Use angle-specific preprocessing
        X_preprocessed, y_encoded, class_labels = prepare_data_for_training(
            data, labels, angle_config, recognition_mode='angle'
        )
    else:
        # Normal preprocessing for other modes
        X_preprocessed, y_encoded, class_labels = prepare_data_for_training(
            data, labels, PREPROCESSING_CONFIG, recognition_mode=args.recognition_mode or RECOGNITION_MODE
        )
    
    print(f"Forme des données prétraitées: {X_preprocessed.shape}")
    if args.recognition_mode == 'multi' or RECOGNITION_MODE == 'multi':
        print(f"Nombre de classes ID: {len(class_labels[0])}")
        if isinstance(class_labels[1], tuple):
            print(f"Plage de distances: {class_labels[1]}")
        else:
            print(f"Nombre de classes distance: {len(class_labels[1])}")
    elif args.recognition_mode == 'height_class' or RECOGNITION_MODE == 'height_class':
        print(f"Mode classification de hauteur: 2 classes (h1, h2)")
        print(f"Classes encodées: {sorted(list(set(y_encoded)))}")
    else:
        if isinstance(class_labels, tuple):
            print(f"Plage de distances: {class_labels}")
        elif class_labels is not None:
            print(f"Nombre de classes: {len(class_labels)}")
        else:
            print("Classes: données préparées pour classification")
    print()
    
    # 3. Visualiser quelques exemples (si demandé)
    if args.visualize:
        print("Visualisation des échantillons avec les étapes de prétraitement...")
        # Échantillon aléatoire
        sample_idx = np.random.randint(len(data))
        sample_data = data[sample_idx]
        
        # Create directories for saving visualization
        vis_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Utiliser la nouvelle fonction qui montre toutes les étapes
        from src.visualization.visualize import plot_spectrum_processing_stages
        plot_spectrum_processing_stages(
            sample_data,
            title=f"Tag ID: {labels[sample_idx]} - Processing Stages",
            save_path=os.path.join(vis_dir, f"sample_{sample_idx}_processing.png")
        )
        
        # Conserver l'affichage du résultat prétraité pour le CNN
        print("Visualisation de l'échantillon prétraité pour le CNN:")
        plot_processed_data(
            X_preprocessed, 
            sample_idx,
            title=f"Tag ID: {labels[sample_idx]} (preprocessed for CNN)",
            save_path=os.path.join(vis_dir, f"sample_{sample_idx}_preprocessed.png")
        )
        
        print(f"Sample visualizations saved to: {vis_dir}")
    
    # Visualiser tous les échantillons si demandé
    if args.visualize_all:
        print(f"Visualisation de tous les échantillons (max: {args.max_samples if args.max_samples else 'tous'})...")
        
        # Créer un dossier pour les visualisations
        visualization_dir = os.path.join("visualization_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Déterminer le nombre d'échantillons à traiter
        num_samples = len(data)
        if args.max_samples and args.max_samples < num_samples:
            num_samples = args.max_samples
            
        print(f"Traitement de {num_samples} échantillons...")
        
        # Traiter chaque échantillon
        from src.visualization.visualize import plot_spectrum_processing_stages
        for i in range(num_samples):
            sample_data = data[i]
            tag_id = labels[i]
            filename = metadata[i].get('file_name', f"sample_{i}")
            
            # Retirer l'extension .csv si présente
            if filename.endswith('.csv'):
                base_filename = filename[:-4]
            else:
                base_filename = filename
                
            # Visualiser les étapes de traitement
            save_path = os.path.join(visualization_dir, f"{base_filename}_processing.png")
            
            # Utiliser le paramètre save_path pour éviter d'afficher le graphique à l'écran
            plot_spectrum_processing_stages(
                sample_data,
                title=f"Tag ID: {tag_id} - Processing Stages",
                save_path=save_path
            )
            
            if (i+1) % 10 == 0 or i == num_samples-1:
                print(f"Progression: {i+1}/{num_samples} échantillons traités")
        
        print(f"Visualisations sauvegardées dans: {visualization_dir}")
      # 4. Construire le modèle
    print("Construction du modèle CNN...")
    input_shape = X_preprocessed.shape[1:]  # (canaux, points)
    
    recognition_mode = args.recognition_mode or RECOGNITION_MODE
    
    # Force height to always be classification
    if recognition_mode == 'height':
        recognition_mode = 'height_class'
        
    if recognition_mode == 'height_class':
        from config import HEIGHT_CLASS_NAMES
        model_classes = len(HEIGHT_CLASS_NAMES)
        class_labels = HEIGHT_CLASS_NAMES
        print(f"Configuration forcée: {model_classes} classes de hauteur pour classification")
    
    # For height_class mode, we need to pass the number of classes (2)
    if recognition_mode == 'height_class':
        model_classes = 2  # Binary classification: h1 vs h2
    else:
        model_classes = class_labels
    
    model = build_cnn_model(input_shape, model_classes, CNN_CONFIG, recognition_mode=recognition_mode)
    model.summary()
    
    # 5. Entraîner le modèle (si demandé)
    if not args.skip_training:
        print("\nEntraînement du modèle...")
        
        # Créer un nom de modèle unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(results_dir, "models", f"cnn_model_{recognition_mode}_{timestamp}.h5")
        
        # Special handling for angle recognition to ensure balanced splits
        if recognition_mode == 'angle':
            print("Angle recognition: Creating balanced train/test split")
            
            # Create stratification keys combining angle and distance
            stratification_keys = []
            for i, meta in enumerate(metadata):
                angle_label = labels[i]
                distance_cm = meta.get('distance_cm', 'unknown')
                # Create a compound key for stratification
                strat_key = f"{angle_label}_{distance_cm}"
                stratification_keys.append(strat_key)
            
            # Check distribution of stratification keys
            from collections import Counter
            strat_counts = Counter(stratification_keys)
            print(f"Stratification key distribution: {dict(strat_counts)}")
            
            # Only use stratification keys that appear multiple times
            valid_strat_keys = [key for key, count in strat_counts.items() if count >= 2]
            
            if len(valid_strat_keys) > 1:
                # Filter data to only include valid stratification keys
                valid_indices = [i for i, key in enumerate(stratification_keys) if key in valid_strat_keys]
                
                if len(valid_indices) >= 10:  # Minimum samples for meaningful split
                    X_filtered = X_preprocessed[valid_indices]
                    y_filtered = y_encoded[valid_indices]
                    strat_filtered = [stratification_keys[i] for i in valid_indices]
                    
                    print(f"Using {len(valid_indices)} samples with balanced angle-distance combinations")
                    
                    # Perform stratified split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_filtered, y_filtered,
                        test_size=TRAIN_CONFIG['test_size'],
                        random_state=TRAIN_CONFIG['random_state'],
                        stratify=strat_filtered
                    )
                    
                    # Print detailed split statistics
                    print("\nBalanced split statistics:")
                    print(f"Train set: {len(X_train)} samples")
                    print(f"Test set: {len(X_test)} samples")
                    
                    # Analyze distributions
                    train_strat = [strat_filtered[i] for i in range(len(y_train))]
                    test_strat = [strat_filtered[i] for i in range(len(y_train), len(strat_filtered))]
                    
                    train_dist = Counter(train_strat)
                    test_dist = Counter(test_strat)
                    
                    print(f"Train set distribution: {dict(train_dist)}")
                    print(f"Test set distribution: {dict(test_dist)}")
                    
                    # Use TensorFlow/Keras training directly instead of train_model_with_presplit
                    from src.models.cnn_model import get_callbacks
                    
                    # Get callbacks
                    callbacks = get_callbacks(model_save_path, patience=TRAIN_CONFIG.get('patience', 10))
                    
                    # Train the model with pre-split data
                    print("\nTraining with balanced pre-split data...")
                    history = model.fit(
                        X_train, y_train,
                        batch_size=TRAIN_CONFIG['batch_size'],
                        epochs=TRAIN_CONFIG['epochs'],
                        validation_split=TRAIN_CONFIG['validation_split'],
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Load best model
                    trained_model = tf.keras.models.load_model(model_save_path)
                    
                else:
                    print("Not enough samples for balanced split, using standard training")
                    history, trained_model, (X_test, y_test) = train_model(
                        X_preprocessed, y_encoded, model, TRAIN_CONFIG, model_save_path, recognition_mode=recognition_mode
                    )
            else:
                print("Cannot create balanced split, using standard training")
                history, trained_model, (X_test, y_test) = train_model(
                    X_preprocessed, y_encoded, model, TRAIN_CONFIG, model_save_path, recognition_mode=recognition_mode
                )
        else:
            # Standard training for other modes
            history, trained_model, (X_test, y_test) = train_model(
                X_preprocessed, y_encoded, model, TRAIN_CONFIG, model_save_path, recognition_mode=recognition_mode
            )
        
        # ✅ FIX: Evaluate model first to get metrics
        print(f"\nÉvaluation finale sur {len(X_test)} échantillons de test")
        
        # Evaluate model (X_test, y_test already defined above)
        metrics = evaluate_model(trained_model, X_test, y_test, class_labels, recognition_mode=recognition_mode)
        
        # Update plot saving paths
        if args.visualize:
            # Save training history plot
            history_plot_path = os.path.join(results_dir, "plots", f"training_history_{recognition_mode}_{timestamp}.png")
            plot_training_history(history, save_path=history_plot_path)
            
            # Save confusion matrix
            if 'confusion_matrix' in metrics:
                cm_plot_path = os.path.join(results_dir, "plots", f"confusion_matrix_{recognition_mode}_{timestamp}.png")
                plot_confusion_matrix(metrics['confusion_matrix'], class_labels, save_path=cm_plot_path)
        
        # Save detailed report (now metrics is defined)
        report_path = os.path.join(results_dir, "reports", f"training_report_{recognition_mode}_{timestamp}.txt")
        try:
            save_training_report(report_path, args, recognition_mode, metrics, class_labels, X_preprocessed.shape, len(data))
        except NameError:
            print("Note: save_training_report function is not defined, skipping report generation")
        
        # Afficher les résultats
        if recognition_mode == 'height_class':
            # Height classification (binary)
            from config import HEIGHT_CLASS_NAMES
            print("\nRapport de classification Height:")
            print(f"Overall accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
            if 'classification_report' in metrics:
                print(f"Report keys: {metrics['classification_report'].keys()}")
                
                # Handle each class
                for i, class_name in enumerate(HEIGHT_CLASS_NAMES):
                    class_key = class_name  # Try class name first
                    if class_key not in metrics['classification_report']:
                        class_key = str(i)  # Try string index
                    if class_key not in metrics['classification_report']:
                        class_key = i  # Try int index
                        
                    if class_key in metrics['classification_report']:
                        report = metrics['classification_report'][class_key]
                        print(f"Classe {class_name}:")
                        print(f"  Précision: {report['precision']:.4f}")
                        print(f"  Rappel: {report['recall']:.4f}")
                        print(f"  F1-score: {report['f1-score']:.4f}")
                
                # Also try macro avg and weighted avg
                for avg_type in ['macro avg', 'weighted avg']:
                    if avg_type in metrics['classification_report']:
                        report = metrics['classification_report'][avg_type]
                        print(f"{avg_type}:")
                        print(f"  Précision: {report['precision']:.4f}")
                        print(f"  Rappel: {report['recall']:.4f}")
                        print(f"  F1-score: {report['f1-score']:.4f}")

            # Plot confusion matrix for height classification
            if 'confusion_matrix' in metrics:
                plot_confusion_matrix(
                    metrics['confusion_matrix'], 
                    HEIGHT_CLASS_NAMES,
                    save_path=os.path.join(results_dir, 'height_confusion_matrix.png')
                )
                print(f"Confusion matrix saved: {os.path.join(results_dir, 'height_confusion_matrix.png')}")                
        elif recognition_mode == 'id' or (recognition_mode == 'height_class' or (recognition_mode == 'distance' and not isinstance(class_labels, tuple))):
            # Classification metrics (including height_class)
            print("\nMétriques de classification:")
            print(f"  Précision: {metrics['accuracy']:.4f}")
            
            # Print detailed classification report
            if 'classification_report' in metrics:
                print("Rapport de classification détaillé:")
                for cls in class_labels:
                    if cls in metrics['classification_report']:
                        report = metrics['classification_report'][cls]
                        print(f"  Classe {cls}:")
                        print(f"    Précision: {report['precision']:.4f}")
                        print(f"    Rappel: {report['recall']:.4f}")
                        print(f"    F1-score: {report['f1-score']:.4f}")
            
            # Tracer la matrice de confusion
            plot_confusion_matrix(
                metrics['confusion_matrix'], 
                class_labels,
                save_path=os.path.join(results_dir, 'confusion_matrix.png')
            )
        elif recognition_mode in ['distance'] and isinstance(class_labels, tuple):
            # Régression
            mode_name = "distance" if recognition_mode == 'distance' else "height"
            print(f"\nMétriques de régression ({mode_name}):")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")
            
            # Plot regression results
            from src.visualization.visualize import plot_regression_results
            title = f"{mode_name.title()} Regression Results"
            plot_regression_results(
                metrics['true_values'],  # Use denormalized true values
                metrics['predictions'],  # Use denormalized predictions
                title=title,
                save_path=os.path.join(results_dir, f'{mode_name}_regression_results.png')
            )
        elif recognition_mode == 'multi':
            # Multi-task results
            # ID results
            print("\nRapport de classification ID:")
            for cls in class_labels[0]:
                if cls in metrics['id_classification_report']:
                    report = metrics['id_classification_report'][cls]
                    print(f"Classe {cls}:")
                    print(f"  Précision: {report['precision']:.4f}")
                    print(f"  Rappel: {report['recall']:.4f}")
                    print(f"  F1-score: {report['f1-score']:.4f}")
            
            # Plot ID confusion matrix
            plot_confusion_matrix(
                metrics['id_confusion_matrix'], 
                class_labels[0],
                save_path=os.path.join(results_dir, 'id_confusion_matrix.png')
            )
            
            # Distance results (either classification or regression)
            if 'distance_classification_report' in metrics:
                print("\nRapport de classification Distance:")
                for cls in class_labels[1]:
                    cls_str = str(cls)
                    if cls_str in metrics['distance_classification_report']:
                        report = metrics['distance_classification_report'][cls_str]
                        print(f"Distance {cls}cm:")
                        print(f"  Précision: {report['precision']:.4f}")
                        print(f"  Rappel: {report['recall']:.4f}")
                        print(f"  F1-score: {report['f1-score']:.4f}")
                
                # Plot distance confusion matrix
                plot_confusion_matrix(
                    metrics['distance_confusion_matrix'], 
                    [str(d) for d in class_labels[1]],
                    save_path=os.path.join(results_dir, 'distance_confusion_matrix.png')
                )
            else:
                # Regression metrics for distance
                print("\nMétriques de régression Distance:")
                print(f"  MAE: {metrics['distance_mae']:.4f}")
                print(f"  MSE: {metrics['distance_mse']:.4f}")

        # Analyze dataset class distribution
        from src.visualization.visualize import plot_class_distribution
        plot_class_distribution(
            labels,
            metadata,
            save_path=os.path.join(results_dir, 'class_distribution.png'),
            recognition_mode=recognition_mode
        )        # Analyser les erreurs
        if recognition_mode == 'distance' and isinstance(class_labels, tuple) and len(class_labels) == 2:
            # For distance regression, we have (min_dist, max_dist) as class_labels
            # We need to pass the evaluation results properly
            if 'true_values' in metrics and 'predictions' in metrics:
                analyze_errors(data, metrics['true_values'], metrics['predictions'], class_labels, metadata)
            else:
                analyze_errors(data, y_test, metrics['predictions'], class_labels, metadata)
        elif recognition_mode == 'height_class':
            # For height classification, use class names  
            from config import HEIGHT_CLASS_NAMES
            print("Height classification error analysis:")
            print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
            # Skip detailed error analysis for now
        elif recognition_mode == 'multi':
            # For multi-task, we analyze ID classification errors
            analyze_errors(data, y_test['id_output'], metrics['id_predictions'], class_labels[0], metadata)
        elif recognition_mode in ['height']:
            # Skip error analysis for regression modes without proper class labels
            print("Skipping error analysis for regression mode")
        else:
            # Standard classification
            if class_labels is not None:
                analyze_errors(data, y_test, metrics['predictions'], class_labels, metadata)        # Évaluer par distance
        if any('distance_cm' in meta for meta in metadata) and recognition_mode in ['distance', 'multi']:
            print("\nÉvaluation par distance:")
            test_metadata = [metadata[i] for i in range(len(y_test))]
            
            # For distance regression, use denormalized values if available
            if recognition_mode == 'distance' and isinstance(class_labels, tuple) and len(class_labels) == 2:
                if 'true_values' in metrics and 'predictions' in metrics:
                    # Use the denormalized values from the evaluation
                    distance_results = evaluate_by_distance(
                        metrics['true_values'], 
                        metrics['predictions'], 
                        test_metadata,
                        class_labels
                    )
                else:
                    # Fallback to original values
                    distance_results = evaluate_by_distance(
                        y_test, 
                        metrics['predictions'], 
                        test_metadata,
                        class_labels
                    )
            else:
                # For classification modes, use original values
                distance_results = evaluate_by_distance(
                    y_test, 
                    metrics['predictions'], 
                    test_metadata,
                    class_labels
                )
              # Print distance results
            print("\nPERFORMANCE BY DISTANCE:")
            is_regression = recognition_mode == 'distance' and isinstance(class_labels, tuple) and len(class_labels) == 2
            
            for distance, metrics in distance_results.items():
                print(f"\nDistance: {distance}cm, Samples: {metrics['sample_count']}")
                
                if is_regression:
                    # Print regression metrics
                    print(f"  MAE: {metrics['mae']:.4f}")
                    print(f"  RMSE: {metrics['rmse']:.4f}")
                    print(f"  R²: {metrics['r2']:.4f}")
                    print(f"  Mean True: {metrics['mean_true']:.2f}")
                    print(f"  Mean Pred: {metrics['mean_pred']:.2f}")
                else:
                    # Print classification metrics
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    if 'tag_distribution' in metrics:
                        print(f"  Tag Distribution: {metrics['tag_distribution']}")
            
            # Plot all distance performance metrics with multiple visualizations
            plot_performance_by_distance(
                distance_results,
                save_path=os.path.join(results_dir, 'distance_performance.png')
            )
            
            print(f"\nResults and visualizations saved in: {results_dir}")
        
        # Save class labels with the model
        labels_save_path = os.path.splitext(model_save_path)[0] + '_labels.npy'
        np.save(labels_save_path, class_labels)
        print(f"Class labels saved to {labels_save_path}")
    
    # Manual visualization for angle recognition
    if recognition_mode == 'angle' and not args.skip_training:
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Save training history plot
        history_plot_path = os.path.join(plots_dir, 'training_history.png')
        if history is not None and hasattr(history, 'history'):
            # Plot training history
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(history_plot_path)
            plt.close()
            print(f"Training history plot saved to: {history_plot_path}")
        
        # 2. Generate confusion matrix even if not in metrics
        cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
        if y_test is not None:
            # Get predictions if not already in metrics
            if 'predictions' not in metrics and trained_model is not None:
                predictions = trained_model.predict(X_test)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    pred_classes = np.argmax(predictions, axis=1)
                else:
                    pred_classes = (predictions > 0.5).astype(int)
            else:
                pred_classes = metrics.get('predictions', [])
            
            if len(pred_classes) > 0:
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, pred_classes)
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Angle Classification Confusion Matrix')
                plt.colorbar()
                
                # Set tick marks
                tick_marks = np.arange(len(class_labels))
                plt.xticks(tick_marks, class_labels, rotation=45)
                plt.yticks(tick_marks, class_labels)
                
                # Add text annotations
                thresh = cm.max() / 2
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True Angle')
                plt.xlabel('Predicted Angle')
                plt.savefig(cm_path)
                plt.close()
                print(f"Confusion matrix saved to: {cm_path}")
    
    print("\nTerminé!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification CNN de tags RFID")
    parser.add_argument('--data_dir', type=str, help='Répertoire des données brutes')
    parser.add_argument('--mapping_file', type=str, help='Fichier de mapping')
    parser.add_argument('--limit', type=int, help='Limite du nombre de fichiers à charger')
    parser.add_argument('--visualize', action='store_true', help='Visualiser les échantillons')
    parser.add_argument('--skip_training', action='store_true', help='Sauter l\'entraînement')
    parser.add_argument('--visualize-all', action='store_true', 
                      help='Visualiser les étapes de prétraitement pour tous les échantillons')
    parser.add_argument('--max-samples', type=int, default=None,
                      help='Nombre maximum d\'échantillons à visualiser (pour --visualize-all)')
    parser.add_argument('--recognition_mode', type=str, choices=['id', 'distance', 'height', 'height_class', 'multi', 'srresnet', 'angle'],
                      help='Mode de reconnaissance (id, distance, height, height_class, multi, srresnet, ou angle)')
    parser.add_argument('--srresnet', action='store_true',
                      help='Utiliser le mode SRRESNET (ajoute des données SRRESNET à l\'entraînement)')
    
    args = parser.parse_args()
    
    # Handle SRRESNET mode
    if args.srresnet or args.recognition_mode == 'srresnet':
        from src.training.srresnet_train import main_srresnet_training
        main_srresnet_training()
    else:
        main(args)