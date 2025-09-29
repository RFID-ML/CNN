"""
Training functions for the RFID CNN classification model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train_model(X, y, model, config, model_save_path, recognition_mode='id'):
    """
    Train the CNN model on preprocessed data
    
    Args:
        X (np.array): Preprocessed data
        y (np.array or tuple): Encoded labels (can be a tuple for multi-task)
        model (tf.keras.Model): Model to train
        config (dict): Training configuration
        model_save_path (str): Path to save the model
        recognition_mode (str): Recognition mode ('id', 'distance', 'height', 'angle', or 'multi')
        
    Returns:
        tuple: (training history, trained model, test data)
    """
    from src.models.cnn_model import get_callbacks
    
    # Step 1: Clean train/test split FIRST (before any augmentation)
    print("Performing clean train/test split...")
    
    if recognition_mode == 'angle':
        # Special balanced splitting for angle recognition mode
        print("Using balanced train/test split for angle recognition mode")
        
        # We need access to metadata to balance by both angle and distance
        # This requires modification to pass metadata through the pipeline
        # For now, we'll use stratified split on angle labels only
        if len(set(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config['test_size'],
                random_state=config['random_state'],
                stratify=y  # Stratify by angle to ensure balanced representation
            )
        else:
            # Fallback if only one class
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config['test_size'],
                random_state=config['random_state']
            )
        
        # Print split statistics for angle mode
        print(f"Train set angle distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test set angle distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    elif recognition_mode == 'multi':
        y_id, y_distance = y
        if len(set(y_id)) > 1:
            X_train, X_test, y_id_train, y_id_test, y_dist_train, y_dist_test = train_test_split(
                X, y_id, y_distance,
                test_size=config['test_size'],
                random_state=config['random_state'],
                stratify=y_id  # Stratify on ID
            )
        else:
            X_train, X_test, y_id_train, y_id_test, y_dist_train, y_dist_test = train_test_split(
                X, y_id, y_distance,
                test_size=config['test_size'], 
                random_state=config['random_state']
            )
        
        y_train = {'id_output': y_id_train, 'distance_output': y_dist_train}
        y_test = {'id_output': y_id_test, 'distance_output': y_dist_test}
        
    else:
        # Standard split with stratification for classification
        if recognition_mode in ['id', 'height_class'] or (recognition_mode == 'distance' and len(set(y)) <= 20):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config['test_size'],
                random_state=config['random_state'],
                stratify=y  # Stratified for balanced classes
            )
        elif recognition_mode == 'distance' and len(set(y)) > 20:
            # Distance regression - create balanced split by distance
            print("Creating balanced train/test split for distance regression...")
            
            # Step 1: Denormalize distances if they were normalized
            if min(y) >= 0 and max(y) <= 1:
                # Assume these are normalized in [0,1] range from the 1-49cm original range
                from config import DISTANCE_RANGE
                min_dist, max_dist = DISTANCE_RANGE
                dist_range = max_dist - min_dist
                # Denormalize to get the actual distances
                distances = np.round(y * dist_range + min_dist).astype(int)
            else:
                # Distances are already in cm
                distances = np.round(y).astype(int)
                
            # Step 2: Group samples by distance
            distance_indices = {}
            for i, dist in enumerate(distances):
                if dist not in distance_indices:
                    distance_indices[dist] = []
                distance_indices[dist].append(i)
            
            # Step 3: Calculate how many samples to take from each distance for test set
            test_size = int(config['test_size'] * len(y))
            unique_distances = sorted(distance_indices.keys())
            n_distances = len(unique_distances)
            
            # Calculate samples per distance (roughly equal)
            samples_per_distance = max(1, test_size // n_distances)
            print(f"Balancing test set: ~{samples_per_distance} samples per distance value")
            
            # Step 4: Create balanced test indices
            test_indices = []
            for dist in unique_distances:
                # Get available indices for this distance
                avail_indices = distance_indices[dist]
                
                # Take up to samples_per_distance indices, or all if fewer are available
                n_samples = min(samples_per_distance, len(avail_indices))
                
                # Randomly select indices without replacement
                import random
                random.seed(config['random_state'])
                selected_indices = random.sample(avail_indices, n_samples)
                test_indices.extend(selected_indices)
            
            # Step 5: Create train indices (all indices not in test)
            all_indices = set(range(len(y)))
            train_indices = list(all_indices - set(test_indices))
            
            # Step 6: Create train/test split using these indices
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            # Print statistics
            print(f"Created balanced split: {len(X_train)} train, {len(X_test)} test samples")
            
            # Print distribution summary
            train_counts = {}
            test_counts = {}
            for dist in unique_distances:
                train_count = sum(1 for i in train_indices if distances[i] == dist)
                test_count = sum(1 for i in test_indices if distances[i] == dist)
                train_counts[dist] = train_count
                test_counts[dist] = test_count
            
            print(f"Distance distribution samples - Training min: {min(train_counts.values())}, max: {max(train_counts.values())}")
            print(f"Distance distribution samples - Test min: {min(test_counts.values())}, max: {max(test_counts.values())}")
        else:
            # Regression - no stratification for other regression modes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config['test_size'],
                random_state=config['random_state']
            )
    
    print(f"Data split: {len(X_train)} training, {len(X_test)} test samples")
    
    # Step 2: For distance regression mode, create a balanced validation split too
    if recognition_mode == 'distance' and len(set(y)) > 20:
        print("Creating balanced validation split for distance regression...")
        
        # Get the validation size from the config
        validation_size = config['validation_split']
        
        # Same approach as test split, but apply to training data
        if min(y_train) >= 0 and max(y_train) <= 1:
            # Denormalize to get actual distances
            from config import DISTANCE_RANGE
            min_dist, max_dist = DISTANCE_RANGE
            dist_range = max_dist - min_dist
            train_distances = np.round(y_train * dist_range + min_dist).astype(int)
        else:
            train_distances = np.round(y_train).astype(int)
        
        # Group by distance
        distance_indices = {}
        for i, dist in enumerate(train_distances):
            if dist not in distance_indices:
                distance_indices[dist] = []
            distance_indices[dist].append(i)
        
        # Calculate samples per distance for validation
        val_size = int(validation_size * len(y_train))
        unique_distances = sorted(distance_indices.keys())
        n_distances = len(unique_distances)
        samples_per_distance = max(1, val_size // n_distances)
        
        print(f"Balancing validation set: ~{samples_per_distance} samples per distance value")
        
        # Create balanced validation indices
        val_indices = []
        for dist in unique_distances:
            avail_indices = distance_indices[dist]
            n_samples = min(samples_per_distance, len(avail_indices))
            
            import random
            random.seed(config['random_state'])
            selected_indices = random.sample(avail_indices, n_samples)
            val_indices.extend(selected_indices)
        
        # Create actual training indices (all train indices not in validation)
        all_train_indices = set(range(len(y_train)))
        actual_train_indices = list(all_train_indices - set(val_indices))
        
        # Split the training data
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[actual_train_indices]
        y_train = y_train[actual_train_indices]
        
        # Print statistics
        print(f"Created balanced validation split: {len(X_train)} actual train, {len(X_val)} validation samples")
        
        # Create counts for printing
        val_counts = {}
        actual_train_counts = {}
        for dist in unique_distances:
            val_count = sum(1 for i in val_indices if train_distances[i] == dist)
            train_count = sum(1 for i in actual_train_indices if train_distances[i] == dist)
            val_counts[dist] = val_count
            actual_train_counts[dist] = train_count
        
        print(f"Distance distribution - Actual train min: {min(actual_train_counts.values())}, max: {max(actual_train_counts.values())}")
        print(f"Distance distribution - Validation min: {min(val_counts.values())}, max: {max(val_counts.values())}")
        
        # Set flag to use explicit validation data
        use_explicit_validation = True
    else:
        # For all other modes, use Keras' built-in validation split
        use_explicit_validation = False
        X_val = None
        y_val = None
    
    # Step 2: Apply augmentation ONLY to training data
    from src.data.preprocessing import apply_data_augmentation_post_split
    
    if recognition_mode == 'multi':
        # For multi-task, augment features but keep both label sets
        X_train_aug, y_id_train_aug = apply_data_augmentation_post_split(
            X_train, y_train['id_output'], config
        )
        # Replicate distance labels to match augmented size
        augmentation_factor = len(X_train_aug) // len(X_train)
        y_dist_train_aug = np.tile(y_train['distance_output'], augmentation_factor)
        
        y_train_aug = {'id_output': y_id_train_aug, 'distance_output': y_dist_train_aug}
    else:
        # Standard augmentation
        X_train_aug, y_train_aug = apply_data_augmentation_post_split(X_train, y_train, config)
    
    # Step 3: Prepare data for TensorFlow
    X_train_tf = np.transpose(X_train_aug, (0, 2, 1))
    X_test_tf = np.transpose(X_test, (0, 2, 1))  # Test data remains UNAUGMENTED
    
    # Remove TensorFlow verbose output
    # Step 4: Train model
    callbacks = get_callbacks(
        model_path=model_save_path,
        patience=config['early_stopping_patience'],
        recognition_mode=recognition_mode
    )
    
    if use_explicit_validation and X_val is not None:
        # Use explicit validation data for balanced split
        X_val_tf = np.transpose(X_val, (0, 2, 1))  # Apply same transpose as for test data
        history = model.fit(
            X_train_tf, y_train_aug,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            validation_data=(X_val_tf, y_val),  # Use explicit validation data
            callbacks=callbacks,
            verbose=0
        )
    else:
        # Use default Keras validation split for other modes
        history = model.fit(
            X_train_tf, y_train_aug,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            validation_split=config['validation_split'],
            callbacks=callbacks,
            verbose=0
        )
    
    # Step 5: Evaluation on clean test set
    if recognition_mode == 'multi':
        test_results = model.evaluate(X_test_tf, y_test, verbose=0)
        if isinstance(test_results, list):
            print(f"Test Results - Total Loss: {test_results[0]:.4f}")
            print(f"ID Accuracy: {test_results[3]:.4f}, Distance Precision: {test_results[5]:.4f}")
    elif recognition_mode in ['distance', 'height'] and hasattr(model.output, 'name') and ('distance_output' in model.output.name or 'height_output' in model.output.name):
        test_results = model.evaluate(X_test_tf, y_test, verbose=0)
        if isinstance(test_results, list) and len(test_results) >= 3:
            test_loss, test_mae, test_mse = test_results[:3]
            mode_name = "distance" if "distance_output" in model.output.name else "height"
            print(f"Test MAE ({mode_name}): {test_mae:.4f}")
            print(f"Test MSE ({mode_name}): {test_mse:.4f}")
    else:
        test_results = model.evaluate(X_test_tf, y_test, verbose=0)
        if isinstance(test_results, list) and len(test_results) >= 2:
            test_loss, test_acc = test_results[:2]
            print(f"✅ Test Accuracy: {test_acc:.4f}")
    
    # FINAL EVALUATION - Make sure this uses X_test (clean, unaugmented)
    test_results = model.evaluate(X_test_tf, y_test, verbose=0)
    
    # Add this debug info in your evaluation
    print(f"Confusion matrix dataset size: {len(y_test)}")
    print(f"Expected test size (20% of 2939): {int(2939 * 0.20)}")
    
    # Return ONLY the clean test data for confusion matrix
    return history, model, (X_test_tf, y_test)  # This should be 588 samples

def plot_training_history(history, save_path=None):
    """
    Saves the training history graphs without displaying them
    Handles both classification (accuracy) and regression (mae, mse) metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Check what metrics are available in the history
    has_accuracy = 'accuracy' in history.history
    has_mae = 'mae' in history.history
    
    # First plot: Accuracy for classification or MAE for regression
    if has_accuracy:
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.legend(['Training', 'Validation'], loc='lower right')
    elif has_mae:
        ax1.plot(history.history['mae'])
        ax1.plot(history.history['val_mae'])
        ax1.set_title('Model Mean Absolute Error')
        ax1.set_ylabel('MAE')
        ax1.legend(['Training', 'Validation'], loc='upper right')
    
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    # Loss is always present
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Training', 'Validation'], loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    # Close the figure to free memory and avoid display
    plt.close(fig)

# Fix the preprocessing to apply augmentation after split
def prepare_data_for_training(data, labels, config, recognition_mode='id'):
    """
    Enhanced preprocessing with post-split augmentation to prevent data leakage
    """
    print("Preprocessing spectra...")
    
    # Step 1: Basic preprocessing (same for all samples)
    processed_spectra = []
    for spectrum_data in data:
        processed_spectrum = preprocess_single_spectrum(spectrum_data, config)
        processed_spectra.append(processed_spectrum)
    
    processed_spectra = np.array(processed_spectra)
    print(f"Basic preprocessing complete: {processed_spectra.shape}")
    
    # Step 2: Handle different recognition modes (encode labels)
    if recognition_mode == 'height_class':
        # Binary classification for heights
        from config import HEIGHT_CLASSES
        encoded_labels = []
        valid_indices = []
        
        for i, label in enumerate(labels):
            if label is not None and label in HEIGHT_CLASSES:
                encoded_labels.append(HEIGHT_CLASSES[label])
                valid_indices.append(i)
        
        # Filter out invalid samples
        processed_spectra = processed_spectra[valid_indices]
        encoded_labels = np.array(encoded_labels)
        class_labels = list(HEIGHT_CLASSES.keys())
        
    elif recognition_mode == 'distance' and len(set(labels)) > 20:
        # Distance regression
        min_dist, max_dist = config.get('distance_range', (1.0, 49.0))
        normalized_labels = [(float(label) - min_dist) / (max_dist - min_dist) for label in labels]
        encoded_labels = np.array(normalized_labels)
        class_labels = (min_dist, max_dist)
        
    elif recognition_mode == 'id':
        # ID classification
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = np.array([label_to_idx[label] for label in labels])
        class_labels = unique_labels
        
    else:
        # Default classification
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = np.array([label_to_idx[label] for label in labels])
        class_labels = unique_labels
    
    print(f"Labels encoded for {recognition_mode} mode: {len(encoded_labels)} samples")
    
    # Step 3: FIXED - Return unaugmented data for proper splitting
    return processed_spectra, encoded_labels, class_labels

def apply_data_augmentation_post_split(X_train, y_train, config):
    """
    Apply data augmentation ONLY to training data after train/test split
    
    Args:
        X_train: Training data
        y_train: Training labels  
        config: Configuration with augmentation settings
        
    Returns:
        tuple: (augmented_X_train, augmented_y_train)
    """
    if not config.get('data_augmentation', False):
        print("Data augmentation disabled")
        return X_train, y_train
    
    augmentation_factor = config.get('augmentation_factor', 2)  # Reduced from 3
    
    print(f"Applying data augmentation to training set only (factor: {augmentation_factor})")
    print(f"Original training size: {len(X_train)}")
    
    # Start with original data
    augmented_X = [X_train]
    augmented_y = [y_train]
    
    # Create additional augmented versions
    for aug_round in range(augmentation_factor - 1):
        print(f"Creating augmentation round {aug_round + 1}/{augmentation_factor - 1}")
        
        aug_X_round = []
        for spectrum in X_train:
            # Apply augmentation to each spectrum
            aug_spectrum = create_augmented_spectrum(
                spectrum, 
                freq_shift_range=config.get('freq_shift_range', 0.005),
                noise_level=config.get('noise_level', 0.02)
            )
            aug_X_round.append(aug_spectrum)
        
        augmented_X.append(np.array(aug_X_round))
        augmented_y.append(y_train.copy())  # Same labels for augmented data
    
    # Combine all augmented data
    final_X = np.concatenate(augmented_X, axis=0)
    final_y = np.concatenate(augmented_y, axis=0)
    
    print(f"Final training size after augmentation: {len(final_X)} (increased by {((len(final_X)/len(X_train)) - 1)*100:.0f}%)")
    
    return final_X, final_y

def train_model_with_presplit(X_train, X_test, y_train, y_test, model, config, model_save_path):
    """
    Train model with pre-split data (for balanced splits)
    
    Args:
        X_train, X_test, y_train, y_test: Pre-split data
        model: Model to train
        config: Training configuration
        model_save_path: Path to save model
        
    Returns:
        tuple: (history, trained_model)
    """
    from src.models.cnn_model import get_callbacks
    
    print(f"Training with pre-split data:")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Get callbacks
    callbacks = get_callbacks(model_save_path, patience=config['patience'])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    trained_model = tf.keras.models.load_model(model_save_path)
    
    return history, trained_model