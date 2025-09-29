"""
SRRESNET Training Module for CNN RFID Classification
Adds SRRESNET data to existing training without disturbing current functionality
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf

# SRRESNET data paths
SRRESNET_BASE_DIR = r"D:\Documents\Mitacs\Model\Data\pre_processed\SRRESNET_ID reduced dataset"
SRRESNET_TRAIN_DIRS = [
    os.path.join(SRRESNET_BASE_DIR, "ENV1"),
    os.path.join(SRRESNET_BASE_DIR, "ENV2"), 
    os.path.join(SRRESNET_BASE_DIR, "ENV3")
]
SRRESNET_TEST_DIRS = [
    os.path.join(SRRESNET_BASE_DIR, "Final_test", "ENV1"),
    os.path.join(SRRESNET_BASE_DIR, "Final_test", "ENV2"),
    os.path.join(SRRESNET_BASE_DIR, "Final_test", "ENV3")
]

def load_srresnet_data(data_dirs, limit=None):
    """
    Load SRRESNET data from specified directories using existing data loading infrastructure
    
    Args:
        data_dirs (list): List of directories containing SRRESNET CSV files
        limit (int): Optional limit on number of files to load
        
    Returns:
        tuple: (data, labels, metadata)
    """
    from src.data.data_loader import load_csv_spectrum
    
    print(f"Loading SRRESNET data from {len(data_dirs)} directories...")
    
    all_data = []
    all_labels = []
    all_metadata = []
    file_count = 0
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory {data_dir} does not exist, skipping...")
            continue
            
        print(f"Processing directory: {data_dir}")
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if limit and file_count >= limit:
            break
            
        for csv_file in csv_files:
            if limit and file_count >= limit:
                break
                
            file_path = os.path.join(data_dir, csv_file)
            
            try:
                # Use the existing CSV loading function which handles the SRRESNET format
                spectrum_data = load_csv_spectrum(file_path)
                
                if spectrum_data is None:
                    print(f"Could not load spectrum data from {csv_file}")
                    continue
                
                # Extract metadata from filename (adapt existing function)
                metadata = extract_srresnet_metadata_from_filename(csv_file)
                metadata['file_path'] = file_path
                metadata['source'] = 'srresnet'
                metadata['environment'] = os.path.basename(data_dir)
                
                # Extract ID label from metadata and normalize format
                tag_id = metadata.get('tag_id', 'unknown')
                
                # Normalize label format to match original data
                if tag_id in ['1', '11', '111']:
                    normalized_label = f'Format_{tag_id}'
                else:
                    normalized_label = tag_id
                
                all_data.append(spectrum_data)
                all_labels.append(normalized_label)
                all_metadata.append(metadata)
                
                file_count += 1
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
    
    print(f"Loaded {len(all_data)} SRRESNET samples")
    return all_data, all_labels, all_metadata

def extract_srresnet_metadata_from_filename(filename):
    """
    Extract metadata from SRRESNET filename
    Expected format: M_Sp_XY_ID_S_Dcm_H_A.csv
    Where:
    - M: measure id
    - Sp: sample id  
    - XY: number of resonances/resonators
    - ID: "1", "11" or "111"
    - S: substrate
    - Dcm: distance (e.g., 45cm)
    - H: height (h1 or h2)
    - A: angle
    
    Note: Uses same keys as original data loader for compatibility
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Parse SRRESNET filename format
    parts = name_without_ext.split('_')
    
    metadata = {
        "file_name": basename,
        "tag_id": "unknown",  # Keep for SRRESNET compatibility
        "tag_format": "unknown",  # Add for original data compatibility
        "distance_cm": None,
        "height_info": None,  # Keep for SRRESNET compatibility
        "height": None,  # Add for original data compatibility
        "substrate": None,
        "measure_id": None,
        "sample_id": None,
        "resonances": None,
        "angle": None
    }
    
    if len(parts) >= 7:  # Minimum expected parts: M_Sp_XY_ID_S_Dcm_H_A
        try:
            # M: measure id
            metadata["measure_id"] = parts[0]
            
            # Sp: sample id  
            metadata["sample_id"] = parts[1]
            
            # XY: resonances/resonators
            metadata["resonances"] = parts[2]
            
            # ID: "1", "11" or "111" - this is the RFID tag ID
            tag_id = parts[3]
            metadata["tag_id"] = tag_id  # For SRRESNET compatibility
            metadata["tag_format"] = tag_id  # For original data compatibility
            
            # S: substrate
            metadata["substrate"] = parts[4]
            
            # Dcm: distance
            distance_part = parts[5]
            if "cm" in distance_part.lower():
                distance_str = distance_part.lower().replace("cm", "")
                metadata["distance_cm"] = float(distance_str)
            
            # H: height
            height = parts[6]
            metadata["height_info"] = height  # For SRRESNET compatibility
            metadata["height"] = height  # For original data compatibility
            
            # A: angle (if present)
            if len(parts) > 7:
                metadata["angle"] = parts[7]
                
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not fully parse filename {filename}: {e}")
            # Fall back to extracting just the ID from parts[3] if available
            if len(parts) > 3:
                tag_id = parts[3]
                metadata["tag_id"] = tag_id
                metadata["tag_format"] = tag_id
    
    return metadata

def combine_datasets(original_data, original_labels, original_metadata, 
                    srresnet_data, srresnet_labels, srresnet_metadata):
    """
    Combine original CNN data with SRRESNET data
    
    Args:
        original_data, original_labels, original_metadata: Original dataset
        srresnet_data, srresnet_labels, srresnet_metadata: SRRESNET dataset
        
    Returns:
        tuple: Combined (data, labels, metadata)
    """
    print("Combining original and SRRESNET datasets...")
    
    combined_data = original_data + srresnet_data
    combined_labels = original_labels + srresnet_labels
    combined_metadata = original_metadata + srresnet_metadata
    
    print(f"Combined dataset: {len(combined_data)} samples")
    print(f"  - Original: {len(original_data)} samples")
    print(f"  - SRRESNET: {len(srresnet_data)} samples")
    
    # Check label compatibility
    original_unique_labels = set(original_labels)
    srresnet_unique_labels = set(srresnet_labels)
    common_labels = original_unique_labels.intersection(srresnet_unique_labels)
    
    print(f"Label analysis:")
    print(f"  - Original unique labels: {len(original_unique_labels)}")
    print(f"  - SRRESNET unique labels: {len(srresnet_unique_labels)}")
    print(f"  - Common labels: {len(common_labels)}")
    
    return combined_data, combined_labels, combined_metadata

def train_model_with_srresnet(X, y, model, config, model_save_path, 
                             use_srresnet=True, recognition_mode='id', srresnet_only_mode=False, class_labels=None):
    """
    Enhanced training function that can include SRRESNET data
    
    Args:
        X (np.array): Original preprocessed data
        y (np.array): Original encoded labels
        model (tf.keras.Model): Model to train
        config (dict): Training configuration
        model_save_path (str): Path to save model
        use_srresnet (bool): Whether to include SRRESNET data
        recognition_mode (str): Recognition mode
        srresnet_only_mode (bool): If True, use SRRESNET Final_test for evaluation
        class_labels (list): List of class labels from training preprocessing
        
    Returns:
        tuple: (history, trained_model, test_data)
    """
    from src.models.cnn_model import get_callbacks
    from src.data.preprocessing import apply_data_augmentation_post_split
    
    print(f"Training with SRRESNET data: {use_srresnet}")
    
    if use_srresnet and recognition_mode == 'id':
        # Load SRRESNET training data
        srresnet_train_data, srresnet_train_labels, srresnet_train_metadata = load_srresnet_data(
            SRRESNET_TRAIN_DIRS
        )
        
        if srresnet_train_data:
            # Preprocess SRRESNET data to match original format
            from src.data.preprocessing import preprocess_single_spectrum
            from config import PREPROCESSING_CONFIG
            
            processed_srresnet = []
            for spectrum_data in srresnet_train_data:
                processed_spectrum = preprocess_single_spectrum(spectrum_data, PREPROCESSING_CONFIG)
                processed_srresnet.append(processed_spectrum)
            
            processed_srresnet = np.array(processed_srresnet)
            
            # Encode SRRESNET labels to match original encoding
            unique_original_labels = sorted(list(set(y)))
            original_to_encoded = {label: idx for idx, label in enumerate(unique_original_labels)}
            
            # Filter SRRESNET data to only include labels present in original data
            valid_srresnet_indices = []
            valid_srresnet_labels = []
            
            for i, label in enumerate(srresnet_train_labels):
                if label in original_to_encoded:
                    valid_srresnet_indices.append(i)
                    valid_srresnet_labels.append(original_to_encoded[label])
            
            if valid_srresnet_indices:
                valid_srresnet_data = processed_srresnet[valid_srresnet_indices]
                valid_srresnet_labels = np.array(valid_srresnet_labels)
                
                # Combine with original data
                X_combined = np.concatenate([X, valid_srresnet_data], axis=0)
                y_combined = np.concatenate([y, valid_srresnet_labels], axis=0)
                
                print(f"Combined training data: {len(X_combined)} samples")
                print(f"  - Original: {len(X)} samples")
                print(f"  - Valid SRRESNET: {len(valid_srresnet_data)} samples")
                
                X = X_combined
                y = y_combined
            else:
                print("No valid SRRESNET data found with matching labels")
        else:
            print("No SRRESNET training data loaded")
    
    # Use separate test set for SRRESNET if available
    if use_srresnet and recognition_mode == 'id':
        srresnet_test_data, srresnet_test_labels, srresnet_test_metadata = load_srresnet_data(
            SRRESNET_TEST_DIRS
        )
        use_srresnet_test = bool(srresnet_test_data)
    else:
        use_srresnet_test = False
    
    # Standard train/validation split for training
    # Check if stratification is possible (all classes need at least 2 samples)
    unique_labels, label_counts = np.unique(y, return_counts=True)
    min_samples_per_class = min(label_counts)
    
    if len(set(y)) > 1 and min_samples_per_class >= 2:
        # Stratified split is possible
        X_train, X_val_temp, y_train, y_val_temp = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y
        )
        print(f"Used stratified split (min samples per class: {min_samples_per_class})")
    else:
        # Random split without stratification
        X_train, X_val_temp, y_train, y_val_temp = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        if min_samples_per_class < 2:
            print(f"Warning: Some classes have only {min_samples_per_class} sample(s), using random split")
        else:
            print("Using random split (single class)")
            
    print(f"Class distribution in training set:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    for label, count in zip(train_unique, train_counts):
        print(f"  Class {label}: {count} samples")
    
    # Determine test set based on mode
    if srresnet_only_mode and use_srresnet_test:
        # In SRRESNET-only mode, always use SRRESNET Final_test data
        print("Using SRRESNET Final_test data for evaluation (SRRESNET-only mode)")
        from src.data.preprocessing import preprocess_single_spectrum
        from config import PREPROCESSING_CONFIG
        
        processed_test_srresnet = []
        for spectrum_data in srresnet_test_data:
            processed_spectrum = preprocess_single_spectrum(spectrum_data, PREPROCESSING_CONFIG)
            processed_test_srresnet.append(processed_spectrum)
        
        processed_test_srresnet = np.array(processed_test_srresnet)
        
        print(f"SRRESNET test labels found: {set(srresnet_test_labels)}")
        
        # Use the same class_labels that were created during training preprocessing
        # to ensure consistent encoding between training and test sets
        if class_labels is not None:
            print(f"Using training class labels for encoding: {class_labels}")
            # Create the same mapping as used in prepare_data_for_training
            label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
            print(f"Label mapping: {label_to_idx}")
            
            # Encode test labels using the training mapping
            y_test_encoded = []
            valid_indices = []
            for i, label in enumerate(srresnet_test_labels):
                if label in label_to_idx:
                    y_test_encoded.append(label_to_idx[label])
                    valid_indices.append(i)
                else:
                    print(f"⚠️ Warning: Test label '{label}' not found in training labels")
            
            if len(valid_indices) > 0:
                X_test = processed_test_srresnet[valid_indices]
                y_test = np.array(y_test_encoded)
                print(f"✅ SRRESNET Final_test data loaded: {len(X_test)} samples")
                print(f"Test class distribution: {np.bincount(y_test)}")
                
                # Verify we have exactly the Final_test samples (should be 198)
                if len(X_test) != 198:
                    print(f"⚠️ Warning: Expected 198 Final_test samples, got {len(X_test)}")
                else:
                    print("✅ Confirmed: Using exactly 198 Final_test samples")
            else:
                print("❌ No valid SRRESNET test data with matching labels, falling back to validation split")
                X_test, y_test = X_val_temp, y_val_temp
        else:
            print("❌ No class_labels provided, falling back to validation split")
            X_test, y_test = X_val_temp, y_val_temp
    elif use_srresnet_test and not srresnet_only_mode:
        # In combined mode, use standard validation split for primary evaluation
        # SRRESNET test will be done separately
        print("Using validation split for primary evaluation (combined mode)")
        X_test, y_test = X_val_temp, y_val_temp
    else:
        # No SRRESNET test data available or not requested
        X_test, y_test = X_val_temp, y_val_temp
    
    print(f"Data split: {len(X_train)} training, {len(X_test)} test samples")
    
    # Apply augmentation only to training data
    X_train_aug, y_train_aug = apply_data_augmentation_post_split(X_train, y_train, config)
    
    # Prepare data for TensorFlow
    X_train_tf = np.transpose(X_train_aug, (0, 2, 1))
    X_test_tf = np.transpose(X_test, (0, 2, 1))
    
    # Train model
    callbacks = get_callbacks(
        model_path=model_save_path,
        patience=config['early_stopping_patience'],
        recognition_mode=recognition_mode
    )
    
    history = model.fit(
        X_train_tf, y_train_aug,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split'],
        callbacks=callbacks,
        verbose=1  # Show progress for SRRESNET training
    )
    
    # Final evaluation
    test_results = model.evaluate(X_test_tf, y_test, verbose=0)
    if isinstance(test_results, list) and len(test_results) >= 2:
        test_loss, test_acc = test_results[:2]
        print(f"✅ Final Test Accuracy: {test_acc:.4f}")
        if srresnet_only_mode:
            print("  (Evaluated on SRRESNET Final_test set)")
        elif use_srresnet_test:
            print("  (Evaluated on validation split - combined mode)")
        else:
            print("  (Evaluated on validation split)")
    
    return history, model, (X_test_tf, y_test)

def check_srresnet_data_availability():
    """
    Check if SRRESNET data directories exist and contain CSV files
    
    Returns:
        tuple: (train_available, test_available, message)
    """
    train_available = False
    test_available = False
    messages = []
    
    # Check training directories
    train_file_count = 0
    for train_dir in SRRESNET_TRAIN_DIRS:
        if os.path.exists(train_dir):
            csv_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
            train_file_count += len(csv_files)
            messages.append(f"  {train_dir}: {len(csv_files)} CSV files")
        else:
            messages.append(f"  {train_dir}: NOT FOUND")
    
    train_available = train_file_count > 0
    
    # Check test directories  
    test_file_count = 0
    for test_dir in SRRESNET_TEST_DIRS:
        if os.path.exists(test_dir):
            csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
            test_file_count += len(csv_files)
            messages.append(f"  {test_dir}: {len(csv_files)} CSV files")
        else:
            messages.append(f"  {test_dir}: NOT FOUND")
    
    test_available = test_file_count > 0
    
    summary = f"SRRESNET Data Status:\n"
    summary += f"Training data: {'✅ Available' if train_available else '❌ Not found'} ({train_file_count} files)\n"
    summary += f"Test data: {'✅ Available' if test_available else '❌ Not found'} ({test_file_count} files)\n"
    summary += "Details:\n" + "\n".join(messages)
    
    return train_available, test_available, summary

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
    
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'model_outputs'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'performance'), exist_ok=True)
    
    # Create a README file with information
    with open(os.path.join(results_dir, 'README.txt'), 'w') as f:
        f.write(f"SRRESNET-Enhanced CNN RFID Classification Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Directory structure:\n")
        f.write("- visualizations: Contains plots of data samples\n")
        f.write("- model_outputs: Contains model predictions and metrics\n")
        f.write("- performance: Contains performance analysis visualizations\n")
        f.write("\nTraining Mode: SRRESNET-Enhanced (combined original + SRRESNET data)\n")
    
    return results_dir

def create_pairing_key(metadata):
    """
    Create a pairing key for matching original and SRRESNET data
    Based on: distance, ID, height, and first two digits of substrate
    
    Args:
        metadata (dict): File metadata
        
    Returns:
        str: Pairing key for matching
    """
    # Handle different key names between original and SRRESNET metadata
    distance = metadata.get('distance_cm', 'unknown')
    
    # Tag ID can be under 'tag_id' (SRRESNET) or 'tag_format' (original)
    tag_id = metadata.get('tag_id') or metadata.get('tag_format', 'unknown')
    
    # Height can be under 'height_info' (SRRESNET) or 'height' (original)
    height = metadata.get('height_info') or metadata.get('height', 'unknown')
    
    substrate = metadata.get('substrate', 'unknown')
    
    # Use first two digits of substrate if available
    substrate_prefix = substrate[:2] if substrate and len(substrate) >= 2 else substrate
    
    # Normalize distance to integer format (handle both float and int)
    if distance != 'unknown':
        distance = int(float(distance))
    
    # Create composite key - normalize the format
    pairing_key = f"{tag_id}_{distance}cm_{height}_{substrate_prefix}"
    return pairing_key

def pair_original_and_srresnet_data(original_data, original_labels, original_metadata,
                                   srresnet_data, srresnet_labels, srresnet_metadata):
    """
    Pair original and SRRESNET data based on matching criteria
    Criteria: distance, ID, height, and first two digits of substrate
    
    Returns:
        tuple: (paired_original_data, paired_srresnet_data, common_labels)
    """
    print("Pairing original and SRRESNET data...")
    
    # Create pairing keys for original data
    original_pairs = {}
    print(f"Processing {len(original_metadata)} original files...")
    for i, metadata in enumerate(original_metadata):
        key = create_pairing_key(metadata)
        if key not in original_pairs:
            original_pairs[key] = []
        original_pairs[key].append(i)
        if i < 3:  # Debug: show first 3 original pairing keys
            print(f"  Original {i}: {metadata.get('file_name', 'unknown')} -> {key}")
    
    # Create pairing keys for SRRESNET data
    srresnet_pairs = {}
    print(f"Processing {len(srresnet_metadata)} SRRESNET files...")
    for i, metadata in enumerate(srresnet_metadata):
        key = create_pairing_key(metadata)
        if key not in srresnet_pairs:
            srresnet_pairs[key] = []
        srresnet_pairs[key].append(i)
        if i < 3:  # Debug: show first 3 SRRESNET pairing keys
            print(f"  SRRESNET {i}: {metadata.get('file_name', 'unknown')} -> {key}")
    
    # Find common pairing keys
    common_keys = set(original_pairs.keys()).intersection(set(srresnet_pairs.keys()))
    print(f"Found {len(common_keys)} matching pairing keys")
    
    if len(common_keys) == 0:
        print("No matching pairs found between original and SRRESNET data")
        print(f"Original keys sample: {list(original_pairs.keys())[:5]}")
        print(f"SRRESNET keys sample: {list(srresnet_pairs.keys())[:5]}")
        return [], [], []
    
    # Extract paired data
    paired_original_indices = []
    paired_srresnet_indices = []
    
    for key in common_keys:
        # Take all original samples with this key
        paired_original_indices.extend(original_pairs[key])
        # Take all SRRESNET samples with this key
        paired_srresnet_indices.extend(srresnet_pairs[key])
    
    paired_original_data = [original_data[i] for i in paired_original_indices]
    paired_original_labels = [original_labels[i] for i in paired_original_indices]
    paired_original_metadata = [original_metadata[i] for i in paired_original_indices]
    
    paired_srresnet_data = [srresnet_data[i] for i in paired_srresnet_indices]
    paired_srresnet_labels = [srresnet_labels[i] for i in paired_srresnet_indices]
    paired_srresnet_metadata = [srresnet_metadata[i] for i in paired_srresnet_indices]
    
    print(f"Paired data:")
    print(f"  - Original samples: {len(paired_original_data)}")
    print(f"  - SRRESNET samples: {len(paired_srresnet_data)}")
    
    # Combine paired data
    combined_data = paired_original_data + paired_srresnet_data
    combined_labels = paired_original_labels + paired_srresnet_labels
    combined_metadata = paired_original_metadata + paired_srresnet_metadata
    
    return combined_data, combined_labels, combined_metadata

def main_srresnet_training(srresnet_only=False):
    """
    Main function for SRRESNET-enhanced training
    
    Args:
        srresnet_only (bool): If True, use only SRRESNET data (ignore original data)
    """
    print("SRRESNET-Enhanced CNN Training for RFID Classification")
    print("=" * 60)
    
    # Check SRRESNET data availability
    train_available, test_available, status_message = check_srresnet_data_availability()
    print(status_message)
    
    if not train_available:
        print("\n❌ No SRRESNET training data found. Please check the data paths.")
        print("Expected directories:")
        for dir_path in SRRESNET_TRAIN_DIRS:
            print(f"  - {dir_path}")
        return
    
    # Import necessary modules
    from src.data.data_loader import load_dataset
    from src.data.preprocessing import prepare_data_for_training
    from src.models.cnn_model import build_cnn_model
    from src.training.train import plot_training_history
    from src.training.evaluate import evaluate_model, plot_confusion_matrix
    from config import (RAW_DATA_DIR, MODELS_DIR, MAPPING_FILE, 
                       TRAIN_CONFIG, CNN_CONFIG, PREPROCESSING_CONFIG)
    
    # Determine if we should load original data
    use_original_data = False
    
    if not srresnet_only:
        # Try to load original dataset, but proceed with SRRESNET only if original fails
        print("\nAttempting to load original dataset...")
        try:
            # Use the correct preprocessed data directory
            original_data_dir = r"D:\Documents\Mitacs\Model\Data\pre_processed\CNN_filtering"
            data, labels, metadata = load_dataset(
                data_dir=original_data_dir,
                mapping_file=MAPPING_FILE,
                recognition_mode='id'
            )
            
            if len(data) == 0:
                print("⚠️ No original data found, proceeding with SRRESNET data only")
                use_original_data = False
            else:
                print(f"✅ Original dataset loaded: {len(data)} samples")
                use_original_data = True
        except Exception as e:
            print(f"⚠️ Failed to load original data: {e}")
            print("Proceeding with SRRESNET data only")
            use_original_data = False
    else:
        print("\n🎯 SRRESNET-only mode: Skipping original data loading")
        use_original_data = False
    
    # Load SRRESNET data
    print("\nLoading SRRESNET training data...")
    srresnet_data, srresnet_labels, srresnet_metadata = load_srresnet_data(SRRESNET_TRAIN_DIRS)
    
    if not srresnet_data:
        print("❌ Failed to load SRRESNET data")
        return
    
    print(f"SRRESNET training data loaded: {len(srresnet_data)} samples")
    
    # Determine which data to use
    if use_original_data:
        print("Using pairing approach for original + SRRESNET data")
        # Use proper pairing instead of simple combining
        paired_data, paired_labels, paired_metadata = pair_original_and_srresnet_data(
            data, labels, metadata,
            srresnet_data, srresnet_labels, srresnet_metadata
        )
        
        if len(paired_data) > 0:
            print(f"Successfully paired data: {len(paired_data)} samples")
            training_data = paired_data
            training_labels = paired_labels
            training_metadata = paired_metadata
        else:
            print("No paired data found, falling back to SRRESNET data only")
            training_data = srresnet_data
            training_labels = srresnet_labels
            training_metadata = srresnet_metadata
    else:
        print("Using SRRESNET data only")
        training_data = srresnet_data
        training_labels = srresnet_labels
        training_metadata = srresnet_metadata
    
    # Preprocess data
    print("Preprocessing data...")
    X_preprocessed, y_encoded, class_labels = prepare_data_for_training(
        training_data, training_labels, PREPROCESSING_CONFIG, recognition_mode='id'
    )
    
    if X_preprocessed.shape[0] == 0:
        print("❌ No valid preprocessed data")
        return
    
    print(f"Preprocessed data shape: {X_preprocessed.shape}")
    print(f"Number of classes: {len(class_labels)}")
    print(f"Class labels: {class_labels}")
    
    # Build model
    print("Building CNN model...")
    input_shape = X_preprocessed.shape[1:]
    model = build_cnn_model(input_shape, class_labels, CNN_CONFIG, recognition_mode='id')
    
    # Train with SRRESNET data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"rfid_cnn_srresnet_id_{timestamp}.h5"
    model_save_path = os.path.join(MODELS_DIR, model_name)
    
    print("Training with SRRESNET data...")
    history, trained_model, (X_test, y_test) = train_model_with_srresnet(
        X_preprocessed, y_encoded, model, TRAIN_CONFIG, model_save_path,
        use_srresnet=True, recognition_mode='id', srresnet_only_mode=(not use_original_data),
        class_labels=class_labels
    )
    
    # Create results directory
    results_dir = create_results_directory(f"srresnet_{timestamp}")
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(results_dir, 'training_history.png'))
    
    # Evaluate model
    evaluation = evaluate_model(trained_model, X_test, y_test, class_labels, recognition_mode='id')
    
    # Print primary results
    print("\nPrimary Training Results:")
    print(f"Test Accuracy: {evaluation.get('accuracy', 'N/A'):.4f}")
    
    # Plot confusion matrix for primary evaluation
    if 'confusion_matrix' in evaluation:
        plot_confusion_matrix(
            evaluation['confusion_matrix'],
            class_labels,
            save_path=os.path.join(results_dir, 'confusion_matrix_primary.png')
        )
    
    # Additional SRRESNET Final_test evaluation (for combined mode or confirmation in SRRESNET-only mode)
    if use_original_data:
        # In combined mode, do additional SRRESNET-only evaluation
        srresnet_evaluation = evaluate_on_srresnet_test(trained_model, class_labels, recognition_mode='id')
        
        if srresnet_evaluation:
            print(f"\nAdditional SRRESNET Final_test Results:")
            print(f"SRRESNET Test Accuracy: {srresnet_evaluation['accuracy']:.4f}")
            
            # Plot confusion matrix for SRRESNET evaluation
            if srresnet_evaluation['detailed_evaluation'] and 'confusion_matrix' in srresnet_evaluation['detailed_evaluation']:
                plot_confusion_matrix(
                    srresnet_evaluation['detailed_evaluation']['confusion_matrix'],
                    class_labels,
                    save_path=os.path.join(results_dir, 'confusion_matrix_srresnet_test.png')
                )
    else:
        # In SRRESNET-only mode, the primary evaluation is already on Final_test data
        print("\n(Primary evaluation was performed on SRRESNET Final_test data)")
    
    # Save model with SRRESNET label
    labels_save_path = os.path.splitext(model_save_path)[0] + '_labels.npy'
    np.save(labels_save_path, class_labels)
    print(f"Class labels saved to {labels_save_path}")
    
    print(f"\nResults saved to: {results_dir}")
    print("SRRESNET training completed!")

def evaluate_on_srresnet_test(model, class_labels, recognition_mode='id'):
    """
    Evaluate trained model on SRRESNET Final_test data
    
    Args:
        model: Trained model
        class_labels: List of class labels
        recognition_mode: Recognition mode
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*50)
    print("ADDITIONAL EVALUATION ON SRRESNET FINAL_TEST DATA")
    print("="*50)
    
    # Load SRRESNET test data
    srresnet_test_data, srresnet_test_labels, srresnet_test_metadata = load_srresnet_data(
        SRRESNET_TEST_DIRS
    )
    
    if not srresnet_test_data:
        print("❌ No SRRESNET test data available")
        return None
    
    print(f"Loaded {len(srresnet_test_data)} SRRESNET test samples")
    
    # Preprocess test data
    from src.data.preprocessing import preprocess_single_spectrum
    from config import PREPROCESSING_CONFIG
    
    processed_test_data = []
    for spectrum_data in srresnet_test_data:
        processed_spectrum = preprocess_single_spectrum(spectrum_data, PREPROCESSING_CONFIG)
        processed_test_data.append(processed_spectrum)
    
    processed_test_data = np.array(processed_test_data)
    
    # Encode test labels
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    
    valid_test_indices = []
    valid_test_labels = []
    
    for i, label in enumerate(srresnet_test_labels):
        if label in label_to_idx:
            valid_test_indices.append(i)
            valid_test_labels.append(label_to_idx[label])
    
    if not valid_test_indices:
        print("❌ No valid test samples with matching labels")
        return None
    
    X_test_srresnet = processed_test_data[valid_test_indices]
    y_test_srresnet = np.array(valid_test_labels)
    
    # Prepare data for TensorFlow
    X_test_tf = np.transpose(X_test_srresnet, (0, 2, 1))
    
    print(f"Evaluating on {len(X_test_tf)} SRRESNET test samples...")
    
    # Evaluate model
    test_results = model.evaluate(X_test_tf, y_test_srresnet, verbose=0)
    
    if isinstance(test_results, list) and len(test_results) >= 2:
        test_loss, test_acc = test_results[:2]
        print(f"✅ SRRESNET Final_test Accuracy: {test_acc:.4f}")
        
        # Additional detailed evaluation
        from src.training.evaluate import evaluate_model
        evaluation = evaluate_model(model, X_test_tf, y_test_srresnet, class_labels, recognition_mode)
        
        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'detailed_evaluation': evaluation,
            'test_data': (X_test_tf, y_test_srresnet)
        }
    else:
        print("❌ Could not evaluate model")
        return None

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='SRRESNET Training for RFID Classification')
    parser.add_argument('--srresnet-only', '-s', action='store_true', 
                       help='Use only SRRESNET data (ignore original data)')
    parser.add_argument('--limit', type=int, help='Limit number of files to load')
    
    args = parser.parse_args()
    
    if args.srresnet_only:
        print("🎯 Running in SRRESNET-only mode")
    
    main_srresnet_training(srresnet_only=args.srresnet_only)
