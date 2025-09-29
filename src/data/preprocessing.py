"""
Fonctions de prétraitement pour les données spectrales RFID
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm

def resample_spectrum(frequencies, spectrum, target_size):
    """
    Resamples the spectrum with absolute preservation of deepest resonances
    """
    if len(frequencies) <= target_size:
        return spectrum  # Already at or below target size
    
    # Create uniform grid as starting point
    master_freq = np.linspace(np.min(frequencies), np.max(frequencies), target_size)
    
    # STEP 1: Find the ABSOLUTE deepest points ONLY
    indices = np.arange(len(spectrum))
    sorted_indices = np.argsort(spectrum)
    deepest_indices = sorted_indices[:5]  # Just the 5 absolute deepest points
    
    # STEP 2: Create interpolation function for base values
    interp_func = interp1d(frequencies, spectrum, kind='cubic', 
                         bounds_error=False, fill_value='extrapolate')
    
    # Get initial interpolated values
    resampled_values = interp_func(master_freq)
    
    # STEP 3: FORCE the deepest points into the grid - with highest priority
    for idx in deepest_indices:
        extremum_freq = frequencies[idx]
        extremum_val = spectrum[idx]  # EXACT original value
        
        # Find the closest point in our grid
        closest_idx = np.abs(master_freq - extremum_freq).argmin()
        
        # CRITICAL: Replace this grid point with EXACT frequency and value
        master_freq[closest_idx] = extremum_freq
        resampled_values[closest_idx] = extremum_val  # EXACT original value
        
        # STEP 4: Now preserve multiple points around each deep point
        peak_indices = []
        for offset in range(-6, 7):  # 6 points before, 6 after
            if offset == 0:
                continue  # Skip the center (already handled)
                
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(frequencies):
                peak_indices.append(neighbor_idx)
                
        # For each point in the resonance curve:
        for peak_idx in peak_indices:
            neighbor_freq = frequencies[peak_idx]
            neighbor_val = spectrum[peak_idx]
            
            # Find a grid point to use (not necessarily the closest)
            distances = np.abs(master_freq - neighbor_freq)
            grid_indices = np.argsort(distances)
            
            # Find first available grid point
            for pot_idx in grid_indices[:10]:  # Try the 10 closest points
                # Check if this grid point is already assigned to a peak point
                already_used = False
                for used_idx in deepest_indices:
                    if abs(master_freq[pot_idx] - frequencies[used_idx]) < 1e-10:
                        already_used = True
                        break
                        
                if not already_used:
                    grid_idx = pot_idx
                    break
            else:
                # If we can't find a free grid point, use the closest anyway
                grid_idx = grid_indices[0]
            
            # Direct copy from original data - EXACT VALUE PRESERVATION
            master_freq[grid_idx] = neighbor_freq
            resampled_values[grid_idx] = neighbor_val
    
    return resampled_values

def smooth_spectrum(spectrum, window_size=5):
    """
    Apply extremely aggressive filtering to non-peak regions, with ZERO filtering on 
    resonance peaks. This approach preserves peaks with absolute fidelity.
    """
    # Copy the original spectrum - we'll modify this
    result = spectrum.copy()
    data_length = len(spectrum)
    
    # STEP 1: Find both minimum and maximum extrema
    indices = np.arange(len(spectrum))
    
    # Get deepest points (minima) - INCREASED from 10 to 20
    min_indices = indices[np.argsort(spectrum)][:20]  # Top 20 deepest points
    
    # Get highest points (maxima) - INCREASED from 10 to 15
    max_indices = indices[np.argsort(-spectrum)][:15]  # Top 15 highest points
    
    # Combine all extrema points
    all_extrema_indices = np.concatenate([min_indices, max_indices])
    all_extrema_indices = np.unique(all_extrema_indices)
    
    # STEP 2: Create wide protected regions around these extrema
    peak_regions = np.zeros_like(spectrum, dtype=bool)
    margin = int(data_length * 0.20)  # 20% of signal length
    
    for idx in all_extrema_indices:
        left = max(0, idx - margin)
        right = min(len(spectrum) - 1, idx + margin)
        peak_regions[left:right+1] = True
    
    # STEP 3: Only apply filtering to non-peak regions
    non_peak_regions = ~peak_regions
    
    if np.any(non_peak_regions):
        # First apply Savitzky-Golay to the entire spectrum
        filtered_temp = signal.savgol_filter(spectrum, 51, 3)
        
        # Then apply multiple passes of moving average for extreme smoothing
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'same') / w
        
        smoothed = filtered_temp.copy()
        # Apply 7 passes with a very wide window
        for _ in range(7):
            smoothed = moving_average(smoothed, 31)
        
        # Copy the smoothed version ONLY to non-peak regions
        # Peak regions remain completely untouched from the original
        result[non_peak_regions] = smoothed[non_peak_regions]
    
    # Additional debug verification
    for idx in all_extrema_indices:
        if idx < len(result) and idx < len(spectrum):
            if not np.isclose(result[idx], spectrum[idx]):
                # Force correction
                result[idx] = spectrum[idx]
    
    # Apply edge correction as final step
    _, result = correct_edge_effects(np.arange(len(result)), result, edge_points=5)
    
    return result

# Add edge correction function to preprocessing.py too
def correct_edge_effects(x, y, lower_freq_limit=3.22e9, upper_freq_limit=5.15e9, transition_points=5, edge_points=None):
    """
    Replaces data points outside the valid frequency range with values close to the boundaries
    instead of removing them completely, which could cause array length issues.
    
    Args:
        x: x-coordinates (frequency values in Hz)
        y: y-coordinates (amplitude/phase values)
        lower_freq_limit: Lower frequency boundary of useful data (Hz)
        upper_freq_limit: Upper frequency boundary of useful data (Hz)
        transition_points: Number of points to use for calculating reference values
        edge_points: Alias for transition_points (for backward compatibility)
        
    Returns:
        tuple: corrected x and y arrays with the same length as inputs
    """
    # Use edge_points if provided (for backward compatibility)
    if edge_points is not None:
        transition_points = edge_points
    
    # Create copy for correction - don't remove points
    y_corrected = y.copy()
    
    # Find indices for frequency limits
    lower_indices = np.where(x < lower_freq_limit)[0]
    upper_indices = np.where(x > upper_freq_limit)[0]
    
    # Safety check - if we would filter out everything, return original data
    if len(lower_indices) + len(upper_indices) >= len(x):
        return x, y
    
    # Only proceed if we have points outside the limits
    if len(lower_indices) > 0:
        # Find reference points just inside the lower limit
        reference_start = lower_indices[-1] + 1
        reference_end = min(reference_start + transition_points, len(y))
        
        # Calculate reference value (average of first few valid points)
        if reference_end > reference_start:
            reference_value = np.mean(y[reference_start:reference_end])
            
            # Replace all values below the limit with the reference value
            y_corrected[lower_indices] = reference_value
    
    if len(upper_indices) > 0:
        # Find reference points just inside the upper limit
        reference_end = upper_indices[0] - 1
        reference_start = max(0, reference_end - transition_points)
        
        # Calculate reference value (average of last few valid points)
        if reference_end > reference_start:
            reference_value = np.mean(y[reference_start:reference_end+1])
            
            # Replace all values above the limit with the reference value
            y_corrected[upper_indices] = reference_value
    
    return x, y_corrected

def normalize_spectrum(spectrum):
    """
    Normalise le spectre entre -1 et 1
    
    Args:
        spectrum (np.array): Données spectrales à normaliser
        
    Returns:
        np.array: Spectre normalisé
    """
    if np.max(spectrum) == np.min(spectrum):
        return np.zeros_like(spectrum)
    return 2 * (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) - 1

def create_augmented_spectrum(spectrum, freq_shift_range=0.005, noise_level=0.02):
    """
    Crée une version augmentée du spectre par décalage de fréquence et bruit
    
    Args:
        spectrum (np.array): Spectre original
        freq_shift_range (float): Amplitude du décalage de fréquence (±%)
        noise_level (float): Niveau de bruit à ajouter
        
    Returns:
        np.array: Spectre augmenté
    """
    # Simuler un petit décalage de fréquence
    n_points = len(spectrum)
    shift = int(n_points * np.random.uniform(-freq_shift_range, freq_shift_range))
    
    if shift > 0:
        shifted_spectrum = np.concatenate([spectrum[shift:], spectrum[:shift]])
    elif shift < 0:
        shifted_spectrum = np.concatenate([spectrum[shift:], spectrum[:shift]])
    else:
        shifted_spectrum = spectrum.copy()
    
    # Ajouter du bruit gaussien
    noise = np.random.normal(0, noise_level, size=n_points)
    augmented_spectrum = shifted_spectrum + noise
    
    return augmented_spectrum

def preprocess_dataset(data, config, augment=True):
    """
    Prétraite l'ensemble du jeu de données pour l'entraînement du CNN
    
    Args:
        data (list): Liste de dictionnaires contenant les données spectrales
        config (dict): Configuration de prétraitement
        augment (bool): Si True, effectue l'augmentation de données
        
    Returns:
        np.array: Données prétraitées prêtes pour le CNN
    """
    target_size = config['num_frequency_points']
    window_size = config['window_size']
    normalize = config['normalize']
    
    processed_data = []
    
    for spectrum_dict in tqdm(data, desc="Prétraitement des données"):
        # Extraction des données
        frequencies = spectrum_dict['frequencies']
        amplitude = spectrum_dict['amplitude']
        phase = spectrum_dict['phase']
        
        # Rééchantillonnage
        amplitude_resampled = resample_spectrum(frequencies, amplitude, target_size)
        phase_resampled = resample_spectrum(frequencies, phase, target_size)
        
        # Lissage
        amplitude_smoothed = smooth_spectrum(amplitude_resampled, window_size)
        phase_smoothed = smooth_spectrum(phase_resampled, window_size)
        
        # Normalisation
        if normalize:
            amplitude_normalized = normalize_spectrum(amplitude_smoothed)
            phase_normalized = normalize_spectrum(phase_smoothed)
        else:
            amplitude_normalized = amplitude_smoothed
            phase_normalized = phase_smoothed
        
        # Format pour CNN: [canaux, fréquence]
        spectrum_processed = np.stack([amplitude_normalized, phase_normalized], axis=0)
        processed_data.append(spectrum_processed)
        
        # Augmentation de données
        if augment and config['data_augmentation']:
            for _ in range(config['augmentation_factor'] - 1):
                # Augmenter amplitude et phase
                aug_amplitude = create_augmented_spectrum(
                    amplitude_normalized, 
                    freq_shift_range=0.01, 
                    noise_level=0.05
                )
                aug_phase = create_augmented_spectrum(
                    phase_normalized,
                    freq_shift_range=0.005,
                    noise_level=0.02
                )
                
                # Format pour CNN
                aug_spectrum = np.stack([aug_amplitude, aug_phase], axis=0)
                processed_data.append(aug_spectrum)
    
    return np.array(processed_data)

def prepare_data_for_training(data, labels, config, recognition_mode='id'):
    """
    Prepare data for training with mode-specific handling
    """
    print("Preprocessing spectra...")
    
    # For angle recognition, use minimal preprocessing
    if recognition_mode == 'angle':
        print("Angle recognition mode: Using minimal preprocessing")
        processed_data = []
        
        # Process ALL spectra, not just the first one
        for i, spectrum in enumerate(data):
            # Only print debug info for first spectrum to avoid spam
            if i == 0:
                print(f"DEBUG: First spectrum shape - frequencies: {len(spectrum['frequencies'])}, amplitudes: {len(spectrum['amplitude'])}")
            
            # Extract raw frequency, amplitude, and phase
            frequencies = np.array(spectrum['frequencies'])
            amplitudes = np.array(spectrum['amplitude'])
            phases = np.array(spectrum['phase'])
            
            # Debug: print shapes before processing
            if i == 0:
                print(f"DEBUG: Raw arrays shape - amplitudes: {amplitudes.shape}, phases: {phases.shape}")
            
            # Simple normalization only
            amplitudes = (amplitudes - np.mean(amplitudes)) / np.std(amplitudes)
            phases = (phases - np.mean(phases)) / np.std(phases)
            
            # CRITICAL: Ensure we create (freq_points, channels) format
            # Create as (2, freq_points) first, then transpose to (freq_points, 2)
            processed_spectrum = np.stack([amplitudes, phases], axis=0)  # Shape: (2, freq_points)
            processed_spectrum = processed_spectrum.T  # Transpose to (freq_points, 2)
            
            if i == 0:
                print(f"DEBUG: Processed spectrum shape: {processed_spectrum.shape}")
            
            processed_data.append(processed_spectrum)
        
        X = np.array(processed_data)
        
        # Encode labels for angle classification
        unique_angles = sorted(list(set(labels)))
        angle_to_idx = {angle: idx for idx, angle in enumerate(unique_angles)}
        y_encoded = np.array([angle_to_idx[label] for label in labels])
        class_labels = unique_angles
        
        print(f"Processed {len(X)} spectra for angle recognition")
        print(f"Data shape: {X.shape}")
        print(f"Input shape for CNN: {X.shape[1:]} (frequency_points, channels)")
        print(f"Angles: {unique_angles}")
        
        return X, y_encoded, class_labels
    
    print("Preprocessing spectra for other modes...")
    # Prétraiter les données
    X_preprocessed = preprocess_dataset(data, config)
    
    if recognition_mode == 'multi':
        # Pour le mode multi, on doit traiter séparément les IDs et les distances
        ids = [label[0] for label in labels]
        distances = [label[1] for label in labels]
        
        # Encoder les ID de tags
        unique_ids = sorted(list(set(ids)))
        id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
        y_id_encoded = np.array([id_to_idx[id_val] for id_val in ids])
          # Encoder les distances (régression ou classification selon le cas)
        unique_distances = sorted(list(set(distances)))
        if len(unique_distances) <= 20:  # Si peu de valeurs, traiter comme classification
            distance_to_idx = {dist: idx for idx, dist in enumerate(unique_distances)}
            y_distance_encoded = np.array([distance_to_idx[dist] for dist in distances])
        else:  # Sinon, traiter comme régression
            # Use hardcoded distance range of 1-49cm for RFID measurements
            from config import DISTANCE_RANGE
            min_dist, max_dist = DISTANCE_RANGE
            range_dist = max_dist - min_dist  # 48.0
            
            # Normalize distances to [0, 1] range using the hardcoded range
            y_distance_encoded = np.array([(dist - min_dist) / range_dist for dist in distances])
            
            print(f"Multi-task distance regression: Using hardcoded range [{min_dist}-{max_dist}]cm")
            print(f"Actual distance range in data: [{min(distances):.1f}-{max(distances):.1f}]cm")
        
        # Si augmentation, répéter les étiquettes
        if config['data_augmentation']:
            y_id_encoded = np.repeat(y_id_encoded, config['augmentation_factor'])[:len(X_preprocessed)]
            y_distance_encoded = np.repeat(y_distance_encoded, config['augmentation_factor'])[:len(X_preprocessed)]
        
        return X_preprocessed, (y_id_encoded, y_distance_encoded), (unique_ids, unique_distances)
    
    else:  # Modes 'id', 'distance', ou 'height'
        # Encoder les étiquettes
        unique_labels = sorted(list(set(labels)))        
        
        # Pour le mode 'distance', traiter comme régression par défaut
        if recognition_mode == 'distance':
            # Use hardcoded distance range of 1-49cm for RFID measurements
            from config import DISTANCE_RANGE
            min_dist, max_dist = DISTANCE_RANGE
            range_dist = max_dist - min_dist  # 48.0
            
            # Normalize distances to [0, 1] range using the hardcoded range
            y_encoded = np.array([(dist - min_dist) / range_dist for dist in labels])
            
            print(f"Distance regression mode: Using hardcoded range [{min_dist}-{max_dist}]cm")
            print(f"Actual distance range in data: [{min(labels):.1f}-{max(labels):.1f}]cm")
            print(f"Normalized distance range: [{min(y_encoded):.3f}-{max(y_encoded):.3f}]")
            
            # Si augmentation, répéter les étiquettes
            if config['data_augmentation']:
                y_encoded = np.repeat(y_encoded, config['augmentation_factor'])[:len(X_preprocessed)]
            
            # Pour la régression, unique_labels est simplement (min, max)
            return X_preprocessed, y_encoded, (min_dist, max_dist)
        
        # Pour le mode 'height', traiter comme régression par défaut
        elif recognition_mode == 'height':
            # Use hardcoded height range of 4-11.5cm for RFID measurements (h1=4cm, h2=11.5cm)
            from config import HEIGHT_RANGE
            min_height, max_height = HEIGHT_RANGE
            range_height = max_height - min_height  # 7.5
            
            # Normalize heights to [0, 1] range using the hardcoded range
            y_encoded = np.array([(height - min_height) / range_height for height in labels])
            
            print(f"Height regression mode: Using hardcoded range [{min_height}-{max_height}]cm")
            print(f"Actual height range in data: [{min(labels):.1f}-{max(labels):.1f}]cm")
            print(f"Normalized height range: [{min(y_encoded):.3f}-{max(y_encoded):.3f}]")
            
            # Si augmentation, répéter les étiquettes
            if config['data_augmentation']:
                y_encoded = np.repeat(y_encoded, config['augmentation_factor'])
            
            return X_preprocessed, y_encoded, (min_height, max_height)
        
        # Pour le mode 'height_class', traiter comme classification (académiquement valide)
        elif recognition_mode == 'height_class':
            from config import HEIGHT_CLASSES
            
            # Convert height class labels to integers (already done in data_loader)
            y_encoded = np.array(labels, dtype=int)
            
            print(f"Height classification mode: {len(set(y_encoded))} classes")
            print(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
            
            # Si augmentation, répéter les étiquettes
            if config['data_augmentation']:
                y_encoded = np.repeat(y_encoded, config['augmentation_factor'])[:len(X_preprocessed)]
            
            return X_preprocessed, y_encoded, None
        
        # Ajout du cas pour la reconnaissance d'angle:
        elif recognition_mode == 'angle':
            # Classification par angle
            unique_angles = sorted(list(set(labels)))
            angle_to_idx = {angle: idx for idx, angle in enumerate(unique_angles)}
            encoded_labels = np.array([angle_to_idx[label] for label in labels])
            class_labels = unique_angles
            
            print(f"Angles encodés : {len(unique_angles)} classes")
            print(f"Classes d'angle : {unique_angles}")
            
            # Si augmentation, répéter les étiquettes
            if config['data_augmentation']:
                encoded_labels = np.repeat(encoded_labels, config['augmentation_factor'])[:len(X_preprocessed)]
            
            return X_preprocessed, encoded_labels, class_labels
        
        else:
            # Classification standard (ID ou distances discrètes)
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y_encoded = np.array([label_to_idx[label] for label in labels])
            
            # Si augmentation, répéter les étiquettes
            if config['data_augmentation']:
                y_encoded = np.repeat(y_encoded, config['augmentation_factor'])[:len(X_preprocessed)]
            
            return X_preprocessed, y_encoded, unique_labels

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
        return X_train, y_train  # Remove print
    
    augmentation_factor = config.get('augmentation_factor', 2)
    
    if augmentation_factor <= 1:
        return X_train, y_train  # Remove warning print
    
    # Keep only essential prints
    print(f"Augmenting training data: {len(X_train)} → {len(X_train) * augmentation_factor} samples")
    
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

def preprocess_single_spectrum(spectrum_data, config):
    """
    Prétraite un spectre individual pour l'entraînement ou la prédiction
    
    Args:
        spectrum_data: Données spectrales (peut être un dict avec frequencies/amplitude/phase 
                      ou directement un array numpy)
        config (dict): Configuration de prétraitement
        
    Returns:
        np.array: Spectre prétraité au format [canaux, points]
    """
    target_size = config['num_frequency_points']
    window_size = config['window_size']
    
    # Handle different input formats
    if isinstance(spectrum_data, dict):
        # Format dict avec frequencies/amplitude/phase
        frequencies = spectrum_data['frequencies']
        amplitude = spectrum_data['amplitude']
        phase = spectrum_data['phase']
    elif isinstance(spectrum_data, np.ndarray):
        # Si c'est déjà un array numpy, on assume que c'est déjà prétraité
        if spectrum_data.shape[0] == 2 and spectrum_data.shape[1] == target_size:
            return spectrum_data  # Déjà au bon format
        elif len(spectrum_data.shape) == 2 and spectrum_data.shape[1] >= 2:
            # Données CSV avec colonnes frequency, amplitude, phase
            frequencies = spectrum_data[:, 0]
            amplitude = spectrum_data[:, 1]
            phase = spectrum_data[:, 2] if spectrum_data.shape[1] > 2 else np.zeros_like(amplitude)
        else:
            raise ValueError(f"Format de données non supporté: {spectrum_data.shape}")
    else:
        raise ValueError(f"Type de données non supporté: {type(spectrum_data)}")
    
    # Rééchantillonnage
    amplitude_resampled = resample_spectrum(frequencies, amplitude, target_size)
    phase_resampled = resample_spectrum(frequencies, phase, target_size)
    
    # Lissage
    amplitude_smoothed = smooth_spectrum(amplitude_resampled, window_size)
    phase_smoothed = smooth_spectrum(phase_resampled, window_size)
    
    # Normalisation
    amplitude_normalized = normalize_spectrum(amplitude_smoothed)
    phase_normalized = normalize_spectrum(phase_smoothed)
    
    # Format pour CNN: [canaux, points]
    spectrum_processed = np.stack([amplitude_normalized, phase_normalized], axis=0)
    
    return spectrum_processed