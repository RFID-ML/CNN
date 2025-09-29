"""
Visualization functions for RFID spectral data and model performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

def plot_spectrum(spectrum_data, title=None, save_path=None):
    """
    Plot and save spectrum data without displaying
    
    Args:
        spectrum_data (dict): Dictionary containing frequencies, amplitude, and phase
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
    """
    frequencies = spectrum_data['frequencies']
    amplitude = spectrum_data['amplitude']
    phase = spectrum_data['phase']
    
    fig = plt.figure(figsize=(12, 8))
    
    # Amplitude plot
    plt.subplot(2, 1, 1)
    plt.plot(frequencies, amplitude)
    plt.title('Amplitude' if title is None else f"{title} - Amplitude")
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True, alpha=0.3)
    
    # Phase plot
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, phase)
    plt.title('Phase')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close figure to free memory and avoid display
    plt.close(fig)

def plot_processed_data(X_processed, sample_idx, title=None, save_path=None):
    """
    Plot and save processed data without displaying
    
    Args:
        X_processed (np.array): Processed data array [samples, channels, features]
        sample_idx (int): Index of sample to plot
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
    """
    # Extract sample
    sample = X_processed[sample_idx]
    
    fig = plt.figure(figsize=(12, 8))
    
    # Amplitude
    plt.subplot(2, 1, 1)
    plt.plot(sample[0])
    plt.title('Processed Amplitude' if title is None else f"{title} - Amplitude")
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Phase
    plt.subplot(2, 1, 2)
    plt.plot(sample[1])
    plt.title('Processed Phase')
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Phase')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close figure to free memory and avoid display
    plt.close(fig)

def plot_spectrum_processing_stages(spectrum_data, title=None, save_path=None):
    """
    Plot and save all stages of spectrum processing without displaying
    
    Args:
        spectrum_data (dict): Dictionary containing frequencies, amplitude, and phase
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
    """
    from src.data.preprocessing import resample_spectrum, smooth_spectrum, normalize_spectrum
    
    # Extract data
    frequencies = spectrum_data['frequencies']
    amplitude = spectrum_data['amplitude']
    phase = spectrum_data['phase']
    
    # Process data through each stage
    target_size = 1000  # From config
    
    # Stage 1: Resample
    amplitude_resampled = resample_spectrum(frequencies, amplitude, target_size)
    phase_resampled = resample_spectrum(frequencies, phase, target_size)
    
    # Stage 2: Smooth
    amplitude_smoothed = smooth_spectrum(amplitude_resampled)
    phase_smoothed = smooth_spectrum(phase_resampled)
    
    # Stage 3: Normalize
    amplitude_normalized = normalize_spectrum(amplitude_smoothed)
    phase_normalized = normalize_spectrum(phase_smoothed)
    
    # Create x-axis for resampled data
    x_resampled = np.linspace(min(frequencies), max(frequencies), target_size)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Original Amplitude
    plt.subplot(3, 2, 1)
    plt.plot(frequencies, amplitude)
    plt.title('Original Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Original Phase
    plt.subplot(3, 2, 2)
    plt.plot(frequencies, phase)
    plt.title('Original Phase')
    plt.grid(True, alpha=0.3)
    
    # Resampled & Smoothed Amplitude
    plt.subplot(3, 2, 3)
    plt.plot(x_resampled, amplitude_resampled, alpha=0.5, label='Resampled')
    plt.plot(x_resampled, amplitude_smoothed, label='Smoothed')
    plt.title('Resampled & Smoothed Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Resampled & Smoothed Phase
    plt.subplot(3, 2, 4)
    plt.plot(x_resampled, phase_resampled, alpha=0.5, label='Resampled')
    plt.plot(x_resampled, phase_smoothed, label='Smoothed')
    plt.title('Resampled & Smoothed Phase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Normalized Amplitude
    plt.subplot(3, 2, 5)
    plt.plot(x_resampled, amplitude_normalized)
    plt.title('Normalized Amplitude (CNN Input)')
    plt.xlabel('Frequency (GHz)')
    plt.grid(True, alpha=0.3)
    
    # Normalized Phase
    plt.subplot(3, 2, 6)
    plt.plot(x_resampled, phase_normalized)
    plt.title('Normalized Phase (CNN Input)')
    plt.xlabel('Frequency (GHz)')
    plt.grid(True, alpha=0.3)
    
    # Main title
    if title:
        plt.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Always close the figure to avoid display
    plt.close(fig)

def plot_performance_by_distance(distance_metrics, save_path=None):
    """
    Create detailed visualizations of model performance by distance
    
    Args:
        distance_metrics: Dictionary containing metrics by distance
        save_path: Path for saving the plot
    """
    # This function is now implemented in the evaluate.py file
    # This stub is here for backwards compatibility
    from src.training.evaluate import plot_performance_by_distance as plot_perf
    return plot_perf(distance_metrics, save_path)

def plot_class_distribution(labels, metadata, save_path=None, recognition_mode='id'):
    """
    Plot the distribution of classes in the dataset
    """
    plt.figure(figsize=(12, 8))
    
    # Special handling for angle mode
    if recognition_mode == 'angle':
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Angle distribution
        angle_counts = {}
        for label in labels:
            angle_counts[label] = angle_counts.get(label, 0) + 1
        
        # Sort angles for better visualization
        sorted_angles = sorted(angle_counts.keys())
        counts = [angle_counts[angle] for angle in sorted_angles]
        
        ax1.bar(sorted_angles, counts)
        ax1.set_title('Angle Distribution')
        ax1.set_xlabel('Angle')
        ax1.set_ylabel('Count')
        
        # Plot 2: Angle-Distance heatmap
        angle_distance_matrix = {}
        for i, label in enumerate(labels):
            angle = label
            distance = metadata[i].get('distance_cm', 'unknown')
            
            if angle not in angle_distance_matrix:
                angle_distance_matrix[angle] = {}
            if distance not in angle_distance_matrix[angle]:
                angle_distance_matrix[angle][distance] = 0
            angle_distance_matrix[angle][distance] += 1
        
        # Convert to numpy array for heatmap
        all_angles = sorted(angle_distance_matrix.keys())
        
        # Handle None values by converting to 'unknown' string
        distance_values = []
        for meta in metadata:
            dist = meta.get('distance_cm')
            if dist is None:
                distance_values.append('unknown')
            else:
                distance_values.append(dist)
        
        # Get unique distances and sort them properly (numeric first, then 'unknown')
        unique_distances = set(distance_values)
        numeric_distances = sorted([d for d in unique_distances if isinstance(d, (int, float))])
        all_distances = numeric_distances + (['unknown'] if 'unknown' in unique_distances else [])
        
        # Create heatmap data
        heatmap_data = np.zeros((len(all_angles), len(all_distances)))
        for i, angle in enumerate(all_angles):
            for j, dist in enumerate(all_distances):
                heatmap_data[i, j] = angle_distance_matrix.get(angle, {}).get(dist, 0)
        
        # Create heatmap
        im = ax2.imshow(heatmap_data, cmap='viridis')
        ax2.set_title('Angle-Distance Distribution')
        ax2.set_xlabel('Distance (cm)')
        ax2.set_ylabel('Angle')
        
        # Set x and y ticks
        ax2.set_xticks(np.arange(len(all_distances)))
        ax2.set_yticks(np.arange(len(all_angles)))
        ax2.set_xticklabels(all_distances)
        ax2.set_yticklabels(all_angles)
        
        # Add colorbar
        fig.colorbar(im, ax=ax2, label='Count')
        
        plt.tight_layout()
        
    elif recognition_mode == 'multi':
        # Extract IDs and distances from label tuples
        ids = [label[0] for label in labels]
        distances = [label[1] for label in labels]
        
        # Plot ID distribution
        plt.subplot(2, 2, 1)
        unique_ids = np.unique(ids)
        id_counts = [ids.count(id_val) for id_val in unique_ids]
        
        sns.barplot(x=unique_ids, y=id_counts)
        plt.title('ID Distribution')
        plt.xlabel('ID')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot distance distribution
        plt.subplot(2, 2, 2)
        unique_distances = np.unique(distances)
        distance_counts = [distances.count(dist) for dist in unique_distances]
        
        sns.barplot(x=unique_distances, y=distance_counts)
        plt.title('Distance Distribution')
        plt.xlabel('Distance (cm)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Plot ID distribution by distance
        plt.subplot(2, 1, 2)
        
        # Group by distance and ID
        import pandas as pd
        data = []
        
        for i in range(len(labels)):
            id_val = ids[i]
            distance = distances[i]
            data.append({'Distance (cm)': distance, 'ID': id_val})
        
        df = pd.DataFrame(data)
        
        # Create count plot
        sns.countplot(data=df, x='Distance (cm)', hue='ID')
        plt.title('ID Distribution by Distance')
        plt.grid(True, alpha=0.3)
        plt.legend(title='ID')
        plt.xticks(rotation=45)
        
    elif recognition_mode == 'distance':
        # Plot distance distribution
        plt.subplot(2, 1, 1)
        unique_distances = np.unique(labels)
        distance_counts = [list(labels).count(dist) for dist in unique_distances]
        
        sns.barplot(x=unique_distances, y=distance_counts)
        plt.title('Distance Distribution')
        plt.xlabel('Distance (cm)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # If metadata available, plot additional information
        if metadata and any('tag_format' in meta for meta in metadata):
            plt.subplot(2, 1, 2)
            
            # Group by distance and tag format
            import pandas as pd
            data = []
            
            for i, distance in enumerate(labels):
                if 'tag_format' in metadata[i]:
                    tag_format = metadata[i]['tag_format']
                    data.append({'Distance (cm)': distance, 'Tag Format': tag_format})
            
            df = pd.DataFrame(data)
            
            # Create count plot
            sns.countplot(data=df, x='Distance (cm)', hue='Tag Format')
            plt.title('Distance Distribution by Tag Format')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Tag Format')
            plt.xticks(rotation=45)
    
    elif recognition_mode == 'angle':
        plt.title('Angle Distribution')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Sample Count')
    
    else:  # Default 'id' mode
        # Plot overall class distribution
        plt.subplot(2, 1, 1)
        unique_labels = np.unique(labels)
        counts = [list(labels).count(label) for label in unique_labels]
        
        sns.barplot(x=unique_labels, y=counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # If metadata available, plot distribution by distance
        if metadata and any('distance_cm' in meta for meta in metadata):
            plt.subplot(2, 1, 2)
            
            # Group by distance and class
            distance_class_counts = {}
            
            for i, label in enumerate(labels):
                if 'distance_cm' in metadata[i]:
                    distance = metadata[i]['distance_cm']
                    
                    # Handle None values by converting to string representation
                    if distance is None:
                        distance = 'unknown'
                    
                    if distance not in distance_class_counts:
                        distance_class_counts[distance] = {}
                    
                    if label not in distance_class_counts[distance]:
                        distance_class_counts[distance][label] = 0
                    
                    distance_class_counts[distance][label] += 1
            
            # Convert to DataFrame for plotting
            import pandas as pd
            data = []
            
            # Define a custom key function for sorting that handles 'unknown'
            def sort_key(x):
                if x == 'unknown' or x is None:
                    return float('inf')  # Put unknown at the end
                return x
            
            # Sort distances with 'unknown' at the end
            sorted_distances = sorted(distance_class_counts.keys(), key=sort_key)
            
            for distance in sorted_distances:
                for label in unique_labels:
                    count = distance_class_counts[distance].get(label, 0)
                    data.append({'Distance (cm)': distance, 'Class': label, 'Count': count})
            
            df = pd.DataFrame(data)
            
            # Plot
            sns.barplot(data=df, x='Distance (cm)', y='Count', hue='Class')
            plt.title('Class Distribution by Distance')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Class')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_regression_results(y_true, y_pred, title="Regression Results", save_path=None):
    """
    Plot regression results showing predicted vs true values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot of true vs predicted
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    # Calculate error stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Add stats to plot
    stats_text = f"MAE: {mae:.2f}cm\nRMSE: {rmse:.2f}cm\nR²: {r2:.3f}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    ax.text(0.15, 0.90, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=bbox_props)
    
    # Add regression line (only if there's variation in the data)
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        from scipy import stats
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
            regression_line = intercept + slope * np.array([min_val, max_val])
            ax.plot([min_val, max_val], regression_line, 'g-', label=f'Regression line (slope={slope:.2f})')
        except ValueError as e:
            print(f"Warning: Could not calculate regression line: {e}")
    else:
        print("Warning: Insufficient variation in data for regression line")
    
    # Labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    
    # Equal axes
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close(fig)