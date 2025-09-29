"""
Functions for evaluating the CNN model performance on RFID tag recognition tasks.

This module contains evaluation tools for assessing model performance across different
recognition modes (ID classification, distance/height estimation, and angle detection).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
import pandas as pd

def evaluate_model(model, X_test, y_test, class_labels, recognition_mode='id'):
    """
    Evaluate model performance and generate standard metrics
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: True labels (can be dict for multi-task)
        class_labels: List of class names or tuple for regression
        recognition_mode: 'id', 'distance', or 'multi'
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # For multi-task models
    if recognition_mode == 'multi':
        y_pred_prob = model.predict(X_test)
        
        # Handle ID predictions
        id_pred_prob = y_pred_prob[0]
        id_pred = np.argmax(id_pred_prob, axis=1)
        
        # Handle distance predictions
        dist_pred_prob = y_pred_prob[1]
        
        # If distance is regression (single value output)
        if dist_pred_prob.shape[1] == 1:
            dist_pred = dist_pred_prob.squeeze()
            
            # If we have range values in class_labels[1], denormalize
            if isinstance(class_labels[1], tuple) and len(class_labels[1]) == 2:
                min_dist, max_dist = class_labels[1]
                dist_range = max_dist - min_dist
                dist_pred = dist_pred * dist_range + min_dist
                
            # For regression, we can't use classification metrics
            id_report = classification_report(y_test['id_output'], id_pred, 
                                              target_names=class_labels[0], output_dict=True)
            id_conf_matrix = confusion_matrix(y_test['id_output'], id_pred)
            id_acc = accuracy_score(y_test['id_output'], id_pred)
            
            # Calculate regression metrics for distance
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            dist_mae = mean_absolute_error(y_test['distance_output'], dist_pred)
            dist_mse = mean_squared_error(y_test['distance_output'], dist_pred)
            
            print(f"ID accuracy: {id_acc:.4f}")
            print(f"Distance MAE: {dist_mae:.4f}, MSE: {dist_mse:.4f}")
            
            return {
                'id_predictions': id_pred,
                'id_probabilities': id_pred_prob,
                'id_classification_report': id_report,
                'id_confusion_matrix': id_conf_matrix,
                'id_accuracy': id_acc,
                'distance_predictions': dist_pred,
                'distance_mae': dist_mae,
                'distance_mse': dist_mse
            }
        else:
            # Both ID and distance are classification
            dist_pred = np.argmax(dist_pred_prob, axis=1)
            
            id_report = classification_report(y_test['id_output'], id_pred, 
                                             target_names=class_labels[0], output_dict=True)
            id_conf_matrix = confusion_matrix(y_test['id_output'], id_pred)
            id_acc = accuracy_score(y_test['id_output'], id_pred)
            
            dist_report = classification_report(y_test['distance_output'], dist_pred, 
                                               target_names=[str(d) for d in class_labels[1]], output_dict=True)
            dist_conf_matrix = confusion_matrix(y_test['distance_output'], dist_pred)
            dist_acc = accuracy_score(y_test['distance_output'], dist_pred)
            
            print(f"ID accuracy: {id_acc:.4f}")
            print(f"Distance accuracy: {dist_acc:.4f}")
            
            return {
                'id_predictions': id_pred,
                'id_probabilities': id_pred_prob,
                'id_classification_report': id_report,
                'id_confusion_matrix': id_conf_matrix,
                'id_accuracy': id_acc,
                'distance_predictions': dist_pred,
                'distance_probabilities': dist_pred_prob,
                'distance_classification_report': dist_report,
                'distance_confusion_matrix': dist_conf_matrix,
                'distance_accuracy': dist_acc
            }    # For distance or height regression
    elif recognition_mode in ['distance', 'height'] and isinstance(class_labels, tuple) and len(class_labels) == 2 and all(isinstance(x, (int, float)) for x in class_labels):
        # Handle regression output
        y_pred_normalized = model.predict(X_test).squeeze()
        
        # Regression mode - we need to denormalize values
        mode_name = 'Distance' if recognition_mode == 'distance' else 'Height'
        
        # Denormalize if class_labels has range values
        if class_labels and len(class_labels) == 2:
            min_val, max_val = class_labels
            val_range = max_val - min_val
            # Denormalize predictions
            y_pred = y_pred_normalized * val_range + min_val
            # Denormalize true values (assuming they were normalized during training)
            y_test_denorm = y_test * val_range + min_val
            
            print(f"Denormalizing predictions: {y_pred_normalized[:5]} -> {y_pred[:5]}")
            print(f"Denormalizing true values: {y_test[:5]} -> {y_test_denorm[:5]}")
        else:
            # Values are already in the correct range
            y_pred = y_pred_normalized
            y_test_denorm = y_test
        
        # Calculate regression metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test_denorm, y_pred)
        mse = mean_squared_error(y_test_denorm, y_pred)
        r2 = r2_score(y_test_denorm, y_pred)
        
        print(f"{mode_name} MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")        
        return {
            'predictions': y_pred,
            'true_values': y_test_denorm,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
    
    # Height classification - binary classification for h1/h2
    elif recognition_mode == 'height_class':
        # Get predictions
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate classification report
        from config import HEIGHT_CLASS_NAMES
        report = classification_report(y_test, y_pred, target_names=HEIGHT_CLASS_NAMES, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        # Format report for easier access
        formatted_report = {}
        for i, class_name in enumerate(HEIGHT_CLASS_NAMES):
            if str(i) in report:
                formatted_report[class_name] = report[str(i)]
            elif i in report:
                formatted_report[class_name] = report[i]
        
        # Add averages
        for avg in ['macro avg', 'weighted avg']:
            if avg in report:
                formatted_report[avg] = report[avg]
                
        print(f"Height classification accuracy: {acc:.4f}")
        
        # Return detailed results
        result = {
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'classification_report': formatted_report,
            'confusion_matrix': conf_matrix,
            'accuracy': acc,
            'class_names': HEIGHT_CLASS_NAMES
        }
        
        # Print detailed confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        print(f"Confusion Matrix Details:")
        print(f"  True Negative (TN): {tn} (Correct h1)")
        print(f"  False Positive (FP): {fp} (h1 misclassified as h2)")
        print(f"  False Negative (FN): {fn} (h2 misclassified as h1)")
        print(f"  True Positive (TP): {tp} (Correct h2)")
        
        # Calculate precision, recall, f1 manually
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        
        return result
    
    # Standard classification for ID or discrete distance
    elif recognition_mode == 'angle':
        # Standard classification approach for angles
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Angle classification accuracy: {acc:.4f}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': acc
        }
    
    else:
        # Get predictions
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Overall accuracy: {acc:.4f}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': acc
        }

def plot_confusion_matrix(conf_matrix, class_labels, save_path=None):
    """
    Plot and save confusion matrix without displaying
    """
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    # Close figure to free memory and avoid display
    plt.close(fig)

def analyze_errors(data, y_test, y_pred, class_labels, metadata):
    """
    Analyze and display information about misclassified samples or regression errors
    
    Args:
        data: Original data
        y_test: True labels or values
        y_pred: Predicted labels or values
        class_labels: List of class names or tuple (min_dist, max_dist) for regression 
        metadata: List of metadata for each sample
    """
    # Check if we're dealing with regression (distance recognition) or classification
    # For regression: class_labels is a tuple (min_dist, max_dist)
    # For classification: class_labels is a list of string labels
    is_regression = isinstance(class_labels, tuple) and len(class_labels) == 2 and all(isinstance(x, (int, float)) for x in class_labels)
    
    if is_regression:
        # For regression models (distance recognition)
        # Calculate absolute errors for each sample
        abs_errors = np.abs(y_test - y_pred)
        
        # Get indices of samples with the largest errors
        largest_errors_idx = np.argsort(abs_errors)[-5:][::-1]  # Top 5 largest errors
        
        print(f"\nLargest distance prediction errors:")
        for idx in largest_errors_idx:
            true_distance = y_test[idx]
            pred_distance = y_pred[idx]
            error = abs_errors[idx]
            
            # Extract useful metadata
            meta = metadata[idx] if idx < len(metadata) else {}
            tag_format = meta.get('tag_format', 'Unknown')
            file_name = meta.get('file_name', f'Sample_{idx}')
            
            print(f"Sample {idx} ({file_name}):")
            print(f"  True distance: {true_distance:.2f} cm")
            print(f"  Predicted distance: {pred_distance:.2f} cm")
            print(f"  Absolute error: {error:.2f} cm")
            print(f"  Tag Format: {tag_format}")
            
        # Calculate error statistics by distance range
        if len(y_test) > 0:
            distance_ranges = {}
            bin_size = 5  # Group by 5cm intervals
            
            for i in range(len(y_test)):
                true_dist = y_test[i]
                pred_dist = y_pred[i]
                bin_key = int(true_dist // bin_size) * bin_size
                
                if bin_key not in distance_ranges:
                    distance_ranges[bin_key] = {'errors': [], 'count': 0}
                    
                distance_ranges[bin_key]['errors'].append(abs(true_dist - pred_dist))
                distance_ranges[bin_key]['count'] += 1
            
            print("\nErrors by distance range:")
            for bin_key in sorted(distance_ranges.keys()):
                errors = distance_ranges[bin_key]['errors']
                count = distance_ranges[bin_key]['count']
                mean_error = np.mean(errors)
                
                print(f"  {bin_key}-{bin_key+bin_size} cm: {mean_error:.2f} cm average error ({count} samples)")
    else:
        # For classification models
        misclassified = np.where(y_test != y_pred)[0]
        
        print(f"\nNumber of misclassified samples: {len(misclassified)} out of {len(y_test)}")
        
        if len(misclassified) == 0:
            print("No misclassified samples found!")
            return
        
        print("\nSample of misclassifications:")
        for idx in misclassified[:5]:  # Show first 5 misclassifications
            true_label = class_labels[y_test[idx]]
            pred_label = class_labels[y_pred[idx]]
            
            # Extract useful metadata
            meta = metadata[idx]
            distance = meta.get('distance_cm', 'Unknown')
            tag_format = meta.get('tag_format', 'Unknown')
            
            print(f"Sample {idx}: True: {true_label}, Predicted: {pred_label}")
            print(f"  Distance: {distance}cm, Tag Format: {tag_format}")

def evaluate_by_distance(y_test, y_pred, metadata, class_labels=None):
    """
    Evaluate model performance stratified by distance
    
    Args:
        y_test: True labels or distance values
        y_pred: Predicted labels or distance values  
        metadata: List of metadata dictionaries
        class_labels: List of class names or tuple (min_dist, max_dist) for regression
        
    Returns:
        dict: Dictionary containing distance-based metrics
    """
    # Check if we're dealing with regression or classification
    is_regression = isinstance(class_labels, tuple) and len(class_labels) == 2 and all(isinstance(x, (int, float)) for x in class_labels)
    
    # Group samples by distance
    distances = {}
    for i in range(len(y_test)):
        meta = metadata[i]
        if 'distance_cm' in meta:
            distance = meta['distance_cm']
            
            if distance not in distances:
                distances[distance] = {
                    'y_true': [],
                    'y_pred': [],
                    'tag_formats': [],
                    'count': 0
                }
            
            distances[distance]['y_true'].append(y_test[i])
            distances[distance]['y_pred'].append(y_pred[i])
            distances[distance]['count'] += 1
            
            if 'tag_format' in meta:
                distances[distance]['tag_formats'].append(meta['tag_format'])
        # --- BEGIN: Balanced evaluation for regression ---
    if is_regression and len(distances) > 1:
        # Find the minimum sample count for any distance
        min_count = min([data['count'] for data in distances.values()])
        print(f"Balanced evaluation: using {min_count} samples per distance for metrics.")
        # For each distance, randomly select min_count samples
        for distance, data in distances.items():
            if data['count'] > min_count:
                np.random.seed(42)
                idx = np.random.choice(len(data['y_true']), min_count, replace=False)
                data['y_true'] = [data['y_true'][i] for i in idx]
                data['y_pred'] = [data['y_pred'][i] for i in idx]
                data['count'] = min_count
    # --- END: Balanced evaluation for regression ---
    # Calculate metrics for each distance
    distance_metrics = {}
    for distance, data in distances.items():
        if data['count'] > 0:
            y_true = np.array(data['y_true'])
            y_pred = np.array(data['y_pred'])
            
            if is_regression:
                # For regression, ensure both true and predicted values are denormalized
                min_dist, max_dist = class_labels
                dist_range = max_dist - min_dist
                
                # Check if true values need denormalization (if they're in 0-1 range)
                if np.all(y_true >= 0) and np.all(y_true <= 1):
                    # True values are normalized, denormalize them
                    y_true_denorm = y_true * dist_range + min_dist
                    print(f"Distance {distance}cm: Denormalizing true values from {y_true[:3]} to {y_true_denorm[:3]}")
                else:
                    y_true_denorm = y_true
                
                # Check if predicted values need denormalization (if they're in 0-1 range)
                if np.all(y_pred >= 0) and np.all(y_pred <= 1):
                    # Predicted values are normalized, denormalize them
                    y_pred_denorm = y_pred * dist_range + min_dist
                    print(f"Distance {distance}cm: Denormalizing predicted values from {y_pred[:3]} to {y_pred_denorm[:3]}")
                else:
                    y_pred_denorm = y_pred
                
                # Calculate regression metrics using denormalized values
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
                mse = mean_squared_error(y_true_denorm, y_pred_denorm)
                r2 = r2_score(y_true_denorm, y_pred_denorm)
                
                # Calculate additional metrics
                rmse = np.sqrt(mse)
                mean_true = np.mean(y_true_denorm)
                mean_pred = np.mean(y_pred_denorm)
                
                distance_metrics[distance] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mean_true': mean_true,
                    'mean_pred': mean_pred,
                    'sample_count': data['count']
                }
            else:
                # Calculate classification metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # Calculate precision for each class and average
                if class_labels:
                    precision_per_class = {}
                    for i, label in enumerate(class_labels):
                        # Check if this class exists in this distance group
                        if i in y_true:
                            # Binary precision for this class (one-vs-rest)
                            y_true_binary = (y_true == i)
                            y_pred_binary = (y_pred == i)
                            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                            precision_per_class[label] = precision
                    
                    # Calculate macro average precision
                    if precision_per_class:
                        macro_precision = np.mean(list(precision_per_class.values()))
                    else:
                        macro_precision = 0
                else:
                    # Use micro average if class labels not provided
                    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    precision_per_class = {}
                    
                # Count tag formats
                tag_counts = {}
                if data['tag_formats']:
                    for tag in data['tag_formats']:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Store metrics
                distance_metrics[distance] = {
                    'accuracy': accuracy,
                    'precision': macro_precision,
                    'precision_per_class': precision_per_class,
                    'sample_count': data['count'],
                    'tag_distribution': tag_counts
                }
    
    # Sort by distance
    distance_metrics = {k: distance_metrics[k] for k in sorted(distance_metrics.keys())}
    
    return distance_metrics

def plot_performance_by_distance(distance_metrics, save_path=None):
    """
    Create visualizations of model performance by distance
    
    Args:
        distance_metrics: Dictionary containing metrics by distance
        save_path: Base path for saving plots
    """
    if not distance_metrics:
        print("No distance metrics available to plot")
        return
    
    # Check if we have regression or classification metrics
    first_metric = next(iter(distance_metrics.values()))
    is_regression = 'mae' in first_metric  # If MAE is present, it's regression
    
    # Extract distances and sample counts (common to both types)
    distances = list(distance_metrics.keys())
    sample_counts = [metrics['sample_count'] for metrics in distance_metrics.values()]
    
    if is_regression:
        # Handle regression metrics
        maes = [metrics['mae'] for metrics in distance_metrics.values()]
        rmses = [metrics['rmse'] for metrics in distance_metrics.values()]
        r2_scores = [metrics['r2'] for metrics in distance_metrics.values()]
        
        # Create regression performance plot
        fig1 = plt.figure(figsize=(14, 10))
        
        # Plot 1: MAE and RMSE
        plt.subplot(2, 2, 1)
        plt.plot(distances, maes, marker='o', color='blue', label='MAE')
        plt.plot(distances, rmses, marker='s', color='red', label='RMSE')
        plt.xlabel('Distance (cm)')
        plt.ylabel('Error (cm)')
        plt.title('Mean Absolute Error and RMSE by Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: R² Score
        plt.subplot(2, 2, 2)
        plt.plot(distances, r2_scores, marker='d', color='green', label='R²')
        plt.xlabel('Distance (cm)')
        plt.ylabel('R² Score')
        plt.title('R² Score by Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Sample count
        plt.subplot(2, 2, 3)
        plt.bar(distances, sample_counts, alpha=0.7, color='gray')
        plt.xlabel('Distance (cm)')
        plt.ylabel('Sample Count')
        plt.title('Sample Distribution by Distance')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Combined view
        plt.subplot(2, 2, 4)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot MAE on left axis
        line1 = ax1.plot(distances, maes, marker='o', color='blue', label='MAE')
        ax1.set_xlabel('Distance (cm)')
        ax1.set_ylabel('MAE (cm)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot sample count on right axis
        bars = ax2.bar(distances, sample_counts, alpha=0.3, color='gray', label='Sample Count')
        ax2.set_ylabel('Sample Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        ax1.set_title('MAE and Sample Distribution by Distance')
        ax1.grid(True, alpha=0.3)
        
    else:
        # Handle classification metrics
        accuracies = [metrics['accuracy'] for metrics in distance_metrics.values()]
        precisions = [metrics['precision'] for metrics in distance_metrics.values()]
        
        # 1. Overall Accuracy and Precision by Distance
        fig1 = plt.figure(figsize=(12, 7))
        
        # Plot bars for sample count on a secondary axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot lines for accuracy and precision
        sns.lineplot(x=distances, y=accuracies, marker='o', color='blue', label='Accuracy', ax=ax1)
        sns.lineplot(x=distances, y=precisions, marker='s', color='red', label='Precision', ax=ax1)
        
        # Plot bars for sample count
        ax2.bar(distances, sample_counts, alpha=0.3, color='gray', label='Sample Count')
        
        # Customize plot
        ax1.set_xlabel('Distance (cm)')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance by Distance')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        ax2.set_ylabel('Sample Count')
          # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1 + ['Sample Count'], labels1 + ['Sample Count'], loc='lower left')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        # Replace extension with _performance.png
        base_path = os.path.splitext(save_path)[0]
        performance_path = f"{base_path}_performance.png"
        plt.savefig(performance_path, dpi=300, bbox_inches='tight')
        print(f"Performance by distance plot saved to {performance_path}")
    
    # Close figure to free memory and avoid display
    plt.close(fig1)