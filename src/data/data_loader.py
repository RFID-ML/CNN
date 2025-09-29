"""
Functions to load RFID spectral data with support for filtered data from processed directory
"""

import os
import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm

# Import paths from config
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def extract_metadata_from_filename(filename):
    """
    Extract metadata from filename following the pattern:
    M_Sp_XY_ID_S_Dcm_H_A.csv
    
    Where:
    - M: ID of the measure
    - Sp: ID of the sample
    - XY: X=Number of resonances, Y=Number of resonators
    - ID: Tag identifier format ("1", "11", or "111")
    - S: Substrate
    - Dcm: Distance in cm
    - H: Height (h1 or h2)
    - A: Angle in degrees
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Remove _filtered suffix if present
    if name_without_ext.endswith("_filtered"):
        name_without_ext = name_without_ext[:-9]
        
    parts = name_without_ext.split('_')
    
    metadata = {
        "file_name": basename,
    }
    
    # Parse parts according to the new convention
    if len(parts) >= 1:
        metadata["measure_id"] = parts[0]
    
    if len(parts) >= 2:
        metadata["sample_id"] = parts[1]
    
    # Extract resonance and resonator info (part 3: XY)
    if len(parts) > 2 and parts[2]:
        xy_info = parts[2]
        if len(xy_info) >= 2:
            metadata["num_resonances"] = int(xy_info[0])
            metadata["num_resonators"] = int(xy_info[1])
    
    # Extract tag format (part 4: ID)
    if len(parts) > 3:
        metadata["tag_format"] = parts[3]
    
    # Extract substrate (part 5: S)
    if len(parts) > 4:
        metadata["substrate"] = parts[4]
    
    # Extract distance (part 6: Dcm)
    if len(parts) > 5:
        distance_str = parts[5]
        if distance_str.endswith('cm'):
            distance_str = distance_str[:-2]
        try:
            distance = float(distance_str)
            metadata["distance_cm"] = distance
        except ValueError:
            metadata["distance_cm"] = None
    
    # Extract height (part 7: H)
    if len(parts) > 6 and parts[6]:
        height_str = parts[6]
        if height_str == 'h1':
            metadata["height_cm"] = 4.0
            metadata["height_class"] = 0
        elif height_str == 'h2':
            metadata["height_cm"] = 11.5
            metadata["height_class"] = 1
        else:
            try:
                height = float(height_str)
                metadata["height_cm"] = height
                metadata["height_class"] = 0 if height < 7.75 else 1
            except ValueError:
                metadata["height_cm"] = None
                metadata["height_class"] = None
    
    # Extract angle (part 8: A) - THIS IS THE KEY ADDITION
    if len(parts) > 7:
        angle_str = parts[7]
        try:
            # Handle angle formats like "30", "-30", "0"
            angle_value = int(angle_str)
            metadata["angle"] = angle_value
        except ValueError:
            # Handle formats like "30deg", "-30deg"
            import re
            angle_match = re.search(r'(-?\d+)', angle_str)
            if angle_match:
                metadata["angle"] = int(angle_match.group(1))
            else:
                metadata["angle"] = None
    
    return metadata

def find_filtered_version(file_path):
    """
    Look for a filtered version of the file in the processed directory
    
    Args:
        file_path (str): Path to the original CSV file
        
    Returns:
        str: Path to the filtered version if it exists, otherwise the original path
    """
    basename = os.path.basename(file_path)
    base_name = os.path.splitext(basename)[0]
    filtered_filename = f"{base_name}_filtered.csv"
    filtered_path = os.path.join(PROCESSED_DATA_DIR, filtered_filename)
    
    if os.path.exists(filtered_path):
        return filtered_path
    
    return file_path

def find_data_columns(df):
    """Find the frequency, amplitude and phase columns in the dataframe."""
    # Initialize column names
    freq_col = None
    amp_col = None
    phase_col = None
    
    # Look for standard column names
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['freq', 'hz', 'ghz']):
            freq_col = col
        elif 'db' in col_lower and ('s21' in col_lower or 's12' in col_lower):
            amp_col = col
        elif 'deg' in col_lower and ('s21' in col_lower or 's12' in col_lower):
            phase_col = col
    
    # If specific columns not found, try general patterns
    if not freq_col:
        freq_col = next((col for col in df.columns if any(x in col.lower() for x in ['freq', 'hz', 'ghz'])), df.columns[0])
    if not amp_col:
        amp_col = next((col for col in df.columns if 'db' in col.lower()), None)
    if not phase_col:
        phase_col = next((col for col in df.columns if 'deg' in col.lower()), None)
    
    # Last resort: use positions
    if not freq_col and len(df.columns) > 0:
        freq_col = df.columns[0]
    if not amp_col and len(df.columns) > 4:
        amp_col = df.columns[4]  # Usually S21(DB) is column 5
    if not phase_col and len(df.columns) > 5:
        phase_col = df.columns[5]  # Usually S21(DEG) is column 6
    
    return freq_col, amp_col, phase_col

def load_csv_spectrum(file_path):
    """
    Loads a CSV file containing RFID spectral data.
    Automatically tries to use the filtered version if available.
    
    Args:
        file_path (str): Path to the original CSV file
        
    Returns:
        dict: Dictionary containing spectral data
    """
    try:
        # First, check if a filtered version exists
        file_path = find_filtered_version(file_path)
        
        # Read the entire file content
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the start and end of the data section
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if 'BEGIN CH1_DATA' in line:
                start_idx = i
            elif 'END' in line and start_idx is not None:
                end_idx = i
                break
        
        # Extract the data section if it's the special format
        if start_idx is not None:
            header_idx = start_idx + 1  # Header is the line after BEGIN
            
            # Extract the data lines
            if end_idx is not None:
                data_lines = lines[header_idx:end_idx]
            else:
                data_lines = lines[header_idx:]
            
            # Parse CSV from extracted lines
            import io
            data = pd.read_csv(io.StringIO(''.join(data_lines)))
        else:
            # Try standard format with comment handling
            try:
                data = pd.read_csv(file_path, comment='!')
            except:
                # If first line starts with //, try skipping it
                if lines and lines[0].strip().startswith('//'):
                    data = pd.read_csv(file_path, skiprows=1)
                else:
                    # Last resort: try without special handling
                    data = pd.read_csv(file_path)
        
        # Find columns for frequency, amplitude, and phase
        freq_col, amp_col, phase_col = find_data_columns(data)
        
        if not freq_col:
            print(f"Error: Could not find frequency column in {file_path}")
            return None
            
        # Convert to numeric values with error handling
        frequencies = pd.to_numeric(data[freq_col], errors='coerce').values
        
        # Initialize amplitude and phase arrays
        amplitude = np.zeros_like(frequencies)
        phase = np.zeros_like(frequencies)
        
        # Extract amplitude and phase if columns are found
        if amp_col:
            amplitude = pd.to_numeric(data[amp_col], errors='coerce').values
        else:
            print(f"Warning: No amplitude column found in {file_path}")
            
        if phase_col:
            phase = pd.to_numeric(data[phase_col], errors='coerce').values
        else:
            print(f"Warning: No phase column found in {file_path}")
        
        # Handle potential NaN values
        frequencies = np.nan_to_num(frequencies)
        amplitude = np.nan_to_num(amplitude)
        phase = np.nan_to_num(phase)
        
        # Convert Hz to GHz if needed
        if np.mean(frequencies) > 1e6:  # Likely in Hz
            frequencies = frequencies / 1e9
        
        return {
            'file_path': file_path,
            'frequencies': frequencies,
            'amplitude': amplitude,
            'phase': phase,
            'is_filtered': '_filtered' in os.path.basename(file_path)
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_mapping_file(mapping_file):
    """
    Load the JSON mapping file that associates file paths with tag IDs.
    
    Args:
        mapping_file (str): Path to the JSON mapping file
        
    Returns:
        dict: Mapping dictionary
    """
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        # Create dictionaries for easy access
        file_to_tag_map = {}
        file_to_metadata_map = {}
        
        for file_entry in mapping_data.get('files', []):
            # Extract the filename without path (for more robustness)
            file_path = os.path.normpath(file_entry['file_path'])
            filename = os.path.basename(file_path)
            tag_id = file_entry['tag_id']
            metadata = file_entry.get('metadata', {})
            
            # Store with and without path for robustness
            file_to_tag_map[file_path] = tag_id
            file_to_tag_map[filename] = tag_id
            file_to_metadata_map[file_path] = metadata
            file_to_metadata_map[filename] = metadata
        
        return {
            'file_to_tag': file_to_tag_map,
            'file_to_metadata': file_to_metadata_map,
            'dataset_info': mapping_data.get('dataset_info', {})
        }
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return {'file_to_tag': {}, 'file_to_metadata': {}, 'dataset_info': {}}

def get_processed_files():
    """
    Get a list of all filtered files in the processed directory
    
    Returns:
        dict: Mapping of original filenames to processed filenames
    """
    processed_files = {}
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        return processed_files
        
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if filename.endswith('_filtered.csv'):
            original_name = filename.replace('_filtered.csv', '.csv')
            processed_files[original_name] = filename
    
    return processed_files

def load_dataset(data_dir=RAW_DATA_DIR, mapping_file=None, limit=None, use_filtered=True, recognition_mode='id'):
    """
    Load the entire dataset from a directory.
    Prioritizes filtered versions of files if available.
    
    Args:
        data_dir (str): Directory containing the CSV files
        mapping_file (str): Path to the mapping file
        limit (int, optional): Limit the number of files to load (for testing)
        use_filtered (bool): Whether to prioritize filtered files
        recognition_mode (str): 'id', 'distance', 'height', 'height_class', 'multi', or 'angle' - determines what to use as labels
        
    Returns:
        tuple: (data, labels, metadata)
    """
    # For angle recognition, always use raw files (no filtering)
    if recognition_mode == 'angle':
        use_filtered = False
        print("Angle recognition mode: Using raw files only (no filtering)")
    
    # Get processed files mapping
    processed_files = get_processed_files() if use_filtered else {}
    
    # Load mapping if provided
    mapping = {}
    if mapping_file and os.path.exists(mapping_file):
        mapping = load_mapping_file(mapping_file)
        file_to_tag = mapping['file_to_tag']
    else:
        file_to_tag = {}
    
    data = []
    labels = []
    metadata_list = []
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Apply limit if specified
    if limit is not None:
        csv_files = csv_files[:limit]
    
    for filename in tqdm(csv_files, desc="Loading files"):
        file_path = os.path.join(data_dir, filename)
        
        # Extract metadata from filename
        file_metadata = extract_metadata_from_filename(filename)
        
        # Skip files without distance_cm metadata if in distance mode
        if recognition_mode in ['distance', 'multi'] and 'distance_cm' not in file_metadata:
            print(f"Skipping {filename} - no distance metadata found")
            continue
            
        # Determine the tag ID - this should be the ID format we want to classify
        tag_id = None
        
        # Try to get tag ID from mapping file first
        if filename in file_to_tag:
            tag_id = file_to_tag[filename]
        elif file_path in file_to_tag:
            tag_id = file_to_tag[file_path]
        else:
            # Fall back to using the tag_format field from metadata
            tag_format = file_metadata.get("tag_format")
            if tag_format:
                tag_id = f"Format_{tag_format}"
            else:
                # If we can't extract the tag format, use a default
                tag_id = "Unknown_Format"
        
        # Load the spectrum data (will automatically use filtered version if available)
        spectrum_data = load_csv_spectrum(file_path)
        
        if spectrum_data is not None:
            data.append(spectrum_data)
            
            # Set label based on recognition mode
            if recognition_mode == 'id':
                labels.append(tag_id)
            elif recognition_mode == 'distance':
                if 'distance_cm' in file_metadata:
                    labels.append(file_metadata['distance_cm'])
                else:
                    print(f"Skipping {filename} - no distance metadata found")
                    continue
            elif recognition_mode == 'height':
                if 'height_cm' in file_metadata:
                    labels.append(file_metadata['height_cm'])
                else:
                    print(f"Skipping {filename} - no height metadata found")
                    continue
            elif recognition_mode == 'height_class':
                if 'height_class' in file_metadata:
                    labels.append(file_metadata['height_class'])
                else:
                    print(f"Skipping {filename} - no height class metadata found")
                    continue
            elif recognition_mode == 'multi':
                if 'distance_cm' in file_metadata:
                    labels.append((tag_id, file_metadata['distance_cm']))
                else:
                    print(f"Skipping {filename} - no distance metadata for multi-task")
                    continue
            elif recognition_mode == 'angle':
                # For angle recognition, extract angle from metadata
                angle = file_metadata.get('angle')
                if angle is not None:
                    # Convert to string to ensure consistent format
                    labels.append(str(angle))
                else:
                    print(f"Skipping {filename} - no angle metadata found")
                    continue
            else:
                # Default to ID recognition
                labels.append(tag_id)
            
            # Merge metadata from mapping file (if available) with filename metadata
            combined_metadata = file_metadata
            if mapping and filename in mapping['file_to_metadata']:
                mapping_metadata = mapping['file_to_metadata'][filename]
                combined_metadata.update(mapping_metadata)
            
            metadata_list.append(combined_metadata)
    
    print(f"Loaded {len(data)} files ({len([d for d in data if d.get('is_filtered', False)])} filtered)")
    
    # Filter for height_class mode to only include files with valid height class
    if recognition_mode == 'height_class':
        print("\nFiltering files with valid height classes...")
        valid_indices = []
        for i, (label, meta) in enumerate(zip(labels, metadata_list)):
            if label is not None and 'height_class' in meta and meta['height_class'] in [0, 1]:
                valid_indices.append(i)
        
        if len(valid_indices) < len(data):
            print(f"Filtering: {len(data)} -> {len(valid_indices)} files with valid height classes")
            data = [data[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            metadata_list = [metadata_list[i] for i in valid_indices]
        
        print(f"Height class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Validate distance ranges for distance-based recognition modes
    if recognition_mode in ['distance', 'multi']:
        print("\nValidating distance ranges...")
        valid_metadata, invalid_count, distance_stats = validate_distance_range(metadata_list)
        
        if invalid_count > 0:
            print(f"Filtering out {invalid_count} files with invalid distances")
            # Filter out invalid files from data, labels, and metadata
            valid_indices = [i for i, meta in enumerate(metadata_list) 
                           if meta in valid_metadata or 'distance_cm' not in meta]
            
            data = [data[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            metadata_list = valid_metadata
        
        print(f"Final dataset: {len(data)} files with distance range [{distance_stats['min']}-{distance_stats['max']}]cm")
        print(f"Unique distances: {distance_stats['unique_values']}")
    
    # Validate height ranges for height-based recognition modes
    elif recognition_mode in ['height', 'height_class']:
        print("\nValidating height ranges...")
        valid_metadata, invalid_count, height_stats = validate_height_range(metadata_list)
        
        if invalid_count > 0:
            print(f"Filtering out {invalid_count} files with invalid heights")
            # Filter out invalid files from data, labels, and metadata
            valid_indices = [i for i, meta in enumerate(metadata_list) 
                           if meta in valid_metadata or 'height_cm' not in meta]
            
            data = [data[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            metadata_list = valid_metadata
        
        print(f"Final dataset: {len(data)} files with height range [{height_stats['min']}-{height_stats['max']}]cm")
        print(f"Unique heights: {height_stats['unique_values']}")
        if recognition_mode == 'height_class':
            print(f"Height classes: {len(set(labels))} classes")
    
    # Keep original height validation for backward compatibility
    elif recognition_mode == 'height':
        print("\nValidating height ranges...")
        valid_metadata, invalid_count, height_stats = validate_height_range(metadata_list)
        
        if invalid_count > 0:
            print(f"Filtering out {invalid_count} files with invalid heights")
            # Filter out invalid files from data, labels, and metadata
            valid_indices = [i for i, meta in enumerate(metadata_list) 
                           if meta in valid_metadata or 'height_cm' not in meta]
            
            data = [data[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            metadata_list = valid_metadata
        
        print(f"Final dataset: {len(data)} files with height range [{height_stats['min']}-{height_stats['max']}]cm")
        print(f"Unique heights: {height_stats['unique_values']}")
    
    # Validate angle values for angle recognition mode
    elif recognition_mode == 'angle':
        print("\nValidating angle values...")
        # Count angle distribution
        angle_counts = {}
        for label in labels:
            angle_counts[label] = angle_counts.get(label, 0) + 1
        
        print(f"Angle distribution: {angle_counts}")
        expected_angles = ['-30', '0', '30']
        found_angles = list(angle_counts.keys())
        
        if not all(angle in found_angles for angle in expected_angles):
            print(f"Warning: Expected angles {expected_angles}, found {found_angles}")
        
        print(f"Final dataset: {len(data)} files with angles")
    
    return data, labels, metadata_list

def validate_distance_range(metadata_list, valid_range=(1, 49)):
    """
    Validate that distance values are within the expected range for RFID measurements
    
    Args:
        metadata_list: List of metadata dictionaries
        valid_range: Tuple of (min_distance, max_distance) in cm
        
    Returns:
        tuple: (valid_metadata, invalid_count, distance_stats)
    """
    valid_metadata = []
    invalid_distances = []
    min_valid, max_valid = valid_range
    
    for meta in metadata_list:
        if 'distance_cm' in meta:
            distance = meta['distance_cm']
            if min_valid <= distance <= max_valid:
                valid_metadata.append(meta)
            else:
                invalid_distances.append((meta['file_name'], distance))
                print(f"Warning: Invalid distance {distance}cm in file {meta['file_name']} - outside range [{min_valid}-{max_valid}]cm")
        else:
            # Files without distance information are kept
            valid_metadata.append(meta)
    
    if invalid_distances:
        print(f"\nFound {len(invalid_distances)} files with distances outside the valid range:")
        for filename, distance in invalid_distances[:5]:  # Show first 5
            print(f"  {filename}: {distance}cm")
        if len(invalid_distances) > 5:
            print(f"  ... and {len(invalid_distances) - 5} more")
    
    # Calculate distance statistics
    valid_distances = [meta['distance_cm'] for meta in valid_metadata if 'distance_cm' in meta]
    distance_stats = {
        'min': min(valid_distances) if valid_distances else None,
        'max': max(valid_distances) if valid_distances else None,
        'count': len(valid_distances),
        'unique_values': len(set(valid_distances)) if valid_distances else 0
    }
    
    return valid_metadata, len(invalid_distances), distance_stats

def validate_height_range(metadata_list, valid_range=(4.0, 11.5)):
    """
    Validate that height values are within the expected range for RFID measurements
    
    Args:
        metadata_list: List of metadata dictionaries
        valid_range: Tuple of (min_height, max_height) in cm
        
    Returns:
        tuple: (valid_metadata, invalid_count, height_stats)
    """
    valid_metadata = []
    invalid_heights = []
    min_valid, max_valid = valid_range
    
    for meta in metadata_list:
        if 'height_cm' in meta:
            height = meta['height_cm']
            if min_valid <= height <= max_valid:
                valid_metadata.append(meta)
            else:
                invalid_heights.append((meta['file_name'], height))
                print(f"Warning: Invalid height {height}cm in file {meta['file_name']} - outside range [{min_valid}-{max_valid}]cm")
        else:
            # Files without height information are kept
            valid_metadata.append(meta)
    
    if invalid_heights:
        print(f"\nFound {len(invalid_heights)} files with heights outside the valid range:")
        for filename, height in invalid_heights[:5]:  # Show first 5
            print(f"  {filename}: {height}cm")
        if len(invalid_heights) > 5:
            print(f"  ... and {len(invalid_heights) - 5} more")
    
    # Calculate height statistics
    valid_heights = [meta['height_cm'] for meta in valid_metadata if 'height_cm' in meta]
    height_stats = {
        'min': min(valid_heights) if valid_heights else None,
        'max': max(valid_heights) if valid_heights else None,
        'count': len(valid_heights),
        'unique_values': len(set(valid_heights)) if valid_heights else 0
    }
    
    return valid_metadata, len(invalid_heights), height_stats