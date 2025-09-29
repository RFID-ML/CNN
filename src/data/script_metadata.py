"""
Script to process spectral data files with the naming convention:
NoMeasure_IDtag_XY_ID_Dcm_S.csv

This script loads the raw files, applies filtering, and saves the processed
data to CSV files with the same name plus "_filtered"
"""

import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from tqdm import tqdm

# Input and output directories
INPUT_DIR = r"D:\Documents\Mitacs\Model\Data\Raw"
OUTPUT_DIR = r"D:\Documents\Mitacs\Model\Data\pre_processed\CNN_filtering"
OUTPUT_DIR_PNG = r"D:\Documents\Mitacs\Model\Data\pre_processed\CNN_filtering\pictures"


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
        name_without_ext = name_without_ext.replace("_filtered", "")
        
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
        match = re.search(r'(\d+)(\d+)', parts[2])
        if match:
            metadata["resonances"] = int(match.group(1))
            metadata["resonators"] = int(match.group(2))
    
    # Extract tag format (part 4: ID)
    if len(parts) > 3:
        metadata["tag_format"] = parts[3]
    
    # Extract substrate (part 5: S)
    if len(parts) > 4:
        metadata["substrate"] = parts[4]
    
    # Extract distance (part 6: Dcm)
    if len(parts) > 5:
        match = re.search(r'(\d+)cm', parts[5], re.IGNORECASE)
        if match:
            metadata["distance_cm"] = int(match.group(1))
    
    # Extract height (part 7: H)
    if len(parts) > 6:
        metadata["height"] = parts[6]
    
    # Extract angle (part 8: A)
    if len(parts) > 7:
        try:
            metadata["angle"] = int(parts[7])
        except ValueError:
            metadata["angle"] = parts[7]
    
    return metadata

def read_csv_file(filename):
    """Read CSV file, handling comment lines and special formats."""
    try:
        # Open file and read content
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Store the header lines for later use
        header_lines = []
        data_lines = []
        has_begin_end = False
        
        # Look for special VNA format with BEGIN/END markers
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if 'BEGIN CH1_DATA' in line:
                start_idx = i
                header_lines = lines[:i+1]  # Include the BEGIN line in header
                has_begin_end = True
            elif 'END' in line and start_idx is not None:
                end_idx = i
                data_lines = lines[start_idx+1:end_idx]  # Data between BEGIN and END
                break
        
        # Extract data section if it's the special format
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
            # For standard format, consider all lines with ! as headers
            header_lines = []
            data_lines = []
            
            for i, line in enumerate(lines):
                if line.startswith('!') or line.startswith('//'):
                    header_lines.append(line)
                else:
                    data_lines = lines[i:]
                    break
            
            # Try standard format with comment handling
            data = pd.read_csv(filename, comment='!')
            
            # If first line starts with //, try skipping it
            if data.empty and lines[0].strip().startswith('//'):
                data = pd.read_csv(filename, skiprows=1)
        
        return data, header_lines, has_begin_end
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, [], False

def find_data_columns(df):
    """Find the frequency, amplitude and phase columns in the dataframe."""
    # Initialize column names
    freq_col = None
    amp_col = None
    phase_col = None
    
    # Look for standard column names
    for col in df.columns:
        if any(x in col.lower() for x in ['freq', 'hz', 'ghz']):
            freq_col = col
        elif 'db' in col.lower() and ('s21' in col.lower() or 's12' in col.lower()):
            amp_col = col
        elif 'deg' in col.lower() and ('s21' in col.lower() or 's12' in col.lower()):
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

def apply_filters(data, window_size=11, poly_order=2):
    """Apply a series of filters to smooth the data."""
    # Make a copy to avoid modifying original
    filtered = np.array(data).copy()
    
    # Apply median filter
    filtered = medfilt(filtered, kernel_size=window_size)
    
    # Apply Savitzky-Golay filter if window is large enough
    if window_size > poly_order + 1:
        filtered = savgol_filter(filtered, window_size, poly_order)
    
    # Apply moving average
    window = np.ones(window_size) / window_size
    padded = np.pad(filtered, (window_size//2, window_size//2), mode='edge')
    filtered = np.convolve(padded, window, mode='valid')
    
    return filtered

def process_file(file_path, output_dir, output_dir_png=OUTPUT_DIR_PNG):
    """Process a single file and save the filtered version."""
    try:
        # Get metadata from filename
        metadata = extract_metadata_from_filename(file_path)
        
        # Load data and preserve headers
        result = read_csv_file(file_path)
        if result is None or result[0] is None:
            return False
            
        df, header_lines, has_begin_end = result
        
        # Find data columns
        freq_col, amp_col, phase_col = find_data_columns(df)
        
        if not freq_col:
            print(f"Could not find frequency column in {file_path}")
            return False
            
        # Create filtered copy
        filtered_df = df.copy()
        
        # Apply filters to amplitude and phase if found
        if amp_col:
            filtered_df[amp_col] = apply_filters(df[amp_col].values)
            
        if phase_col:
            filtered_df[phase_col] = apply_filters(df[phase_col].values)
        
        # Create output filename (preserve the same naming convention with _filtered suffix)
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_filtered.csv")
        
        # Save filtered data with original headers
        with open(output_path, 'w') as f:
            # Write original header lines
            for line in header_lines:
                f.write(line)
            
            # Write the filtered data
            if has_begin_end:
                # If original had BEGIN/END format, write data without header row
                filtered_df.to_csv(f, index=False, header=True)
                f.write("END\n")
            else:
                # Standard format - just write the data
                filtered_df.to_csv(f, index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Add title with relevant metadata
        plot_title = os.path.basename(file_path)
        if "tag_format" in metadata:
            plot_title += f" - Format: {metadata['tag_format']}"
        if "distance_cm" in metadata:
            plot_title += f" - {metadata['distance_cm']}cm"
        if "angle" in metadata:
            plot_title += f" - {metadata['angle']}°"
            
        plt.suptitle(plot_title, fontsize=12)
        
        if amp_col:
            plt.subplot(2, 1, 1)
            plt.plot(df[freq_col], df[amp_col], 'b-', alpha=0.5, label='Original')
            plt.plot(filtered_df[freq_col], filtered_df[amp_col], 'r-', label='Filtered')
            plt.title(f'Amplitude - {os.path.basename(file_path)}')
            plt.ylabel('Amplitude (dB)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        if phase_col:
            plt.subplot(2, 1, 2)
            plt.plot(df[freq_col], df[phase_col], 'b-', alpha=0.5, label='Original')
            plt.plot(filtered_df[freq_col], filtered_df[phase_col], 'r-', label='Filtered')
            plt.title('Phase')
            plt.xlabel('Frequency')
            plt.ylabel('Phase (deg)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir_png, f"{base_name}_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to process all files."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all CSV files from input directory
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    successful = 0
    for filename in tqdm(csv_files, desc="Processing files"):
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Extract metadata for display
        metadata = extract_metadata_from_filename(filename)
        info_parts = []
        
        if "tag_format" in metadata:
            info_parts.append(f"Tag: {metadata['tag_format']}")
        if "resonances" in metadata and "resonators" in metadata:
            info_parts.append(f"Res: {metadata['resonances']}/{metadata['resonators']}")
        if "distance_cm" in metadata:
            info_parts.append(f"{metadata['distance_cm']}cm")
        if "angle" in metadata:
            info_parts.append(f"{metadata['angle']}°")
        if "height" in metadata:
            info_parts.append(f"{metadata['height']}")
        if "substrate" in metadata:
            info_parts.append(f"Sub: {metadata['substrate']}")
        
        file_info = ", ".join(info_parts) if info_parts else "No metadata extracted"
        tqdm.write(f"Processing {filename} ({file_info})")
        
        if process_file(file_path, OUTPUT_DIR):
            successful += 1
    
    print(f"Processing complete. {successful}/{len(csv_files)} files successfully processed.")
    print(f"Filtered data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()