"""
Script to generate the rfid_mapping.json file compatible with data_loader.py
from CSV files with the format NoMeasure_IDtag_XY_ID_Dcm_S.csv
"""

import os
import json
import re
import argparse

# Path to raw data
RAW_DATA_DIR = r"D:\Documents\Mitacs\Model\Data\Raw"
# Path to output file
OUTPUT_FILE = r"D:\Documents\Mitacs\Model\Models\Classification\CNN\data\mapping\rfid_mapping.json"

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

def create_mapping(raw_data_dir, output_file):
    """
    Creates a mapping file in the format expected by data_loader.py
    """
    # Check that the directory exists
    if not os.path.exists(raw_data_dir):
        print(f"Error: Directory {raw_data_dir} does not exist")
        return False
    
    # Get the list of CSV files
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {raw_data_dir}")
        return False
    
    print(f"Analyzing {len(csv_files)} CSV files...")
    
    # Format expected by data_loader.py
    mapping_data = {
        "files": [],
        "dataset_info": {
            "description": "Mapping of RFID tags for CNN classification",
            "format": "NoMeasure_IDtag_XY_ID_Dcm_S.csv"
        }
    }
    
    # Build the entry for each file
    for filename in csv_files:
        # Extract metadata using our updated function
        metadata = extract_metadata_from_filename(filename)
        
        # For classification, we use the tag_format as the primary identifier
        tag_format = metadata.get("tag_format")
        if tag_format:
            # Relative path as expected by the data_loader
            relative_path = os.path.normpath(os.path.join('data', 'raw', filename))
            
            # Create the entry for this file
            file_entry = {
                "file_path": relative_path,
                "tag_id": f"Format_{tag_format}",  # Use the tag format as the primary identifier
                "metadata": metadata
            }
            
            mapping_data["files"].append(file_entry)
        else:
            print(f"Warning: File {filename} does not contain valid tag format")
    
    if not mapping_data["files"]:
        print("No files matching the format could be processed")
        return False
    
    # Create the output directory if necessary
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the mapping
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=4)
    
    # Display a summary
    unique_tags = set(entry["tag_id"] for entry in mapping_data["files"])
    print(f"Mapping created successfully:")
    print(f"- {len(mapping_data['files'])} files mapped")
    print(f"- {len(unique_tags)} unique tags identified")
    print(f"File saved: {output_file}")
    
    # Show some examples
    print("\nExample entries:")
    for entry in mapping_data["files"][:3]:
        print(f"  {entry['file_path']} -> {entry['tag_id']} (metadata: {entry['metadata']})")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create RFID mapping file")
    parser.add_argument('--data_dir', type=str, default=RAW_DATA_DIR, 
                        help='Directory of raw data')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, 
                        help='Path to output file')
    
    args = parser.parse_args()
    create_mapping(args.data_dir, args.output)