#!/usr/bin/env python3
"""
SECS Visualization (secs_viz.py)
================================

This script generates RGB images from SECS magnetic field components (Be, Bn, Bu).
Each component is mapped to a color channel in the RGB image:
- Red: Eastward component (Be)
- Green: Northward component (Bn)
- Blue: Upward component (Bu)

The script reads magnetic field data for a specified time range and creates
visualizations that help identify patterns in the magnetic field variations.
Values are discretized to 256 levels (8-bit per channel) between -1250 nT and +1250 nT.

Usage:
------
python secs_viz.py --start_date "2024-02-05 12:00" --end_date "2024-02-05 13:00" \
                  --data_dir /path/to/secs/data

For detailed parameter descriptions, run:
python secs_viz.py --help
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image

# Import helper functions
from helper_functions import create_dir

def parse_arguments():
    """
    Parse command line arguments for the SECS visualization script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate RGB images from SECS magnetic field components')
    
    # Time range parameters
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Start date and time (format: "YYYY-MM-DD HH:MM")')
    parser.add_argument('--end_date', type=str, required=True, 
                        help='End date and time (format: "YYYY-MM-DD HH:MM")')
    
    # Data directory
    parser.add_argument('--data_dir', type=str, default='/Users/akv020/Tensorflow/fennomag-net/data/secs', 
                        help='Directory containing SECS data')
    
    args = parser.parse_args()
    return args

def load_magnetic_component(timestamp, component_name, data_dir):
    """
    Load magnetic field component data for a specific timestamp.
    
    Args:
        timestamp (datetime): Timestamp to load
        component_name (str): Component name ('Be', 'Bn', or 'Bu')
        data_dir (str): Directory containing SECS data
        
    Returns:
        numpy.ndarray: 2D array of magnetic field values
    """
    year = str(timestamp.year)
    filename = f"{component_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(data_dir, year, component_name, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Magnetic field component file not found: {file_path}")
    
    # Load the CSV file (no headers, just data)
    component_data = np.loadtxt(file_path, delimiter=',')
    
    return component_data

def create_rgb_image(be, bn, bu, vmin=-1250, vmax=1250):
    """
    Create an RGB image from magnetic field components.
    
    The function maps each component to a color channel:
    - Red: Eastward component (Be)
    - Green: Northward component (Bn)
    - Blue: Upward component (Bu)
    
    Values are discretized to 256 levels (8-bit per channel) between vmin and vmax.
    
    Args:
        be (numpy.ndarray): Eastward magnetic field component
        bn (numpy.ndarray): Northward magnetic field component
        bu (numpy.ndarray): Upward magnetic field component
        vmin (float): Minimum value for normalization
        vmax (float): Maximum value for normalization
        
    Returns:
        numpy.ndarray: RGB image array with values in [0, 255] (uint8)
    """
    # Linear mapping from [vmin, vmax] to [0, 255]
    def normalize_to_uint8(data):
        # Clip values to the specified range
        data_clipped = np.clip(data, vmin, vmax)
        
        # Map from [vmin, vmax] to [0, 255]
        normalized = ((data_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        return normalized
    
    # Normalize each component to 8-bit (0-255)
    r = normalize_to_uint8(be)
    g = normalize_to_uint8(bn)
    b = normalize_to_uint8(bu)
    
    # Stack the components to create an RGB image
    rgb = np.stack([r, g, b], axis=2)
    
    return rgb

def save_rgb_image(rgb_image, timestamp, data_dir):
    """
    Save RGB image to file.
    
    Args:
        rgb_image (numpy.ndarray): RGB image array with values in [0, 255]
        timestamp (datetime): Timestamp for the image
        data_dir (str): Base directory for SECS data
        
    Returns:
        str: Path to the saved image
    """
    year = str(timestamp.year)
    figures_dir = os.path.join(data_dir, year, 'figures')
    create_dir(figures_dir)
    
    # Create filename with timestamp
    filename = f"secs_rgb_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(figures_dir, filename)
    
    # Save using PIL
    img = Image.fromarray(rgb_image)
    img.save(save_path)
    
    return save_path

def generate_timestamp_list(start_date, end_date):
    """
    Generate a list of timestamps between start_date and end_date.
    
    Args:
        start_date (datetime): Start date and time
        end_date (datetime): End date and time
        
    Returns:
        list: List of datetime objects
    """
    timestamps = []
    current = start_date
    
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(minutes=1)  # Assuming 1-minute resolution
    
    return timestamps

def main():
    """
    Main function to generate RGB images from SECS magnetic field components.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M')
    
    # Print configuration
    print("\n=== SECS Visualization Configuration ===")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Data directory: {args.data_dir}")
    print(f"Value range: -1250 nT to +1250 nT (discretized to 256 levels)")
    print("========================================\n")
    
    # Generate list of timestamps to process
    timestamps = generate_timestamp_list(start_date, end_date)
    
    # Process each timestamp
    total_timestamps = len(timestamps)
    print(f"Processing {total_timestamps} timestamps...")
    
    for i, timestamp in enumerate(timestamps):
        try:
            print(f"Processing timestamp {i+1}/{total_timestamps}: {timestamp}")
            
            # Load magnetic field components
            be = load_magnetic_component(timestamp, 'Be', args.data_dir)
            bn = load_magnetic_component(timestamp, 'Bn', args.data_dir)
            bu = load_magnetic_component(timestamp, 'Bu', args.data_dir)
            
            # Create RGB image
            rgb_image = create_rgb_image(be, bn, bu, vmin=-1250, vmax=1250)
            
            # Save RGB image
            save_path = save_rgb_image(rgb_image, timestamp, args.data_dir)
            
            print(f"  Image saved to: {save_path}")
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nProcessing completed successfully.")

if __name__ == "__main__":
    main()