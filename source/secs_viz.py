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

Usage:
------
python secs_viz.py --start_date "2024-02-05 12:00" --end_date "2024-02-05 13:00" \
                  --output_dir /path/to/output

For detailed parameter descriptions, run:
python secs_viz.py --help
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default='/Users/akv020/Tensorflow/fennomag-net/data/secs', 
                        help='Directory containing SECS data')
    parser.add_argument('--output_dir', type=str, 
                        default='/Users/akv020/Tensorflow/fennomag-net/data/secs', 
                        help='Output directory for figures (will create a "figures" subdirectory)')
    
    args = parser.parse_args()
    return args

def read_grid_metadata(data_dir, year):
    """
    Read grid metadata from the grid_metadata.txt file.
    
    Args:
        data_dir (str): Directory containing SECS data
        year (int): Year of the data
        
    Returns:
        dict: Grid metadata including shape, resolution, and center
    """
    metadata_file = os.path.join(data_dir, str(year), 'grid_metadata.txt')
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Grid metadata file not found: {metadata_file}")
    
    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'Grid shape':
                    # Parse "30×18 points" format
                    shape_str = value.split(' ')[0]
                    ns, ew = shape_str.split('×')
                    metadata['grid_shape'] = (int(ns), int(ew))
                elif key == 'Grid resolution':
                    # Parse "100.0 km" format
                    resolution = float(value.split(' ')[0])
                    metadata['grid_resolution'] = resolution
                elif key == 'Grid center':
                    # Parse "17.0°E, 67.0°N" format
                    lon = float(value.split('°')[0])
                    lat = float(value.split(',')[1].split('°')[0])
                    metadata['grid_center'] = (lon, lat)
    
    return metadata

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
    
    Args:
        be (numpy.ndarray): Eastward component (mapped to red)
        bn (numpy.ndarray): Northward component (mapped to green)
        bu (numpy.ndarray): Upward component (mapped to blue)
        vmin (float): Minimum value for normalization
        vmax (float): Maximum value for normalization
        
    Returns:
        numpy.ndarray: RGB image array with values in [0, 1]
    """
    # Create a normalization function to map values to [0, 1]
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Normalize each component
    r = norm(be)
    g = norm(bn)
    b = norm(bu)
    
    # Clip values to [0, 1] range
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    # Stack the components to create an RGB image
    rgb = np.stack([r, g, b], axis=2)
    
    return rgb

def save_rgb_image(rgb_image, timestamp, output_dir):
    """
    Save RGB image to file.
    
    Args:
        rgb_image (numpy.ndarray): RGB image array with values in [0, 1]
        timestamp (datetime): Timestamp for the image
        output_dir (str): Output directory for figures
        
    Returns:
        str: Path to the saved image
    """
    year = str(timestamp.year)
    figures_dir = os.path.join(output_dir, year, 'figures')
    create_dir(figures_dir)
    
    # Create filename with timestamp
    filename = f"secs_rgb_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(figures_dir, filename)
    
    # Convert to 8-bit RGB (0-255)
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    
    # Save using PIL (better quality control than matplotlib)
    img = Image.fromarray(rgb_uint8)
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
    
    # Read grid metadata
    grid_metadata = read_grid_metadata(args.data_dir, start_date.year)
    
    # Print configuration
    print("\n=== SECS Visualization Configuration ===")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Grid shape: {grid_metadata['grid_shape'][0]}×{grid_metadata['grid_shape'][1]} points")
    print(f"Grid resolution: {grid_metadata['grid_resolution']} km")
    print(f"Grid center: {grid_metadata['grid_center'][0]}°E, {grid_metadata['grid_center'][1]}°N")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
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
            save_path = save_rgb_image(rgb_image, timestamp, args.output_dir)
            
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