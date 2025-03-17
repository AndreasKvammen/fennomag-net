#!/usr/bin/env python3
"""
SECS Generator (secs_gen.py)
============================

This script calculates and saves Spherical Elementary Current Systems (SECS) matrices
and magnetic field components based on magnetometer data. It performs SECS inversion
to estimate ionospheric currents and calculates the resulting magnetic field.

The script follows a modular approach with most functionality implemented in helper_functions.py,
making the main workflow clear and easy to follow.

Table of Contents:
-----------------
1. Import Libraries and Parse Arguments
2. Main Function
   a. Configuration and Data Loading
   b. Station Filtering and Grid Setup
   c. SECS Matrix Calculation
   d. Timestamp Processing
   e. Results Saving

Usage:
------
python secs_gen.py --start_date "2024-02-05 12:00" --end_date "2024-02-05 13:00" \
                  --grid_center 17.0 67.0 --grid_shape 30 18 --grid_resolution 100 \
                  --l0 1e-2 --l1 1e-2 --mirror_depth 1000 --time_resolution 1

For detailed parameter descriptions, run:
python secs_gen.py --help
"""

# 1. Import Libraries and Parse Arguments
# --------------------------------------
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import traceback

# Import helper functions that implement most of the functionality
from helper_functions import (
    create_dir, load_magnetometer_data, trailing_average, get_common_stations,
    create_secs_directories, save_magnetic_component, save_grid_metadata,
    setup_secs_grid, filter_stations_by_grid, calculate_secs_matrices, process_timestamp
)

def parse_arguments():
    """
    Parse command line arguments for SECS generation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate SECS matrices and magnetic field components')
    
    # Time-related parameters
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Start date and time (YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_date', type=str, required=True, 
                        help='End date and time (YYYY-MM-DD HH:MM)')
    parser.add_argument('--time_resolution', type=int, default=1, 
                        help='Time resolution in minutes (default: 1)')
    
    # Grid-related parameters
    parser.add_argument('--grid_center', type=float, nargs=2, required=True, 
                        help='Grid center [longitude, latitude] in degrees')
    parser.add_argument('--grid_shape', type=int, nargs=2, required=True, 
                        help='Grid shape [N-S points, E-W points]')
    parser.add_argument('--grid_resolution', type=float, required=True, 
                        help='Grid resolution in kilometers')
    
    # Regularization parameters
    parser.add_argument('--l0', type=float, default=1e-2, 
                        help='Zero-order (amplitude) regularization strength (default: 1e-2)')
    parser.add_argument('--l1', type=float, default=1e-2, 
                        help='First-order (smoothness) regularization strength (default: 1e-2)')
    
    # Mirror method parameters
    parser.add_argument('--mirror_depth', type=int, default=9999, 
                        help='Mirror depth in km (9999 to disable, default: 9999)')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default='/Users/akv020/Tensorflow/fennomag-net/data', 
                        help='Base directory for data')
    parser.add_argument('--output_dir', type=str, 
                        default='/Users/akv020/Tensorflow/fennomag-net/data/secs', 
                        help='Output directory for SECS data')
    
    args = parser.parse_args()
    
    # Convert grid resolution from km to meters
    args.grid_resolution = args.grid_resolution * 1000
    
    return args

def main():
    """
    Main function to generate SECS matrices and magnetic field components.
    
    This function implements the complete workflow:
    1. Parse arguments and load magnetometer data
    2. Filter stations and set up the SECS grid
    3. Calculate SECS basis matrices
    4. Process each timestamp to calculate currents and fields
    5. Save results to files
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M')
    
    # Print configuration
    print("\n=== SECS Generator Configuration ===")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Time resolution: {args.time_resolution} minutes")
    print(f"Grid center: {args.grid_center[0]}°E, {args.grid_center[1]}°N")
    print(f"Grid shape: {args.grid_shape[0]}×{args.grid_shape[1]} points")
    print(f"Grid resolution: {args.grid_resolution/1000:.1f} km")
    print(f"Regularization: l0={args.l0}, l1={args.l1}")
    print(f"Mirror depth: {'Disabled' if args.mirror_depth == 9999 else f'{args.mirror_depth} km'}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=====================================\n")
    
    # Define data directories for X, Y, Z components
    Xdir = os.path.join(args.data_dir, 'XYZmagnetometer/X')
    Ydir = os.path.join(args.data_dir, 'XYZmagnetometer/Y')
    Zdir = os.path.join(args.data_dir, 'XYZmagnetometer/Z')
    
    # Step 1: Load magnetometer data for all components within specified time range
    print("Loading magnetometer data...")
    data = load_magnetometer_data(start_date, end_date, Xdir, Ydir, Zdir)
    if any(data[comp] is None for comp in ['X', 'Y', 'Z']):
        raise ValueError("Could not load all magnetometer data files")
    
    # Display data summary
    for comp in ['X', 'Y', 'Z']:
        print(f"{comp} component data shape: {data[comp].shape}")
        print(f"Time range: {data[comp]['timestamp'].min()} to {data[comp]['timestamp'].max()}")
        print(f"Number of stations: {len(data[comp].columns) - 1}")  # -1 for timestamp column
    
    # Step 2: Load station coordinates and prepare for processing
    station_coords_file = os.path.join(args.data_dir, 'magnetometer/station_coordinates.csv')
    station_coordinates = pd.read_csv(station_coords_file)
    station_coordinates.set_index('station', inplace=True)
    
    print(f"Total stations in coordinates file: {len(station_coordinates)}")
    
    # Filter stations to keep only those with both coordinates and measurements
    common_stations = get_common_stations(data, station_coordinates)
    print(f"Number of common stations with both coordinates and measurements: {len(common_stations)}")
    
    for comp in ['X', 'Y', 'Z']:
        columns_to_keep = ['timestamp'] + common_stations
        data[comp] = data[comp][columns_to_keep]
    station_coordinates = station_coordinates.loc[common_stations]
    
    # Step 3: Set up SECS grid
    print("\nSetting up SECS grid...")
    grid = setup_secs_grid(
        grid_center=args.grid_center,
        grid_shape=args.grid_shape,
        grid_resolution=args.grid_resolution,
        ionosphere=True
    )
    
    print(f"Grid created with {grid.size} points")
    print(f"Grid spans approximately {args.grid_resolution/1000 * args.grid_shape[1]:.1f} km E-W × "
          f"{args.grid_resolution/1000 * args.grid_shape[0]:.1f} km N-S")
    
    # Filter stations to keep only those inside the grid
    print("Checking which stations are inside the grid...")
    stations_inside = filter_stations_by_grid(common_stations, station_coordinates, grid)
    
    print(f"Stations inside grid: {len(stations_inside)}/{len(common_stations)}")
    if len(stations_inside) < len(common_stations):
        print(f"Dropping {len(common_stations) - len(stations_inside)} stations outside grid")
    
    # Filter data and coordinates to keep only stations inside the grid
    for comp in ['X', 'Y', 'Z']:
        columns_to_keep = ['timestamp'] + stations_inside
        data[comp] = data[comp][columns_to_keep]
    station_coordinates = station_coordinates.loc[stations_inside]
    
    # Step 4: Calculate SECS basis matrices
    print("\nCalculating SECS basis matrices...")
    if args.mirror_depth != 9999:
        print(f"Using mirror method with superconducting layer at depth {args.mirror_depth} km")
    else:
        print("Mirror method disabled")
    
    secs_matrices = calculate_secs_matrices(
        grid=grid,
        station_coordinates=station_coordinates,
        stations=stations_inside,
        mirror_depth=args.mirror_depth
    )
    
    # Step 5: Apply time averaging to smooth the data
    print("Applying trailing average...")
    for comp in ['X', 'Y', 'Z']:
        data[comp] = trailing_average(data[comp], interval_minutes=args.time_resolution)
    
    timestamps = pd.to_datetime(data['X']['timestamp'])
    print(f"Number of timestamps after averaging: {len(timestamps)}")
    print(f"First timestamp: {timestamps.min()}")
    print(f"Last timestamp: {timestamps.max()}")
    
    # Step 6: Create directories for SECS components
    create_secs_directories(args.output_dir, start_date.year)
    
    # Save grid metadata once at the start
    save_grid_metadata(grid, args.output_dir, args.grid_resolution/1000, args.grid_shape, start_date.year)
    
    # Prepare regularization parameters
    regularization_params = {
        'l0': args.l0,
        'l1': args.l1
    }
    
    # Step 7: Process each timestamp
    total_timestamps = len(timestamps)
    print(f"\nProcessing {total_timestamps} timestamps...")
    
    for i, this_time in enumerate(timestamps):
        try:
            print(f"Processing timestamp {i+1}/{total_timestamps}: {this_time}")
            
            # Process this timestamp to get currents and magnetic fields
            I_timestamp, Be, Bn, Bu = process_timestamp(
                timestamp_idx=i,
                timestamp=this_time,
                data=data,
                good_stations=stations_inside,
                station_coordinates=station_coordinates,
                grid=grid,
                secs_matrices=secs_matrices,
                regularization_params=regularization_params
            )
            
            # Save each magnetic field component
            save_magnetic_component(Be, this_time, 'Be', args.output_dir)
            save_magnetic_component(Bn, this_time, 'Bn', args.output_dir)
            save_magnetic_component(Bu, this_time, 'Bu', args.output_dir)
            
            print(f"  Timestamp {this_time} processed successfully")
                
        except Exception as e:
            print(f"Error processing timestamp {this_time}: {str(e)}")
            traceback.print_exc()
    
    print("\nProcessing completed successfully.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()