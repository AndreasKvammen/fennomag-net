#!/usr/bin/env python3
"""
SECS Generator (secs_gen.py)
============================

This script calculates and saves Spherical Elementary Current Systems (SECS) matrices
and magnetic field components based on magnetometer data. It performs SECS inversion
to estimate ionospheric currents and calculates the resulting magnetic field.

Table of Contents:
-----------------
1. Import Libraries and Parse Arguments
2. Load and Process Magnetometer Data
3. Set Up SECS Grid
4. Calculate SECS Basis Matrices
5. Build Covariance Matrix
6. Process Timestamps and Calculate Fields
7. Save Results

Usage:
------
python secs_gen.py --start_date "2024-02-05 12:00" --end_date "2024-02-05 13:00" \
                  --grid_center 17.0 67.0 --grid_shape 30 18 --grid_resolution 100 \
                  --l0 1e-2 --l1 1e-2 --mirror_depth 1000 --time_resolution 1
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

# Import SECSY library components for Spherical Elementary Current Systems calculations
from secsy import cubedsphere as cs  # Handles cubed sphere projections and grid operations
from secsy import get_SECS_B_G_matrices  # Calculates SECS basis function matrices

# Import helper functions
from helper_functions import (
    create_dir, load_magnetometer_data, trailing_average, get_common_stations,
    create_secs_directories, save_magnetic_component, save_grid_metadata
)

def parse_arguments():
    """Parse command line arguments for SECS generation."""
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
    """Main function to generate SECS matrices and magnetic field components."""
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
    
    # 2. Load and Process Magnetometer Data
    # ------------------------------------
    RE = 6371.2e3  # Earth radius in meters
    
    # Define data directories for X, Y, Z components
    Xdir = os.path.join(args.data_dir, 'XYZmagnetometer/X')
    Ydir = os.path.join(args.data_dir, 'XYZmagnetometer/Y')
    Zdir = os.path.join(args.data_dir, 'XYZmagnetometer/Z')
    
    # Load magnetometer data for all components within specified time range
    print("Loading magnetometer data...")
    data = load_magnetometer_data(start_date, end_date, Xdir, Ydir, Zdir)
    if any(data[comp] is None for comp in ['X', 'Y', 'Z']):
        raise ValueError("Could not load all magnetometer data files")
    
    # Display data summary
    for comp in ['X', 'Y', 'Z']:
        print(f"{comp} component data shape: {data[comp].shape}")
        print(f"Time range: {data[comp]['timestamp'].min()} to {data[comp]['timestamp'].max()}")
        print(f"Number of stations: {len(data[comp].columns) - 1}")  # -1 for timestamp column
    
    # Load station coordinates and prepare for processing
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
    
    # 3. Set Up SECS Grid
    # ------------------
    # Define Earth radius and grid altitude
    ionosphere = True
    Rgrid = RE + 110e3 if ionosphere else RE  # Place grid at ionospheric altitude or ground
    
    # Set up cubed sphere projection
    print("\nSetting up SECS grid...")
    projection = cs.CSprojection(
        position=(args.grid_center[0], args.grid_center[1]),  # (lon, lat) in degrees
        orientation=0.0  # orientation of the local x-axis on the cubed-sphere face
    )
    
    # Create the grid on the cubed-sphere
    grid = cs.CSgrid(
        projection,
        L=args.grid_shape[1] * args.grid_resolution,  # East-West extent in meters
        W=args.grid_shape[0] * args.grid_resolution,  # North-South extent in meters
        Lres=args.grid_shape[1],  # E-W resolution (number of points)
        Wres=args.grid_shape[0],  # N-S resolution (number of points)
        R=Rgrid,                  # Grid radius (Earth radius + ionosphere height)
        wshift=1e3                # Small shift in grid placement (meters)
    )
    
    print(f"Grid created with {grid.size} points")
    print(f"Grid spans approximately {args.grid_resolution/1000 * args.grid_shape[1]:.1f} km E-W × "
          f"{args.grid_resolution/1000 * args.grid_shape[0]:.1f} km N-S")
    
    # Check which stations are inside the grid
    print("Checking which stations are inside the grid...")
    stations_inside = []
    stations_outside = []
    
    for station in common_stations:
        if station in station_coordinates.index:
            lon = station_coordinates.loc[station, 'longitude']
            lat = station_coordinates.loc[station, 'latitude']
            is_inside = grid.ingrid(lon, lat)
            
            if is_inside:
                stations_inside.append(station)
            else:
                stations_outside.append(station)
    
    print(f"Stations inside grid: {len(stations_inside)}/{len(common_stations)}")
    if stations_outside:
        print(f"Dropping {len(stations_outside)} stations outside grid")
    
    # Filter data and coordinates to keep only stations inside the grid
    for comp in ['X', 'Y', 'Z']:
        columns_to_keep = ['timestamp'] + stations_inside
        data[comp] = data[comp][columns_to_keep]
    station_coordinates = station_coordinates.loc[stations_inside]
    
    # 4. Calculate SECS Basis Matrices
    # ------------------------------
    # Get station coordinates for SECS basis function calculation
    lon_mag = station_coordinates.longitude.values
    lat_mag = station_coordinates.latitude.values
    
    # Calculate SECS basis function matrices relating current amplitudes to magnetic field components
    print("\nCalculating SECS basis matrices for stations...")
    
    # Set up mirror method for induction nullification if enabled
    induction_nullification_radius = RE - args.mirror_depth * 1000 if args.mirror_depth != 9999 else None
    if induction_nullification_radius is not None:
        print(f"Using mirror method with superconducting layer at depth {args.mirror_depth} km")
    else:
        print("Mirror method disabled")
    
    # Calculate SECS basis matrices for stations
    # These matrices relate SECS amplitudes to magnetic field components at station locations
    GeB_mag, GnB_mag, GuB_mag = get_SECS_B_G_matrices(
        lat_mag, lon_mag, RE, grid.lat, grid.lon,
        induction_nullification_radius=induction_nullification_radius
    )
    
    print(f"SECS basis matrices for stations calculated with shapes: {GeB_mag.shape}, {GnB_mag.shape}, {GuB_mag.shape}")
    
    # Calculate SECS matrices for the full grid
    # These matrices will be used to calculate magnetic field components at all grid points
    print("\nCalculating SECS matrices for the full grid...")
    GeB_full, GnB_full, GuB_full = get_SECS_B_G_matrices(
        grid.lat_mesh.flatten(),
        grid.lon_mesh.flatten(),
        RE,
        grid.lat,
        grid.lon,
        induction_nullification_radius=induction_nullification_radius
    )
    print("SECS matrices for full grid calculated successfully")
    print(f"Matrix shapes: {GeB_full.shape}, {GnB_full.shape}, {GuB_full.shape}")
    
    # 5. Build Covariance Matrix
    # ------------------------
    # Get list of stations for inversion
    good_stations = [col for col in data['X'].columns if col != 'timestamp']
    print(f"\nNumber of stations for inversion: {len(good_stations)}")
    
    # Apply time averaging to smooth the data
    print("Applying trailing average...")
    for comp in ['X', 'Y', 'Z']:
        data[comp] = trailing_average(data[comp], interval_minutes=args.time_resolution)
    
    timestamps = pd.to_datetime(data['X']['timestamp'])
    print(f"Number of timestamps after averaging: {len(timestamps)}")
    print(f"First timestamp: {timestamps.min()}")
    print(f"Last timestamp: {timestamps.max()}")
    
    # Build covariance matrix from magnetic field components
    print("\nBuilding covariance matrix from magnetic field components...")
    
    # Extract magnetic field data for all stations
    By_day = data['Y'][good_stations].values
    Bx_day = data['X'][good_stations].values
    Bz_day = -data['Z'][good_stations].values  # Note: Z component is negated
    
    # Combine all magnetic field components for covariance calculation
    D_all = np.hstack([By_day, Bx_day, Bz_day])
    mask = np.isnan(D_all)
    D_all_ma = np.ma.array(D_all, mask=mask)
    
    # Calculate correlation coefficient matrix between components
    cv = np.ma.corrcoef(D_all_ma.T)
    cv = np.ma.filled(cv, fill_value=0.0)
    
    # Calculate inverse of correlation matrix for weighting
    cvinv = np.linalg.lstsq(cv, np.eye(cv.shape[0]), rcond=None)[0]
    
    print(f"Correlation matrix shape: {cv.shape}")
    
    # Calculate regularization matrices using grid's built-in methods
    De, Dn = grid.get_Le_Ln()  # Get matrices for E-W and N-S derivatives
    DTD = De.T @ De + Dn.T @ Dn  # Combined regularization matrix
    
    # 6. Process Timestamps and Calculate Fields
    # ---------------------------------------
    # Initialize storage for results
    I_timestamps = {}  # Store current amplitudes for each timestamp
    total_timestamps = len(timestamps)
    
    # Create directories for SECS components
    create_secs_directories(args.output_dir, start_date.year)
    
    # Save grid metadata once at the start
    save_grid_metadata(grid, args.output_dir, args.grid_resolution/1000, args.grid_shape, start_date.year)
    
    # Process each timestamp
    print(f"\nProcessing {total_timestamps} timestamps...")
    
    for i, this_time in enumerate(timestamps):
        try:
            print(f"Processing timestamp {i+1}/{total_timestamps}: {this_time}")
            
            # Extract magnetic field components for current timestamp
            By_i = data['Y'].iloc[i][good_stations].astype(float).values
            Bx_i = data['X'].iloc[i][good_stations].astype(float).values
            Bz_i = -data['Z'].iloc[i][good_stations].astype(float).values  # Note: Z component is negated
            
            # Create mask for valid measurements (not NaN)
            valid_mask = ~(np.isnan(By_i) | np.isnan(Bx_i) | np.isnan(Bz_i))
            
            # Filter measurements and combine into observation vector
            By_i = By_i[valid_mask]
            Bx_i = Bx_i[valid_mask]
            Bz_i = Bz_i[valid_mask]
            d = np.hstack([By_i, Bx_i, Bz_i])
            
            # Get valid stations and their indices
            valid_stations = [st for j, st in enumerate(good_stations) if valid_mask[j]]
            idx = [station_coordinates.index.get_loc(st) for st in valid_stations]
            
            # Select relevant rows from SECS matrices
            GeB_i = GeB_mag[idx, :]
            GnB_i = GnB_mag[idx, :]
            GuB_i = GuB_mag[idx, :]
            
            # Combine matrices for all components
            G = np.vstack([GeB_i, GnB_i, GuB_i])
            
            # Use identity matrix for weighting (simpler approach)
            cvinv_valid = np.eye(len(d))
            
            # Apply weighting to form normal equations
            # GTG = G^T * cvinv * G (coefficient matrix)
            # GTd = G^T * cvinv * d (right-hand side)
            GTG = G.T @ cvinv_valid @ G
            GTd = G.T @ cvinv_valid @ d
            
            # Calculate regularization scaling factors
            # These ensure proper balance between data fit and regularization
            scale_gtg = np.median(np.diag(GTG))
            scale_dtd = np.median(np.diag(DTD)) if DTD.size > 0 else 1
            
            # Compute regularization terms
            # T0: Zero-order term (amplitude damping)
            # T1: First-order term (smoothness constraint)
            T0 = args.l0 * scale_gtg
            T1 = args.l1 * scale_gtg / (scale_dtd if scale_dtd != 0 else 1e-10)
            
            # Combined regularization matrix
            R = T0 * np.eye(grid.size) + T1 * DTD
            
            # Solve regularized inverse problem
            # (GTG + R)^-1 * GTd = I
            Cmpost = np.linalg.lstsq(GTG + R, np.eye(GTG.shape[0]), rcond=None)[0]
            I_timestamp = Cmpost @ GTd
            I_timestamps[str(this_time)] = I_timestamp.tolist()  # Store for potential later use
            
            # Calculate magnetic field components at all grid points
            # B = G * I
            Be = GeB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)
            Bn = GnB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)
            Bu = GuB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)
            
            # 7. Save Results
            # -------------
            # Save each magnetic field component
            save_magnetic_component(Be, this_time, 'Be', args.output_dir)
            save_magnetic_component(Bn, this_time, 'Bn', args.output_dir)
            save_magnetic_component(Bu, this_time, 'Bu', args.output_dir)
            
            print(f"  Timestamp {this_time} processed successfully")
                
        except Exception as e:
            print(f"Error processing timestamp {this_time}: {str(e)}")
            traceback.print_exc()
    
    # Save current amplitudes to a JSON file for potential later use
    currents_file = os.path.join(args.output_dir, str(start_date.year), 
                               f"currents_{start_date.strftime('%Y%m%d_%H%M')}_to_{end_date.strftime('%Y%m%d_%H%M')}.json")
    with open(currents_file, 'w') as f:
        json.dump(I_timestamps, f)
    
    print("\nProcessing completed successfully.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()