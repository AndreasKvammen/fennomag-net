"""
SECS-Based Magnetometer Data Analysis Script

This script processes magnetometer data using Spherical Elementary Current Systems (SECS)
to analyze magnetic field components and saves the results.

The script heavily relies on the SECSY library, particularly:
- cubedsphere.py: Handles the cubed sphere projection and grid operations
- csplot.py: Provides plotting utilities for the cubed sphere projection

The cubed sphere projection is used because it:
1. Avoids pole-related numerical issues
2. Maintains relatively uniform spatial resolution across the grid
3. Is well-suited for regional analyses like SECS
"""

# Imports and Setup
import os
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.linalg
import argparse

# Import SECSY library components for Spherical Elementary Current Systems calculations
from secsy import cubedsphere as cs  # Handles cubed sphere projections and grid operations
from secsy import get_SECS_B_G_matrices  # Calculates SECS basis function matrices

# Import helper functions
from helper_functions import (
    create_dir, load_magnetometer_data, trailing_average, get_common_stations,
    create_secs_directories, save_magnetic_component, save_grid_metadata, save_secs_currents
)

def main():
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description='Process magnetometer data for a date range.')
    
    # Time-related arguments
    parser.add_argument('--start', type=str, required=True,
                      help='Start datetime in "YYYY-MM-DD HH:MM" format')
    parser.add_argument('--end', type=str, required=True,
                      help='End datetime in "YYYY-MM-DD HH:MM" format')
    parser.add_argument('--time-resolution', type=int, default=10,
                      help='Time resolution in minutes (default: 10)')
    
    # Grid-related arguments
    parser.add_argument('--grid-center', type=float, nargs=2, default=[17.0, 65.0],
                      help='Grid center as (longitude, latitude) in degrees (default: 17.0 65.0)')
    parser.add_argument('--grid-shape', type=int, nargs=2, default=[32, 32],
                      help='Grid shape as (N-S points, E-W points) (default: 32 32)')
    parser.add_argument('--grid-resolution', type=float, default=0.5e5,
                      help='Grid resolution in meters (default: 0.5e5)')
    
    # Add regularization parameters
    parser.add_argument('--l0', type=float, default=1e-2,
                      help='Zero-order (amplitude) regularization strength (default: 1e-2)')
    parser.add_argument('--l1', type=float, default=1e-2,
                      help='First-order (smoothness) regularization strength (default: 1e-2)')

    # Add new argument for mirror method
    parser.add_argument('--mirror-depth', type=int, default=9999,
                      help="Depth (measured in km, positive from Earth's surface) at which to place superconducting layer")

    args = parser.parse_args()
    
    # Parse and validate input dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d %H:%M')
        end_date = datetime.strptime(args.end, '%Y-%m-%d %H:%M')
    except ValueError:
        print("Error: Dates must be in 'YYYY-MM-DD HH:MM' format")
        return
        
    # Validate time resolution
    if args.time_resolution <= 0:
        print("Error: Time resolution must be a positive number of minutes")
        return
        
    # Data Loading and Processing
    RE = 6371e3  # Earth radius in meters
    # Define data directories for X, Y, Z components
    base_dir = '/Users/akv020/Tensorflow/Bcast/data'
    Xdir = os.path.join(base_dir, 'XYZmagnetometer/X')
    Ydir = os.path.join(base_dir, 'XYZmagnetometer/Y')
    Zdir = os.path.join(base_dir, 'XYZmagnetometer/Z')
    
    # Load magnetometer data for all components within specified time range
    data = load_magnetometer_data(start_date, end_date, Xdir, Ydir, Zdir)
    if any(data[comp] is None for comp in ['X', 'Y', 'Z']):
        print("Error: Could not load all magnetometer data files")
        return
    
    # Load station coordinates and prepare for processing
    station_coords_file = os.path.join(base_dir, 'magnetometer/station_coordinates.csv')
    station_coordinates = pd.read_csv(station_coords_file)
    station_coordinates.set_index('station', inplace=True)
    
    # Filter stations to keep only those with both coordinates and measurements
    common_stations = get_common_stations(data, station_coordinates)
    for comp in ['X', 'Y', 'Z']:
        columns_to_keep = ['timestamp'] + common_stations
        data[comp] = data[comp][columns_to_keep]
    station_coordinates = station_coordinates.loc[common_stations]
    
    # Apply time averaging to reduce noise in measurements
    for comp in ['X', 'Y', 'Z']:
        data[comp] = trailing_average(data[comp], interval_minutes=args.time_resolution)

    timestamps = pd.to_datetime(data['X']['timestamp'])
    
    # SECS Grid Setup
    # -----------------------
    # User-Editable Grid Settings
    # -----------------------
    grid_center_lon = args.grid_center[0]   # degrees East
    grid_center_lat = args.grid_center[1]   # degrees North
    grid_orientation = 0.0   # orientation of the local x-axis on the cubed-sphere face

    # Number of grid cells in latitude (N) and longitude (M) directions
    grid_shape = tuple(args.grid_shape)    # (North-South, East-West) points

    # Physical size in each dimension (approx. region coverage), in meters
    grid_dim_ns = grid_shape[0] * args.grid_resolution  # North-South dimension
    grid_dim_ew = grid_shape[1] * args.grid_resolution  # East-West dimension

    # Set up grid altitude
    RE = 6371.2e3  # Earth radius in meters
    ionosphere = True
    Rgrid = RE + 110e3 if ionosphere else RE  # Place grid at ionospheric altitude or ground

    # Small shift in grid placement, in meters, along the cube's W direction
    wshift_meters = 1e3

    # Create the cubed-sphere projection
    # Uses CSprojection class from cubedsphere.py to handle coordinate transformations
    projection = cs.CSprojection(
        position=(grid_center_lon, grid_center_lat),
        orientation=grid_orientation
    )

    # Create the grid on the cubed-sphere
    # Uses CSgrid class from cubedsphere.py to set up computational grid
    grid = cs.CSgrid(
        projection,
        L=grid_dim_ew,      # East-West extent
        W=grid_dim_ns,      # North-South extent
        Lres=grid_shape[1], # E-W resolution
        Wres=grid_shape[0], # N-S resolution
        R=Rgrid,           
        wshift=wshift_meters 
    )

    # SECS Inversion Setup
    # Define regularization parameters for the inverse problem
    l0 = args.l0  # Zero-order (amplitude) regularization strength
    l1 = args.l1  # First-order (smoothness) regularization strength

    # Calculate regularization matrices using grid's built-in methods
    # These matrices help control the smoothness of the solution
    De, Dn = grid.get_Le_Ln()  # Get matrices for E-W and N-S derivatives
    DTD = De.T @ De + Dn.T @ Dn  # Combined regularization matrix

    # Get station coordinates for SECS basis function calculation
    lon_mag = station_coordinates.longitude.values
    lat_mag = station_coordinates.latitude.values
    
    # Calculate SECS basis function matrices relating current amplitudes to magnetic field components
    GeB_mag, GnB_mag, GuB_mag = get_SECS_B_G_matrices(
        lat_mag, lon_mag, RE, grid.lat, grid.lon,
        induction_nullification_radius = RE - args.mirror_depth * 1000 if args.mirror_depth != 9999 else None
    )
    
    # Get list of stations for inversion
    station_cols = [col for col in data['X'].columns if col != 'timestamp']
    good_stations = station_cols
    
    # Build covariance matrix from magnetic field components
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

    # Initialize storage for results
    I_timestamps = {}  # Store current amplitudes for each timestamp
    
    # Initialize magnetic field component arrays
    Be = np.full(grid.lat_mesh.shape, np.nan)
    Bn = np.full(grid.lat_mesh.shape, np.nan)
    Bu = np.full(grid.lat_mesh.shape, np.nan)

    # Calculate SECS matrices for the full grid
    print(f"\nCalculating SECS matrices for {grid.size} grid points...")
    GeB_full, GnB_full, GuB_full = get_SECS_B_G_matrices(
        grid.lat_mesh,
        grid.lon_mesh,
        RE,
        grid.lat,
        grid.lon,
        induction_nullification_radius = RE - args.mirror_depth * 1000 if args.mirror_depth != 9999 else None
    )
    print("SECS matrices calculated successfully")
    print(f"Matrix shapes: {GeB_full.shape}, {GnB_full.shape}, {GuB_full.shape}")

    # Process Timestamps and Calculate Fields
    base_path = '/Users/akv020/Tensorflow/Bcast/data/secs'
    total_timestamps = len(timestamps)
    
    # Save grid metadata once at the start
    save_grid_metadata(grid, base_path, args.grid_resolution, grid_shape, start_date.year)
    
    # Create directories for SECS components
    create_secs_directories(base_path, start_date.year)
    
    # Process each timestamp
    for i, this_time in enumerate(timestamps):
        try:
            print(f"Processing timestamp {i+1}/{total_timestamps}: {this_time}")
            
            # Extract magnetic field components for current timestamp
            By_i = data['Y'].iloc[i][good_stations].astype(float).values
            Bx_i = data['X'].iloc[i][good_stations].astype(float).values
            Bz_i = -data['Z'].iloc[i][good_stations].astype(float).values
            
            # Create mask for valid measurements
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
            Ge_sel = GeB_mag[idx, :]
            Gn_sel = GnB_mag[idx, :]
            Gu_sel = GuB_mag[idx, :]
            G = np.vstack([Ge_sel, Gn_sel, Gu_sel])
            
            # Select relevant parts of correlation matrix
            cvinv_sel = cvinv[np.repeat(valid_mask, 3)][:, np.repeat(valid_mask, 3)]
            
            # Calculate normal equations with correlation weighting
            GTG = G.T @ cvinv_sel @ G
            GTd = G.T @ cvinv_sel @ d
            
            # Calculate regularization scaling factors
            scale_gtg = np.median(np.diag(GTG))
            scale_dtd = np.median(np.diag(DTD)) if DTD.size > 0 else 1
            
            # Compute regularization terms
            T0 = l0 * scale_gtg                                           # Zero-order term
            T1 = l1 * scale_gtg / (scale_dtd if scale_dtd != 0 else 1e-10)  # First-order term
            R = T0 * np.eye(grid.size) + T1 * DTD                        # Combined regularization matrix
            
            # Solve regularized inverse problem
            Cmpost = np.linalg.lstsq(GTG + R, np.eye(GTG.shape[0]), rcond=None)[0]
            I_timestamp = Cmpost @ GTd
            I_timestamps[i] = I_timestamp

            # Calculate magnetic field components
            Be = GeB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)
            Bn = GnB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)
            Bu = GuB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)
            
            # Save each component
            save_magnetic_component(Be, this_time, 'Be', base_path)
            save_magnetic_component(Bn, this_time, 'Bn', base_path)
            save_magnetic_component(Bu, this_time, 'Bu', base_path)
                
        except Exception as e:
            print(f"Error processing timestamp {this_time}: {e}")

    print("\nProcessing completed successfully.")

if __name__ == "__main__":
    main()