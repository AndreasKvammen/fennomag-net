#!/usr/bin/env python3
"""
SECS Comparison (secs_comp.py)
==============================

This script compares SECS-predicted magnetic field components with actual magnetometer
observations. It creates visualizations showing:
- SECS-predicted horizontal magnetic field (black arrows)
- SECS-predicted vertical magnetic field (color map)
- Observed magnetometer horizontal field (blue arrows)
- Magnetometer station locations (blue dots)

The script processes data for a specified time range and saves comparison plots
that help evaluate the accuracy of the SECS model.

Usage:
------
python secs_comp.py --start_date "2024-02-05 12:00" --end_date "2024-02-05 13:00" \
                   --data_dir /path/to/data --secs_dir /path/to/secs/data \
                   --vmin -750 --vmax 750

For detailed parameter descriptions, run:
python secs_comp.py --help
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from secsy import cubedsphere as cs
from secsy import CSplot
import warnings

# Import helper functions
from helper_functions import (
    create_dir, load_magnetic_component, filter_stations_by_grid, setup_secs_grid
)

# Suppress specific pcolormesh warnings
warnings.filterwarnings('ignore', 
                       message='The input coordinates to pcolormesh are interpreted as cell*',
                       category=UserWarning)

def parse_arguments():
    """
    Parse command line arguments for the SECS comparison script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Compare SECS predictions with magnetometer observations')
    
    # Time range parameters
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Start date and time (format: "YYYY-MM-DD HH:MM")')
    parser.add_argument('--end_date', type=str, required=True, 
                        help='End date and time (format: "YYYY-MM-DD HH:MM")')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default='/Users/akv020/Tensorflow/fennomag-net/data', 
                        help='Base directory for magnetometer data')
    parser.add_argument('--secs_dir', type=str, default='/Users/akv020/Tensorflow/fennomag-net/data/secs', 
                        help='Directory containing SECS data')
    
    # Value range for color mapping
    parser.add_argument('--vmin', type=float, default=-750,
                        help='Minimum value for vertical field color mapping (default: -750 nT)')
    parser.add_argument('--vmax', type=float, default=750,
                        help='Maximum value for vertical field color mapping (default: 750 nT)')
    
    args = parser.parse_args()
    return args

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

def load_station_coordinates(data_dir):
    """
    Load station coordinates from CSV file.
    
    Args:
        data_dir (str): Base directory for magnetometer data
        
    Returns:
        pandas.DataFrame: DataFrame with station coordinates
    """
    station_coords_file = os.path.join(data_dir, 'magnetometer/station_coordinates.csv')
    
    if not os.path.exists(station_coords_file):
        raise FileNotFoundError(f"Station coordinates file not found: {station_coords_file}")
    
    station_coordinates = pd.read_csv(station_coords_file)
    station_coordinates.set_index('station', inplace=True)
    
    return station_coordinates

def load_grid_metadata(data_dir, year):
    """
    Load grid metadata from text file.
    
    Args:
        data_dir (str): Directory containing SECS data
        year (int): Year for which to load metadata
        
    Returns:
        dict: Dictionary with grid metadata
    """
    metadata_file = os.path.join(data_dir, str(year), 'grid_metadata.txt')
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Grid metadata file not found: {metadata_file}")
    
    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            # Try to convert to float, but handle array values
            try:
                # Check if it's an array representation
                if '[' in value and ']' in value:
                    # Parse array values
                    value = value.strip('[]').split()
                    metadata[key] = np.array([float(v) for v in value])
                else:
                    metadata[key] = float(value)
            except ValueError:
                # If conversion fails, keep as string
                metadata[key] = value
    
    return metadata

def setup_grid_from_metadata(metadata):
    """
    Set up SECS grid from metadata.
    
    Args:
        metadata (dict): Dictionary with grid metadata
        
    Returns:
        cs.CSgrid: Configured grid object
    """
    grid_center = (metadata['grid_center_lon'], metadata['grid_center_lat'])
    grid_shape = (int(metadata['grid_shape_NS']), int(metadata['grid_shape_EW']))
    grid_resolution = metadata['grid_resolution'] * 1000  # Convert from km to meters
    
    # Set up cubed sphere projection
    projection = cs.CSprojection(
        position=grid_center,
        orientation=metadata['grid_orientation']
    )
    
    # Create the grid on the cubed-sphere
    grid = cs.CSgrid(
        projection,
        L=grid_shape[1] * grid_resolution,  # East-West extent in meters
        W=grid_shape[0] * grid_resolution,  # North-South extent in meters
        Lres=grid_shape[0],  # E-W resolution (number of points)
        Wres=grid_shape[1],  # N-S resolution (number of points)
        R=metadata['Rgrid'],  # Grid radius (Earth radius + ionosphere height)
        wshift=1e3           # Small shift in grid placement (meters)
    )
    
    return grid

def load_magnetometer_data(timestamp, data_dir):
    """
    Load magnetometer data for a specific timestamp.
    
    Args:
        timestamp (datetime): Timestamp to load
        data_dir (str): Base directory for magnetometer data
        
    Returns:
        tuple: (obs_x, obs_y, obs_z, stations) - Observed magnetic field components and station names
    """
    # Construct file paths
    date_str = timestamp.strftime('%Y-%m-%d')
    x_file = os.path.join(data_dir, 'XYZmagnetometer/X', f"magnetometer_X_{date_str}.csv")
    y_file = os.path.join(data_dir, 'XYZmagnetometer/Y', f"magnetometer_Y_{date_str}.csv")
    z_file = os.path.join(data_dir, 'XYZmagnetometer/Z', f"magnetometer_Z_{date_str}.csv")
    
    # Check if files exist
    for file_path in [x_file, y_file, z_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Magnetometer data file not found: {file_path}")
    
    # Load data
    x_data = pd.read_csv(x_file)
    y_data = pd.read_csv(y_file)
    z_data = pd.read_csv(z_file)
    
    # Convert timestamp to string format used in CSV
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    # Find the row with the matching timestamp
    x_row = x_data[x_data['timestamp'] == timestamp_str]
    y_row = y_data[y_data['timestamp'] == timestamp_str]
    z_row = z_data[z_data['timestamp'] == timestamp_str]
    
    if len(x_row) == 0 or len(y_row) == 0 or len(z_row) == 0:
        raise ValueError(f"No data found for timestamp: {timestamp_str}")
    
    # Get station names (columns except 'timestamp')
    stations = x_data.columns[1:].tolist()
    
    # Extract values for each component
    obs_x = x_row.iloc[0, 1:].values
    obs_y = y_row.iloc[0, 1:].values
    obs_z = z_row.iloc[0, 1:].values
    
    return obs_x, obs_y, obs_z, stations

def plot_comparison(Be, Bn, Bu, lon_mesh, lat_mesh, timestamp, 
                   station_coords, stations_inside, obs_x, obs_y, obs_z,
                   vmin, vmax, grid):
    """
    Create a comparison plot of SECS predictions and observations.
    
    Args:
        Be, Bn, Bu (numpy.ndarray): SECS magnetic field components
        lon_mesh, lat_mesh (numpy.ndarray): Longitude and latitude meshgrids
        timestamp (datetime): Timestamp for the plot
        station_coords (pandas.DataFrame): DataFrame with station coordinates
        stations_inside (list): List of stations inside the grid
        obs_x, obs_y, obs_z (numpy.ndarray): Observed magnetic field components
        vmin, vmax (float): Minimum and maximum values for vertical field color mapping
        grid (cs.CSgrid): SECS grid object
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure and axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Create CSplot object for plotting on the cubed sphere
    csax = CSplot(ax=ax, csgrid=grid)
    
    # Plot vertical component as color map
    pcm = csax.pcolormesh(
        lon_mesh, lat_mesh, Bu,
        vmin=vmin, vmax=vmax,
        cmap='RdBu_r',
        zorder=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label('Vertical Magnetic Field (nT)')
    
    # Plot horizontal component as quiver (black arrows)
    step = 1  # Downsample for clarity
    csax.quiver(
        Be[::step, ::step], Bn[::step, ::step],
        lon_mesh[::step, ::step], lat_mesh[::step, ::step],
        scale=500,
        scale_units='inches',
        color='black',
        width=0.0015,
        alpha=0.5,
        headwidth=3,
        headlength=5,
        label='SECS Prediction',
        zorder=3
    )
    
    # Plot stations and observations
    if stations_inside:
        # Get station indices
        station_indices = [station_coords.index.get_loc(st) if st in station_coords.index else -1 
                          for st in stations_inside]
        
        # Filter valid indices
        valid_indices = [i for i in station_indices if i >= 0]
        
        if valid_indices:
            # Get station coordinates
            station_lons = station_coords.iloc[valid_indices]['longitude'].values
            station_lats = station_coords.iloc[valid_indices]['latitude'].values
            
            # Plot station locations
            csax.scatter(
                station_lons,
                station_lats,
                c='blue', marker='.', s=30, alpha=0.5, zorder=11,
                label='Magnetometer Stations'
            )
            
            # Get observations for these stations
            valid_obs_x = np.array([obs_x[stations_inside.index(st)] if st in stations_inside else np.nan 
                                  for st in stations_inside])
            valid_obs_y = np.array([obs_y[stations_inside.index(st)] if st in stations_inside else np.nan 
                                  for st in stations_inside])
            
            # Filter out NaN values
            nan_mask = ~(np.isnan(valid_obs_x) | np.isnan(valid_obs_y))
            
            if np.sum(nan_mask) > 0:
                # Plot observations (blue arrows)
                obs_quiver = csax.quiver(
                    valid_obs_y[nan_mask], valid_obs_x[nan_mask],
                    station_lons[nan_mask], station_lats[nan_mask],
                    scale=500,
                    scale_units='inches',
                    color='blue',
                    width=0.005,
                    alpha=0.5,
                    headwidth=4,
                    headlength=6,
                    label='Observations',
                    zorder=12
                )
                
                # Add quiver key for scale reference
                ax.quiverkey(obs_quiver, 0.13, 0.975, 500, '500 nT',
                            labelpos='E',
                            coordinates='figure')
    
    # Add coastlines and title
    csax.add_coastlines(color='grey', resolution='50m')
    plt.title(f'SECS Magnetic Field Comparison - {timestamp}')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    return fig

def main():
    """
    Main function to compare SECS predictions with magnetometer observations.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M')
    
    # Print configuration
    print("\n=== SECS Comparison Configuration ===")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Data directory: {args.data_dir}")
    print(f"SECS directory: {args.secs_dir}")
    print(f"Vertical field range: {args.vmin} nT to {args.vmax} nT")
    print("======================================\n")
    
    # Load station coordinates
    print("Loading station coordinates...")
    station_coords = load_station_coordinates(args.data_dir)
    print(f"Loaded coordinates for {len(station_coords)} stations")
    
    # Load grid metadata
    print("Loading grid metadata...")
    grid_metadata = load_grid_metadata(args.secs_dir, start_date.year)
    
    # Set up grid from metadata
    print("Setting up SECS grid...")
    grid = setup_grid_from_metadata(grid_metadata)
    print(f"Grid spans approximately {grid_metadata['grid_resolution'] * grid_metadata['grid_shape_EW']:.1f} km E-W × "
          f"{grid_metadata['grid_resolution'] * grid_metadata['grid_shape_NS']:.1f} km N-S")
    
    # Create output directory
    output_dir = os.path.join(args.secs_dir, str(start_date.year), 'comparison')
    create_dir(output_dir)
    print(f"Saving comparison plots to: {output_dir}")
    
    # Generate list of timestamps to process
    timestamps = generate_timestamp_list(start_date, end_date)
    
    # Process each timestamp
    total_timestamps = len(timestamps)
    print(f"Processing {total_timestamps} timestamps...")
    
    for i, timestamp in enumerate(timestamps):
        try:
            print(f"Processing timestamp {i+1}/{total_timestamps}: {timestamp}")
            
            # Load SECS magnetic field components
            Be = load_magnetic_component(timestamp, 'Be', args.secs_dir)
            Bn = load_magnetic_component(timestamp, 'Bn', args.secs_dir)
            Bu = load_magnetic_component(timestamp, 'Bu', args.secs_dir)
            
            # Load magnetometer data
            obs_x, obs_y, obs_z, stations = load_magnetometer_data(timestamp, args.data_dir)
            
            # Filter stations to keep only those inside the grid
            stations_inside = filter_stations_by_grid(stations, station_coords, grid)
            print(f"  {len(stations_inside)}/{len(stations)} stations inside grid")
            
            # Create comparison plot
            fig = plot_comparison(
                Be, Bn, Bu, grid.lon_mesh, grid.lat_mesh, timestamp,
                station_coords, stations_inside, obs_x, obs_y, obs_z,
                args.vmin, args.vmax, grid
            )
            
            # Save plot
            output_file = os.path.join(output_dir, f"secs_comp_{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Plot saved to: {output_file}")
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nProcessing completed successfully.")

if __name__ == "__main__":
    main()