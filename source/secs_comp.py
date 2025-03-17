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
    create_dir, load_magnetic_component
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
        data_dir (str): Base directory for data
        
    Returns:
        pandas.DataFrame: DataFrame with station coordinates
    """
    station_coords_file = os.path.join(data_dir, 'magnetometer/station_coordinates.csv')
    station_coordinates = pd.read_csv(station_coords_file)
    station_coordinates.set_index('station', inplace=True)
    
    return station_coordinates

def load_grid_metadata(secs_dir, year):
    """
    Load SECS grid metadata from file.
    
    Args:
        secs_dir (str): Directory containing SECS data
        year (int or str): Year for the data
        
    Returns:
        dict: Dictionary with grid metadata
    """
    metadata_path = os.path.join(secs_dir, str(year), 'grid_metadata.txt')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Grid metadata file not found: {metadata_path}")
    
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            try:
                # Try to convert to float if possible
                metadata[key] = float(value)
            except ValueError:
                # Otherwise keep as string
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
    # Set up cubed sphere projection
    projection = cs.CSprojection(
        position=(metadata['grid_center_lon'], metadata['grid_center_lat']),
        orientation=metadata['grid_orientation']
    )
    
    # Create the grid on the cubed-sphere
    grid = cs.CSgrid(
        projection,
        L=metadata['grid_shape_EW'] * metadata['grid_resolution'] * 1000,  # East-West extent in meters
        W=metadata['grid_shape_NS'] * metadata['grid_resolution'] * 1000,  # North-South extent in meters
        Lres=int(metadata['grid_shape_NS']),  # E-W resolution (number of points)
        Wres=int(metadata['grid_shape_EW']),  # N-S resolution (number of points)
        R=metadata['Rgrid'],                  # Grid radius
        wshift=1e3                            # Small shift in grid placement (meters)
    )
    
    return grid

def load_magnetometer_data(timestamp, data_dir):
    """
    Load magnetometer data for a specific timestamp.
    
    Args:
        timestamp (datetime): Timestamp to load data for
        data_dir (str): Base directory for data
        
    Returns:
        tuple: (obs_x, obs_y, obs_z, stations) - Observed magnetic field components and station names
    """
    Xdir = os.path.join(data_dir, 'XYZmagnetometer/X')
    Ydir = os.path.join(data_dir, 'XYZmagnetometer/Y')
    Zdir = os.path.join(data_dir, 'XYZmagnetometer/Z')
    
    date_str = timestamp.strftime('%Y-%m-%d')
    
    # Construct file paths
    x_file = os.path.join(Xdir, f"magnetometer_X_{date_str}.csv")
    y_file = os.path.join(Ydir, f"magnetometer_Y_{date_str}.csv")
    z_file = os.path.join(Zdir, f"magnetometer_Z_{date_str}.csv")
    
    if not os.path.exists(x_file) or not os.path.exists(y_file) or not os.path.exists(z_file):
        raise FileNotFoundError(f"Magnetometer data files not found for {date_str}")
    
    # Load data
    x_data = pd.read_csv(x_file)
    y_data = pd.read_csv(y_file)
    z_data = pd.read_csv(z_file)
    
    # Convert timestamps
    x_data['timestamp'] = pd.to_datetime(x_data['timestamp'])
    y_data['timestamp'] = pd.to_datetime(y_data['timestamp'])
    z_data['timestamp'] = pd.to_datetime(z_data['timestamp'])
    
    # Get data for specific timestamp
    x_row = x_data[x_data['timestamp'] == timestamp]
    y_row = y_data[y_data['timestamp'] == timestamp]
    z_row = z_data[z_data['timestamp'] == timestamp]
    
    if x_row.empty or y_row.empty or z_row.empty:
        raise ValueError(f"No magnetometer data found for timestamp {timestamp}")
    
    # Get station columns and extract observations
    station_cols = [col for col in x_data.columns if col != 'timestamp']
    obs_x = x_row[station_cols].values.flatten()
    obs_y = y_row[station_cols].values.flatten()
    obs_z = z_row[station_cols].values.flatten()
    
    return obs_x, obs_y, obs_z, station_cols

def filter_stations_by_grid(stations, station_coordinates, grid):
    """
    Filter stations to keep only those inside the grid area.
    
    Args:
        stations (list): List of station names
        station_coordinates (DataFrame): DataFrame with station coordinates
        grid (cs.CSgrid): SECS grid object
        
    Returns:
        list: List of stations inside the grid
    """
    stations_inside = []
    
    for station in stations:
        if station in station_coordinates.index:
            lon = station_coordinates.loc[station, 'longitude']
            lat = station_coordinates.loc[station, 'latitude']
            
            # Check if station is inside grid
            if grid.lon_mesh.min() <= lon <= grid.lon_mesh.max() and \
               grid.lat_mesh.min() <= lat <= grid.lat_mesh.max():
                stations_inside.append(station)
    
    return stations_inside

def plot_comparison(Be, Bn, Bu, lon_mesh, lat_mesh, timestamp, 
                   station_coords, stations_inside, obs_x, obs_y, obs_z,
                   vmin, vmax):
    """
    Create comparison plot of SECS predictions and observations.
    
    Args:
        Be, Bn, Bu (numpy.ndarray): SECS magnetic field components
        lon_mesh, lat_mesh (numpy.ndarray): Longitude and latitude meshes
        timestamp (datetime): Timestamp for the plot
        station_coords (DataFrame): DataFrame with station coordinates
        stations_inside (list): List of stations inside the grid
        obs_x, obs_y, obs_z (numpy.ndarray): Observed magnetic field components
        vmin, vmax (float): Min and max values for vertical field color mapping
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Create cubed sphere projection using the same parameters as the grid
    projection = cs.CSprojection(
        position=(lon_mesh.mean(), lat_mesh.mean()),
        orientation=0.0
    )
    
    # Create SECS grid
    RE = 6371e3
    grid = cs.CSgrid(
        projection,
        L=32 * 0.5e5,
        W=32 * 0.5e5,
        Lres=32,
        Wres=32,
        R=RE + 110e3,
        wshift=1e3
    )
    
    csax = CSplot(ax, grid)
    
    # Plot vertical component as color map
    im = csax.pcolormesh(lon_mesh, lat_mesh, Bu, 
                        cmap=plt.cm.bwr,
                        vmin=vmin,
                        vmax=vmax)
    plt.colorbar(im, label='Vertical Magnetic Field (nT)')
    
    # Plot SECS horizontal predictions (black arrows)
    step = 2  # Plot every nth arrow to avoid crowding
    secs_quiver = csax.quiver(
        Be[::step, ::step], Bn[::step, ::step],
        lon_mesh[::step, ::step], lat_mesh[::step, ::step],
        scale=500,
        scale_units='inches',
        color='black',
        width=0.003,
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
                c='blue', marker='.', s=30, zorder=11,
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
                    headwidth=4,
                    headlength=6,
                    label='Observations',
                    zorder=12
                )
                
                # Add quiver key for scale reference
                ax.quiverkey(obs_quiver, 0.1, 0.9, 200, '200 nT',
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
                args.vmin, args.vmax
            )
            
            # Save plot
            output_file = os.path.join(output_dir, f"secs_comp_{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
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