"""
Helper functions for SECS-Based Magnetometer Data Analysis

This module contains utility functions for:
- Directory creation and file management
- Data loading and processing
- Station management
- SECS data saving
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

def create_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_magnetometer_data(start_date, end_date, x_dir, y_dir, z_dir):
    """
    Load magnetometer data for a specified date range from CSV files.
    Each component (X, Y, Z) is stored in separate directories with daily files.
    
    Args:
        start_date (datetime): Start date for data loading
        end_date (datetime): End date for data loading
        x_dir, y_dir, z_dir (str): Directories containing X, Y, Z component data
    
    Returns:
        dict: Dictionary containing X, Y, Z component dataframes, where each dataframe
             has timestamps as rows and station measurements as columns
    """
    data = {'X': [], 'Y': [], 'Z': []}
    
    # Get unique dates needed
    dates_needed = pd.date_range(start_date.date(), end_date.date(), freq='D')
    
    # Iterate through each day in the date range
    for current_date in dates_needed:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Construct file paths for each component
        x_file = os.path.join(x_dir, f"magnetometer_X_{date_str}.csv")
        y_file = os.path.join(y_dir, f"magnetometer_Y_{date_str}.csv")
        z_file = os.path.join(z_dir, f"magnetometer_Z_{date_str}.csv")
        
        # Load data if files exist
        if os.path.exists(x_file):
            data['X'].append(pd.read_csv(x_file))
        if os.path.exists(y_file):
            data['Y'].append(pd.read_csv(y_file))
        if os.path.exists(z_file):
            data['Z'].append(pd.read_csv(z_file))
    
    # Concatenate all days into single dataframes
    data['X'] = pd.concat(data['X'], ignore_index=True) if data['X'] else None
    data['Y'] = pd.concat(data['Y'], ignore_index=True) if data['Y'] else None
    data['Z'] = pd.concat(data['Z'], ignore_index=True) if data['Z'] else None
    
    # Convert timestamps and filter for the exact time range
    for comp in ['X', 'Y', 'Z']:
        if data[comp] is not None:
            data[comp]['timestamp'] = pd.to_datetime(data[comp]['timestamp'])
            mask = (data[comp]['timestamp'] >= start_date) & (data[comp]['timestamp'] <= end_date)
            data[comp] = data[comp][mask].reset_index(drop=True)
    
    return data

def trailing_average(df_original, interval_minutes=10):
    """
    Compute trailing average for the data at specified time intervals.
    This smooths the data and reduces noise in the measurements.
    Uses only past values for each point (true trailing average).
    
    Args:
        df_original (DataFrame): Original data with timestamp column and station measurements
        interval_minutes (int): Time interval for averaging in minutes
    
    Returns:
        DataFrame: Averaged data at specified intervals
    """
    df = df_original.copy()
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # First handle any duplicate timestamps by taking their mean
    df = df.groupby(level=0).mean()
    
    # Calculate trailing mean over specified time window (only using past values)
    df_rolled = df.rolling(f'{interval_minutes}T', closed='right').mean()
    
    # Resample to regular intervals
    df_rolled = df_rolled.resample(f'{interval_minutes}T').mean()
    
    # Forward fill any missing values after resampling
    df_rolled = df_rolled.fillna(method='ffill')
    
    df_rolled.reset_index(inplace=True)
    return df_rolled

def get_common_stations(data, station_coordinates):
    """
    Get stations that exist in both magnetometer data and coordinate information.
    This ensures we only use stations with known locations and available measurements.
    
    Args:
        data (dict): Dictionary containing magnetometer data for X, Y, Z components
        station_coordinates (DataFrame): Station coordinate information with lat/lon
    
    Returns:
        list: List of common station names
    """
    data_stations = set(data['X'].columns) - {'timestamp'}
    coord_stations = set(station_coordinates.index)
    return list(data_stations.intersection(coord_stations))

def create_secs_directories(base_path, year):
    """
    Create directory structure for storing SECS magnetic component data.
    
    Args:
        base_path (str): Root directory for SECS data
        year (str): Year for which to create directories
    
    Creates directories in structure:
        base_path/year/Be
        base_path/year/Bn
        base_path/year/Bu
    """
    components = ['Be', 'Bn', 'Bu']
    year_path = os.path.join(base_path, str(year))
    
    for component in components:
        component_path = os.path.join(year_path, component)
        if not os.path.exists(component_path):
            os.makedirs(component_path)
            print(f"Created directory: {component_path}")

def save_magnetic_component(component_data, timestamp, component_name, base_path):
    """
    Save magnetic component data as a simple 2D matrix in L-W coordinates.
    
    Args:
        component_data (numpy.ndarray): 2D array of magnetic field values [L, W]
        timestamp (pandas.Timestamp): Timestamp for the data
        component_name (str): Name of component ('Be', 'Bn', or 'Bu')
        base_path (str): Base directory for saving files
    """
    # Create filename with timestamp
    filename = f"{component_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    year = str(timestamp.year)
    save_path = os.path.join(base_path, year, component_name, filename)
    
    # Save the 2D matrix directly to CSV (no headers, no index)
    np.savetxt(save_path, component_data, delimiter=',')
    
    return save_path

def save_grid_metadata(grid, base_path, grid_resolution, grid_shape, year):
    """Save grid metadata to a single text file within the year folder."""
    metadata = {
        'grid_center_lon': grid.projection.lon0,
        'grid_center_lat': grid.projection.lat0,
        'grid_orientation': grid.projection.orientation,
        'grid_resolution': grid_resolution,  # This matches the input resolution
        'grid_shape_NS': grid_shape[0],
        'grid_shape_EW': grid_shape[1],
        'Rgrid': grid.R
    }
    
    # Save metadata to text file in year folder
    year_path = os.path.join(base_path, str(year))
    create_dir(year_path)  # Ensure year directory exists
    metadata_path = os.path.join(year_path, 'grid_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")