"""
Helper Functions for SECS-Based Magnetometer Data Analysis
=========================================================

This module contains utility functions for the SECS Generator script (secs_gen.py).
It implements the core functionality for processing magnetometer data and calculating
Spherical Elementary Current Systems (SECS).

Table of Contents:
-----------------
1. File and Directory Management
   - create_dir: Create directory if it doesn't exist
   - create_secs_directories: Create directory structure for SECS components
   - save_magnetic_component: Save magnetic field component data
   - save_grid_metadata: Save grid configuration information

2. Data Loading and Processing
   - load_magnetometer_data: Load X, Y, Z component data from files
   - trailing_average: Apply time averaging to smooth data
   - get_common_stations: Find stations with both data and coordinates

3. SECS Grid and Station Management
   - setup_secs_grid: Configure SECS grid with cubed sphere projection
   - filter_stations_by_grid: Keep only stations inside the grid area

4. SECS Calculations
   - calculate_secs_matrices: Calculate SECS basis matrices
   - process_timestamp: Process data for a single timestamp

5. Visualization
   - create_rgb_image: Create RGB image from magnetic field components
   - save_rgb_image: Save RGB image to file

These functions are designed to be modular and reusable, with clear documentation
to help users understand the SECS calculation process.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from secsy import cubedsphere as cs  # Handles cubed sphere projections and grid operations
from secsy import get_SECS_B_G_matrices  # Calculates SECS basis function matrices
from PIL import Image

# =====================================================================
# 1. File and Directory Management
# =====================================================================

def create_dir(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def create_secs_directories(base_path, year):
    """
    Create directory structure for storing SECS magnetic component data.
    
    The function creates three subdirectories for the east, north, and
    upward magnetic field components (Be, Bn, Bu) within the year folder.
    
    Args:
        base_path (str): Root directory for SECS data
        year (int or str): Year for which to create directories
    
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
    
    The function saves a 2D array of magnetic field values to a CSV file
    with a filename that includes the component name and timestamp.
    Data is saved in single precision (float32) to save disk space.
    
    Args:
        component_data (numpy.ndarray): 2D array of magnetic field values [L, W]
        timestamp (pandas.Timestamp): Timestamp for the data
        component_name (str): Name of component ('Be', 'Bn', or 'Bu')
        base_path (str): Base directory for saving files
        
    Returns:
        str: Path to the saved file
    """
    # Create filename with timestamp
    filename = f"{component_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    year = str(timestamp.year)
    save_path = os.path.join(base_path, year, component_name, filename)
    
    # Convert to single precision (float32) to save space
    component_data_float32 = component_data.astype(np.float32)
    
    # Save the 2D matrix directly to CSV (no headers, no index)
    np.savetxt(save_path, component_data_float32, delimiter=',')
    
    return save_path

def save_grid_metadata(grid, base_path, grid_resolution, grid_shape, year):
    """
    Save grid metadata to a single text file within the year folder.
    
    This function saves important grid configuration parameters to a text file,
    which can be used later to reconstruct the grid or interpret the results.
    
    Args:
        grid (cs.CSgrid): SECS grid object
        base_path (str): Base directory for SECS data
        grid_resolution (float): Grid resolution in kilometers
        grid_shape (tuple): Grid shape as (N-S points, E-W points)
        year (int or str): Year for the data
    """
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

# =====================================================================
# 2. Data Loading and Processing
# =====================================================================
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

def load_magnetometer_data(start_date, end_date, x_dir, y_dir, z_dir):
    """
    Load magnetometer data for a specified date range from CSV files.
    
    This function loads X, Y, Z component data from separate directories,
    where each component has daily CSV files. It handles multiple days by
    concatenating the data and filters to the exact time range requested.
    
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
    
    This function smooths the data by calculating a trailing average over a
    specified time window. It uses only past values for each point (true trailing
    average) to avoid using future information that wouldn't be available in
    real-time applications.
    
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
    
    This function finds the intersection of stations that have both measurement data
    and known coordinates, ensuring we only use stations with complete information.
    
    Args:
        data (dict): Dictionary containing magnetometer data for X, Y, Z components
        station_coordinates (DataFrame): Station coordinate information with lat/lon
    
    Returns:
        list: List of common station names
    """
    data_stations = set(data['X'].columns) - {'timestamp'}
    coord_stations = set(station_coordinates.index)
    return list(data_stations.intersection(coord_stations))

# =====================================================================
# 3. SECS Grid and Station Management
# =====================================================================

def setup_secs_grid(grid_center, grid_shape, grid_resolution, ionosphere=True):
    """
    Set up a SECS grid using cubed sphere projection.
    
    This function creates a grid for SECS calculations using a cubed sphere
    projection, which provides a more accurate representation of the spherical
    Earth surface than a simple Cartesian grid.
    
    Args:
        grid_center (tuple): Grid center (longitude, latitude) in degrees
        grid_shape (tuple): Grid shape (N-S points, E-W points)
        grid_resolution (float): Grid resolution in meters
        ionosphere (bool): Whether to place grid at ionospheric altitude (110km) or ground
        
    Returns:
        cs.CSgrid: Configured grid object
    """
    RE = 6371e3  # Earth radius in meters
    Rgrid = RE + 110e3 if ionosphere else RE  # Place grid at ionospheric altitude or ground
    
    # Set up cubed sphere projection
    projection = cs.CSprojection(
        position=(grid_center[0], grid_center[1]),  # (lon, lat) in degrees
        orientation=0.0  # orientation of the local x-axis on the cubed-sphere face
    )
    
    # Create the grid on the cubed-sphere
    grid = cs.CSgrid(
        projection,
        L=grid_shape[1] * grid_resolution,  # East-West extent in meters
        W=grid_shape[0] * grid_resolution,  # North-South extent in meters
        Lres=grid_shape[0],  # E-W resolution (number of points)
        Wres=grid_shape[1],  # N-S resolution (number of points)
        R=Rgrid,             # Grid radius (Earth radius + ionosphere height)
        wshift=1e3           # Small shift in grid placement (meters)
    )
    
    return grid

def filter_stations_by_grid(stations, station_coordinates, grid):
    """
    Filter stations to keep only those inside the grid.
    
    This function checks each station's coordinates to determine if it falls
    within the boundaries of the SECS grid, and returns only those stations
    that are inside the grid area.
    
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
            if grid.ingrid(lon, lat):
                stations_inside.append(station)
    
    return stations_inside

# =====================================================================
# 4. SECS Calculations
# =====================================================================

def calculate_secs_matrices(grid, station_coordinates, stations, mirror_depth=9999):
    """
    Calculate SECS basis matrices for stations and full grid.
    
    This function calculates the SECS basis matrices that relate current amplitudes
    to magnetic field components. It calculates matrices both for the station
    locations (for inversion) and for the full grid (for forward calculation).
    
    The mirror method can be used to account for induction effects by placing a
    mirror current system below the Earth's surface.
    
    Args:
        grid (cs.CSgrid): SECS grid object
        station_coordinates (DataFrame): DataFrame with station coordinates
        stations (list): List of station names to include
        mirror_depth (int): Mirror depth in km (9999 to disable)
        
    Returns:
        tuple: (station_matrices, grid_matrices) where each is a tuple of (GeB, GnB, GuB)
    """
    RE = 6371e3  # Earth radius in meters
    
    # Get station coordinates for SECS basis function calculation
    lon_mag = station_coordinates.loc[stations, 'longitude'].values
    lat_mag = station_coordinates.loc[stations, 'latitude'].values
    
    # Set up mirror method for induction nullification if enabled
    induction_nullification_radius = RE - mirror_depth * 1000 if mirror_depth != 9999 else None
    
    # Calculate SECS basis matrices for stations
    GeB_mag, GnB_mag, GuB_mag = get_SECS_B_G_matrices(
        lat_mag, lon_mag, RE, grid.lat, grid.lon,
        induction_nullification_radius=induction_nullification_radius
    )
    
    # Calculate SECS matrices for the full grid
    GeB_full, GnB_full, GuB_full = get_SECS_B_G_matrices(
        grid.lat_mesh.flatten(),
        grid.lon_mesh.flatten(),
        RE,
        grid.lat,
        grid.lon,
        induction_nullification_radius=induction_nullification_radius
    )
    
    return (GeB_mag, GnB_mag, GuB_mag), (GeB_full, GnB_full, GuB_full)

def process_timestamp(timestamp_idx, timestamp, data, good_stations, station_coordinates, 
                     grid, secs_matrices, regularization_params):
    """
    Process a single timestamp to calculate SECS currents and magnetic fields.
    
    This function performs the core SECS calculation for a single timestamp by solving
    an inverse problem: given magnetic field measurements at ground stations, what are
    the ionospheric current amplitudes that produced these measurements?
    
    The SECS method involves:
    1. Forward problem: Given current amplitudes, calculate magnetic fields (G * I = B)
    2. Inverse problem: Given magnetic fields, find current amplitudes (I = G⁻¹ * B)
    
    Since the inverse problem is highly underdetermined (more unknowns than measurements),
    regularization is necessary to obtain a physically meaningful solution.
    
    Args:
        timestamp_idx (int): Index of the timestamp in the data
        timestamp (datetime): Timestamp to process
        data (dict): Dictionary with X, Y, Z component data
        good_stations (list): List of stations to use
        station_coordinates (DataFrame): DataFrame with station coordinates
        grid (cs.CSgrid): SECS grid object
        secs_matrices (tuple): Tuple of (station_matrices, grid_matrices)
        regularization_params (dict): Dictionary with regularization parameters
        
    Returns:
        tuple: (I_timestamp, Be, Bn, Bu) - current amplitudes and magnetic field components
    """
    # Unpack SECS matrices
    (GeB_mag, GnB_mag, GuB_mag), (GeB_full, GnB_full, GuB_full) = secs_matrices
    
    # Extract magnetic field components for current timestamp
    By_i = data['Y'].iloc[timestamp_idx][good_stations].astype(float).values  # East component
    Bx_i = data['X'].iloc[timestamp_idx][good_stations].astype(float).values  # North component
    Bz_i = -data['Z'].iloc[timestamp_idx][good_stations].astype(float).values  # Vertical component (negated)
    
    # Create mask for valid measurements (not NaN)
    valid_mask = ~(np.isnan(By_i) | np.isnan(Bx_i) | np.isnan(Bz_i))
    
    # Filter measurements and combine into observation vector
    By_i = By_i[valid_mask]
    Bx_i = Bx_i[valid_mask]
    Bz_i = Bz_i[valid_mask]
    d = np.hstack([By_i, Bx_i, Bz_i])  # Combined observation vector
    
    # Get valid stations and their indices
    valid_stations = [st for j, st in enumerate(good_stations) if valid_mask[j]]
    idx = [station_coordinates.index.get_loc(st) for st in valid_stations]
    
    # Select relevant rows from SECS matrices for valid stations
    GeB_i = GeB_mag[idx, :]  # East component forward matrix
    GnB_i = GnB_mag[idx, :]  # North component forward matrix
    GuB_i = GuB_mag[idx, :]  # Vertical component forward matrix
    
    # Combine matrices for all components to form the complete forward matrix G
    # G relates current amplitudes to magnetic field measurements: B = G * I
    # Each row corresponds to a measurement, each column to a SECS pole
    G = np.vstack([GeB_i, GnB_i, GuB_i])
    
    # Create measurement covariance matrix (here simplified as identity)
    # cvinv_valid represents the inverse of measurement error covariance
    # Using identity matrix assumes equal and uncorrelated errors in all measurements
    cvinv_valid = np.eye(len(d))
    
    # Form normal equations for weighted least squares
    # GTG is G^T * C^-1 * G, the normal equation matrix
    # GTd is G^T * C^-1 * d, the right-hand side vector
    GTG = G.T @ cvinv_valid @ G
    GTd = G.T @ cvinv_valid @ d
    
    # Get regularization matrices from the grid
    # De and Dn are differentiation matrices in east and north directions
    # They approximate spatial derivatives of the current distribution
    De, Dn = grid.get_Le_Ln()
    
    # Form combined regularization matrix for smoothness constraint
    # DTD represents the squared gradient operator (Laplacian-like)
    DTD = De.T @ De + Dn.T @ Dn
    
    # Calculate regularization scaling factors to balance data fit vs. regularization
    scale_gtg = np.median(np.diag(GTG))  # Scale for data fit term
    scale_dtd = np.median(np.diag(DTD)) if DTD.size > 0 else 1  # Scale for smoothness term
    
    # Compute regularization terms with user-provided parameters
    # l0 controls amplitude damping (zeroth-order Tikhonov regularization)
    #   - Large l0: Solutions with smaller amplitudes are preferred
    #   - Small l0: Allows larger amplitudes, potentially more detailed but noisier solution
    # 
    # l1 controls smoothness (first-order Tikhonov regularization)
    #   - Large l1: Smoother solutions with less spatial variation
    #   - Small l1: Allows sharper gradients, potentially more detailed but less stable
    T0 = regularization_params['l0'] * scale_gtg
    T1 = regularization_params['l1'] * scale_gtg / (scale_dtd if scale_dtd != 0 else 1e-10)
    
    # Combined regularization matrix R
    # R = T0*I + T1*DTD combines amplitude damping and smoothness constraints
    # This implements a mixed zeroth and first order Tikhonov regularization
    R = T0 * np.eye(grid.size) + T1 * DTD
    
    # Solve regularized inverse problem
    # The regularized normal equation is: (GTG + R) * I = GTd
    # This balances data fit against the regularization constraints
    # We compute the posterior model covariance matrix Cmpost first
    Cmpost = np.linalg.lstsq(GTG + R, np.eye(GTG.shape[0]), rcond=None)[0]
    
    # Then calculate the model parameters (current amplitudes)
    I_timestamp = Cmpost @ GTd
    
    # Forward calculation: Calculate magnetic field components at all grid points
    # This uses the full grid matrices to compute B = G * I for visualization
    Be = GeB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)  # East component
    Bn = GnB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)  # North component
    Bu = GuB_full.dot(I_timestamp).reshape(grid.lat_mesh.shape)  # Vertical component
    
    return I_timestamp, Be, Bn, Bu

# =====================================================================
# 5. Visualization
# =====================================================================

def create_rgb_image(be, bn, bu, vmin=-1250, vmax=1250, std=500):
    """
    Create an RGB image from magnetic field components using statistical normalization.
    
    The function maps each component to a color channel using a normal CDF mapping:
    - Red: Eastward component (Be)
    - Green: Northward component (Bn)
    - Blue: Upward component (Bu)
    
    Statistical normalization using the normal CDF:
    1. For each value x, calculate Φ(x/σ) where Φ is the normal CDF
    2. This maps the range (-∞,+∞) to (0,1) with 0 mapping to 0.5
    3. Values within ±1σ use ~68% of the pixel range
    4. Values within ±2σ use ~95% of the pixel range
    
    Args:
        be (numpy.ndarray): Eastward magnetic field component
        bn (numpy.ndarray): Northward magnetic field component
        bu (numpy.ndarray): Upward magnetic field component
        vmin (float): Minimum expected value (for reference only)
        vmax (float): Maximum expected value (for reference only)
        std (float): Standard deviation parameter for normalization
                     Controls the allocation of pixel values:
                     - Smaller values give higher resolution near zero but compress extremes more
                     - Larger values give more even distribution but less resolution near zero
        
    Returns:
        numpy.ndarray: RGB image array with values in [0, 255] (uint8)
    """
    # Import scipy.stats for the normal CDF
    from scipy.stats import norm
    
    def normalize_to_uint8_normal_dist(data):
        # Calculate CDF values (between 0 and 1)
        # This maps the data to a probability using the normal distribution
        cdf_values = norm.cdf(data / std)
        
        # Map to 0-255 range for 8-bit pixel values
        pixel_values = (cdf_values * 255).astype(np.uint8)
        
        return pixel_values
    
    # Normalize each component to 8-bit (0-255)
    r = normalize_to_uint8_normal_dist(be)
    g = normalize_to_uint8_normal_dist(bn)
    b = normalize_to_uint8_normal_dist(bu)
    
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