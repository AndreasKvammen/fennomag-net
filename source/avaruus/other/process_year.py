#!/usr/bin/env python3
"""
Process a full year of data with SECS Generator

This script processes magnetometer data for an entire year using the SECS (Spherical Elementary
Current Systems) method. It can process the data either sequentially or in parallel (one process
per month), making it faster on multi-core systems.

The script divides the year into daily chunks and calls the secs_gen.py script for each day.
"""

import os
import subprocess
import argparse
from datetime import datetime, timedelta
import multiprocessing
from pathlib import Path

def process_month(year, month, grid_params, reg_params, data_dirs):
    """
    Process a full month of data in daily chunks.
    
    This function processes all days in a given month by calling secs_gen.py for each day.
    
    Args:
        year (int): Year to process (e.g., 2023)
        month (int): Month to process (1-12)
        grid_params (dict): Grid parameters for SECS calculation
        reg_params (dict): Regularization parameters for SECS calculation
        data_dirs (dict): Data directories for input and output
    """
    # Calculate the first and last day of the month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    end_date = end_date.replace(hour=23, minute=59)
    
    # Process each day in the month
    current_date = start_date
    while current_date <= end_date:
        # Define start and end of current day (from 00:00 to 23:59)
        day_start = current_date
        day_end = current_date.replace(hour=23, minute=59)
        
        # Format dates for command line
        start_str = day_start.strftime('%Y-%m-%d %H:%M')
        end_str = day_end.strftime('%Y-%m-%d %H:%M')
        
        # Build command to run secs_gen.py with all necessary parameters
        cmd = [
            'python', os.path.join(os.path.dirname(__file__), 'secs_gen.py'),
            '--start_date', f'"{start_str}"',
            '--end_date', f'"{end_str}"',
            '--grid_center', f'{grid_params["center_lon"]} {grid_params["center_lat"]}',
            '--grid_shape', f'{grid_params["shape_ns"]} {grid_params["shape_ew"]}',
            '--grid_resolution', f'{grid_params["resolution"]}',
            '--l0', f'{reg_params["l0"]}',
            '--l1', f'{reg_params["l1"]}',
            '--mirror_depth', f'{reg_params["mirror_depth"]}',
            '--time_resolution', f'{grid_params["time_resolution"]}',
            '--data_dir', f'{data_dirs["data_dir"]}',
            '--output_dir', f'{data_dirs["output_dir"]}'
        ]
        
        # Execute command and show which day is being processed
        print(f"Processing {day_start.strftime('%Y-%m-%d')}...")
        subprocess.run(' '.join(cmd), shell=True)
        
        # Move to next day
        current_date += timedelta(days=1)

def process_year(year, grid_params, reg_params, data_dirs, parallel=True):
    """
    Process a full year of data, either sequentially or in parallel.
    
    Args:
        year (int): Year to process (e.g., 2023)
        grid_params (dict): Grid parameters for SECS calculation
        reg_params (dict): Regularization parameters for SECS calculation
        data_dirs (dict): Data directories for input and output
        parallel (bool): If True, process all months in parallel (faster but uses more CPU cores)
                         If False, process days sequentially (slower but uses less resources)
    """
    if parallel:
        # PARALLEL MODE: Process each month in parallel (12 processes)
        print(f"Processing year {year} in parallel mode (one process per month)...")
        processes = []
        
        # Start one process for each month (1-12)
        for month in range(1, 13):
            print(f"Starting process for month {month}...")
            p = multiprocessing.Process(
                target=process_month,
                args=(year, month, grid_params, reg_params, data_dirs)
            )
            processes.append(p)
            p.start()
            
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"All months of {year} have been processed.")
    else:
        # SEQUENTIAL MODE: Process one day at a time for the whole year
        print(f"Processing year {year} in sequential mode...")
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59)
        
        current_date = start_date
        while current_date <= end_date:
            # Define start and end of current day
            day_start = current_date
            day_end = current_date.replace(hour=23, minute=59)
            
            # Format dates for command line
            start_str = day_start.strftime('%Y-%m-%d %H:%M')
            end_str = day_end.strftime('%Y-%m-%d %H:%M')
            
            # Build command to run secs_gen.py with all necessary parameters
            cmd = [
                'python', os.path.join(os.path.dirname(__file__), 'secs_gen.py'),
                '--start_date', f'"{start_str}"',
                '--end_date', f'"{end_str}"',
                '--grid_center', f'{grid_params["center_lon"]} {grid_params["center_lat"]}',
                '--grid_shape', f'{grid_params["shape_ns"]} {grid_params["shape_ew"]}',
                '--grid_resolution', f'{grid_params["resolution"]}',
                '--l0', f'{reg_params["l0"]}',
                '--l1', f'{reg_params["l1"]}',
                '--mirror_depth', f'{reg_params["mirror_depth"]}',
                '--time_resolution', f'{grid_params["time_resolution"]}',
                '--data_dir', f'{data_dirs["data_dir"]}',
                '--output_dir', f'{data_dirs["output_dir"]}'
            ]
            
            # Execute command and show which day is being processed
            print(f"Processing {day_start.strftime('%Y-%m-%d')}...")
            subprocess.run(' '.join(cmd), shell=True)
            
            # Move to next day
            current_date += timedelta(days=1)

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Process a year of magnetometer data with SECS Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define command line arguments
    parser.add_argument('--year', type=int, default=2024, 
                        help='Year to process (e.g., 2023)')
    parser.add_argument('--parallel', action='store_true', 
                        help='Process months in parallel (uses 12 CPU cores)')
    parser.add_argument('--data_dir', type=str, 
                        help='Directory containing magnetometer data')
    parser.add_argument('--output_dir', type=str, 
                        help='Directory for output files')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Define SECS grid parameters (these control the spatial resolution and extent)
    grid_params = {
        "center_lon": 17.0,       # Center longitude of the grid (degrees)
        "center_lat": 67.0,       # Center latitude of the grid (degrees)
        "shape_ns": 32,           # Number of grid points in north-south direction
        "shape_ew": 32,           # Number of grid points in east-west direction
        "resolution": 100,        # Grid spacing in kilometers
        "time_resolution": 1      # Time resolution in minutes
    }
    
    # Define regularization parameters (these control the smoothness of the solution)
    reg_params = {
        "l0": 1e-2,               # Zero-order regularization strength
        "l1": 1e-2,               # First-order regularization strength
        "mirror_depth": 1000      # Depth of the mirror current in kilometers
    }
    
    # Set up data directories with relative paths if not provided
    # This makes the script work after cloning the repository
    data_dirs = {
        "data_dir": args.data_dir if args.data_dir else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "data", "XYZmagnetometer"
        ),
        "output_dir": args.output_dir if args.output_dir else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "data", "secs"
        )
    }
    
    # Print configuration information
    print("\n=== SECS Year Processing Configuration ===")
    print(f"Year to process: {args.year}")
    print(f"Parallel processing: {'Enabled (12 cores)' if args.parallel else 'Disabled (sequential)'}")
    print(f"Input data directory: {data_dirs['data_dir']}")
    print(f"Output directory: {data_dirs['output_dir']}")
    print("==========================================\n")
    
    # Ensure output directory exists
    os.makedirs(data_dirs["output_dir"], exist_ok=True)
    
    # Process the year
    process_year(args.year, grid_params, reg_params, data_dirs, parallel=args.parallel)
    
    print(f"\nProcessing of year {args.year} completed!")
    print(f"Results are saved in: {data_dirs['output_dir']}") 