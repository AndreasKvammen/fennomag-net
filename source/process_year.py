#!/usr/bin/env python3
"""
Process a full year of data with SECS Generator
"""

import os
import subprocess
from datetime import datetime, timedelta

def process_year(year, grid_params, reg_params, data_dirs):
    """
    Process a full year of data in daily chunks.
    
    Args:
        year (int): Year to process
        grid_params (dict): Grid parameters
        reg_params (dict): Regularization parameters
        data_dirs (dict): Data directories
    """
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
        
        # Build command
        cmd = [
            'python', 'secs_gen.py',
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
        
        # Execute command
        print(f"Processing {day_start.strftime('%Y-%m-%d')}...")
        subprocess.run(' '.join(cmd), shell=True)
        
        # Move to next day
        current_date += timedelta(days=1)

if __name__ == "__main__":
    # Define parameters
    grid_params = {
        "center_lon": 17.0,
        "center_lat": 67.0,
        "shape_ns": 32,
        "shape_ew": 32,
        "resolution": 100,  # km
        "time_resolution": 1  # minutes
    }
    
    reg_params = {
        "l0": 1e-2,
        "l1": 1e-2,
        "mirror_depth": 1000
    }
    
    data_dirs = {
        "data_dir": "/Users/akv020/Tensorflow/fennomag-net/data/XYZmagnetometer",
        "output_dir": "/Users/akv020/Tensorflow/fennomag-net/data/secs"
    }
    
    # Process year 2024
    process_year(2024, grid_params, reg_params, data_dirs) 