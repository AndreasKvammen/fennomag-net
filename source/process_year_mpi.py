#!/usr/bin/env python3
"""
MPI-based script to process a full year of data with SECS Generator

This script uses MPI to distribute the processing of days across multiple processes,
making it suitable for high-performance computing environments with many cores/threads.

Requirements:
- mpi4py (pip install mpi4py)
- An MPI implementation (OpenMPI, MPICH, etc.)

Run with: mpiexec -n <num_processes> python process_year_mpi.py --year 2024
"""

import os
# Limit OpenMP threading
os.environ["OMP_NUM_THREADS"] = "1"
# Limit MKL threading
os.environ["MKL_NUM_THREADS"] = "1"
# Limit BLAS threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"
# Limit OpenMP threading
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import subprocess
import argparse
from datetime import datetime, timedelta
import calendar
from mpi4py import MPI

def get_days_in_year(year):
    """Generate a list of all days in the given year."""
    days = []
    for month in range(1, 13):
        num_days = calendar.monthrange(year, month)[1]
        for day in range(1, num_days + 1):
            days.append((year, month, day))
    return days

def process_day(year, month, day, grid_params, reg_params, data_dirs):
    """Process a single day with secs_gen.py."""
    # Format the date strings
    start_date = f"{year}-{month:02d}-{day:02d} 00:00"
    end_date = f"{year}-{month:02d}-{day:02d} 23:59"
    
    # Build command to run secs_gen.py
    cmd = [
        'python', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'secs_gen.py'),
        '--start_date', f'"{start_date}"',
        '--end_date', f'"{end_date}"',
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
    cmd_str = ' '.join(cmd)
    print(f"Process {MPI.COMM_WORLD.Get_rank()}: Processing {year}-{month:02d}-{day:02d}")
    result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"Error processing {year}-{month:02d}-{day:02d}: {result.stderr.decode()}")
        return False
    return True

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parse arguments on the root process only
    if rank == 0:
        parser = argparse.ArgumentParser(description='Process a year of data with SECS Generator using MPI')
        parser.add_argument('--year', type=int, default=2024, help='Year to process')
        parser.add_argument('--data_dir', type=str, 
                            default='/home/akv020/fennomag/fennomag-net/data',
                            help='Directory containing magnetometer data')
        parser.add_argument('--output_dir', type=str, 
                            default=None,
                            help='Directory for output files')
        args = parser.parse_args()
        
        # Set default output directory if not specified
        if args.output_dir is None:
            args.output_dir = f'/home/akv020/fennomag/fennomag-net/data/secs_{args.year}'
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print configuration
        print(f"\n=== SECS Year Processing with MPI ===")
        print(f"Year to process: {args.year}")
        print(f"Number of MPI processes: {size}")
        print(f"Input data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print("======================================\n")
        
        # Get all days in the year
        all_days = get_days_in_year(args.year)
        print(f"Total days to process: {len(all_days)}")
        
        # Define parameters
        grid_params = {
            "center_lon": 17.0,
            "center_lat": 67.0,
            "shape_ns": 32,
            "shape_ew": 32,
            "resolution": 100,
            "time_resolution": 1
        }
        
        reg_params = {
            "l0": 1e-2,
            "l1": 1e-2,
            "mirror_depth": 1000
        }
        
        data_dirs = {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir
        }
        
        # Package data for broadcasting
        data = {
            'year': args.year,
            'all_days': all_days,
            'grid_params': grid_params,
            'reg_params': reg_params,
            'data_dirs': data_dirs
        }
    else:
        data = None
    
    # Broadcast data to all processes
    data = comm.bcast(data, root=0)
    
    # Distribute days among processes
    all_days = data['all_days']
    days_per_process = len(all_days) // size
    remainder = len(all_days) % size
    
    # Calculate start and end indices for this process
    start_idx = rank * days_per_process + min(rank, remainder)
    end_idx = start_idx + days_per_process + (1 if rank < remainder else 0)
    
    # Get days for this process
    my_days = all_days[start_idx:end_idx]
    
    # Process assigned days
    successful_days = 0
    for year, month, day in my_days:
        success = process_day(
            year, month, day, 
            data['grid_params'], 
            data['reg_params'], 
            data['data_dirs']
        )
        if success:
            successful_days += 1
    
    # Gather results
    results = comm.gather(successful_days, root=0)
    
    # Print summary on root process
    if rank == 0:
        total_successful = sum(results)
        print(f"\nProcessing completed!")
        print(f"Successfully processed {total_successful}/{len(all_days)} days")
        print(f"Results saved to: {data['data_dirs']['output_dir']}")

if __name__ == "__main__":
    main() 