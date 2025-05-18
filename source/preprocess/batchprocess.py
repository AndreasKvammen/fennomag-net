#!/usr/bin/env python3
"""
Unified Batch Processing (batchprocess.py)
========================================

This script creates aligned batches of data for a two-branch neural network that forecasts
magnetic field variations at Earth's surface. The two branches are:

Branch 1: Solar wind and global-scale geophysical data
    - 24-hour lookback windows at 15-minute resolution
    - Features include: SME, SML, SMU, SYM-D, SYM-H, etc.
    - Data source: geodata.csv

Branch 2: High-resolution SECS-processed magnetometer data
    - 3-hour lookback windows at 1-minute resolution
    - RGB images (20x20x3) representing magnetic field components
    - Data source: PNG files (YYYYMMDD_HHMMSS.png)

The script:
1. Loads and validates both data sources
2. Finds timestamps where both sources have complete data
3. Creates random batches of aligned observations
4. Saves data in TFRecord format for efficient training
5. Generates comprehensive metadata

Table of Contents:
-----------------
1. Imports and Constants
2. Argument Parsing
3. UnifiedDataProcessor Class
   3.1 Initialization
   3.2 Data Loading Functions
   3.3 Window Creation Functions
   3.4 TFRecord Creation Functions
   3.5 Batch Processing Functions
   3.6 Metadata Functions
4. Main Execution
"""

#------------------------------------------------------------------------------
# 1. Imports and Constants
#------------------------------------------------------------------------------
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from glob import glob
import json
from PIL import Image
import re

#------------------------------------------------------------------------------
# 2. Argument Parsing
#------------------------------------------------------------------------------
def parse_arguments():
    """
    Parse command line arguments for batch processing configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments including:
            - geodata_path: Path to the geodata CSV file
            - secs_dir: Directory containing SECS RGB images
            - output_dir: Directory to save processed batches
            - batch_size: Number of observations per batch
            - train_ratio: Proportion of data for training
            - val_ratio: Proportion of data for validation
            - test_mode: Whether to use only test year data
            - test_year: Specific year to use in test mode
    """
    parser = argparse.ArgumentParser(
        description='Process geodata and SECS data into aligned batches'
    )
    
    # Data paths
    parser.add_argument('--geodata_path', type=str,
                    default='/Users/akv020/Tensorflow/fennomag-net/data/geophysical/geodata.csv',
                    help='Path to geodata CSV file')
    parser.add_argument('--secs_dir', type=str,
                    default='/Users/akv020/Tensorflow/fennomag-net/data/secs/test_mode_2024/figures',
                    help='Directory containing SECS RGB images')
    parser.add_argument('--output_dir', type=str,
                    default='/Users/akv020/Tensorflow/fennomag-net/data/processed',
                    help='Directory to save processed batches')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=96,
                    help='Number of observations per batch')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                    help='Ratio of data to use for validation (default: 0.15)')
    
    # Test mode parameters
    parser.add_argument('--test_mode', action='store_true',
                    help='Use only test year data')
    parser.add_argument('--test_year', type=int, default=2024,
                    help='Year to use for test mode (default: 2024)')
    
    return parser.parse_args()

#------------------------------------------------------------------------------
# 3. UnifiedDataProcessor Class
#------------------------------------------------------------------------------
class UnifiedDataProcessor:
    """
    A class to process and align data from both branches of the neural network.
    
    This class handles:
    - Loading and validating data from both sources
    - Creating aligned observation windows
    - Generating random batches
    - Saving data in TFRecord format
    """
    def __init__(self, geodata_path, secs_dir, output_dir, batch_size=96,
                 train_ratio=0.7, val_ratio=0.15, test_mode=False, test_year=2024):
        """
        Initialize the unified data processor.
        """
        self.geodata_path = geodata_path
        self.secs_dir = secs_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_mode = test_mode
        self.test_year = test_year
        
        # Constants
        self.geo_lookback = 96  # 24 hours at 15-min resolution
        self.secs_lookback = 180  # 3 hours at 1-min resolution
        
        # Create output directories
        for branch in [1, 2]:
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(output_dir, f'branch{branch}', split), exist_ok=True)

    #--------------------------------------------------------------------------
    # 3.1 Data Loading Functions
    #--------------------------------------------------------------------------
    def load_data(self):
        """
        Load and validate both data sources.
        
        For Branch 1 (geodata):
        - Loads CSV file containing geophysical parameters
        - Converts DateTime column to pandas datetime index
        - Filters for test year if in test mode
        
        For Branch 2 (SECS):
        - Scans directory for PNG files
        - Creates mapping of timestamps to image paths
        - Validates image filename format
        
        Prints summary of loaded data for verification.
        """
        print("Loading data sources...")
        
        # Load geodata
        self.geodata = pd.read_csv(self.geodata_path)
        self.geodata['DateTime'] = pd.to_datetime(self.geodata['DateTime'])
        self.geodata.set_index('DateTime', inplace=True)
        
        if self.test_mode:
            self.geodata = self.geodata[self.geodata.index.year == self.test_year]
        
        # Load SECS image paths
        image_files = glob(os.path.join(self.secs_dir, "*.png"))
        self.secs_map = {}
        timestamp_pattern = re.compile(r"(\d{8}_\d{6})\.png$")
        
        for image_path in image_files:
            match = timestamp_pattern.search(os.path.basename(image_path))
            if match:
                timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                self.secs_map[timestamp] = image_path
        
        print(f"Loaded {len(self.geodata)} geodata records")
        print(f"Loaded {len(self.secs_map)} SECS images")

    def find_valid_timestamps(self):
        """
        Find timestamps that have valid data in both sources.
        
        For each potential end timestamp, verifies:
        1. Branch 1: Has complete 24-hour lookback (96 15-min timesteps)
        2. Branch 2: Has complete 3-hour lookback (180 1-min timesteps)
        
        Returns:
            list: Sorted list of timestamps that have complete data for both branches
        """
        print("Finding valid timestamps...")
        
        valid_timestamps = []
        
        # Check each potential end timestamp
        for end_time in self.geodata.index:
            # Check if we have enough lookback data for geodata
            geo_start = end_time - timedelta(minutes=15 * (self.geo_lookback - 1))
            geo_times = pd.date_range(geo_start, end_time, freq='15T')
            
            # Check if we have enough lookback data for SECS
            secs_start = end_time - timedelta(minutes=self.secs_lookback - 1)
            secs_times = pd.date_range(secs_start, end_time, freq='1T')
            
            # Verify data availability
            if (all(t in self.geodata.index for t in geo_times) and
                all(t in self.secs_map for t in secs_times)):
                valid_timestamps.append(end_time)
        
        print(f"Found {len(valid_timestamps)} valid timestamps")
        return sorted(valid_timestamps)

    #--------------------------------------------------------------------------
    # 3.2 Window Creation Functions
    #--------------------------------------------------------------------------
    def get_geodata_window(self, end_time):
        """
        Get geodata window for given end timestamp.
        
        Creates a 24-hour lookback window of geophysical parameters.
        
        Args:
            end_time (datetime): End timestamp for the window
            
        Returns:
            np.array: Array of shape [96, num_features] containing normalized parameters
        """
        start_time = end_time - timedelta(minutes=15 * (self.geo_lookback - 1))
        window_data = self.geodata.loc[start_time:end_time].values
        return window_data.astype(np.float32)

    def get_secs_window(self, end_time):
        """
        Get SECS image window for given end timestamp.
        
        Creates a 3-hour lookback window of SECS RGB images.
        
        Args:
            end_time (datetime): End timestamp for the window
            
        Returns:
            np.array: Array of shape [180, 20, 20, 3] containing normalized RGB values
        """
        start_time = end_time - timedelta(minutes=self.secs_lookback - 1)
        timestamps = pd.date_range(start_time, end_time, freq='1T')
        
        # Load and stack all images in the window
        images = []
        for ts in timestamps:
            with Image.open(self.secs_map[ts]) as img:
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
        
        return np.stack(images)

    #--------------------------------------------------------------------------
    # 3.3 TFRecord Creation Functions
    #--------------------------------------------------------------------------
    def create_tf_example(self, window_data, end_timestamp, branch):
        """
        Create TFRecord example for either branch.
        
        Args:
            window_data (np.array): Window data from either branch
            end_timestamp (datetime): End timestamp for the window
            branch (int): Branch number (1 for geodata, 2 for SECS)
            
        Returns:
            tf.train.Example: TFRecord example containing:
                - window: Flattened window data
                - timestamp: Unix timestamp
                - Branch-specific shape information
        """
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
        # Convert timestamp to unix timestamp
        ts = int(pd.Timestamp(end_timestamp).timestamp())
        
        feature = {
            'window': _float_feature(window_data),
            'timestamp': _int64_feature(ts)
        }
        
        # Add branch-specific shape information
        if branch == 1:
            feature.update({
                'lookback': _int64_feature(self.geo_lookback),
                'num_features': _int64_feature(window_data.shape[1])
            })
        else:  # branch == 2
            feature.update({
                'lookback': _int64_feature(self.secs_lookback),
                'height': _int64_feature(window_data.shape[1]),
                'width': _int64_feature(window_data.shape[2]),
                'channels': _int64_feature(window_data.shape[3])
            })
        
        return tf.train.Example(features=tf.train.Features(feature=feature))

    #--------------------------------------------------------------------------
    # 3.4 Batch Processing Functions
    #--------------------------------------------------------------------------
    def split_timestamps(self, valid_timestamps):
        """
        Split timestamps into train/val/test sets.
        
        Performs chronological split to maintain temporal order:
        - Training: First 70% of timestamps
        - Validation: Next 15% of timestamps
        - Testing: Final 15% of timestamps
        
        Args:
            valid_timestamps (list): List of timestamps with complete data
            
        Returns:
            tuple: (train_times, val_times, test_times)
        """
        print("Splitting timestamps into train/val/test sets...")
        
        # Sort timestamps to ensure deterministic splits
        timestamps = sorted(valid_timestamps)
        n = len(timestamps)
        
        # Calculate split indices
        train_idx = int(n * self.train_ratio)
        val_idx = int(n * (self.train_ratio + self.val_ratio))
        
        # Split the timestamps
        train_times = timestamps[:train_idx]
        val_times = timestamps[train_idx:val_idx]
        test_times = timestamps[val_idx:]
        
        print(f"Train set: {len(train_times)} timestamps")
        print(f"Validation set: {len(val_times)} timestamps")
        print(f"Test set: {len(test_times)} timestamps")
        
        return train_times, val_times, test_times

    def create_batches(self, timestamps):
        """
        Create random batches from timestamps.
        
        1. Randomly shuffles timestamps
        2. Creates batches of specified size
        3. Sorts timestamps within each batch for consistent processing
        
        Args:
            timestamps (list): List of timestamps to batch
            
        Returns:
            list: List of batches, each containing batch_size timestamps
        """
        # Shuffle timestamps
        timestamps = np.random.permutation(timestamps)
        
        # Create batches
        num_batches = len(timestamps) // self.batch_size
        batches = []
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_times = timestamps[start_idx:end_idx]
            batches.append(sorted(batch_times))  # Sort within batch for consistent processing
        
        return batches

    def save_batch(self, batch_timestamps, branch, split, batch_num):
        """
        Save a batch of data as TFRecord.
        
        Creates TFRecord file containing:
        - Window data for each timestamp in the batch
        - Associated metadata and shape information
        
        Args:
            batch_timestamps (list): List of timestamps in this batch
            branch (int): Branch number (1 or 2)
            split (str): Data split ('train', 'val', or 'test')
            batch_num (int): Batch number for filename
            
        Returns:
            str: Path to saved TFRecord file
        """
        output_path = os.path.join(
            self.output_dir,
            f"branch{branch}",
            split,
            f"batch_{batch_num:04d}.tfrecord"
        )
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for end_time in batch_timestamps:
                # Get window data
                if branch == 1:
                    window_data = self.get_geodata_window(end_time)
                else:  # branch == 2
                    window_data = self.get_secs_window(end_time)
                
                # Create and write example
                example = self.create_tf_example(window_data, end_time, branch)
                writer.write(example.SerializeToString())
        
        return output_path

    #--------------------------------------------------------------------------
    # 3.5 Metadata Functions
    #--------------------------------------------------------------------------
    def save_metadata(self, split_info):
        """
        Save processing metadata as JSON.
        
        Saves comprehensive information about:
        - Processing configuration
        - Data splits and sizes
        - Date ranges
        - Data dimensions and resolutions
        
        Args:
            split_info (dict): Information about data splits and batches
        """
        metadata = {
            'test_mode': self.test_mode,
            'test_year': self.test_year if self.test_mode else None,
            'batch_size': self.batch_size,
            'splits': {
                'train': {
                    'num_batches': len(split_info['train']),
                    'num_samples': len(split_info['train']) * self.batch_size,
                    'date_range': [
                        str(min(split_info['train'][0])),
                        str(max(split_info['train'][-1]))
                    ]
                },
                'val': {
                    'num_batches': len(split_info['val']),
                    'num_samples': len(split_info['val']) * self.batch_size,
                    'date_range': [
                        str(min(split_info['val'][0])),
                        str(max(split_info['val'][-1]))
                    ]
                },
                'test': {
                    'num_batches': len(split_info['test']),
                    'num_samples': len(split_info['test']) * self.batch_size,
                    'date_range': [
                        str(min(split_info['test'][0])),
                        str(max(split_info['test'][-1]))
                    ]
                }
            },
            'data_info': {
                'branch1': {
                    'lookback': self.geo_lookback,
                    'resolution': '15min'
                },
                'branch2': {
                    'lookback': self.secs_lookback,
                    'resolution': '1min',
                    'dimensions': '20x20x3'
                }
            }
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def process(self):
        """
        Main processing function that orchestrates the entire pipeline.
        
        Steps:
        1. Load and validate data sources
        2. Find valid timestamps with complete data
        3. Split timestamps into train/val/test
        4. For each split:
           - Create random batches
           - Process and save both branches
        5. Save metadata
        """
        print("\nStarting unified data processing...")
        
        # Load data sources
        self.load_data()
        
        # Find valid timestamps
        valid_timestamps = self.find_valid_timestamps()
        
        # Split timestamps
        train_times, val_times, test_times = self.split_timestamps(valid_timestamps)
        
        # Process each split
        split_info = {}
        for split_name, timestamps in [
            ('train', train_times),
            ('val', val_times),
            ('test', test_times)
        ]:
            print(f"\nProcessing {split_name} split...")
            
            # Create random batches
            batches = self.create_batches(timestamps)
            split_info[split_name] = batches
            
            # Process each batch for both branches
            for batch_num, batch_times in enumerate(batches):
                print(f"Processing batch {batch_num + 1}/{len(batches)}", end='\r')
                
                # Save Branch 1 (geodata)
                self.save_batch(batch_times, branch=1, split=split_name, batch_num=batch_num)
                
                # Save Branch 2 (SECS)
                self.save_batch(batch_times, branch=2, split=split_name, batch_num=batch_num)
            
            print(f"Completed {split_name} split: {len(batches)} batches")
        
        # Save metadata
        self.save_metadata(split_info)
        
        print("\nProcessing complete!")
        print(f"Output saved to: {self.output_dir}")

#------------------------------------------------------------------------------
# 4. Main Execution
#------------------------------------------------------------------------------
def main():
    """
    Main execution function.
    
    Example usage:
        # Process test year only
        python batchprocess.py --test_mode
        
        # Process specific year
        python batchprocess.py --test_mode --test_year 2024
        
        # Process full dataset with custom parameters
        python batchprocess.py --batch_size 64 --train_ratio 0.8
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create processor instance
    processor = UnifiedDataProcessor(
        geodata_path=args.geodata_path,
        secs_dir=args.secs_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_mode=args.test_mode,
        test_year=args.test_year
    )
    
    # Process data
    processor.process()

if __name__ == "__main__":
    main()
