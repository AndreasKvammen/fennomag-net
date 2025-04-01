#!/usr/bin/env python3
"""
SECS Data Preprocessing (secsdata.py)
====================================

This script preprocesses SECS RGB image data for Branch 2 of the magnetic field forecasting model.
It implements:
1. Loading and organizing SECS RGB images
2. Time-based train/validation/test splitting
3. Creation of windowed observations with 3-hour lookback
4. Temporal alignment with geodata windows
5. Visualization of the data splits and window examples

Usage:
------
python secsdata.py --data_dir path/to/secs/figures --output_dir path/to/output
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from glob import glob
import re

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess SECS data for magnetic field forecasting')
    
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/akv020/Tensorflow/fennomag-net/data/secs/2024/figures',
                        help='Directory containing SECS RGB images')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Directory to save processed data and visualizations')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation (default: 0.15)')
    parser.add_argument('--lookback_minutes', type=int, default=180,
                        help='Lookback window size in minutes (default: 180)')
    parser.add_argument('--stride_minutes', type=int, default=15,
                        help='Stride between windows in minutes (default: 15)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use only 1 year of data for testing')
    parser.add_argument('--test_year', type=int, default=None,
                        help='Specific year to use for test mode (default: most recent year)')
    
    return parser.parse_args()

def load_and_check_images(data_dir, test_mode=False, test_year=None):
    """
    Load and organize SECS RGB images, checking for missing timestamps.
    
    Args:
        data_dir (str): Directory containing SECS RGB images
        test_mode (bool): Whether to use only 1 year of data
        test_year (int): Specific year to use for test mode
    
    Returns:
        dict: Dictionary mapping timestamps to image paths
        list: List of missing timestamps
    """
    print(f"Loading SECS images from {data_dir}...")
    
    # Get all PNG files in the directory
    image_files = glob(os.path.join(data_dir, "*.png"))
    
    if not image_files:
        raise ValueError(f"No PNG files found in {data_dir}")
    
    # Extract timestamps from filenames and create mapping
    timestamp_pattern = re.compile(r"(\d{8}_\d{6})\.png$")
    image_map = {}
    
    for image_path in image_files:
        match = timestamp_pattern.search(os.path.basename(image_path))
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            # Filter for test_mode if enabled
            if test_mode:
                if test_year is None:
                    # Find the most recent year
                    test_year = max(int(os.path.basename(f)[:4]) for f in image_files)
                    print(f"No test year specified. Using the most recent year: {test_year}")
                
                if timestamp.year != test_year:
                    continue
            
            image_map[timestamp] = image_path
    
    if not image_map:
        raise ValueError("No valid images found after filtering")
    
    # Check for missing timestamps
    timestamps = sorted(image_map.keys())
    expected_timestamps = []
    current = timestamps[0]
    
    while current <= timestamps[-1]:
        expected_timestamps.append(current)
        current += timedelta(minutes=1)
    
    missing_timestamps = [ts for ts in expected_timestamps if ts not in image_map]
    
    print(f"Found {len(image_map)} images")
    print(f"Missing timestamps: {len(missing_timestamps)}")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
    
    return image_map, missing_timestamps

def create_windowed_dataset(image_map, lookback_minutes, stride_minutes):
    """
    Create windowed observations from SECS images, aligned with geodata timestamps.
    
    Args:
        image_map (dict): Dictionary mapping timestamps to image paths
        lookback_minutes (int): Size of lookback window in minutes
        stride_minutes (int): Stride between windows in minutes
    
    Returns:
        list: List of tuples (window_paths, end_timestamp) for valid windows
    """
    print("\nCreating windowed dataset...")
    
    # Sort timestamps
    timestamps = sorted(image_map.keys())
    windows = []
    
    # Ensure end timestamps align with 15-minute intervals
    for end_time in timestamps:
        # Check if timestamp is aligned with stride interval
        if end_time.minute % stride_minutes != 0 or end_time.second != 0:
            continue
            
        # Calculate start time for this window
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Generate list of required timestamps for this window
        window_timestamps = []
        current = start_time
        while current <= end_time:
            window_timestamps.append(current)
            current += timedelta(minutes=1)
        
        # Check if all required timestamps are available
        if all(ts in image_map for ts in window_timestamps):
            window_paths = [image_map[ts] for ts in window_timestamps]
            windows.append((window_paths, end_time))
    
    print(f"Created {len(windows)} valid windows")
    return windows

def split_data(windows, train_ratio=0.7, val_ratio=0.15):
    """
    Split windows into training, validation, and test sets based on time.
    
    Args:
        windows (list): List of (window_paths, end_timestamp) tuples
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
    
    Returns:
        tuple: (train_windows, val_windows, test_windows)
    """
    print("\nSplitting data into train/validation/test sets...")
    
    # Sort windows by end timestamp
    windows.sort(key=lambda x: x[1])
    
    # Calculate split indices
    n = len(windows)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    # Split the data
    train_windows = windows[:train_idx]
    val_windows = windows[train_idx:val_idx]
    test_windows = windows[val_idx:]
    
    print(f"Training set: {len(train_windows)} windows")
    print(f"Validation set: {len(val_windows)} windows")
    print(f"Test set: {len(test_windows)} windows")
    
    return train_windows, val_windows, test_windows

def load_and_process_image(image_path):
    """
    Load and process a single RGB image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        np.array: Processed image array
    """
    with Image.open(image_path) as img:
        # Convert to numpy array and normalize to [0, 1]
        return np.array(img, dtype=np.float32) / 255.0

def save_tfrecord(windows, output_path):
    """
    Save windowed SECS data as TFRecord file.
    
    Args:
        windows (list): List of (window_paths, end_timestamp) tuples
        output_path (str): Path to save the TFRecord file
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for window_paths, end_timestamp in windows:
            # Load and stack all images in the window
            window_data = np.stack([load_and_process_image(path) for path in window_paths])
            
            # Convert timestamp to unix timestamp
            ts = int(end_timestamp.timestamp())
            
            # Create feature dictionary
            feature = {
                'window': _bytes_feature(window_data.tobytes()),
                'window_shape': _int64_feature(len(window_paths)),
                'timestamp': _int64_feature(ts),
                'height': _int64_feature(window_data.shape[1]),
                'width': _int64_feature(window_data.shape[2]),
                'channels': _int64_feature(window_data.shape[3])
            }
            
            # Create Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Write Example
            writer.write(example.SerializeToString())
    
    print(f"Saved {len(windows)} windows to {output_path}")

def visualize_windows(windows, num_examples=3, save_path=None):
    """
    Visualize example windows to verify correct windowing.
    
    Args:
        windows (list): List of (window_paths, end_timestamp) tuples
        num_examples (int): Number of example windows to show
        save_path (str, optional): Path to save the figure
    """
    # Pick random windows
    indices = np.random.choice(len(windows), min(num_examples, len(windows)), replace=False)
    
    fig = plt.figure(figsize=(15, num_examples * 5))
    
    for i, idx in enumerate(indices):
        window_paths, end_timestamp = windows[idx]
        
        # Sample images from the window (start, middle, end)
        sample_indices = [0, len(window_paths)//2, -1]
        
        for j, img_idx in enumerate(sample_indices):
            plt.subplot(num_examples, 3, i*3 + j + 1)
            img = load_and_process_image(window_paths[img_idx])
            plt.imshow(img)
            
            timestamp = end_timestamp - timedelta(minutes=len(window_paths)-1-img_idx)
            plt.title(f'Window {i+1}\n{timestamp}')
            plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Window visualization saved to {save_path}")
    
    plt.show()

def main():
    """Main function to preprocess SECS data."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output subdirectory based on test mode
    if args.test_mode:
        output_subdir = 'test_mode'
        if args.test_year:
            output_subdir = f'test_mode_{args.test_year}'
    else:
        output_subdir = 'full_dataset'
    
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load and check images
    image_map, missing_timestamps = load_and_check_images(
        args.data_dir, 
        args.test_mode, 
        args.test_year
    )
    
    # Create windows
    windows = create_windowed_dataset(
        image_map, 
        args.lookback_minutes, 
        args.stride_minutes
    )
    
    # Split data
    train_windows, val_windows, test_windows = split_data(
        windows, 
        args.train_ratio, 
        args.val_ratio
    )
    
    # Save as TFRecord files
    train_tfrecord = os.path.join(output_dir, 'train.tfrecord')
    val_tfrecord = os.path.join(output_dir, 'val.tfrecord')
    test_tfrecord = os.path.join(output_dir, 'test.tfrecord')
    
    save_tfrecord(train_windows, train_tfrecord)
    save_tfrecord(val_windows, val_tfrecord)
    save_tfrecord(test_windows, test_tfrecord)
    
    # Visualize example windows
    window_viz_path = os.path.join(output_dir, 'window_examples.png')
    visualize_windows(train_windows, num_examples=3, save_path=window_viz_path)
    
    # Save metadata
    metadata = {
        'test_mode': args.test_mode,
        'test_year': args.test_year if args.test_mode else 'N/A',
        'lookback_minutes': args.lookback_minutes,
        'stride_minutes': args.stride_minutes,
        'image_dimensions': '20x20x3',
        'num_train_windows': len(train_windows),
        'num_val_windows': len(val_windows),
        'num_test_windows': len(test_windows),
        'train_period': f"{train_windows[0][1]} to {train_windows[-1][1]}",
        'val_period': f"{val_windows[0][1]} to {val_windows[-1][1]}",
        'test_period': f"{test_windows[0][1]} to {test_windows[-1][1]}",
        'missing_timestamps': len(missing_timestamps)
    }
    
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    # Save list of missing timestamps if any
    if missing_timestamps:
        with open(os.path.join(output_dir, 'missing_timestamps.txt'), 'w') as f:
            for ts in missing_timestamps:
                f.write(f"{ts}\n")
    
    print("\nPreprocessing complete!")
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    main()
