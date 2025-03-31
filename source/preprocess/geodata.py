#!/usr/bin/env python3
"""
Geodata Preprocessing (geodata.py)
==================================

This script preprocesses the geodata CSV file for Branch 1 of the magnetic field forecasting model.
It implements:
1. Loading and initial preprocessing of geodata
2. Time-based train/validation/test splitting
3. Creation of windowed observations with 24-hour lookback
4. Visualization of the data splits and window examples

Usage:
------
python geodata.py --data_path path/to/geodata.csv --output_dir path/to/output
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess geodata for magnetic field forecasting')
    
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the geodata CSV file')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Directory to save processed data and visualizations')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation (default: 0.15)')
    parser.add_argument('--lookback_hours', type=int, default=24,
                        help='Lookback window size in hours (default: 24)')
    parser.add_argument('--time_resolution_minutes', type=int, default=15,
                        help='Time resolution in minutes (default: 15)')
    parser.add_argument('--stride_minutes', type=int, default=15,
                        help='Stride between windows in minutes (default: 15)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use only 1 year of data for testing different network structures')
    parser.add_argument('--test_year', type=int, default=None,
                        help='Specific year to use for test mode (default: most recent year)')
    
    return parser.parse_args()

def load_and_preprocess_data(data_path, test_mode=False, test_year=None):
    """
    Load and preprocess the geodata CSV file.
    
    Args:
        data_path (str): Path to the geodata CSV file
        test_mode (bool): Whether to use only 1 year of data
        test_year (int): Specific year to use for test mode
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with datetime index
    """
    print(f"Loading data from {data_path}...")
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
    # Check if data is already preprocessed
    print(f"Data shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    
    # Check for timestamp or DateTime column
    timestamp_col = None
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    elif 'DateTime' in df.columns:
        timestamp_col = 'DateTime'
    else:
        # Try to find any column that might be a timestamp
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                print(f"Using column '{col}' as timestamp")
                break
    
    if timestamp_col is None:
        raise ValueError("No timestamp column found in the data. Please specify a column with datetime information.")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df[timestamp_col])
    
    # If the timestamp column wasn't already named 'timestamp', drop the original column
    if timestamp_col != 'timestamp':
        df = df.drop(columns=[timestamp_col])
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # If test_mode is True, filter data for only one year
    if test_mode:
        if test_year is None:
            # Find the most recent full year in the data
            years = df.index.year.unique()
            years.sort()
            test_year = years[-1]
            print(f"No test year specified. Using the most recent year: {test_year}")
        
        # Filter data for specified year
        df = df[df.index.year == test_year]
        print(f"Test mode: Using data from year {test_year} only.")
        print(f"Filtered data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split the data into training, validation, and test sets based on time.
    
    Args:
        df (pd.DataFrame): Dataframe with datetime index
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
    
    Returns:
        tuple: (train_df, val_df, test_df) - Split dataframes
    """
    print("\nSplitting data into train/validation/test sets...")
    
    # Sort data by timestamp
    df = df.sort_index()
    
    # Calculate split indices
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    # Split the data
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    print(f"Training set: {len(train_df)} samples from {train_df.index.min()} to {train_df.index.max()}")
    print(f"Validation set: {len(val_df)} samples from {val_df.index.min()} to {val_df.index.max()}")
    print(f"Test set: {len(test_df)} samples from {test_df.index.min()} to {test_df.index.max()}")
    
    return train_df, val_df, test_df

def create_windowed_dataset(df, lookback_timesteps, stride_timesteps=1):
    """
    Create windowed observations from sequential data.
    
    Args:
        df (pd.DataFrame): Dataframe with datetime index
        lookback_timesteps (int): Number of timesteps to look back
        stride_timesteps (int): Number of timesteps to stride between windows
    
    Returns:
        tuple: (windows, timestamps) - numpy array of windows and corresponding end timestamps
    """
    data = df.values
    n_samples = len(data)
    
    # Calculate how many windows we can create
    n_windows = max(0, (n_samples - lookback_timesteps) // stride_timesteps + 1)
    
    # Create windows
    windows = []
    timestamps = []
    
    for i in range(n_windows):
        start_idx = i * stride_timesteps
        end_idx = start_idx + lookback_timesteps
        
        if end_idx <= n_samples:
            window = data[start_idx:end_idx]
            windows.append(window)
            timestamps.append(df.index[end_idx-1])  # Use the last timestamp in the window
    
    return np.array(windows), np.array(timestamps)

def visualize_data_splits(train_df, val_df, test_df, save_path=None):
    """
    Visualize the data splits to verify correct splitting.
    
    Args:
        train_df, val_df, test_df (pd.DataFrame): Split dataframes
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(15, 8))
    
    # Select a representative column to plot
    representative_col = train_df.columns[0]
    print(f"Visualizing splits using column: {representative_col}")
    
    # Plot data splits
    plt.plot(train_df.index, train_df[representative_col], 'b-', alpha=0.7, label='Training')
    plt.plot(val_df.index, val_df[representative_col], 'g-', alpha=0.7, label='Validation')
    plt.plot(test_df.index, test_df[representative_col], 'r-', alpha=0.7, label='Test')
    
    # Add vertical lines to mark the split points
    plt.axvline(x=train_df.index[-1], color='k', linestyle='--')
    plt.axvline(x=val_df.index[-1], color='k', linestyle='--')
    
    # Formatting
    plt.xlabel('Date')
    plt.ylabel(representative_col)
    plt.title('Data Split Visualization')
    plt.legend()
    
    # Format x-axis to show dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Split visualization saved to {save_path}")
    
    plt.show()

def visualize_windows(windows, timestamps, columns, num_examples=3, save_path=None):
    """
    Visualize example windows to verify correct windowing.
    
    Args:
        windows (np.array): Array of windowed data
        timestamps (np.array): Array of window end timestamps
        columns (list): List of column names
        num_examples (int): Number of example windows to show
        save_path (str, optional): Path to save the figure
    """
    # Pick a few random windows
    indices = np.random.choice(len(windows), min(num_examples, len(windows)), replace=False)
    
    plt.figure(figsize=(15, 5 * num_examples))
    
    for i, idx in enumerate(indices):
        window = windows[idx]
        end_time = timestamps[idx]
        
        # Create a subplot for each window
        plt.subplot(num_examples, 1, i+1)
        
        # Plot each feature in the window
        for j, col in enumerate(columns):
            plt.plot(np.arange(window.shape[0]), window[:, j], label=col if i == 0 else None)
        
        # Only show legend on the first subplot
        if i == 0:
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=max(1, len(columns)//10))
        
        plt.title(f'Example Window (Ending at {end_time})')
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Window visualization saved to {save_path}")
    
    plt.show()

def save_tfrecord(windows, timestamps, output_path):
    """
    Save windowed data as TFRecord file.
    
    Args:
        windows (np.array): Array of windowed data
        timestamps (np.array): Array of window end timestamps
        output_path (str): Path to save the TFRecord file
    """
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for i, (window, timestamp) in enumerate(zip(windows, timestamps)):
            # Convert timestamp to unix timestamp
            ts = int(pd.Timestamp(timestamp).timestamp())
            
            # Create feature dictionary
            feature = {
                'window': _float_feature(window),
                'window_shape': _int64_feature(window.shape[0] * window.shape[1]),
                'timestamp': _int64_feature(ts)
            }
            
            # Create Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Write Example
            writer.write(example.SerializeToString())
    
    print(f"Saved {len(windows)} windows to {output_path}")

def main():
    """Main function to preprocess geodata."""
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
    
    # Load and preprocess data
    df = load_and_preprocess_data(args.data_path, args.test_mode, args.test_year)
    
    # Split data
    train_df, val_df, test_df = split_data(df, args.train_ratio, args.val_ratio)
    
    # Visualize data splits
    split_viz_path = os.path.join(output_dir, 'data_splits.png')
    visualize_data_splits(train_df, val_df, test_df, split_viz_path)
    
    # Calculate lookback and stride in timesteps
    minutes_per_timestep = args.time_resolution_minutes
    lookback_timesteps = args.lookback_hours * 60 // minutes_per_timestep
    stride_timesteps = args.stride_minutes // minutes_per_timestep
    
    # Ensure stride is at least 1 timestep
    stride_timesteps = max(1, stride_timesteps)
    
    print(f"\nCreating windows with {lookback_timesteps} timesteps ({args.lookback_hours} hours)")
    print(f"Using stride of {stride_timesteps} timesteps ({args.stride_minutes} minutes)")
    
    # Create windowed datasets
    train_windows, train_timestamps = create_windowed_dataset(train_df, lookback_timesteps, stride_timesteps)
    val_windows, val_timestamps = create_windowed_dataset(val_df, lookback_timesteps, stride_timesteps)
    test_windows, test_timestamps = create_windowed_dataset(test_df, lookback_timesteps, stride_timesteps)
    
    print(f"Created {len(train_windows)} training windows")
    print(f"Created {len(val_windows)} validation windows")
    print(f"Created {len(test_windows)} test windows")
    
    # Visualize example windows
    window_viz_path = os.path.join(output_dir, 'window_examples.png')
    visualize_windows(train_windows, train_timestamps, df.columns, num_examples=3, save_path=window_viz_path)
    
    # Save processed data as TFRecord files
    train_tfrecord = os.path.join(output_dir, 'train.tfrecord')
    val_tfrecord = os.path.join(output_dir, 'val.tfrecord')
    test_tfrecord = os.path.join(output_dir, 'test.tfrecord')
    
    save_tfrecord(train_windows, train_timestamps, train_tfrecord)
    save_tfrecord(val_windows, val_timestamps, val_tfrecord)
    save_tfrecord(test_windows, test_timestamps, test_tfrecord)
    
    # Save column names for future reference
    with open(os.path.join(output_dir, 'columns.txt'), 'w') as f:
        f.write('\n'.join(df.columns))
    
    # Save metadata
    metadata = {
        'test_mode': args.test_mode,
        'test_year': args.test_year if args.test_mode else 'N/A',
        'num_features': len(df.columns),
        'lookback_timesteps': lookback_timesteps,
        'time_resolution_minutes': minutes_per_timestep,
        'stride_minutes': args.stride_minutes,
        'num_train_windows': len(train_windows),
        'num_val_windows': len(val_windows),
        'num_test_windows': len(test_windows),
        'train_period': f"{train_df.index.min()} to {train_df.index.max()}",
        'val_period': f"{val_df.index.min()} to {val_df.index.max()}",
        'test_period': f"{test_df.index.min()} to {test_df.index.max()}"
    }
    
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print("\nPreprocessing complete!")
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    main()
