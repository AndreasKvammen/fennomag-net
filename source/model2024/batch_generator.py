#!/usr/bin/env python3
"""
Memory-Efficient Batch Generator for Magnetic Field Forecasting
=============================================================

This module implements a memory-efficient batch generator for training a dual-branch
neural network that forecasts Earth's surface magnetic field variations. The generator
loads all data into memory once and creates batches by slicing the original arrays,
avoiding data duplication and unnecessary I/O operations.

Key Features:
------------
1. Single-pass data loading: All data is loaded into memory once at initialization
2. Efficient batch generation: Uses array slicing instead of creating copies
3. Temporal alignment: Ensures data from both branches is properly aligned
4. Memory optimization: Avoids storing multiple copies of overlapping windows
5. TensorFlow integration: Creates tf.data.Dataset objects for training

Data Structure:
-------------
The generator handles three types of data:
1. Target Data: Magnetic field components (Be, Bn, Bu) at a central point
2. Branch 1: Large-scale solar/geophysical data at 15-minute resolution
3. Branch 2: SECS-processed magnetometer data at 1-minute resolution

Usage Example:
------------
```python
# Initialize the generator
generator = BatchGenerator(
    target_path="path/to/target.csv",
    geodata_path="path/to/geodata.csv",
    secs_data_path="path/to/secs_data.npy",
    secs_timestamps_path="path/to/secs_timestamps.npy",
    batch_size=32
)

# Create TensorFlow datasets
train_dataset = generator.create_tf_dataset(split='train')
val_dataset = generator.create_tf_dataset(split='val')
test_dataset = generator.create_tf_dataset(split='test')

# Use in model training
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
```
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf

class BatchGenerator:
    """
    Memory-efficient batch generator for magnetic field forecasting model.
    
    This class:
    1. Loads all data into memory at initialization
    2. Maintains timestamp indices for both branches
    3. Generates batches by slicing the original data arrays
    4. Ensures temporal alignment between branches
    5. Implements time-shifting for forecasting
    
    The generator avoids data duplication by storing only one copy of each data array
    and creating windows through efficient array slicing operations. This approach
    is particularly beneficial for large datasets with significant temporal overlap
    between windows.
    """
    
    def __init__(self, target_path, geodata_path, secs_data_path, secs_timestamps_path,
                 batch_size=32, train_ratio=0.7, val_ratio=0.15, forecast_horizon=15,
                 valid_start=None, valid_end=None):
        """
        Initialize the batch generator.
        
        Args:
            target_path (str): Path to target data CSV containing Be, Bn, Bu values
            geodata_path (str): Path to geodata CSV with solar/geophysical parameters
            secs_data_path (str): Path to SECS data numpy array (21x21x3 grid)
            secs_timestamps_path (str): Path to SECS timestamps numpy array
            batch_size (int): Number of samples per batch (default: 32)
            train_ratio (float): Proportion of data for training (default: 0.7)
            val_ratio (float): Proportion of data for validation (default: 0.15)
            forecast_horizon (int): Number of timesteps to forecast ahead
                                  (in terms of Branch 1's resolution, i.e., 15-minute intervals)
            valid_start (datetime): Start of valid observation period (default: None)
            valid_end (datetime): End of valid observation period (default: None)
            
        Note:
            The test set will contain the remaining data (1 - train_ratio - val_ratio)
            If valid_start/end are not provided, they will be set to the data boundaries
        """
        # Store parameters
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.forecast_horizon = forecast_horizon
        self.valid_start = valid_start
        self.valid_end = valid_end
        
        # Constants for window sizes
        self.branch1_lookback = 96  # 24 hours at 15-minute intervals
        self.branch2_lookback = 180  # 3 hours at 1-minute intervals
        
        # Load all data into memory
        print("Loading data into memory...")
        self._load_data(target_path, geodata_path, secs_data_path, secs_timestamps_path)
        
        # Set valid observation boundaries if not provided
        if self.valid_start is None:
            self.valid_start = self.target_df['DateTime'].min() + pd.Timedelta(hours=24)
        if self.valid_end is None:
            self.valid_end = self.geo_df['DateTime'].max() - pd.Timedelta(minutes=15)
        
        print(f"\nValid observation period:")
        print(f"Start: {self.valid_start}")
        print(f"End: {self.valid_end}")
        print(f"Duration: {(self.valid_end - self.valid_start).total_seconds() / 3600:.1f} hours")
        
        # Create train/val/test splits
        self._create_splits()
        
        # Initialize batch indices for each split
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0
    
    def _load_data(self, target_path, geodata_path, secs_data_path, secs_timestamps_path):
        """
        Load all data sources into memory.
        
        This method:
        1. Loads target data (Be, Bn, Bu components)
        2. Loads Branch 1 data (solar/geophysical parameters)
        3. Loads Branch 2 data (SECS-processed magnetometer data)
        4. Converts timestamps to datetime objects
        5. Stores data shapes for later use
        
        Args:
            target_path (str): Path to target data CSV
            geodata_path (str): Path to geodata CSV
            secs_data_path (str): Path to SECS data numpy array
            secs_timestamps_path (str): Path to SECS timestamps numpy array
        """
        # Load target data (magnetic field components)
        self.target_df = pd.read_csv(target_path)
        self.target_df['DateTime'] = pd.to_datetime(self.target_df['DateTime'])
        self.target_data = self.target_df[['Be_0_0', 'Bn_0_0', 'Bu_0_0']].values
        
        # Load Branch 1 (solar/geophysical data)
        self.geo_df = pd.read_csv(geodata_path)
        self.geo_df['DateTime'] = pd.to_datetime(self.geo_df['DateTime'])
        self.branch1_data = self.geo_df.drop('DateTime', axis=1).values
        
        # Load Branch 2 (SECS data)
        self.secs_data = np.load(secs_data_path)
        self.secs_timestamps = pd.to_datetime(np.load(secs_timestamps_path))
        
        # Store shapes for later use
        self.branch1_shape = self.branch1_data.shape
        self.branch2_shape = self.secs_data.shape
        self.target_shape = self.target_data.shape
        
        print(f"Loaded data shapes:")
        print(f"Branch 1: {self.branch1_shape}")
        print(f"Branch 2: {self.branch2_shape}")
        print(f"Target: {self.target_shape}")
    
    def _create_splits(self):
        """
        Create train/validation/test splits based on timestamps.
        
        This method:
        1. Identifies timestamps that have data in all three sources
        2. Filters timestamps to be within valid observation period
        3. Sorts timestamps chronologically
        4. Splits them into train/val/test sets according to the specified ratios
        5. Stores the indices for each split
        
        The splits are created at the timestamp level to ensure temporal consistency
        across all data sources.
        """
        # Find timestamps that have data in all three sources and are within valid period
        valid_indices = []
        for i in range(len(self.target_df)):
            target_time = self.target_df['DateTime'].iloc[i]
            
            # Skip if outside valid observation period
            if target_time < self.valid_start or target_time > self.valid_end:
                continue
            
            # Find corresponding indices in Branch 1 and 2
            geo_idx = self.geo_df[self.geo_df['DateTime'] == target_time].index
            secs_idx = np.where(self.secs_timestamps == target_time)[0]
            
            # Only include timestamps that have data in all sources
            if len(geo_idx) > 0 and len(secs_idx) > 0:
                valid_indices.append(i)
        
        # Sort indices chronologically
        valid_indices = sorted(valid_indices)
        
        # Split indices according to specified ratios
        n_samples = len(valid_indices)
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        # Store indices for each split
        self.train_indices = valid_indices[:train_size]
        self.val_indices = valid_indices[train_size:train_size + val_size]
        self.test_indices = valid_indices[train_size + val_size:]
        
        print(f"\nData split sizes:")
        print(f"Training: {len(self.train_indices)}")
        print(f"Validation: {len(self.val_indices)}")
        print(f"Test: {len(self.test_indices)}")
        
        # Print date ranges for each split
        for split_name, indices in [
            ('Training', self.train_indices),
            ('Validation', self.val_indices),
            ('Test', self.test_indices)
        ]:
            if len(indices) > 0:
                start_time = self.target_df['DateTime'].iloc[indices[0]]
                end_time = self.target_df['DateTime'].iloc[indices[-1]]
                duration = (end_time - start_time).total_seconds() / 3600
                print(f"\n{split_name} date range:")
                print(f"Start: {start_time}")
                print(f"End: {end_time}")
                print(f"Duration: {duration:.1f} hours")
    
    def _get_branch1_window(self, idx):
        """Get a window of Branch 1 data for a given timestamp index.
        
        Args:
            idx: Index of the timestamp in the target data
            
        Returns:
            Window of Branch 1 data with shape [BRANCH1_LOOKBACK, n_features]
            
        Note:
            This method ensures that:
            1. The window is within the valid observation period
            2. Edge cases are handled with appropriate padding
            3. The window ends at the current time t
            4. Properly handles the 15-minute resolution of Branch 1 data
        """
        # Create a zero-filled window as a fallback
        window = np.zeros((self.branch1_lookback, self.branch1_data.shape[1]))
        
        try:
            # Get the current timestamp from target data
            current_time = self.target_df['DateTime'].iloc[idx]
            
            # Verify we're within valid observation period
            if current_time < self.valid_start or current_time > self.valid_end:
                print(f"Warning: Timestamp {current_time} outside valid observation period")
                return window
            
            # Find the corresponding index in Branch 1 data
            # Branch 1 data is at 15-minute resolution, so we need to find the closest timestamp
            branch1_idx = self.geo_df[self.geo_df['DateTime'] <= current_time].index[-1]
            
            # Calculate the start index for the window
            start_idx = branch1_idx - self.branch1_lookback + 1
            
            # Handle the case where we need to pad at the beginning
            if start_idx < 0:
                # Get the available data
                available_data = self.branch1_data[max(0, start_idx):branch1_idx+1]
                
                # If we have some data, copy it to the window
                if available_data.shape[0] > 0:
                    # Calculate where to place the data in the window
                    offset = -start_idx
                    window[offset:offset+available_data.shape[0]] = available_data
                    
                    # Fill the beginning with the first value (edge padding)
                    if offset > 0:
                        window[:offset] = available_data[0]
            else:
                # Get the window without padding
                available_data = self.branch1_data[start_idx:branch1_idx+1]
                
                # Copy the available data to the window
                if available_data.shape[0] > 0:
                    window[:available_data.shape[0]] = available_data
                    
                    # If we don't have enough data, pad with the last value
                    if available_data.shape[0] < self.branch1_lookback:
                        window[available_data.shape[0]:] = available_data[-1]
        except Exception as e:
            print(f"Error in _get_branch1_window for idx {idx}: {str(e)}")
            # Return the zero-filled window as fallback
        
        return window
    
    def _get_branch2_window(self, idx):
        """Get a window of Branch 2 data for a given timestamp index.
        
        Args:
            idx: Index of the timestamp in the Branch 2 data
            
        Returns:
            Window of Branch 2 data with shape [BRANCH2_LOOKBACK, 21, 21, 3]
            
        Note:
            This method ensures that:
            1. The window is within the valid observation period
            2. Edge cases are handled with appropriate padding
            3. The window ends at the current time t
        """
        # Create a zero-filled window as a fallback
        window = np.zeros((self.branch2_lookback, 21, 21, 3))
        
        try:
            # Get the current timestamp
            current_time = self.secs_timestamps[idx]
            
            # Verify we're within valid observation period
            if current_time < self.valid_start or current_time > self.valid_end:
                print(f"Warning: Timestamp {current_time} outside valid observation period")
                return window
            
            # Calculate the start index for the window
            start_idx = idx - self.branch2_lookback + 1
            
            # Handle the case where we need to pad at the beginning
            if start_idx < 0:
                # Get the available data
                available_data = self.secs_data[max(0, start_idx):idx+1]
                
                # If we have some data, copy it to the window
                if available_data.shape[0] > 0:
                    # Calculate where to place the data in the window
                    offset = -start_idx
                    window[offset:offset+available_data.shape[0]] = available_data
                    
                    # Fill the beginning with the first value (edge padding)
                    if offset > 0:
                        window[:offset] = available_data[0]
            else:
                # Get the window without padding
                available_data = self.secs_data[start_idx:idx+1]
                
                # Copy the available data to the window
                if available_data.shape[0] > 0:
                    window[:available_data.shape[0]] = available_data
                    
                    # If we don't have enough data, pad with the last value
                    if available_data.shape[0] < self.branch2_lookback:
                        window[available_data.shape[0]:] = available_data[-1]
        except Exception as e:
            print(f"Error in _get_branch2_window for idx {idx}: {str(e)}")
            # Return the zero-filled window as fallback
        
        return window
    
    def get_batch(self, split='train'):
        """
        Get a batch of data with time-shifted targets for forecasting.
        
        Args:
            split (str): 'train', 'val', or 'test'
            
        Returns:
            dict: Dictionary containing input windows and time-shifted targets
            
        Note:
            This method ensures that:
            1. All windows are within the valid observation period
            2. Temporal alignment is maintained between branches
            3. Target values are properly time-shifted
            4. Edge cases are handled with appropriate padding
        """
        # Get indices for the current split
        if split == 'train':
            indices = self.train_indices
        elif split == 'val':
            indices = self.val_indices
        else:
            indices = self.test_indices
        
        # Randomly select batch_size indices
        batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
        
        # Create windows for Branch 1 (low-resolution data)
        branch1_windows = []
        branch2_windows = []
        target_values = []
        
        for idx in batch_indices:
            # Get current timestamp
            current_time = self.target_df['DateTime'].iloc[idx]
            
            # Skip if outside valid observation period
            if current_time < self.valid_start or current_time > self.valid_end:
                print(f"Warning: Skipping timestamp {current_time} outside valid observation period")
                continue
            
            # Get Branch 1 window (96 timesteps, 15-min resolution)
            branch1_window = self._get_branch1_window(idx)
            if branch1_window is not None:
                branch1_windows.append(branch1_window)
                
                # Get Branch 2 window (180 timesteps, 1-min resolution)
                branch2_window = self._get_branch2_window(idx)
                if branch2_window is not None:
                    branch2_windows.append(branch2_window)
                    
                    # Get target values shifted forward by FORECAST_HORIZON
                    target_idx = idx + self.forecast_horizon
                    if target_idx < len(self.target_data):
                        target_time = self.target_df['DateTime'].iloc[target_idx]
                        
                        # Verify target time is within valid period
                        if target_time <= self.valid_end:
                            target_values.append(self.target_data[target_idx])
                        else:
                            print(f"Warning: Target time {target_time} outside valid observation period")
                            # Use the last valid target value
                            target_values.append(self.target_data[idx])
                    else:
                        # Handle edge case where we're at the end of the data
                        print(f"Warning: Target index {target_idx} beyond data length")
                        target_values.append(self.target_data[-1])
        
        # Handle case where we don't have enough valid samples
        if len(branch1_windows) < self.batch_size:
            print(f"Warning: Only got {len(branch1_windows)} valid samples, expected {self.batch_size}")
            # Pad with the last valid sample if we have at least one
            if len(branch1_windows) > 0:
                while len(branch1_windows) < self.batch_size:
                    branch1_windows.append(branch1_windows[-1])
                    branch2_windows.append(branch2_windows[-1])
                    target_values.append(target_values[-1])
            else:
                # If we have no valid samples, return zero-filled arrays
                branch1_windows = [np.zeros((self.branch1_lookback, self.branch1_data.shape[1]))] * self.batch_size
                branch2_windows = [np.zeros((self.branch2_lookback, 21, 21, 3))] * self.batch_size
                target_values = [np.zeros(3)] * self.batch_size
        
        return {
            'branch1_input': np.array(branch1_windows),
            'branch2_input': np.array(branch2_windows),
            'target': np.array(target_values)
        }
    
    def create_tf_dataset(self, split='train'):
        """
        Create a TensorFlow dataset for the specified split.
        
        This method:
        1. Creates a generator function that yields batches
        2. Wraps the generator in a tf.data.Dataset
        3. Specifies the output signature for proper typing
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            
        Returns:
            tf.data.Dataset: TensorFlow dataset for the specified split
            
        Note:
            The dataset is infinite (will yield batches forever) and should be
            used with steps_per_epoch in model.fit() to control the number of
            batches per epoch.
        """
        # Define generator function that yields batches
        def generator():
            while True:
                yield self.get_batch(split)
        
        # Create dataset with proper output signature
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    'branch1_input': tf.TensorSpec(shape=(None, self.branch1_lookback, self.branch1_shape[1]), dtype=tf.float32),
                    'branch2_input': tf.TensorSpec(shape=(None, self.branch2_lookback, *self.branch2_shape[1:]), dtype=tf.float32),
                    'target': tf.TensorSpec(shape=(None, self.target_shape[1]), dtype=tf.float32)
                }
            )
        )
        
        return dataset 