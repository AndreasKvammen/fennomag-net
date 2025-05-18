#!/usr/bin/env python3
"""
Model2024 Testing Script
=======================

This script provides functionality to test the trained Model2024 neural network,
which forecasts Earth's surface magnetic field variations. The script loads the
trained model and evaluates its performance on test data.

Key Features:
------------
1. Loads trained model with custom objects (CrossAttention, custom_loss)
2. Processes test data using the BatchGenerator
3. Generates predictions and compares with actual values
4. Creates visualizations of predictions vs actual values
5. Saves results and metrics for analysis

Usage:
------
python model2024_test.py

The script assumes the following directory structure:
    /path/to/project/
    ├── source/
    │   └── model2024/
    │       ├── model2024_test.py
    │       ├── batch_generator.py
    │       └── best_model.h5
    └── model_outputs/
        ├── model_config.json
        └── data_statistics.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import batch_generator
sys.path.append(str(Path(__file__).parent.parent))
from model2024.batch_generator import BatchGenerator

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'model_outputs', 'model_config.json')
DATA_STATS_PATH = os.path.join(BASE_DIR, 'model_outputs', 'data_statistics.json')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
DATA_DIR = os.path.join(BASE_DIR, 'data')
TARGET_PATH = os.path.join(DATA_DIR, 'target.csv')
GEODATA_PATH = os.path.join(DATA_DIR, 'geodata.csv')
SECS_DATA_PATH = os.path.join(DATA_DIR, 'secs_data.npy')
SECS_TIMESTAMPS_PATH = os.path.join(DATA_DIR, 'secs_timestamps.npy')

# Define output directory for test results
OUTPUT_DIR = os.path.join(BASE_DIR, 'model_outputs', 'test_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define colors for visualization
COLORS = {
    'Be': '#1f77b4',  # Blue
    'Bn': '#2ca02c',  # Green
    'Bu': '#ff7f0e',  # Orange
    'true': '#1f77b4',  # Blue
    'pred': '#ff7f0e'   # Orange
}

class CrossAttention(tf.keras.layers.Layer):
    """Custom layer implementing cross-attention between two feature vectors."""
    
    def __init__(self, num_heads=4, key_dim=16, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def build(self, input_shape):
        self.query_dense = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.key_dense = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.value_dense = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.output_dense = tf.keras.layers.Dense(input_shape[0][-1])
        super(CrossAttention, self).build(input_shape)
        
    def call(self, inputs):
        query, key_value = inputs
        query = self.query_dense(query)
        key = self.key_dense(key_value)
        value = self.value_dense(key_value)
        
        batch_size = tf.shape(query)[0]
        query = tf.reshape(query, [batch_size, 1, self.num_heads, self.key_dim])
        key = tf.reshape(key, [batch_size, 1, self.num_heads, self.key_dim])
        value = tf.reshape(value, [batch_size, 1, self.num_heads, self.key_dim])
        
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.reshape(attention_output, [batch_size, self.num_heads * self.key_dim])
        output = self.output_dense(attention_output)
        
        return output
    
    def get_config(self):
        config = super(CrossAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

def custom_loss(y_true, y_pred):
    """Custom loss function combining MSE for each magnetic field component."""
    be_pred, bn_pred, bu_pred = tf.unstack(y_pred, axis=1)
    be_true, bn_true, bu_true = tf.unstack(y_true, axis=1)
    
    be_loss = tf.keras.losses.mean_squared_error(be_true, be_pred)
    bn_loss = tf.keras.losses.mean_squared_error(bn_true, bn_pred)
    bu_loss = tf.keras.losses.mean_squared_error(bu_true, bu_pred)
    
    return be_loss + bn_loss + bu_loss

def load_model_config():
    """Load model configuration from JSON file."""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config

def load_data_statistics():
    """Load data statistics from JSON file."""
    with open(DATA_STATS_PATH, 'r') as f:
        stats = json.load(f)
    return stats

def setup_plotting():
    """Configure matplotlib plotting style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16

def load_model():
    """Load the trained model with custom objects."""
    print("Loading trained model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'CrossAttention': CrossAttention,
            'custom_loss': custom_loss
        }
    )
    print("Model loaded successfully.")
    return model

def prepare_test_data():
    """Prepare test data using a small subset for quick testing."""
    print("Preparing test data...")
    
    # Load model configuration
    config = load_model_config()
    
    # Define a smaller test period (e.g., 3 days)
    test_start = pd.Timestamp('2024-01-02 00:00:00')
    test_end = pd.Timestamp('2024-01-05 00:00:00')  # 3 days of data
    
    # Initialize batch generator with smaller test period
    batch_generator = BatchGenerator(
        target_path=TARGET_PATH,
        geodata_path=GEODATA_PATH,
        secs_data_path=SECS_DATA_PATH,
        secs_timestamps_path=SECS_TIMESTAMPS_PATH,
        batch_size=config['temporal_params']['batch_size'],
        train_ratio=0.0,  # No training data
        val_ratio=0.0,    # No validation data
        forecast_horizon=config['temporal_params']['forecast_horizon'],
        valid_start=test_start,
        valid_end=test_end
    )
    
    # Create test dataset
    test_dataset = batch_generator.create_tf_dataset(split='test')
    
    print(f"\nTest data period: {test_start} to {test_end}")
    print(f"Test samples: {len(batch_generator.test_indices)}")
    
    return test_dataset, batch_generator

def generate_predictions(model, test_dataset, batch_generator):
    """Generate predictions for test data."""
    print("Generating predictions...")
    
    # Prepare dataset for model input
    def prepare_dataset(dataset):
        return dataset.map(lambda x: (
            {'branch1_input': x['branch1_input'], 'branch2_input': x['branch2_input']},
            x['target']
        ))
    
    test_dataset_prepared = prepare_dataset(test_dataset)
    
    # Initialize lists to store results
    all_predictions = []
    all_targets = []
    all_timestamps = []
    
    # Process each batch
    for batch in test_dataset_prepared.take(10):  # Limit to 10 batches for testing
        inputs, targets = batch
        predictions = model.predict(inputs, verbose=0)
        
        # Get timestamps for this batch
        batch_indices = batch_generator.test_indices[:len(targets)]
        batch_timestamps = pd.to_datetime(batch_generator.target_df['DateTime'].iloc[batch_indices])
        
        all_predictions.extend(predictions)
        all_targets.extend(targets.numpy())
        all_timestamps.extend(batch_timestamps)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'DateTime': all_timestamps,
        'Be_true': [t[0] for t in all_targets],
        'Bn_true': [t[1] for t in all_targets],
        'Bu_true': [t[2] for t in all_targets],
        'Be_pred': [p[0] for p in all_predictions],
        'Bn_pred': [p[1] for p in all_predictions],
        'Bu_pred': [p[2] for p in all_predictions]
    })
    
    print(f"Generated predictions for {len(results_df)} samples")
    return results_df

def visualize_predictions(results_df):
    """Visualize model predictions for each day separately."""
    # Print overall data shape information
    print("\nData Shape Information:")
    print(f"Total number of samples: {len(results_df)}")
    print(f"Date range: {results_df['DateTime'].min()} to {results_df['DateTime'].max()}")
    print(f"Sample frequency: {(results_df['DateTime'].iloc[1] - results_df['DateTime'].iloc[0]).total_seconds()/60:.0f} minutes")
    
    # Get unique days in the results
    unique_days = results_df['DateTime'].dt.date.unique()
    print(f"\nFound {len(unique_days)} unique days in the results")
    
    # Create a separate figure for each day
    for day in unique_days:
        # Get data for this day
        day_data = results_df[results_df['DateTime'].dt.date == day]
        day_times = day_data['DateTime']
        
        # Print shape information for this day
        print(f"\nDay {day}:")
        print(f"Number of samples: {len(day_data)}")
        print(f"Time range: {day_times.min().strftime('%H:%M')} to {day_times.max().strftime('%H:%M')}")
        print(f"Expected samples per day: {24 * 60 / 15}")  # 15-minute intervals
        
        # Create figure for this day
        plt.figure(figsize=(20, 15))
        
        # Components and their labels
        components = ['Be', 'Bn', 'Bu']
        component_labels = ['East Component (Be)', 'North Component (Bn)', 'Up Component (Bu)']
        
        # Plot each component
        for i, (comp, label) in enumerate(zip(components, component_labels)):
            plt.subplot(3, 1, i+1)
            
            # Get the data for this component
            true_values = day_data[f'{comp}_true']
            pred_values = day_data[f'{comp}_pred']
            
            # Print shape information for this component
            print(f"\n{comp} Component:")
            print(f"True values shape: {true_values.shape}")
            print(f"Predicted values shape: {pred_values.shape}")
            print(f"Value range (true): {true_values.min():.2f} to {true_values.max():.2f} nT")
            print(f"Value range (pred): {pred_values.min():.2f} to {pred_values.max():.2f} nT")
            
            # Calculate metrics for this component
            rmse = np.sqrt(np.mean((true_values - pred_values)**2))
            mae = np.mean(np.abs(true_values - pred_values))
            corr = np.corrcoef(true_values, pred_values)[0, 1]
            
            # Plot true and predicted values with proper time handling
            plt.plot(day_times, true_values, 'b-', label='True', linewidth=1.5, alpha=0.8)
            plt.plot(day_times, pred_values, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
            
            # Add metrics box
            metrics_text = f'RMSE: {rmse:.2f} nT\nMAE: {mae:.2f} nT\nCorr: {corr:.2f}'
            plt.text(0.02, 0.98, metrics_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Customize plot
            plt.title(f'{label} - {day}')
            plt.xlabel('Time')
            plt.ylabel('Magnetic Field (nT)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis to show hours with proper spacing
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show ticks every 2 hours
            plt.xticks(rotation=45)
            
            # Adjust y-axis limits to show more detail
            y_min = min(true_values.min(), pred_values.min())
            y_max = max(true_values.max(), pred_values.max())
            y_range = y_max - y_min
            plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
            
            # Add minor grid lines for better readability
            plt.minorticks_on()
            plt.grid(which='minor', alpha=0.2)
        
        # Add overall title
        plt.suptitle(f'Magnetic Field Predictions - {day}', fontsize=16, y=1.02)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(pad=3.0)
        
        # Save the plot
        output_file = os.path.join(OUTPUT_DIR, f'predictions_{day}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot for {day} to {output_file}")
        
        # Show the plot
        plt.show()
        plt.close()

def main():
    """Main function to run the testing process."""
    print("Starting Model2024 testing process...")
    
    # Setup
    setup_plotting()
    
    # Load model
    model = load_model()
    
    # Prepare test data
    test_dataset, batch_generator = prepare_test_data()
    
    # Generate predictions
    results_df = generate_predictions(model, test_dataset, batch_generator)
    
    # Create visualizations
    visualize_predictions(results_df)
    
    print("\nTesting process completed successfully.")

if __name__ == "__main__":
    main() 