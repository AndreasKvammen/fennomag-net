# fennomag-net

A deep learning framework to forecast magnetic disturbances over Fennoscandia.

## Overview

FennoMag-Net is a project aimed at forecasting geomagnetic disturbances over the Fennoscandian region using a combination of physics-based modeling and deep learning techniques. The system processes magnetometer data to reconstruct and predict the ionospheric current systems and resulting magnetic field variations.

## Key Components

### SECS Generator (secs_gen.py)

The SECS Generator processes magnetometer data using Spherical Elementary Current Systems (SECS) to reconstruct the ionospheric current system and calculate the resulting magnetic field components on a regular grid.

Features:
- Processes magnetometer data from multiple stations
- Applies SECS inversion with regularization
- Outputs gridded magnetic field components (Be, Bn, Bu)
- Handles data gaps and station filtering

### SECS Visualization (secs_viz.py)

The visualization tool converts the magnetic field components into RGB images for easier interpretation and analysis.

Features:
- Maps magnetic field components to color channels:
  - Red: Eastward component (Be)
  - Green: Northward component (Bn)
  - Blue: Upward component (Bu)
- Uses statistical normalization based on the normal CDF for improved dynamic range
- Generates time series of images for animation
