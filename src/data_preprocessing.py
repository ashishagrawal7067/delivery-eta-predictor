"""
data_preprocessing.py
---------------------
Handles loading and stateless data cleaning of food delivery records.

Methodological Highlights:
  - Clean stateless row-level transformations:
      * Works for both labelled training data and unlabelled single-order inference.
      * Target column 'Time_taken(min)' is cleaned if present, but is NOT required
        for inference on new orders.
  - Geographic coordinate validation:
      * Rectifies negative coordinate values using .abs() (presumed sign errors).
      * Explicitly removes invalid (0, 0) coordinates (unrecorded sensor values).
      * Validates coordinates to ensure intra-city deliveries (< 100 km).
  - Feature extraction:
      * Haversine great-circle distance (km).
      * Temporal features: order_hour (0–23), day_of_week (0=Mon…6=Sun), is_weekend.
      * City prefix extracted from Delivery_person_ID (e.g. 'BANG', 'MUM').
"""

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    return pd.read_csv(filepath)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance (km) between two coordinate series."""
    R = 6371.0  # Earth's radius in kilometres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform stateless cleaning, coordinate validation, and raw feature extraction.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe. Can be labelled (training) or unlabelled (inference).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature preprocessing.
    """
    df = df.copy()

    # --- 1. Weatherconditions: regex-safe extraction ---
    if 'Weatherconditions' in df.columns:
        df['Weatherconditions'] = (
            df['Weatherconditions']
            .astype(str)
            .str.replace(r'^[A-Za-z]+\s+', '', regex=True)
            .str.strip()
            .replace({'NaN': np.nan, 'nan': np.nan, '': np.nan})
        )

    # --- 2. Target cleanup: optional (present during training, absent during inference) ---
    if 'Time_taken(min)' in df.columns:
        df['Time_taken(min)'] = (
            df['Time_taken(min)']
            .astype(str)
            .str.extract(r'(\d+)', expand=False)
            .astype(float)
        )

    # --- 3. Strip whitespace and normalize string columns ---
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip().replace({'NaN': np.nan, 'nan': np.nan, '': np.nan})

    # --- 4. Numeric type coercion & domain sanity bounds ---
    if 'Delivery_person_Age' in df.columns:
        df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')

    if 'Delivery_person_Ratings' in df.columns:
        # Standard 5-star rating scale: clip anomalous values > 5.0 to 5.0
        df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce').clip(1.0, 5.0)

    if 'multiple_deliveries' in df.columns:
        df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

    if 'Vehicle_condition' in df.columns:
        df['Vehicle_condition'] = pd.to_numeric(df['Vehicle_condition'], errors='coerce')

    # --- 5. GPS Coordinate Validation & Haversine Distance ---
    coord_cols = [
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude'
    ]
    if all(c in df.columns for c in coord_cols):
        # Presumed sign errors: negative coordinates rectified via .abs()
        lat1 = df['Restaurant_latitude'].abs()
        lon1 = df['Restaurant_longitude'].abs()
        lat2 = df['Delivery_location_latitude'].abs()
        lon2 = df['Delivery_location_longitude'].abs()

        # Filter invalid (0, 0) coordinates (unrecorded sensor failures)
        valid_gps = (lat1 > 0) & (lon1 > 0) & (lat2 > 0) & (lon2 > 0)
        df = df[valid_gps].copy()
        lat1, lon1 = lat1[valid_gps], lon1[valid_gps]
        lat2, lon2 = lat2[valid_gps], lon2[valid_gps]

        df['distance_km'] = haversine_distance(lat1, lon1, lat2, lon2)
        # Filter physical outliers (> 100 km intra-city food delivery)
        df = df[df['distance_km'] < 100].copy()

    # --- 6. Timestamp & Calendar Features ---
    if 'Time_Orderd' in df.columns:
        df['order_hour'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.hour

    if 'Order_Date' in df.columns:
        order_dt = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
        df['day_of_week'] = order_dt.dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d in [4, 5, 6] else 0)

    # --- 7. City Code Extraction ---
    if 'Delivery_person_ID' in df.columns:
        df['city_code'] = df['Delivery_person_ID'].astype(str).str.split('RES').str[0].str.strip()

    # If target is present, drop rows missing target value
    if 'Time_taken(min)' in df.columns:
        df = df.dropna(subset=['Time_taken(min)']).reset_index(drop=True)

    return df
