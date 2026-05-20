"""
feature_engineering.py
-----------------------
Transforms the cleaned dataframe into a model-ready feature matrix.

Steps performed:
  1. Compute haversine distance (km) between restaurant and delivery location
  2. Extract order_hour from the Time_Orderd column
  3. Drop columns not useful for modelling
  4. Encode categoricals (ordinal for traffic density, one-hot for the rest)
  5. Scale numeric features with StandardScaler
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (km) between two points
    on the Earth using the Haversine formula.
    """
    R = 6371  # Earth's radius in kilometres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame):
    """
    Engineer features from the cleaned dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of data_preprocessing.clean_data).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable (Time_taken in minutes).
    scaler : StandardScaler
        Fitted scaler (needed at inference time).
    """
    df = df.copy()

    # --- 1. Haversine distance ---
    df['distance_km'] = haversine_distance(
        df['Restaurant_latitude'],
        df['Restaurant_longitude'],
        df['Delivery_location_latitude'],
        df['Delivery_location_longitude']
    )

    # --- 2. Extract order hour from Time_Orderd ---
    df['order_hour'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.hour
    df['order_hour'] = df['order_hour'].fillna(df['order_hour'].median())

    # --- 3. Drop columns not used for modelling ---
    drop_cols = [
        'ID', 'Delivery_person_ID', 'Order_Date',
        'Time_Orderd', 'Time_Order_picked',
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude',
        'Time_taken(min)'
    ]
    y = df['Time_taken(min)']
    X = df.drop(columns=drop_cols, errors='ignore')

    # --- 4. Encode categoricals ---
    # Ordinal encoding for traffic density (meaningful order)
    traffic_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
    X['Road_traffic_density'] = X['Road_traffic_density'].map(traffic_map)

    # One-hot encode the remaining categoricals
    X = pd.get_dummies(X, columns=[
        'Weatherconditions', 'Type_of_order',
        'Type_of_vehicle', 'Festival', 'City'
    ], drop_first=True)

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- 5. Scale numeric features ---
    numeric_features = [
        'Delivery_person_Age', 'Delivery_person_Ratings',
        'Vehicle_condition', 'multiple_deliveries',
        'distance_km', 'order_hour'
    ]
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    return X, y, scaler
