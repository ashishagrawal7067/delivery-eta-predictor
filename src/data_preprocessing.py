"""
data_preprocessing.py
---------------------
Handles loading and cleaning of the raw food delivery dataset.

Steps performed:
  1. Load the CSV file
  2. Fix free-text formatting quirks (e.g. "conditions Sunny" -> "Sunny")
  3. Strip whitespace and replace literal 'NaN' strings with real nulls
  4. Convert numeric-looking object columns to proper numeric types
  5. Impute missing values (median for numerics, mode for categoricals)
  6. Clean the target column  (e.g. "(min) 24" -> 24)
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = df.copy()

    # --- Fix Weatherconditions: "conditions Sunny" -> "Sunny" ---
    df['Weatherconditions'] = df['Weatherconditions'].apply(
        lambda x: x.split(" ")[1] if pd.notnull(x) else x
    )

    # --- Strip whitespace and replace literal 'NaN' strings with real nulls ---
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace('NaN', np.nan)

    # --- Convert numeric-looking object columns to numeric types ---
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

    # --- Clean target column: "(min) 24" -> 24 ---
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(
        lambda x: int(str(x).split(" ")[1]) if pd.notnull(x) else x
    )
    df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'], errors='coerce')

    # --- Impute missing values ---
    # Numeric columns: fill with median
    numeric_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill with mode
    categorical_cols = ['Weatherconditions', 'Road_traffic_density', 'Festival', 'City']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
