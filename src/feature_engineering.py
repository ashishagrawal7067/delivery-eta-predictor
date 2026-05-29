"""
feature_engineering.py
-----------------------
Provides leakage-free, batch-invariant feature preprocessing and scaling.

Methodological Highlights:
  - Scikit-Learn OneHotEncoder: Stores training category vocabularies and encodes
    consistently with drop='first' regardless of whether transforming a full batch
    or a single prediction row (solves batch-size dependency).
  - Scope: Pure Checkout-Time (Order-Placement) ETA. Excludes actual pickup timestamps
    and kitchen durations to prevent future-lookahead target leakage.
  - Selective Scaling: StandardScaler is applied strictly to continuous numerical
    features (NUMERIC_FEATURES). Binary one-hot dummy columns remain unscaled to
    preserve sparsity and direct coefficient interpretability.
  - Standardized ordinal mapping for road traffic density: Low=1, Medium=2, High=3, Jam=4.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = [
    'Delivery_person_Age',
    'Delivery_person_Ratings',
    'distance_km',
    'order_hour',
    'day_of_week',
    'is_weekend',
    'Vehicle_condition',
    'multiple_deliveries',
]

CATEGORICAL_FEATURES = [
    'Weatherconditions',
    'Road_traffic_density',
    'Festival',
    'City',
]

ONE_HOT_FEATURES = [
    'Weatherconditions',
    'Type_of_order',
    'Type_of_vehicle',
    'Festival',
    'City',
    'city_code',
]

TRAFFIC_MAP = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Jam': 4,
}


class DeliveryPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free, batch-invariant preprocessor conforming to the Scikit-Learn API.
    Can be seamlessly placed inside an sklearn Pipeline.
    """

    def __init__(self):
        self.num_medians_ = {}
        self.cat_modes_ = {}
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """Learn medians, modes, OneHotEncoder categories, and scaler parameters from training data."""
        X_df = X.copy()

        # 1. Learn numeric medians
        for col in NUMERIC_FEATURES:
            if col in X_df.columns:
                self.num_medians_[col] = float(X_df[col].median()) if not X_df[col].dropna().empty else 0.0

        # 2. Learn categorical modes
        for col in CATEGORICAL_FEATURES:
            if col in X_df.columns:
                modes = X_df[col].dropna().mode()
                self.cat_modes_[col] = modes.iloc[0] if not modes.empty else 'Missing'

        # Impute missing values internally to fit OHE and Scaler
        for col, med in self.num_medians_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(med)

        for col, mode_val in self.cat_modes_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(mode_val)

        if 'Road_traffic_density' in X_df.columns:
            X_df['Road_traffic_density'] = X_df['Road_traffic_density'].map(TRAFFIC_MAP).fillna(2)

        # 3. Fit OneHotEncoder
        self.ohe.fit(X_df[ONE_HOT_FEATURES].astype(str))
        ohe_cols = list(self.ohe.get_feature_names_out(ONE_HOT_FEATURES))

        # 4. Fit StandardScaler on continuous numeric features only
        self.scaler.fit(X_df[NUMERIC_FEATURES])

        # Feature order
        self.feature_names_ = list(NUMERIC_FEATURES) + ['Road_traffic_density'] + ohe_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X using learned statistics with strict batch invariance."""
        X_df = X.copy()

        # Impute numeric
        for col, med in self.num_medians_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(med)

        # Impute categorical
        for col, mode_val in self.cat_modes_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(mode_val)

        # Ordinal traffic mapping
        if 'Road_traffic_density' in X_df.columns:
            X_df['Road_traffic_density'] = X_df['Road_traffic_density'].map(TRAFFIC_MAP).fillna(2)

        # Scale numeric features
        scaled_nums = self.scaler.transform(X_df[NUMERIC_FEATURES])
        num_df = pd.DataFrame(scaled_nums, columns=NUMERIC_FEATURES, index=X_df.index)

        # Traffic feature
        traffic_df = pd.DataFrame(X_df[['Road_traffic_density']], index=X_df.index)

        # OneHotEncode features
        ohe_arr = self.ohe.transform(X_df[ONE_HOT_FEATURES].astype(str))
        ohe_cols = list(self.ohe.get_feature_names_out(ONE_HOT_FEATURES))
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X_df.index)

        # Concatenate into final feature matrix
        out = pd.concat([num_df, traffic_df, ohe_df], axis=1)
        return out[self.feature_names_]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit on X and transform X."""
        return self.fit(X, y).transform(X)


def prepare_raw_feature_matrix(df: pd.DataFrame):
    """
    Select checkout-time modeling features and target variable from cleaned dataframe.

    Features available at checkout:
      - Delivery_person_Age, Delivery_person_Ratings, distance_km, order_hour
      - day_of_week, is_weekend, Vehicle_condition, multiple_deliveries
      - Road_traffic_density, Weatherconditions, Type_of_order, Type_of_vehicle
      - Festival, City, city_code

    Returns
    -------
    X : pd.DataFrame
        Checkout-time feature columns (15 features).
    y : pd.Series
        Target delivery time (minutes).
    """
    feature_cols = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'distance_km',
        'order_hour',
        'day_of_week',
        'is_weekend',
        'Vehicle_condition',
        'multiple_deliveries',
        'Road_traffic_density',
        'Weatherconditions',
        'Type_of_order',
        'Type_of_vehicle',
        'Festival',
        'City',
        'city_code',
    ]
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    y = df['Time_taken(min)'].copy() if 'Time_taken(min)' in df.columns else None
    return X, y
