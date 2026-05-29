"""
model_training.py
-----------------
Trains, evaluates, and serializes models for food delivery ETA prediction.

Methodological Highlights:
  - Preprocessor inside Cross-Validation: DeliveryPreprocessor is encapsulated
    directly inside an sklearn Pipeline before passing to RandomizedSearchCV.
    Imputation, encoding, and scaling are re-fitted strictly within each CV fold.
  - Evaluation on held-out test split (20%):
      * Linear Regression (Baseline)
      * Decision Tree Regressor
      * Random Forest Regressor
      * XGBoost Regressor (Tuned via 3-fold cross-validation)
  - Metrics: RMSE, MAE (mean minutes error), R² Score.
  - Serialization: Complete unified Pipeline (preprocessor + model + metadata)
    saved using joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

from src.feature_engineering import DeliveryPreprocessor, NUMERIC_FEATURES


def evaluate_pipeline(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    """Train pipeline, calculate train/test metrics, and return evaluation dictionary."""
    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Train R²  : {train_r2:.4f}")
    print(f"  Test  R²  : {test_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f} min")
    print(f"  Test  MAE : {test_mae:.4f} min  ← avg error per delivery")

    return {
        'name': name,
        'pipeline': pipeline,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'test_preds': test_preds,
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """
    Split data into 80/20 train/test, train all models using pipelines,
    tune XGBoost with preprocessing inside CV, and return artifacts.
    """
    # 1. Train / Test Split FIRST (Held-out outer test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # --- 1. Linear Regression Pipeline ---
    lr_pipe = Pipeline([
        ('preprocessor', DeliveryPreprocessor()),
        ('regressor', LinearRegression())
    ])
    results.append(evaluate_pipeline("Linear Regression", lr_pipe, X_train, X_test, y_train, y_test))

    # --- 2. Decision Tree Pipeline ---
    dt_pipe = Pipeline([
        ('preprocessor', DeliveryPreprocessor()),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    results.append(evaluate_pipeline("Decision Tree", dt_pipe, X_train, X_test, y_train, y_test))

    # --- 3. Random Forest Pipeline ---
    rf_pipe = Pipeline([
        ('preprocessor', DeliveryPreprocessor()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    results.append(evaluate_pipeline("Random Forest", rf_pipe, X_train, X_test, y_train, y_test))

    # --- 4. Tuned XGBoost Pipeline with Preprocessing INSIDE Cross-Validation ---
    print("\n[*] Tuning XGBoost with RandomizedSearchCV (preprocessor inside CV) ...")
    xgb_base_pipe = Pipeline([
        ('preprocessor', DeliveryPreprocessor()),
        ('regressor', XGBRegressor(random_state=42, verbosity=0, n_jobs=1, nthread=1))
    ])

    xgb_param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__subsample': [0.7, 0.8, 1.0],
        'regressor__colsample_bytree': [0.7, 0.8, 1.0],
        'regressor__reg_alpha': [0, 0.1, 0.5],
        'regressor__reg_lambda': [1, 1.5, 2],
    }

    xgb_search = RandomizedSearchCV(
        xgb_base_pipe,
        param_distributions=xgb_param_grid,
        n_iter=15,
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb_pipe = xgb_search.best_estimator_
    print(f"  Best params: {xgb_search.best_params_}")

    # Evaluate best XGBoost pipeline
    train_preds = best_xgb_pipe.predict(X_train)
    test_preds = best_xgb_pipe.predict(X_test)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    print(f"\n{'='*55}")
    print(f"  XGBoost (Tuned)")
    print(f"{'='*55}")
    print(f"  Train R²  : {train_r2:.4f}")
    print(f"  Test  R²  : {test_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f} min")
    print(f"  Test  MAE : {test_mae:.4f} min  ← avg error per delivery")

    results.append({
        'name': "XGBoost (Tuned)",
        'pipeline': best_xgb_pipe,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_preds))),
        'test_rmse': float(test_rmse),
        'train_mae': float(mean_absolute_error(y_train, train_preds)),
        'test_mae': float(test_mae),
        'test_preds': test_preds,
    })

    fitted_preprocessor = best_xgb_pipe.named_steps['preprocessor']
    feature_names = fitted_preprocessor.feature_names_
    return results, best_xgb_pipe, fitted_preprocessor, feature_names, X_test, y_test


def plot_results(results: list, y_test, save_path: str = 'outputs/model_comparison.png'):
    """Generate 3-panel comparison chart and scatter plot."""
    names = [r['name'] for r in results]
    r2s = [r['test_r2'] for r in results]
    rmses = [r['test_rmse'] for r in results]
    maes = [r['test_mae'] for r in results]

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R²
    axes[0].barh(names, r2s, color=colors)
    axes[0].set_xlabel('Test R²')
    axes[0].set_title('Model Comparison — R² Score')
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

    # RMSE
    axes[1].barh(names, rmses, color=colors)
    axes[1].set_xlabel('Test RMSE (minutes)')
    axes[1].set_title('Model Comparison — RMSE')
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')

    # MAE
    axes[2].barh(names, maes, color=colors)
    axes[2].set_xlabel('Test MAE (minutes)')
    axes[2].set_title('Model Comparison — MAE')
    for i, v in enumerate(maes):
        axes[2].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[*] Comparison plot saved to: {save_path}")
    plt.close(fig)

    # Actual vs Predicted
    best = results[-1]
    fig_pred = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best['test_preds'], alpha=0.3, s=10, color='steelblue')
    lims = [min(y_test.min(), best['test_preds'].min()),
            max(y_test.max(), best['test_preds'].max())]
    plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    plt.xlabel('Actual Delivery Time (min)')
    plt.ylabel('Predicted Delivery Time (min)')
    plt.title(f'Actual vs Predicted — {best["name"]}')
    plt.legend()
    plt.tight_layout()

    scatter_dir = os.path.dirname(save_path) if save_path else 'outputs'
    scatter_path = os.path.join(scatter_dir, 'actual_vs_predicted.png')
    os.makedirs(scatter_dir, exist_ok=True)
    plt.savefig(scatter_path, dpi=150)
    print(f"[*] Actual vs Predicted plot saved to: {scatter_path}")
    plt.close(fig_pred)


def save_model(pipeline: Pipeline, feature_names: list, results: list, path: str = 'models/xgb_model.pkl'):
    """Serialize the complete trained Pipeline bundle using joblib."""
    bundle = {
        'pipeline': pipeline,
        'model': pipeline.named_steps['regressor'],
        'preprocessor': pipeline.named_steps['preprocessor'],
        'scaler': pipeline.named_steps['preprocessor'].scaler,
        'feature_names': feature_names,
        'numeric_features': NUMERIC_FEATURES,
        'metrics': {r['name']: {'r2': r['test_r2'], 'rmse': r['test_rmse'], 'mae': r['test_mae']} for r in results},
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    joblib.dump(bundle, path)
    print(f"[*] Complete pipeline bundle saved using joblib to: {path}")


def load_model(path: str = 'models/xgb_model.pkl') -> dict:
    """Load model bundle from disk using joblib."""
    return joblib.load(path)
