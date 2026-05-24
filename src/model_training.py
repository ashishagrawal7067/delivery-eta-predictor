"""
model_training.py
-----------------
Trains, evaluates, and saves the best model for food delivery time prediction.

Models compared:
  - Linear Regression  (baseline)
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor  (tuned with RandomizedSearchCV — best performer)

Metrics used: RMSE and R² Score.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train, predict, and print RMSE + R² for a given model."""
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_r2   = r2_score(y_train, train_preds)
    test_r2    = r2_score(y_test,  test_preds)
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Train R2  : {train_r2:.4f}")
    print(f"  Test  R2  : {test_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f} min")

    return {
        'name'      : name,
        'model'     : model,
        'train_r2'  : train_r2,
        'test_r2'   : test_r2,
        'test_rmse' : test_rmse,
        'test_preds': test_preds
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate(X, y):
    """
    Split data, train all models, tune the best one, and return results.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (output of feature_engineering.build_features).
    y : pd.Series
        Target variable.

    Returns
    -------
    results : list[dict]
        Evaluation metrics for all models.
    best_model : fitted estimator
        The tuned XGBoost model.
    X_test, y_test : split test data (used for plotting)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # --- Baseline: Linear Regression ---
    results.append(evaluate_model(
        "Linear Regression",
        LinearRegression(),
        X_train, X_test, y_train, y_test
    ))

    # --- Decision Tree ---
    results.append(evaluate_model(
        "Decision Tree",
        DecisionTreeRegressor(random_state=42),
        X_train, X_test, y_train, y_test
    ))

    # --- Random Forest ---
    results.append(evaluate_model(
        "Random Forest",
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        X_train, X_test, y_train, y_test
    ))

    # --- XGBoost (with RandomizedSearchCV tuning) ---
    print("\n[*] Tuning XGBoost with RandomizedSearchCV ...")
    xgb_param_grid = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [3, 5, 7, 9],
        'learning_rate'   : [0.01, 0.05, 0.1, 0.2],
        'subsample'       : [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha'       : [0, 0.1, 0.5],
        'reg_lambda'      : [1, 1.5, 2],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_distributions=xgb_param_grid,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  Best params: {xgb_search.best_params_}")

    results.append(evaluate_model(
        "XGBoost (Tuned)",
        best_xgb,
        X_train, X_test, y_train, y_test
    ))

    best_model = best_xgb

    return results, best_model, X_test, y_test


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list, y_test, save_path: str = None):
    """
    Plot model comparison bar chart and actual-vs-predicted scatter plot
    for the best model.
    """
    # --- Bar chart: Test R² comparison ---
    names  = [r['name'] for r in results]
    r2s    = [r['test_r2'] for r in results]
    rmses  = [r['test_rmse'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(names, r2s, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    axes[0].set_xlabel('Test R²')
    axes[0].set_title('Model Comparison — R² Score')
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')

    axes[1].barh(names, rmses, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    axes[1].set_xlabel('Test RMSE (minutes)')
    axes[1].set_title('Model Comparison — RMSE')
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.05, i, f'{v:.2f}', va='center')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[*] Comparison plot saved to: {save_path}")
    plt.close(fig)

    # --- Actual vs Predicted for best model (XGBoost) ---
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
    scatter_path = os.path.join(os.path.dirname(save_path) if save_path else 'outputs', 'actual_vs_predicted.png')
    os.makedirs(os.path.dirname(scatter_path) or '.', exist_ok=True)
    plt.savefig(scatter_path, dpi=150)
    print(f"[*] Actual vs Predicted plot saved to: {scatter_path}")
    plt.close(fig_pred)


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

def save_model(model, path: str = 'models/xgb_model.pkl'):
    """Serialize the trained model to disk using pickle."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[*] Model saved to: {path}")


def load_model(path: str = 'models/xgb_model.pkl'):
    """Load a serialized model from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
