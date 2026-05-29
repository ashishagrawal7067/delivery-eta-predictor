"""
main.py
-------
End-to-end pipeline runner for the Delivery ETA Predictor.

Usage:
    python main.py --data data/delivery_data.csv

Pipeline Architecture:
  1. data_preprocessing  — load & clean raw CSV (coordinate validation, regex parsing)
  2. feature_engineering — assemble feature matrix (15 checkout-time features)
  3. model_training      — train/test split → Pipeline with preprocessor inside CV
                           → evaluate 4 models → tune XGBoost → plot comparisons
                           → serialize unified Pipeline bundle via Joblib
"""

import argparse
import os
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import prepare_raw_feature_matrix
from src.model_training import train_and_evaluate, plot_results, save_model


def main(data_path: str):
    print("\n" + "="*60)
    print("  Delivery ETA Predictor — Production Pipeline")
    print("="*60)

    # --- Step 1: Load & clean ---
    print("\n[1/4] Loading and cleaning data ...")
    df_raw = load_data(data_path)
    df_cleaned = clean_data(df_raw)
    print(f"      Loaded {len(df_raw):,} rows. Valid GPS & cleaned rows: {len(df_cleaned):,} rows.")

    # --- Step 2: Feature selection (15 checkout-time features, no future leakage) ---
    print("\n[2/4] Assembling checkout-time feature matrix ...")
    X, y = prepare_raw_feature_matrix(df_cleaned)
    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Target range: {y.min():.0f}–{y.max():.0f} min")
    print(f"      Raw features: {list(X.columns)}")

    # --- Step 3: Train & evaluate models (Pipeline inside CV) ---
    print("\n[3/4] Training and evaluating models with Pipeline CV ...")
    results, best_pipe, preprocessor, feature_names, X_test, y_test = train_and_evaluate(X, y)

    # --- Step 4: Plot & save ---
    print("\n[4/4] Generating comparison plots & serializing pipeline bundle ...")
    os.makedirs("outputs", exist_ok=True)
    plot_results(results, y_test, save_path="outputs/model_comparison.png")

    os.makedirs("models", exist_ok=True)
    save_model(best_pipe, feature_names, results, path="models/xgb_model.pkl")

    # --- Summary ---
    best = results[-1]
    print("\n" + "="*60)
    print("  Final Results Summary")
    print("="*60)
    for r in results:
        print(f"  {r['name']:<25} | R²: {r['test_r2']:.3f} | RMSE: {r['test_rmse']:.2f} min | MAE: {r['test_mae']:.2f} min")
    print(f"\n  [+] Best model: {best['name']}  (R²={best['test_r2']:.3f}, RMSE={best['test_rmse']:.2f} min, MAE={best['test_mae']:.2f} min)")
    print("\nDone! Complete Pipeline bundle saved using Joblib to models/xgb_model.pkl\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery ETA Predictor — Production Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/delivery_data.csv",
        help="Path to the raw delivery CSV file (default: data/delivery_data.csv)"
    )
    args = parser.parse_args()
    main(args.data)
