"""
main.py
-------
End-to-end pipeline runner for the Delivery ETA Predictor.

Usage:
    python main.py --data data/delivery_data.csv

This script ties together the three pipeline modules:
  1. data_preprocessing  — load & clean the raw CSV
  2. feature_engineering — build features and encode/scale
  3. model_training      — train, compare, tune, and save the best model
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
from src.feature_engineering import build_features
from src.model_training import train_and_evaluate, plot_results, save_model


def main(data_path: str):
    print("\n" + "="*60)
    print("  Delivery ETA Predictor — Training Pipeline")
    print("="*60)

    # --- Step 1: Load & clean ---
    print("\n[1/4] Loading and cleaning data ...")
    df_raw     = load_data(data_path)
    df_cleaned = clean_data(df_raw)
    print(f"      Loaded {len(df_raw):,} rows. After cleaning: {len(df_cleaned):,} rows.")

    # --- Step 2: Feature engineering ---
    print("\n[2/4] Engineering features ...")
    X, y, scaler = build_features(df_cleaned)
    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Target range: {y.min():.0f}–{y.max():.0f} min")

    # --- Step 3: Train & evaluate all models ---
    print("\n[3/4] Training and evaluating models ...")
    results, best_model, X_test, y_test = train_and_evaluate(X, y)

    # --- Step 4: Plot & save ---
    print("\n[4/4] Generating comparison plots ...")
    os.makedirs("outputs", exist_ok=True)
    plot_results(results, y_test, save_path="outputs/model_comparison.png")

    save_model(best_model, path="models/xgb_model.pkl")

    # --- Summary ---
    best = results[-1]
    print("\n" + "="*60)
    print("  Final Results Summary")
    print("="*60)
    for r in results:
        print(f"  {r['name']:<25} | R2: {r['test_r2']:.3f} | RMSE: {r['test_rmse']:.2f} min")
    print(f"\n  [+] Best model: {best['name']}  (R2={best['test_r2']:.3f}, RMSE={best['test_rmse']:.2f} min)")
    print("\nDone! Model saved to models/xgb_model.pkl\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery ETA Predictor — Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/delivery_data.csv",
        help="Path to the raw delivery CSV file (default: data/delivery_data.csv)"
    )
    args = parser.parse_args()
    main(args.data)
