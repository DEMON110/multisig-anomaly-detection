"""
Step 04: LightGBM Model Training
================================
Train LightGBM with exact hyperparameters from Table 4 of the paper.

Key hyperparameters:
  - n_estimators: 1000 (early stopping at 50, best: 554)
  - learning_rate: 0.05
  - max_depth: 8
  - scale_pos_weight: 8.4 (imbalance ratio)
  - eval_metric: aucpr
  - random_state: 43

Produces probabilistic anomaly scores with native SHAP support.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Exact LightGBM hyperparameters from Table 4
LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "scale_pos_weight": 8.4,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 43,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbose": -1,
}


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """
    Train LightGBM with AUCPR optimization and early stopping.
    
    Expected performance (from paper):
      - ROC-AUC: 0.9971 (95% CI: 0.9961–0.9980)
      - F1-Score: 0.9739 (95% CI: 0.9691–0.9784)
      - Best iteration: ~554
    """
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    print(f"\nTraining complete.")
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Best score: {model.best_score_['valid_0']['auc']:.4f}")
    
    return model


def save_model(model, output_path="models/lightgbm_model.pkl"):
    """Save trained model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")


def main():
    """Run LightGBM training."""
    train = pd.read_csv("data/processed/train_features.csv")
    val = pd.read_csv("data/processed/validation_features.csv")
    
    features = CONFIG["features"]["raw_onchain_metrics"] + \
               CONFIG["features"]["temporal_indicators"] + \
               CONFIG["features"]["normalized_deviation"]
    
    X_train = train[features]
    y_train = train["is_anomaly"]
    X_val = val[features]
    y_val = val["is_anomaly"]
    
    print(f"Training LightGBM on {len(X_train)} samples...")
    print(f"Features: {len(features)}")
    print(f"Class imbalance (neg:pos): {(y_train==0).sum()}:{(y_train==1).sum()} "
          f"= {((y_train==0).sum()/(y_train==1).sum()):.1f}:1")
    
    model = train_lightgbm(X_train, y_train, X_val, y_val, features)
    save_model(model)
    
    # Print expected performance for verification
    print("\nExpected test performance (from paper):")
    print("  ROC-AUC: 0.9971 (95% CI: 0.9961–0.9980)")
    print("  F1-Score: 0.9739 (95% CI: 0.9691–0.9784)")
    print("  MCC: 0.9690 (95% CI: 0.9638–0.9740)")


if __name__ == "__main__":
    main()
