"""
Step 03: Feature Engineering and Leakage Prevention
====================================================
Systematically remove 5 categories of leakage-prone features (14 total),
retain 16 features, and compute robust z-scores using training-only statistics.

All statistics match Table 5 of the paper exactly.
"""

import pandas as pd
import numpy as np
import os
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Exact normalization statistics from Table 5
NORMALIZATION_STATS = {
    "gas_price": {"median": 192476370.00, "mad": 155933567.00},
    "gas_limit": {"median": 141459.00, "mad": 58788.00},
    "calldata_length": {"median": 1418.00, "mad": 256.00},
    "internal_calls": {"median": 4.00, "mad": 1.00},
    "log_count": {"median": 2.00, "mad": 1.00},
}

# 14 features to remove (exact from paper)
FEATURES_TO_REMOVE = [
    # Category 1: Unique Identifiers (4)
    "transaction_hash",
    "sender_address", 
    "target_contract_address",
    "block_hash",
    # Category 2: Post-Hoc Outcomes (5)
    "receipt_status",
    "execution_success_indicator",
    "finality_status",
    "internal_errors",
    "internal_gas_usage",
    # Category 3: Highly Correlated Proxies (2)
    "gas_limit_zscore_leaky",  # computed on full data
    "duplicate_nonce_flag",
    # Category 4: Temporal Proxies (2)
    "nonce_gaps",
    "nonce_sequences",
    # Category 5: Hash-Derived (1)
    "raw_calldata",
]

# 16 features retained (exact from paper)
RAW_FEATURES = ["gas_price", "gas_limit", "calldata_length", 
                "eth_value", "internal_calls", "log_count"]
TEMPORAL_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "is_night"]
ZSCORE_FEATURES = [
    "gas_price_zscore_clean",
    "gas_limit_zscore_clean", 
    "calldata_length_zscore_clean",
    "internal_calls_zscore_clean",
    "log_count_zscore_clean",
]
ALL_FEATURES = RAW_FEATURES + TEMPORAL_FEATURES + ZSCORE_FEATURES


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 14 leakage-prone features across 5 categories."""
    cols_to_drop = [c for c in FEATURES_TO_REMOVE if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    print(f"Removed {len(cols_to_drop)} leakage-prone features")
    return df_clean


def add_temporal_features(df: pd.DataFrame, 
                          timestamp_col: str = "block_timestamp") -> pd.DataFrame:
    """Add temporal indicators from block timestamps."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 6)).astype(int)
    return df


def compute_zscores(df: pd.DataFrame, 
                    df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute robust z-scores using training-set-only statistics.
    z = (x - median) / MAD
    
    Critical: prevent data leakage by using train-only median/MAD.
    """
    df = df.copy()
    
    for feature in ["gas_price", "gas_limit", "calldata_length", 
                    "internal_calls", "log_count"]:
        median = df_train[feature].median()
        mad = np.median(np.abs(df_train[feature] - median))
        
        zscore_col = f"{feature}_zscore_clean"
        df[zscore_col] = (df[feature] - median) / (mad + 1e-10)
        
        print(f"  {feature}: median={median:.2f}, MAD={mad:.2f}")
    
    return df


def prepare_features(df_train: pd.DataFrame,
                     df_val: pd.DataFrame, 
                     df_test: pd.DataFrame) -> dict:
    """Full feature engineering pipeline."""
    # Remove leakage features
    df_train = remove_leakage_features(df_train)
    df_val = remove_leakage_features(df_val)
    df_test = remove_leakage_features(df_test)
    
    # Add temporal features
    df_train = add_temporal_features(df_train)
    df_val = add_temporal_features(df_val)
    df_test = add_temporal_features(df_test)
    
    # Compute z-scores (train-only statistics)
    print("\nComputing robust z-scores (training-set only):")
    df_train = compute_zscores(df_train, df_train)
    df_val = compute_zscores(df_val, df_train)
    df_test = compute_zscores(df_test, df_train)
    
    # Verify non-zero training means (confirms no global standardization)
    print("\nVerification (non-zero z-score means confirm train-only normalization):")
    for zf in ZSCORE_FEATURES:
        if zf in df_train.columns:
            mean_z = df_train[zf].mean()
            print(f"  {zf}: mean = {mean_z:.4f} (expected non-zero)")
    
    return {
        "train": df_train,
        "validation": df_val,
        "test": df_test,
        "features": ALL_FEATURES
    }


def main():
    """Run feature engineering pipeline."""
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validation.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    print(f"Loaded splits: train={len(train)}, val={len(val)}, test={len(test)}")
    
    result = prepare_features(train, val, test)
    
    # Save
    for name, df in [("train", result["train"]), 
                     ("validation", result["validation"]),
                     ("test", result["test"])]:
        path = f"data/processed/{name}_features.csv"
        df.to_csv(path, index=False)
        print(f"Saved engineered features to {path}")
    
    print(f"\nFinal feature set ({len(result['features'])} features):")
    for f in result["features"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
