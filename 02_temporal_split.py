"""
Step 02: Temporal Block-Height-Based Dataset Partitioning
=========================================================
Algorithm 1 from the paper: strict chronological train-validation-test split.

Enforces:
  max(B_i in D_train) < min(B_j in D_val) < min(B_k in D_test)

Exact splits from paper:
  - Train: 60% (blocks [23586544, 24469834]) — Jan-Aug 2023
  - Validation: 15% (blocks [24469842, 24663889]) — Aug-Oct 2023
  - Test: 25% (blocks [24663902, 24881458]) — Oct-Dec 2023
"""

import pandas as pd
import numpy as np
import os
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


def temporal_split(df: pd.DataFrame, 
                   block_col: str = "block_number",
                   label_col: str = "is_anomaly") -> dict:
    """
    Algorithm 1: Temporal Block-Height-Based Dataset Partitioning
    
    Steps:
    1. Sort dataset in ascending order of block numbers
    2. Compute partition indices: i_tr = floor(0.60 * n), i_val = floor(0.75 * n)
    3. Split: D_tr = D[1:i_tr], D_val = D[i_tr+1:i_val], D_te = D[i_val+1:n]
    4. Enforce: max(blocks in D_tr) < min(blocks in D_val) < min(blocks in D_te)
    """
    n = len(df)
    
    # Step 1: Sort by block number
    df_sorted = df.sort_values(by=block_col, ascending=True).reset_index(drop=True)
    
    # Step 2: Compute partition indices (exact from paper)
    i_tr = int(np.floor(0.60 * n))
    i_val = int(np.floor(0.75 * n))
    
    # Step 3: Split
    df_train = df_sorted.iloc[:i_tr].copy()
    df_val = df_sorted.iloc[i_tr:i_val].copy()
    df_test = df_sorted.iloc[i_val:].copy()
    
    # Step 4: Enforce temporal separation
    assert df_train[block_col].max() < df_val[block_col].min(), \
        "Assertion 1 failed: max(train_blocks) < min(val_blocks)"
    assert df_val[block_col].max() < df_test[block_col].min(), \
        "Assertion 2 failed: max(val_blocks) < min(test_blocks)"
    
    print("Temporal separation verified:")
    print(f"  Train:   blocks [{df_train[block_col].min()}, {df_train[block_col].max()}] "
          f"— {len(df_train)} samples")
    print(f"  Val:     blocks [{df_val[block_col].min()}, {df_val[block_col].max()}] "
          f"— {len(df_val)} samples")
    print(f"  Test:    blocks [{df_test[block_col].min()}, {df_test[block_col].max()}] "
          f"— {len(df_test)} samples")
    
    # Step 5: Verify class distributions (natural temporal variation)
    for name, split_df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        prevalence = split_df[label_col].mean() * 100
        print(f"  {name} prevalence: {prevalence:.2f}% "
              f"({split_df[label_col].sum()} anomalies / {len(split_df)} total)")
    
    return {
        "train": df_train,
        "validation": df_val,
        "test": df_test
    }


def save_splits(splits: dict, output_dir: str = "data/processed"):
    """Save temporal splits to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in splits.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {name} split to {path}")


def main():
    """Run temporal splitting pipeline."""
    df = pd.read_csv("data/processed/labeled_transactions.csv")
    print(f"Loaded {len(df)} labeled transactions")
    
    splits = temporal_split(df)
    save_splits(splits)
    
    print("\nTemporal validation complete.")
    print("Expected distributions from paper:")
    print("  Train: 10.69%, Validation: 7.67%, Test: 9.12%")


if __name__ == "__main__":
    main()
