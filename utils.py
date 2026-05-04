"""
Utility functions for Ethereum multisig anomaly detection.
All constants match the paper exactly.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Fixed random seeds for exact reproducibility
SEED_TRAINING = CONFIG["seeds"]["training"]
SEED_EVALUATION = CONFIG["seeds"]["evaluation"]
SEED_BOOTSTRAP = CONFIG["seeds"]["bootstrap"]

# =============================================================================
# Dataset Constants (Exact from Paper)
# =============================================================================
DATASET_SIZE = CONFIG["dataset"]["total_samples"]  # 100,000

SPLIT_RATIOS = {
    "train": CONFIG["dataset"]["temporal_split"]["train"]["ratio"],      # 0.60
    "validation": CONFIG["dataset"]["temporal_split"]["validation"]["ratio"],  # 0.15
    "test": CONFIG["dataset"]["temporal_split"]["test"]["ratio"]         # 0.25
}

# Block ranges for temporal validation (exact from Table 2)
BLOCK_RANGES = {
    "train": CONFIG["dataset"]["temporal_split"]["train"]["block_range"],
    "validation": CONFIG["dataset"]["temporal_split"]["validation"]["block_range"],
    "test": CONFIG["dataset"]["temporal_split"]["test"]["block_range"]
}

# Class distributions (exact from Table 2)
CLASS_DISTRIBUTIONS = {
    "train": {
        "total": CONFIG["dataset"]["temporal_split"]["train"]["samples"],
        "anomaly": CONFIG["dataset"]["temporal_split"]["train"]["anomalies"],
        "benign": CONFIG["dataset"]["temporal_split"]["train"]["benign"],
        "prevalence": CONFIG["dataset"]["temporal_split"]["train"]["prevalence_pct"] / 100
    },
    "validation": {
        "total": CONFIG["dataset"]["temporal_split"]["validation"]["samples"],
        "anomaly": CONFIG["dataset"]["temporal_split"]["validation"]["anomalies"],
        "benign": CONFIG["dataset"]["temporal_split"]["validation"]["benign"],
        "prevalence": CONFIG["dataset"]["temporal_split"]["validation"]["prevalence_pct"] / 100
    },
    "test": {
        "total": CONFIG["dataset"]["temporal_split"]["test"]["samples"],
        "anomaly": CONFIG["dataset"]["temporal_split"]["test"]["anomalies"],
        "benign": CONFIG["dataset"]["temporal_split"]["test"]["benign"],
        "prevalence": CONFIG["dataset"]["temporal_split"]["test"]["prevalence_pct"] / 100
    }
}

# =============================================================================
# Features (Exact from Paper)
# =============================================================================
# 14 features removed across 5 categories
LEAKAGE_REMOVED_FEATURES = CONFIG["leakage_removed"]

# 16 features retained
RAW_ONCHAIN_FEATURES = CONFIG["features"]["raw_onchain_metrics"]
TEMPORAL_FEATURES = CONFIG["features"]["temporal_indicators"]
ZSCORE_FEATURES = CONFIG["features"]["normalized_deviation"]

ALL_RETAINED_FEATURES = RAW_ONCHAIN_FEATURES + TEMPORAL_FEATURES + ZSCORE_FEATURES

# Training-set normalization statistics (exact from Table 5)
NORMALIZATION_STATS = CONFIG["normalization_stats"]

# =============================================================================
# Labeling Thresholds (Exact from Paper)
# =============================================================================
LABELING_CONFIG = CONFIG["labeling"]

# =============================================================================
# Model Hyperparameters (Exact from Paper)
# =============================================================================
LIGHTGBM_PARAMS = CONFIG["lightgbm"]
BASELINE_CONFIGS = CONFIG["baselines"]

# =============================================================================
# Evaluation Constants (Exact from Paper)
# =============================================================================
BOOTSTRAP_ITERATIONS = CONFIG["evaluation"]["bootstrap"]["n_iterations"]  # 1000
BONFERRONI_M = CONFIG["evaluation"]["statistical_tests"]["bonferroni_correction"]["m"]  # 7
ALPHA_ADJUSTED = CONFIG["evaluation"]["statistical_tests"]["bonferroni_correction"]["alpha_adjusted"]  # 0.05/7

# =============================================================================
# Hybrid Scoring Parameters
# =============================================================================
HYBRID_LAMBDA = CONFIG["hybrid_scoring"]["lambda"]

# =============================================================================
# Utility Functions
# =============================================================================

def compute_robust_zscore(x: np.ndarray, median: float, mad: float) -> np.ndarray:
    """
    Compute robust z-score using median and MAD.
    z = (x - median) / MAD
    
    All normalization parameters computed exclusively from training data.
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    return (x - median) / (mad + epsilon)


def compute_median_mad(x: np.ndarray) -> Tuple[float, float]:
    """Compute median and median absolute deviation."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return median, mad


def set_random_seeds():
    """Set all random seeds for exact reproducibility."""
    np.random.seed(SEED_TRAINING)
    import random
    random.seed(SEED_TRAINING)
    try:
        import torch
        torch.manual_seed(SEED_TRAINING)
        torch.cuda.manual_seed_all(SEED_TRAINING)
    except ImportError:
        pass


def assert_temporal_separation(df_train: pd.DataFrame, 
                                df_val: pd.DataFrame, 
                                df_test: pd.DataFrame,
                                block_col: str = "block_number") -> bool:
    """
    Verify strict temporal separation between splits.
    
    Assertions from Algorithm 1:
    - max(train_blocks) < min(val_blocks)
    - max(val_blocks) < min(test_blocks)
    """
    assert df_train[block_col].max() < df_val[block_col].min(), \
        "Temporal separation violated: train max >= val min"
    assert df_val[block_col].max() < df_test[block_col].min(), \
        "Temporal separation violated: val max >= test min"
    return True


def check_class_distribution(df: pd.DataFrame, label_col: str = "is_anomaly") -> Dict:
    """Check class distribution matches expected values."""
    total = len(df)
    anomalies = df[label_col].sum()
    benign = total - anomalies
    prevalence = anomalies / total
    return {
        "total": total,
        "anomaly": int(anomalies),
        "benign": int(benign),
        "prevalence": prevalence
    }


def format_metric_with_ci(mean: float, ci_lower: float, ci_upper: float, 
                          decimals: int = 4) -> str:
    """Format metric with 95% bootstrap CI."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} (95% CI: {fmt.format(ci_lower)}–{fmt.format(ci_upper)})"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function for hybrid scoring."""
    return 1.0 / (1.0 + np.exp(-x))
