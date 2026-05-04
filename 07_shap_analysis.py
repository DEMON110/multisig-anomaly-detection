"""
Step 07: SHAP Explainability Analysis
=====================================
Generate SHAP-based feature importance for global and local explainability.

Global: Mean absolute SHAP values (Table 16 from paper)
Local: Waterfall plots for individual transactions

Top 10 features (exact from paper):
  1. gas_limit: 1.391
  2. internal_calls: 1.160
  3. log_count: 0.925
  4. gas_limit_zscore_clean: 0.607
  5. gas_price: 0.471
  6. calldata_length: 0.416
  7. log_count_zscore_clean: 0.192
  8. internal_calls_zscore_clean: 0.182
  9. day_of_week: 0.179
  10. gas_price_zscore_clean: 0.165
"""

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Exact SHAP values from Table 16
PAPER_SHAP_IMPORTANCE = {
    "gas_limit": 1.391,
    "internal_calls": 1.160,
    "log_count": 0.925,
    "gas_limit_zscore_clean": 0.607,
    "gas_price": 0.471,
    "calldata_length": 0.416,
    "log_count_zscore_clean": 0.192,
    "internal_calls_zscore_clean": 0.182,
    "day_of_week": 0.179,
    "gas_price_zscore_clean": 0.165,
}


def compute_global_shap(model, X, feature_names):
    """
    Compute global feature importance using mean absolute SHAP values.
    
    Paper uses TreeExplainer for exact SHAP computation.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For binary classification, shap_values returns list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use anomaly class
    
    # Mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    
    return importance_df, shap_values, explainer


def plot_global_summary(importance_df, output_path="results/shap_global_summary.png"):
    """Plot global SHAP summary (Figure 8 from paper)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(10)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
    
    bars = ax.barh(range(len(top_features)), top_features["mean_abs_shap"].values, 
                   color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Global SHAP Feature Importance", fontsize=14, fontweight="bold")
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row["mean_abs_shap"] + 0.02, i, f"{row['mean_abs_shap']:.3f}", 
                va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved global SHAP summary to {output_path}")


def plot_local_waterfall(explainer, shap_values, X, instance_idx=0,
                         output_path="results/shap_local_waterfall.png"):
    """Plot local SHAP waterfall for a single transaction (Figure 11 from paper)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[instance_idx],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X.iloc[instance_idx] if hasattr(X, 'iloc') else X[instance_idx],
        feature_names=X.columns if hasattr(X, 'columns') else None
    ))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved local SHAP waterfall to {output_path}")


def verify_shap_alignment(importance_df):
    """Verify computed SHAP values align with paper."""
    print("\nSHAP Alignment Verification:")
    print(f"{'Feature':<30} {'Computed':>10} {'Paper':>10} {'Diff':>10}")
    print("-" * 62)
    
    for feature, paper_value in PAPER_SHAP_IMPORTANCE.items():
        computed_row = importance_df[importance_df["feature"] == feature]
        if not computed_row.empty:
            computed_value = computed_row["mean_abs_shap"].values[0]
            diff = abs(computed_value - paper_value)
            status = "✓" if diff < 0.05 else "⚠"
            print(f"{feature:<30} {computed_value:>10.3f} {paper_value:>10.3f} "
                  f"{diff:>10.3f} {status}")


def main():
    """Run SHAP analysis pipeline."""
    test = pd.read_csv("data/processed/test_features.csv")
    features = CONFIG["features"]["raw_onchain_metrics"] + \
               CONFIG["features"]["temporal_indicators"] + \
               CONFIG["features"]["normalized_deviation"]
    
    X_test = test[features]
    
    # Load LightGBM model
    with open("models/lightgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    print("Computing SHAP values with TreeExplainer...")
    importance_df, shap_values, explainer = compute_global_shap(model, X_test, features)
    
    print("\nGlobal Feature Importance (Top 10):")
    print(importance_df.head(10).to_string(index=False))
    
    # Verify against paper
    verify_shap_alignment(importance_df)
    
    # Generate plots
    print("\nGenerating SHAP visualizations...")
    plot_global_summary(importance_df)
    plot_local_waterfall(explainer, shap_values, X_test, instance_idx=0)
    
    # Save importance table
    os.makedirs("results", exist_ok=True)
    importance_df.to_csv("results/shap_feature_importance.csv", index=False)
    print("\nSaved SHAP results to results/")


if __name__ == "__main__":
    main()
